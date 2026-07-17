"""Shared building blocks for the Overcooked V3 macro-action MAPPO baselines.

The three trainers intentionally keep their rollout semantics in separate
files. This module contains only representation, optimization, and return
calculation code that should remain identical between experiments.
"""

from functools import partial
import json
import os
from typing import Callable, Dict, Tuple

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState

import jaxmarl
from jaxmarl.environments import spaces
from jaxmarl.wrappers.baselines import JaxMARLWrapper, LogWrapper


_RUN_CONTEXT = {}


class MacroWorldStateWrapper(JaxMARLWrapper):
    """Add macro context for actors and a global state for the MAPPO critic."""

    def __init__(self, env):
        super().__init__(env)
        base_shape = env.observation_space(env.agents[0]).shape
        self.base_obs_size = int(np.prod(base_shape))
        self.actor_obs_size = self.base_obs_size + env.num_macro_actions + 2
        self._world_state_size = self.actor_obs_size * env.num_agents + env.num_agents

    def _augment(self, obs, state):
        macro_one_hot = jax.nn.one_hot(
            state.current_macro_actions,
            self._env.num_macro_actions,
            dtype=jnp.float32,
        )
        macro_done = state.macro_action_done.astype(jnp.float32)
        macro_progress = state.macro_step_count.astype(jnp.float32) / max(
            self._env.max_macro_steps, 1
        )

        augmented = {}
        for index, agent in enumerate(self._env.agents):
            augmented[agent] = jnp.concatenate(
                (
                    obs[agent].reshape(-1).astype(jnp.float32),
                    macro_one_hot[index],
                    macro_done[index, None],
                    macro_progress[index, None],
                )
            )

        global_obs = jnp.concatenate(
            [augmented[agent] for agent in self._env.agents]
        )
        augmented["world_state"] = jnp.concatenate(
            (
                jnp.repeat(global_obs[None, :], self._env.num_agents, axis=0),
                jnp.eye(self._env.num_agents, dtype=jnp.float32),
            ),
            axis=-1,
        )
        augmented["macro_done"] = state.macro_action_done
        augmented["current_macro"] = state.current_macro_actions
        available_actions = self._env.get_avail_actions(state)
        augmented["action_mask"] = jnp.stack(
            [available_actions[agent] for agent in self._env.agents], axis=0
        )
        return augmented

    @partial(jax.jit, static_argnums=0)
    def reset(self, key):
        obs, state = self._env.reset(key)
        return self._augment(obs, state), state

    @partial(jax.jit, static_argnums=0)
    def step(self, key, state, action):
        obs, next_state, reward, done, info = self._env.step(key, state, action)
        return self._augment(obs, next_state), next_state, reward, done, info

    def observation_space(self, agent_id=""):
        del agent_id
        return spaces.Box(0.0, 1.0, (self.actor_obs_size,), dtype=jnp.float32)

    def world_state_size(self):
        return self._world_state_size


class Actor(nn.Module):
    action_dim: int
    hidden_size: int

    @nn.compact
    def __call__(self, obs):
        x = nn.Dense(
            self.hidden_size,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(obs)
        x = nn.tanh(x)
        x = nn.Dense(
            self.hidden_size,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = nn.tanh(x)
        return nn.Dense(
            self.action_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
        )(x)


class ReplanActor(nn.Module):
    """Actor with separate macro-selection and continue/replan heads."""

    action_dim: int
    hidden_size: int

    @nn.compact
    def __call__(self, obs):
        x = nn.Dense(
            self.hidden_size,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(obs)
        x = nn.tanh(x)
        x = nn.Dense(
            self.hidden_size,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = nn.tanh(x)
        macro_logits = nn.Dense(
            self.action_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
            name="macro_head",
        )(x)
        replan_logits = nn.Dense(
            2,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
            name="replan_head",
        )(x)
        return macro_logits, replan_logits


class Critic(nn.Module):
    hidden_size: int

    @nn.compact
    def __call__(self, world_state):
        x = nn.Dense(
            self.hidden_size,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(world_state)
        x = nn.tanh(x)
        x = nn.Dense(
            self.hidden_size,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = nn.tanh(x)
        return nn.Dense(
            1,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
        )(x).squeeze(-1)


def build_env(config: Dict):
    env = jaxmarl.make(config["ENV_NAME"], **config.get("ENV_KWARGS", {}))
    env = MacroWorldStateWrapper(env)
    return LogWrapper(env)


def batchify(values: Dict, agents, num_actors: int):
    """Convert agent-keyed, environment-batched values to agent-major actors."""
    return jnp.stack([values[agent] for agent in agents]).reshape(
        (num_actors,) + jnp.asarray(values[agents[0]]).shape[1:]
    )


def unbatchify(values, agents, num_envs: int):
    values = values.reshape((len(agents), num_envs) + values.shape[1:])
    return {agent: values[index] for index, agent in enumerate(agents)}


def metadata_batch(values, num_actors: int):
    """Convert arrays shaped [environment, agent, ...] to agent-major actors."""
    axes = (1, 0) + tuple(range(2, values.ndim))
    return values.transpose(axes).reshape((num_actors,) + values.shape[2:])


def masked_mean(values, mask):
    mask = mask.astype(values.dtype)
    return jnp.sum(values * mask) / jnp.maximum(jnp.sum(mask), 1.0)


def normalize_masked(values, mask):
    mean = masked_mean(values, mask)
    variance = masked_mean(jnp.square(values - mean), mask)
    return (values - mean) / jnp.sqrt(variance + 1e-8)


def clipped_actor_loss(
    new_log_prob,
    old_log_prob,
    advantages,
    entropy,
    mask,
    clip_eps: float,
    entropy_coefficient: float,
):
    advantages = normalize_masked(advantages, mask)
    log_ratio = new_log_prob - old_log_prob
    ratio = jnp.exp(log_ratio)
    unclipped = ratio * advantages
    clipped = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    policy_loss = -masked_mean(jnp.minimum(unclipped, clipped), mask)
    entropy_loss = masked_mean(entropy, mask)
    total = policy_loss - entropy_coefficient * entropy_loss
    metrics = {
        "policy_loss": policy_loss,
        "entropy": entropy_loss,
        "approx_kl": masked_mean((ratio - 1.0) - log_ratio, mask),
        "clip_fraction": masked_mean(
            jnp.abs(ratio - 1.0) > clip_eps, mask
        ),
    }
    return total, metrics


def clipped_value_loss(new_value, old_value, targets, mask, clip_eps: float):
    clipped_value = old_value + jnp.clip(
        new_value - old_value, -clip_eps, clip_eps
    )
    loss = jnp.maximum(
        jnp.square(new_value - targets),
        jnp.square(clipped_value - targets),
    )
    return 0.5 * masked_mean(loss, mask)


def calculate_gae(reward, done, value, last_value, gamma: float, gae_lambda: float):
    """Primitive-time generalized advantage estimation."""

    def step(carry, transition):
        gae, next_value = carry
        step_reward, step_done, step_value = transition
        not_done = 1.0 - step_done
        delta = step_reward + gamma * next_value * not_done - step_value
        gae = delta + gamma * gae_lambda * not_done * gae
        return (gae, step_value), gae

    _, advantages = jax.lax.scan(
        step,
        (jnp.zeros_like(last_value), last_value),
        (reward, done, value),
        reverse=True,
    )
    return advantages, advantages + value


def calculate_smdp_gae(
    reward,
    duration,
    done,
    value,
    valid,
    gamma: float,
    gae_lambda: float,
):
    """Event-time GAE for variable-duration macro transitions.

    ``reward`` must already be discounted within each macro. Invalid fixed-size
    buffer slots are skipped without changing the reverse-scan carry.
    """

    def step(carry, transition):
        gae, next_value = carry
        step_reward, step_duration, step_done, step_value, step_valid = transition
        discount = jnp.power(gamma, step_duration)
        trace_discount = jnp.power(gamma * gae_lambda, step_duration)
        not_done = 1.0 - step_done
        delta = step_reward + discount * next_value * not_done - step_value
        candidate_gae = delta + trace_discount * not_done * gae
        next_gae = jnp.where(step_valid, candidate_gae, gae)
        next_value = jnp.where(step_valid, step_value, next_value)
        output = jnp.where(step_valid, candidate_gae, 0.0)
        return (next_gae, next_value), output

    zeros = jnp.zeros_like(value[0])
    _, advantages = jax.lax.scan(
        step,
        (zeros, zeros),
        (reward, duration, done, value, valid),
        reverse=True,
    )
    return advantages, advantages + value


def make_train_state(network, params, config: Dict, total_updates: int):
    if config.get("ANNEAL_LR", True):
        learning_rate = optax.linear_schedule(
            config["LR"],
            0.0,
            max(total_updates * config["UPDATE_EPOCHS"] * config["NUM_MINIBATCHES"], 1),
        )
    else:
        learning_rate = config["LR"]
    optimizer = optax.chain(
        optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
        optax.adam(learning_rate, eps=1e-5),
    )
    return TrainState.create(apply_fn=network.apply, params=params, tx=optimizer)


def initialize_actor_critic(
    actor,
    critic,
    actor_input,
    critic_input,
    rng,
    config: Dict,
):
    """Initialize actor, critic, and optimizer state on the training device."""
    rng, actor_rng, critic_rng = jax.random.split(rng, 3)
    actor_params = actor.init(actor_rng, actor_input)
    critic_params = critic.init(critic_rng, critic_input)
    return (
        rng,
        make_train_state(actor, actor_params, config, config["NUM_UPDATES"]),
        make_train_state(critic, critic_params, config, config["NUM_UPDATES"]),
    )


def categorical(logits):
    return distrax.Categorical(logits=logits)


def masked_categorical(logits, action_mask):
    """Categorical policy with impossible actions removed."""
    return categorical(jnp.where(action_mask.astype(jnp.bool_), logits, -1e9))


def add_annealed_shaped_reward(
    reward: Dict,
    shaped_reward: Dict,
    primitive_timestep,
    shaping_horizon: float,
):
    """Add shaped rewards with a linear 1-to-0 primitive-step anneal."""
    horizon = jnp.asarray(shaping_horizon, dtype=jnp.float32)
    coefficient = jnp.where(
        horizon > 0,
        jnp.clip(
            1.0
            - jnp.asarray(primitive_timestep, dtype=jnp.float32)
            / jnp.maximum(horizon, 1.0),
            0.0,
            1.0,
        ),
        0.0,
    )
    combined_reward = jax.tree.map(
        lambda sparse, shaped: sparse + coefficient * shaped,
        reward,
        shaped_reward,
    )
    return combined_reward, coefficient


def deterministic_evaluation(
    env,
    actor_params,
    select_actions: Callable,
    config: Dict,
    key,
):
    """Evaluate a deterministic policy for one full environment horizon."""
    num_envs = int(config.get("NUM_EVAL_ENVS", 8))
    num_actors = num_envs * env.num_agents
    reset_keys = jax.random.split(key, num_envs)
    obs, env_state = jax.vmap(env.reset)(reset_keys)

    def eval_step(carry, _):
        obs, env_state, rng = carry
        obs_batch = batchify(obs, env.agents, num_actors)
        action_mask = metadata_batch(obs["action_mask"], num_actors).astype(
            jnp.bool_
        )
        macro_done = metadata_batch(obs["macro_done"], num_actors)
        current_macro = metadata_batch(obs["current_macro"], num_actors)
        action = select_actions(
            actor_params,
            obs_batch,
            action_mask,
            macro_done,
            current_macro,
        )
        env_action = unbatchify(action, env.agents, num_envs)
        rng, step_rng = jax.random.split(rng)
        step_keys = jax.random.split(step_rng, num_envs)
        next_obs, next_env_state, reward, _, _ = jax.vmap(
            env.step, in_axes=(0, 0, 0)
        )(step_keys, env_state, env_action)
        mean_team_reward = jnp.mean(
            jnp.stack([reward[agent] for agent in env.agents], axis=-1),
            axis=-1,
        )
        return (next_obs, next_env_state, rng), mean_team_reward

    _, rewards = jax.lax.scan(
        eval_step,
        (obs, env_state, key),
        None,
        int(config.get("ENV_KWARGS", {}).get("max_steps", 400)),
    )
    return jnp.mean(jnp.sum(rewards, axis=0))


def _atomic_write(path, data: bytes):
    temporary_path = f"{path}.tmp-{os.getpid()}"
    with open(temporary_path, "wb") as stream:
        stream.write(data)
    os.replace(temporary_path, path)


def _atomic_savez(path, leaves):
    temporary_path = f"{path}.tmp-{os.getpid()}"
    with open(temporary_path, "wb") as stream:
        np.savez(stream, *[np.asarray(leaf) for leaf in leaves])
    os.replace(temporary_path, path)


def _host_log_metrics(update_index, metrics, steps_per_update):
    wandb = _RUN_CONTEXT.get("wandb")
    if wandb is None or wandb.run is None:
        return
    completed_updates = int(np.asarray(update_index)) + 1
    values = {
        key: float(np.asarray(value)) for key, value in metrics.items()
    }
    values["primitive_steps"] = completed_updates * int(
        np.asarray(steps_per_update)
    )
    wandb.log(values, step=values["primitive_steps"])


def emit_live_metrics(update_index, metrics, steps_per_update: int, config: Dict):
    interval = max(int(config.get("LOG_INTERVAL_UPDATES", 1)), 1)
    completed_updates = update_index + 1
    should_log = (completed_updates % interval == 0) | (
        completed_updates == int(config["NUM_UPDATES"])
    )

    def log(_):
        jax.debug.callback(
            _host_log_metrics,
            update_index,
            metrics,
            jnp.asarray(steps_per_update),
            ordered=True,
        )
        return jnp.int32(0)

    jax.lax.cond(should_log, log, lambda _: jnp.int32(0), operand=None)


def _host_save_checkpoint(completed_updates, runner):
    output_dir = _RUN_CONTEXT.get("output_dir")
    if output_dir is None:
        return
    completed_updates = int(np.asarray(completed_updates))
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"checkpoint_{completed_updates:08d}.npz"
    leaves = jax.tree.leaves(runner)
    _atomic_savez(checkpoint_path, leaves)
    latest = json.dumps(
        {"completed_updates": completed_updates, "path": checkpoint_path.name}
    ).encode("utf-8")
    _atomic_write(checkpoint_dir / "latest.json", latest)


def maybe_checkpoint(update_index, runner, config: Dict):
    interval = int(config.get("CHECKPOINT_INTERVAL_UPDATES", 0))
    if interval <= 0 or not config.get("SAVE_PATH"):
        return
    completed_updates = update_index + 1
    should_save = (completed_updates % interval == 0) | (
        completed_updates == int(config["NUM_UPDATES"])
    )

    def save(_):
        jax.debug.callback(
            _host_save_checkpoint,
            completed_updates,
            runner,
            ordered=True,
        )
        return jnp.int32(0)

    jax.lax.cond(should_save, save, lambda _: jnp.int32(0), operand=None)


def _host_maybe_save_best(eval_return, actor_params, critic_params):
    output_dir = _RUN_CONTEXT.get("output_dir")
    if output_dir is None:
        return
    score = float(np.asarray(eval_return))
    if score <= _RUN_CONTEXT.get("best_eval_return", -np.inf):
        return
    from jaxmarl.wrappers.baselines import save_params

    _RUN_CONTEXT["best_eval_return"] = score
    save_params(actor_params, output_dir / "best_actor.safetensors")
    save_params(critic_params, output_dir / "best_critic.safetensors")
    _atomic_write(
        output_dir / "best_eval.json",
        json.dumps({"eval_return": score}).encode("utf-8"),
    )


def maybe_evaluate_and_save_best(
    update_index,
    actor_state,
    critic_state,
    evaluate: Callable,
    config: Dict,
):
    interval = int(config.get("EVAL_INTERVAL_UPDATES", 0))
    if interval <= 0:
        return jnp.asarray(jnp.nan, dtype=jnp.float32)
    completed_updates = update_index + 1
    should_evaluate = (completed_updates % interval == 0) | (
        completed_updates == int(config["NUM_UPDATES"])
    )

    def run_eval(_):
        score = evaluate(actor_state.params, completed_updates)
        if config.get("SAVE_PATH"):
            jax.debug.callback(
                _host_maybe_save_best,
                score,
                actor_state.params,
                critic_state.params,
                ordered=True,
            )
        return score

    return jax.lax.cond(
        should_evaluate,
        run_eval,
        lambda _: jnp.asarray(jnp.nan, dtype=jnp.float32),
        operand=None,
    )


def restore_training_checkpoint(runner, config: Dict):
    """Restore a full runner and return its completed-update index."""
    resume_from = config.get("RESUME_FROM")
    if not resume_from:
        return runner, 0
    from pathlib import Path

    path = Path(resume_from)
    if path.is_dir():
        metadata = json.loads((path / "latest.json").read_text())
        checkpoint_path = path / metadata["path"]
        completed_updates = int(metadata["completed_updates"])
    else:
        checkpoint_path = path
        completed_updates = int(path.stem.rsplit("_", 1)[-1])
    target_leaves, tree_definition = jax.tree.flatten(runner)
    with np.load(checkpoint_path, allow_pickle=False) as archive:
        restored_leaves = [archive[f"arr_{i}"] for i in range(len(archive.files))]
    if len(restored_leaves) != len(target_leaves):
        raise ValueError("Checkpoint structure does not match the initialized runner")
    for restored_leaf, target_leaf in zip(restored_leaves, target_leaves):
        if restored_leaf.shape != np.asarray(target_leaf).shape:
            raise ValueError("Checkpoint array shapes do not match the initialized runner")
    restored = jax.tree.unflatten(tree_definition, restored_leaves)
    if completed_updates > int(config["NUM_UPDATES"]):
        raise ValueError("Checkpoint is beyond the configured training budget")
    return restored, completed_updates


def initialize_config(config: Dict, env):
    config = dict(config)
    config["NUM_ACTORS"] = env.num_agents * int(config["NUM_ENVS"])
    config["NUM_UPDATES"] = int(config["TOTAL_TIMESTEPS"]) // (
        int(config["NUM_STEPS"]) * int(config["NUM_ENVS"])
    )
    steps_per_update = int(config["NUM_STEPS"]) * int(config["NUM_ENVS"])
    if int(config["TOTAL_TIMESTEPS"]) % steps_per_update != 0:
        raise ValueError(
            "TOTAL_TIMESTEPS must be divisible by NUM_STEPS * NUM_ENVS; "
            "refusing to silently truncate the training budget"
        )
    config["BATCH_SIZE"] = config["NUM_ACTORS"] * int(config["NUM_STEPS"])
    if config["BATCH_SIZE"] % int(config["NUM_MINIBATCHES"]) != 0:
        raise ValueError("BATCH_SIZE must be divisible by NUM_MINIBATCHES")
    return config


def minibatches(rng, batch, num_minibatches: int):
    """Shuffle a flat batch and add a leading minibatch axis."""
    size = jax.tree.leaves(batch)[0].shape[0]
    permutation = jax.random.permutation(rng, size)
    shuffled = jax.tree.map(lambda x: x[permutation], batch)
    return jax.tree.map(
        lambda x: x.reshape((num_minibatches, -1) + x.shape[1:]), shuffled
    )


def update_ppo(
    rng,
    actor_state,
    critic_state,
    batch,
    actor_loss_fn: Callable,
    config: Dict,
):
    """Run shared shuffled PPO epochs with a variant-specific actor loss."""

    def update_minibatch(states, minibatch):
        current_actor_state, current_critic_state = states

        actor_grad_fn = jax.value_and_grad(actor_loss_fn, has_aux=True)
        (actor_loss, actor_metrics), actor_grads = actor_grad_fn(
            current_actor_state.params, minibatch
        )

        def critic_loss_fn(params):
            prediction = current_critic_state.apply_fn(
                params, minibatch["world_state"]
            )
            value_loss = clipped_value_loss(
                prediction,
                minibatch["old_value"],
                minibatch["target"],
                minibatch["loss_mask"],
                config["CLIP_EPS"],
            )
            return config["VF_COEF"] * value_loss, value_loss

        (critic_loss, value_loss), critic_grads = jax.value_and_grad(
            critic_loss_fn, has_aux=True
        )(current_critic_state.params)
        current_actor_state = current_actor_state.apply_gradients(grads=actor_grads)
        current_critic_state = current_critic_state.apply_gradients(
            grads=critic_grads
        )
        metrics = {
            **actor_metrics,
            "actor_loss": actor_loss,
            "critic_loss": critic_loss,
            "value_loss": value_loss,
        }
        return (current_actor_state, current_critic_state), metrics

    def update_epoch(carry, unused):
        epoch_rng, current_actor_state, current_critic_state = carry
        epoch_rng, shuffle_rng = jax.random.split(epoch_rng)
        epoch_minibatches = minibatches(
            shuffle_rng, batch, int(config["NUM_MINIBATCHES"])
        )
        (current_actor_state, current_critic_state), metrics = jax.lax.scan(
            update_minibatch,
            (current_actor_state, current_critic_state),
            epoch_minibatches,
        )
        return (epoch_rng, current_actor_state, current_critic_state), metrics

    (rng, actor_state, critic_state), metrics = jax.lax.scan(
        update_epoch,
        (rng, actor_state, critic_state),
        None,
        int(config["UPDATE_EPOCHS"]),
    )
    metrics = jax.tree.map(jnp.mean, metrics)
    return rng, actor_state, critic_state, metrics


def run_experiment(config: Dict, make_train: Callable, experiment_name: str):
    """Run seeds with live metrics and local-only policy persistence."""
    from pathlib import Path

    import wandb
    from omegaconf import OmegaConf

    from jaxmarl.wrappers.baselines import save_params

    rngs = jax.random.split(
        jax.random.PRNGKey(int(config["SEED"])), int(config["NUM_SEEDS"])
    )
    results = []
    for seed_index, rng in enumerate(rngs):
        layout_name = config.get("ENV_KWARGS", {}).get("layout", "unknown-layout")
        output_dir = None
        if config.get("SAVE_PATH"):
            output_dir = (
                Path(config["SAVE_PATH"])
                / experiment_name
                / f"seed_{seed_index}"
            )
            output_dir.mkdir(parents=True, exist_ok=True)
            OmegaConf.save(OmegaConf.create(config), output_dir / "config.yaml")

        wandb.init(
            entity=config.get("ENTITY", ""),
            project=config.get("PROJECT", ""),
            tags=["MAPPO", "macro-actions", experiment_name, layout_name],
            name=(
                f"{experiment_name}_{config['ENV_NAME']}_{layout_name}"
                f"_seed_{seed_index}"
            ),
            config=config,
            mode=config.get("WANDB_MODE", "disabled"),
            save_code=False,
            reinit=True,
        )
        _RUN_CONTEXT.clear()
        _RUN_CONTEXT.update(
            {
                "wandb": wandb,
                "output_dir": output_dir,
                "best_eval_return": -np.inf,
            }
        )

        result = make_train(config)(rng)
        result = jax.block_until_ready(result)
        results.append(result)

        if output_dir is not None:
            actor_state, critic_state = result["runner_state"][:2]
            save_params(actor_state.params, output_dir / "final_actor.safetensors")
            save_params(critic_state.params, output_dir / "final_critic.safetensors")

        wandb.finish()
        _RUN_CONTEXT.clear()

    return results[0] if len(results) == 1 else results
