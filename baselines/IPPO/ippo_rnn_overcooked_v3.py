"""IPPO (RNN) training script for Overcooked V3.

Closely follows ippo_cnn_overcooked.py (the original, working IPPO):
  - Fully JIT'd training loop via jax.lax.scan over all updates
  - Reward shaping anneal via optax.linear_schedule (REW_SHAPING_HORIZON)
  - jax.debug.callback for W&B logging inside the scan
  - update_step counter tracked in runner_state

GRU replaces nn.scan (broken in Flax 0.10.4 / JAX 0.4.38) with:
  - Pre-computed Dense input projections outside lax.scan
  - Raw weight matrices (self.param) for recurrent ops inside lax.scan
  - W&B sweep support with Bayesian optimization
"""
import sys
import os
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import Callable, Sequence, NamedTuple, Any, Dict
from flax.training.train_state import TrainState
from flax import serialization
import distrax
import jaxmarl
from jaxmarl.wrappers.baselines import LogWrapper
from jaxmarl.viz.overcooked_v3_visualizer import OvercookedV3Visualizer
import hydra
from omegaconf import OmegaConf
import wandb
from baselines.IC3Net.monitor import TrainingMonitorInterface
from baselines.IPPO import wandb_logging as wandb_logs
from baselines.IPPO.ippo_cnn_overcooked_v3 import (
    EVENT_METRIC_NAMES,
    _log_reward_structure_to_wandb,
)


def _save_model_params(params, save_path):
    os.makedirs(save_path, exist_ok=True)
    model_path = os.path.join(save_path, "model.msgpack")
    with open(model_path, "wb") as f:
        f.write(serialization.to_bytes({"params": params}))
    return model_path


def _log_training_metrics(metric: Dict[str, Any]) -> None:
    wandb_logs.log_training_metrics(metric)


def _log_history_table_to_wandb() -> None:
    wandb_logs.log_history_table_to_wandb()


# ── Network Architecture ───────────────────────────────────────────────


class CNN(nn.Module):
    """CNN encoder for grid observations (matches v2 architecture)."""
    output_size: int = 64
    activation: Callable[..., Any] = nn.relu

    @nn.compact
    def __call__(self, x, train=False):
        # 1x1 convs for channel mixing
        x = nn.Conv(features=128, kernel_size=(1, 1),
                     kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(x)
        x = self.activation(x)
        x = nn.Conv(features=128, kernel_size=(1, 1),
                     kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(x)
        x = self.activation(x)
        x = nn.Conv(features=8, kernel_size=(1, 1),
                     kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(x)
        x = self.activation(x)
        # Spatial convs
        x = nn.Conv(features=16, kernel_size=(3, 3),
                     kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(x)
        x = self.activation(x)
        x = nn.Conv(features=32, kernel_size=(3, 3),
                     kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(x)
        x = self.activation(x)
        x = nn.Conv(features=32, kernel_size=(3, 3),
                     kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(x)
        x = self.activation(x)
        # Flatten: (batch, H', W', C) -> (batch, H'*W'*C)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(features=self.output_size,
                      kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(x)
        x = self.activation(x)
        return x


class ActorCriticRNN(nn.Module):
    """Actor-Critic with CNN encoder and GRU via jax.lax.scan.

    Avoids Flax nn.scan bug by pre-computing input projections (Dense) for
    all timesteps OUTSIDE scan and using raw weight matrices inside scan.
    """
    action_dim: Sequence[int]
    config: Dict

    @nn.compact
    def __call__(self, hidden, x):
        obs, dones = x
        T = obs.shape[0]
        hidden_dim = self.config.get("GRU_HIDDEN_DIM", 128)

        if self.config.get("ACTIVATION", "relu") == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        # CNN embed: vmap over T, CNN handles actor batch dim
        embed_model = CNN(output_size=hidden_dim, activation=activation)
        embedding = jax.vmap(embed_model)(obs)  # (T, num_actors, hidden_dim)
        embedding = nn.LayerNorm()(embedding)

        # GRU input projections — Dense applied to all timesteps at once (outside scan)
        num_actors = obs.shape[1]
        flat_emb = embedding.reshape(-1, hidden_dim)

        Wi_z = nn.Dense(hidden_dim, use_bias=False, name='gru_Wi_z')(flat_emb)
        Wi_r = nn.Dense(hidden_dim, use_bias=False, name='gru_Wi_r')(flat_emb)
        Wi_h = nn.Dense(hidden_dim, use_bias=False, name='gru_Wi_h')(flat_emb)

        Wi_z = Wi_z.reshape(T, num_actors, hidden_dim)
        Wi_r = Wi_r.reshape(T, num_actors, hidden_dim)
        Wi_h = Wi_h.reshape(T, num_actors, hidden_dim)

        # Recurrent weight matrices as raw params (safe inside lax.scan)
        Wh_z = self.param('gru_Wh_z', nn.initializers.orthogonal(), (hidden_dim, hidden_dim))
        Wh_r = self.param('gru_Wh_r', nn.initializers.orthogonal(), (hidden_dim, hidden_dim))
        Wh_h = self.param('gru_Wh_h', nn.initializers.orthogonal(), (hidden_dim, hidden_dim))
        b_z = self.param('gru_b_z', nn.initializers.zeros_init(), (hidden_dim,))
        b_r = self.param('gru_b_r', nn.initializers.zeros_init(), (hidden_dim,))
        b_h = self.param('gru_b_h', nn.initializers.zeros_init(), (hidden_dim,))

        def _gru_step(h, inp):
            wiz_t, wir_t, wih_t, done_t = inp
            # Reset hidden on episode boundaries
            h = jnp.where(done_t[:, None], jnp.zeros_like(h), h)
            z = jax.nn.sigmoid(wiz_t + h @ Wh_z + b_z)
            r = jax.nn.sigmoid(wir_t + h @ Wh_r + b_r)
            h_hat = jnp.tanh(wih_t + (r * h) @ Wh_h + b_h)
            new_h = (1 - z) * h + z * h_hat
            return new_h, new_h

        final_hidden, embedding = jax.lax.scan(
            _gru_step, hidden, (Wi_z, Wi_r, Wi_h, dones)
        )  # embedding: (T, num_actors, hidden_dim)

        # Actor head — applied directly to (T, num_actors, hidden_dim)
        actor_mean = nn.Dense(
            self.config.get("FC_DIM_SIZE", 128),
            kernel_init=orthogonal(2), bias_init=constant(0.0),
        )(embedding)
        actor_mean = nn.relu(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0),
        )(actor_mean)

        pi = distrax.Categorical(logits=actor_mean)

        # Critic head — applied directly to (T, num_actors, hidden_dim)
        critic = nn.Dense(
            self.config.get("FC_DIM_SIZE", 128),
            kernel_init=orthogonal(2), bias_init=constant(0.0),
        )(embedding)
        critic = nn.relu(critic)
        critic = nn.Dense(
            1, kernel_init=orthogonal(1.0), bias_init=constant(0.0),
        )(critic)

        return final_hidden, pi, jnp.squeeze(critic, axis=-1)


# ── Utilities ──────────────────────────────────────────────────────────


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray


def batchify(x: dict, agent_list, num_actors):
    x = jnp.stack([x[a] for a in agent_list])
    return x.reshape((num_actors, -1))


def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}


def get_rollout(params, config):
    """Run one sampled episode after training for GIF generation."""
    env = jaxmarl.make(config["ENV_NAME"], **config.get("ENV_KWARGS", {}))
    hidden_dim = config.get("GRU_HIDDEN_DIM", 128)
    network = ActorCriticRNN(env.action_space(env.agents[0]).n, config=config)

    key = jax.random.PRNGKey(0)
    key, key_r = jax.random.split(key)
    obs, state = env.reset(key_r)
    hstate = jnp.zeros((env.num_agents, hidden_dim))
    done_batch = jnp.zeros((env.num_agents,), dtype=bool)
    state_seq = [state]
    done = False

    while not done:
        key, key_a, key_s = jax.random.split(key, 3)
        obs_batch = jnp.stack([obs[a] for a in env.agents]).reshape(
            -1, *env.observation_space(env.agents[0]).shape
        )
        ac_in = (obs_batch[np.newaxis, :], done_batch[np.newaxis, :])
        hstate, pi, _ = network.apply(params, hstate, ac_in)
        action = pi.sample(seed=key_a).squeeze(0)
        env_act = {a: action[i] for i, a in enumerate(env.agents)}
        obs, state, reward, done, info = env.step(key_s, state, env_act)
        done_batch = jnp.array([done[a] for a in env.agents])
        done = done["__all__"]
        state_seq.append(state)

    return state_seq


# ── Training ───────────────────────────────────────────────────────────


def make_train(config):
    """Create the fully JIT'd IPPO training function for overcooked_v3.

    Mirrors the original ippo_cnn_overcooked.py structure:
      - jax.lax.scan over _update_step (all updates compiled)
      - optax.linear_schedule for reward shaping anneal
      - jax.debug.callback for W&B logging
    """
    env = jaxmarl.make(config["ENV_NAME"], **config.get("ENV_KWARGS", {}))

    config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ACTORS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )

    obs_shape = env.observation_space(env.agents[0]).shape
    action_dim = env.action_space(env.agents[0]).n
    num_agents = env.num_agents

    env = LogWrapper(env, replace_info=False)

    # Shaped reward coefficient:
    # v3 shaped rewards are 0.1-0.3; v1 baseline uses 3.0-5.0.
    # COEFF=30 brings v3 to parity (30x0.1=3.0), matching v1 gradient signal.
    shaped_reward_coeff = config.get("SHAPED_REWARD_COEFF", 30.0)
    rew_shaping_min_coeff = config.get("REW_SHAPING_MIN_COEFF", 0.0)

    # Reward shaping anneal: linearly decay from 1.0 to 0.0, with an optional
    # floor to match the CNN experiment path.
    rew_shaping_anneal = optax.linear_schedule(
        init_value=1.0,
        end_value=0.0,
        transition_steps=config["REW_SHAPING_HORIZON"],
    )

    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    def train(rng):
        # INIT NETWORK
        hidden_dim = config.get("GRU_HIDDEN_DIM", 128)
        network = ActorCriticRNN(action_dim, config=config)

        rng, _rng = jax.random.split(rng)
        init_x = (
            jnp.zeros((1, config["NUM_ENVS"], *obs_shape)),
            jnp.zeros((1, config["NUM_ENVS"])),
        )
        init_hstate = jnp.zeros((config["NUM_ENVS"], hidden_dim))
        network_params = network.init(_rng, init_hstate, init_x)

        # Optional warm-start: initialize from a previously trained checkpoint.
        load_path = config.get("LOAD_PATH")
        if load_path:
            model_file = os.path.join(load_path, "model.msgpack")
            with open(model_file, "rb") as f:
                restored = serialization.msgpack_restore(f.read())
            loaded = restored.get("params", restored) if isinstance(restored, dict) else restored
            network_params = jax.tree_util.tree_map(jnp.asarray, loaded)
            print(f"[warm-start] IPPO initialized from {model_file}", flush=True)

        if config.get("ANNEAL_LR", True):
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )
        train_state = TrainState.create(
            apply_fn=network.apply, params=network_params, tx=tx,
        )

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset)(reset_rng)
        init_hstate = jnp.zeros((config["NUM_ACTORS"], hidden_dim))

        # TRAIN LOOP — fully inside jax.lax.scan (matches original)
        def _update_step(runner_state, unused):
            train_state, env_state, last_obs, last_done, update_step, hstate, rng = (
                runner_state
            )

            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, last_done, update_step, hstate, rng = (
                    runner_state
                )

                rng, _rng = jax.random.split(rng)
                obs_batch = jnp.stack(
                    [last_obs[a] for a in env.agents]
                ).reshape(-1, *obs_shape)

                ac_in = (obs_batch[np.newaxis, :], last_done[np.newaxis, :])
                hstate, pi, value = network.apply(
                    train_state.params, hstate, ac_in
                )
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)
                env_act = unbatchify(
                    action, env.agents, config["NUM_ENVS"], num_agents
                )
                env_act = {k: v.flatten() for k, v in env_act.items()}

                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(env.step)(
                    rng_step, env_state, env_act
                )

                # Shaped reward with anneal (matches original ippo_cnn_overcooked)
                original_reward = jnp.array([reward[a] for a in env.agents])
                shaped_reward = info.pop("shaped_reward")
                current_timestep = (
                    update_step * config["NUM_STEPS"] * config["NUM_ENVS"]
                )
                anneal_factor = (
                    rew_shaping_min_coeff
                    + (1.0 - rew_shaping_min_coeff)
                    * rew_shaping_anneal(current_timestep)
                )
                reward = jax.tree.map(
                    lambda x, y: x + shaped_reward_coeff * y * anneal_factor,
                    reward,
                    shaped_reward,
                )

                shaped_reward_arr = jnp.array([shaped_reward[a] for a in env.agents])
                combined_reward = jnp.array([reward[a] for a in env.agents])
                info["shaped_reward"] = shaped_reward_arr
                info["original_reward"] = original_reward
                info["combined_reward"] = combined_reward
                info["anneal_factor"] = jnp.full_like(shaped_reward_arr, anneal_factor)

                agent_major_info_keys = {
                    "shaped_reward",
                    "original_reward",
                    "combined_reward",
                    "anneal_factor",
                }

                def _flatten_info_value(key, value):
                    if (
                        key not in agent_major_info_keys
                        and len(value.shape) >= 2
                        and value.shape[0] == config["NUM_ENVS"]
                        and value.shape[1] == num_agents
                    ):
                        value = jnp.swapaxes(value, 0, 1)
                    return value.reshape((config["NUM_ACTORS"],) + value.shape[2:])

                info = {key: _flatten_info_value(key, value) for key, value in info.items()}
                done_batch = jnp.stack(
                    [done[a] for a in env.agents]
                ).reshape(config["NUM_ACTORS"])

                transition = Transition(
                    jnp.tile(done["__all__"], num_agents),
                    action.squeeze(),
                    value.squeeze(),
                    batchify(reward, env.agents, config["NUM_ACTORS"]).squeeze(),
                    log_prob.squeeze(),
                    obs_batch,
                    info,
                )
                runner_state = (
                    train_state, env_state, obsv, done_batch,
                    update_step, hstate, rng,
                )
                return runner_state, transition

            initial_hstate = hstate
            runner_state, traj_batch = jax.lax.scan(
                _env_step,
                (train_state, env_state, last_obs, last_done,
                 update_step, hstate, rng),
                None,
                config["NUM_STEPS"],
            )
            train_state, env_state, last_obs, last_done, update_step, hstate, rng = (
                runner_state
            )

            # CALCULATE ADVANTAGE (GAE)
            last_obs_batch = jnp.stack(
                [last_obs[a] for a in env.agents]
            ).reshape(-1, *obs_shape)
            ac_in = (last_obs_batch[np.newaxis, :], last_done[np.newaxis, :])
            _, _, last_val = network.apply(train_state.params, hstate, ac_in)
            last_val = last_val.squeeze()

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    delta = (
                        reward + config["GAMMA"] * next_value * (1 - done) - value
                    )
                    gae = (
                        delta
                        + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    )
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)

            # Normalise advantages at BATCH level (before minibatch split).
            # Per-minibatch normalization causes catastrophic collapse on
            # sparse-reward maps where most minibatches have zero rewards.
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

            # UPDATE NETWORK (PPO epochs)
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    init_hstate, traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, init_hstate, traj_batch, gae, targets, ent_coef):
                        _, pi, value = network.apply(
                            params,
                            init_hstate[0],
                            (traj_batch.obs, traj_batch.done),
                        )
                        log_prob = pi.log_prob(traj_batch.action)

                        # Value loss (clipped)
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(
                            value_pred_clipped - targets
                        )
                        value_loss = (
                            0.5
                            * jnp.maximum(
                                value_losses, value_losses_clipped
                            ).mean()
                        )

                        # Actor loss (clipped)
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        # NOTE: gae is already normalised at batch level
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config["CLIP_EPS"],
                                1.0 + config["CLIP_EPS"],
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        # ent_coef is annealed by the caller (linear ENT_COEF ->
                        # ENT_COEF_MIN over training) so the policy explores early
                        # and sharpens late instead of staying jittery.
                        entropy_floor = config.get("ENTROPY_FLOOR", 0.0)
                        entropy_floor_coef = config.get("ENTROPY_FLOOR_COEF", 0.0)
                        entropy_deficit = jnp.maximum(0.0, entropy_floor - entropy)

                        total_loss = (
                            loss_actor
                            + config["VF_COEF"] * value_loss
                            - ent_coef * entropy
                            + entropy_floor_coef * entropy_deficit
                        )
                        return total_loss, (value_loss, loss_actor, entropy)

                    # Linearly anneal the entropy coefficient from ENT_COEF to
                    # ENT_COEF_MIN over the course of training (keyed to the
                    # optimizer step), so exploration is high early and the
                    # policy can sharpen (less jitter) as it converges.
                    total_grad_steps = (
                        config["NUM_UPDATES"]
                        * config["UPDATE_EPOCHS"]
                        * config["NUM_MINIBATCHES"]
                    )
                    ent_frac = jnp.clip(
                        train_state.step.astype(jnp.float32) / total_grad_steps,
                        0.0,
                        1.0,
                    )
                    ent_coef = config["ENT_COEF"] + ent_frac * (
                        config.get("ENT_COEF_MIN", 0.0) - config["ENT_COEF"]
                    )

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params,
                        init_hstate,
                        traj_batch,
                        advantages,
                        targets,
                        ent_coef,
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                (
                    train_state,
                    init_hstate,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                ) = update_state
                rng, _rng = jax.random.split(rng)

                init_hstate_r = jnp.reshape(
                    init_hstate, (1, config["NUM_ACTORS"], -1)
                )
                batch = (
                    init_hstate_r,
                    traj_batch,
                    advantages.squeeze(),
                    targets.squeeze(),
                )
                permutation = jax.random.permutation(_rng, config["NUM_ACTORS"])

                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=1), batch
                )

                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.swapaxes(
                        jnp.reshape(
                            x,
                            [x.shape[0], config["NUM_MINIBATCHES"], -1]
                            + list(x.shape[2:]),
                        ),
                        1,
                        0,
                    ),
                    shuffled_batch,
                )

                train_state, total_loss = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )
                update_state = (
                    train_state,
                    init_hstate,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                )
                return update_state, total_loss

            update_state = (
                train_state,
                initial_hstate,
                traj_batch,
                advantages,
                targets,
                rng,
            )
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            train_state = update_state[0]
            metric = traj_batch.info
            rng = update_state[-1]

            update_step = update_step + 1
            metric = jax.tree_util.tree_map(lambda x: x.mean(), metric)
            for event_name in EVENT_METRIC_NAMES:
                event_key = f"event/{event_name}"
                if event_key in traj_batch.info:
                    event_values = traj_batch.info[event_key]
                    metric[event_key] = event_values.sum()
                    metric[f"event_rate/{event_name}"] = event_values.mean()
            if "event/delivery" in traj_batch.info:
                delivery_values = traj_batch.info["event/delivery"]
                metric["delivery"] = delivery_values.sum()
                metric["delivery_count.agent_0"] = delivery_values[:, :config["NUM_ENVS"]].sum()
                metric["delivery_count.agent_1"] = delivery_values[:, config["NUM_ENVS"]:].sum()
            metric["update_step"] = update_step
            metric["env_step"] = (
                update_step * config["NUM_STEPS"] * config["NUM_ENVS"]
            )
            metric["base_reward"] = traj_batch.info["original_reward"].sum()
            metric["base_reward_per_step"] = traj_batch.info["original_reward"].mean()
            metric["combined_reward"] = traj_batch.reward.sum()
            metric["combined_reward_per_step"] = traj_batch.reward.mean()
            metric["mean_reward"] = traj_batch.reward.mean()
            metric["max_reward"] = traj_batch.reward.max()
            metric["reward_sum"] = traj_batch.reward.sum()
            metric["loss/total"] = loss_info[0].mean()
            metric["loss/value"] = loss_info[1][0].mean()
            metric["loss/policy"] = loss_info[1][1].mean()
            metric["loss/entropy"] = loss_info[1][2].mean()
            jax.debug.callback(_log_training_metrics, metric)

            runner_state = (
                train_state,
                env_state,
                last_obs,
                last_done,
                update_step,
                hstate,
                rng,
            )
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (
            train_state,
            env_state,
            obsv,
            jnp.zeros((config["NUM_ACTORS"]), dtype=bool),
            0,
            init_hstate,
            _rng,
        )
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metrics": metric}

    return train


# ── Sweep Configuration ───────────────────────────────────────────────


def _build_overcooked_v3_sweep_configuration():
    """W&B sweep config for IPPO on overcooked_v3."""
    return {
        "method": "bayes",
        "metric": {"goal": "maximize", "name": "returned_episode_returns"},
        "parameters": {
            "LR": {"distribution": "log_uniform_values", "min": 5e-5, "max": 1e-3},
            "ENT_COEF": {
                "distribution": "log_uniform_values",
                "min": 1e-3,
                "max": 1e-2,
            },
            "VF_COEF": {"values": [0.25, 0.5, 0.75, 1.0]},
            "MAX_GRAD_NORM": {"values": [0.25, 0.5]},
            "GAMMA": {"values": [0.99, 0.995]},
            "GAE_LAMBDA": {"values": [0.95, 0.98]},
            "CLIP_EPS": {"values": [0.1, 0.2]},
            "UPDATE_EPOCHS": {"values": [2, 4]},
            "GRU_HIDDEN_DIM": {"values": [128, 256]},
            "REW_SHAPING_HORIZON": {"values": [10000000, 20000000, 40000000]},
            "SHAPED_REWARD_COEFF": {"values": [0.5, 1.0, 2.0, 3.0]},
        },
    }


def run_wandb_sweep(base_config):
    """Run a W&B sweep using current config as base defaults."""
    project = base_config.get("WANDB_PROJECT", "jaxmarl-ippo")
    sweep_count = int(base_config.get("WANDB_SWEEP_COUNT", 20))
    sweep_configuration = _build_overcooked_v3_sweep_configuration()

    def _objective():
        with wandb.init(project=project, config=base_config, mode="online") as run:
            wandb_logs.reset_wandb_logging(False)
            wandb_logs.define_wandb_metrics()
            run_config = dict(run.config)
            train_config = dict(base_config)
            train_config.update(run_config)
            train_config["WANDB_MODE"] = "online"

            train_jit = jax.jit(make_train(train_config))
            rng = jax.random.PRNGKey(train_config.get("SEED", 42))
            output = train_jit(rng)

            runner_state = output["runner_state"]
            train_state = runner_state[0]
            params = train_state.params

            base_save_path = train_config.get("SAVE_PATH", "checkpoints/ippo_overcooked_v3")
            run_save_path = os.path.join(base_save_path, f"sweep_{run.id}")
            model_path = _save_model_params(params, run_save_path)
            wandb.run.summary["saved_model_path"] = model_path

    sweep_id = wandb.sweep(sweep=sweep_configuration, project=project)
    wandb.agent(sweep_id, function=_objective, count=sweep_count)


@hydra.main(
    version_base=None, config_path="config", config_name="ippo_rnn_overcooked_v3"
)
def main(config):
    """Main training entry point."""
    config = OmegaConf.to_container(config, resolve=True)
    capture_wandb_history_table = bool(config.get("WANDB_LOG_HISTORY_TABLE", False))
    wandb_logs.reset_wandb_logging(capture_wandb_history_table)

    if config.get("WANDB_SWEEP", False):
        run_wandb_sweep(config)
        return

    layout_name = config.get("ENV_KWARGS", {}).get("layout", "unknown")

    wandb.init(
        entity=config.get("ENTITY", ""),
        project=config.get("WANDB_PROJECT", "jaxmarl-ippo"),
        tags=["IPPO", "RNN", "OvercookedV3"],
        config=config,
        mode=config.get("WANDB_MODE", "disabled"),
        name=config.get("WANDB_NAME") or f"ippo_rnn_overcooked_v3_{layout_name}",
    )
    if wandb.run is not None:
        wandb_logs.define_wandb_metrics()

    rng = jax.random.PRNGKey(config.get("SEED", 42))
    train_fn = make_train(config)
    if wandb.run is not None:
        wandb.config.update(
            {
                "NUM_UPDATES": config["NUM_UPDATES"],
                "NUM_ACTORS": config["NUM_ACTORS"],
                "MINIBATCH_SIZE": config["MINIBATCH_SIZE"],
            },
            allow_val_change=True,
        )
        _log_reward_structure_to_wandb(config)

    monitor_config = {
        "layout": layout_name,
        "total_timesteps": config["TOTAL_TIMESTEPS"],
        "completed_env_steps": config["NUM_UPDATES"] * config["NUM_STEPS"] * config["NUM_ENVS"],
        "num_updates": config["NUM_UPDATES"],
        "num_envs": config["NUM_ENVS"],
        "num_steps": config["NUM_STEPS"],
        "wandb_step": "env_step",
    }

    try:
        with TrainingMonitorInterface(config["NUM_UPDATES"], monitor_config) as monitor:
            wandb_logs.set_active_monitor(monitor)
            monitor.log(
                "Compiling + training IPPO RNN v3 on "
                f"{layout_name} for {config['TOTAL_TIMESTEPS']:,} env steps "
                f"({config['NUM_UPDATES']:,} PPO updates)."
            )
            train_jit = jax.jit(train_fn)
            out = jax.block_until_ready(train_jit(rng))

            monitor.log("Training finished; saving checkpoint and GIF.")
            runner_state = out["runner_state"]
            train_state = runner_state[0]
            params = train_state.params
            save_path = config.get("SAVE_PATH", "checkpoints/ippo_overcooked_v3")
            model_path = _save_model_params(params, save_path)
            print(f"Saved model checkpoint to: {model_path}", flush=True)

            if wandb.run is not None:
                wandb.run.summary["saved_model_path"] = model_path

            gif_path = config.get("SAVE_GIF_PATH")
            if gif_path:
                state_seq_list = get_rollout(params, config)
                state_seq = jax.tree.map(lambda *xs: jnp.stack(xs), *state_seq_list)
                env_viz = jaxmarl.make(config["ENV_NAME"], **config.get("ENV_KWARGS", {}))
                viz = OvercookedV3Visualizer(env_viz)
                viz.animate(state_seq, filename=gif_path)
                print(f"Saved GIF to: {gif_path}", flush=True)

            _log_history_table_to_wandb()
    finally:
        wandb_logs.set_active_monitor(None)
        wandb.finish()


if __name__ == "__main__":
    main()
