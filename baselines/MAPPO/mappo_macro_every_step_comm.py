"""Discrete learned-communication module trained on top of a frozen MAPPO
macro-action policy. 2 agents, shared weights, RIAL-style message loss.

Requires a checkpoint from `mappo_macro_every_step.py` (frozen Actor +
Critic params). Only the communication module (message encoder +
correction head) is trained; the underlying macro policy never changes.

`load_frozen_macro_params` loads `final_actor.safetensors` /
`final_critic.safetensors` — the exported final weights, not the
`checkpoints/*.npz` resume checkpoints (those carry optimizer state and
env_state meant for `restore_training_checkpoint`, not for this).
"""

from pathlib import Path
from typing import Dict

import hydra
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn
from flax.linen.initializers import constant, orthogonal
from flax.traverse_util import unflatten_dict
from flax.training.train_state import TrainState
from omegaconf import OmegaConf
from safetensors.flax import load_file as load_safetensors

from mappo_macro_common import (
    Actor,
    Critic,
    add_annealed_shaped_reward,
    batchify,
    build_env,
    calculate_gae,
    categorical,
    clipped_actor_loss,
    deterministic_evaluation,
    emit_live_metrics,
    initialize_config,
    masked_categorical,
    maybe_checkpoint,
    maybe_evaluate_and_save_best,
    metadata_batch,
    run_experiment,
    unbatchify,
    update_ppo,
)


# --------------------------------------------------------------------------
# Communication module: a message encoder (obs -> outgoing message logits)
# and a correction head (obs + received message -> logit bias on the frozen
# actor's macro-action logits). Both trained jointly, one param tree.
# --------------------------------------------------------------------------
class CommModule(nn.Module):
    hidden_size: int
    vocab_size: int
    action_dim: int
    message_embed_dim: int

    def setup(self):
        self.msg_dense1 = nn.Dense(
            self.hidden_size,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )
        self.msg_dense2 = nn.Dense(
            self.vocab_size,
            kernel_init=orthogonal(0.0),
            bias_init=constant(0.0),
        )
        self.msg_embed = nn.Embed(self.vocab_size, self.message_embed_dim)
        self.corr_dense1 = nn.Dense(
            self.hidden_size,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )
        self.corr_dense2 = nn.Dense(
            self.action_dim,
            kernel_init=orthogonal(0.0),
            bias_init=constant(0.0),
        )

    def encode_message(self, obs):
        x = nn.tanh(self.msg_dense1(obs))
        return self.msg_dense2(x)

    def correction(self, obs, received_message):
        embed = self.msg_embed(received_message)
        x = jnp.concatenate([obs, embed], axis=-1)
        x = nn.tanh(self.corr_dense1(x))
        return self.corr_dense2(x)

    def __call__(self, obs, received_message):
        # Only used at init time so both branches get params created.
        return self.encode_message(obs), self.correction(obs, received_message)


def swap_two_agent_messages(values, num_envs: int):
    """Swap agent0<->agent1 values within each env. Assumes batchify's
    agent-major layout: [agent0_env0..agent0_envN, agent1_env0..agent1_envN].
    2-agent only."""
    agent0 = values[:num_envs]
    agent1 = values[num_envs : 2 * num_envs]
    return jnp.concatenate([agent1, agent0], axis=0)


def load_frozen_macro_params(config):
    """Load (actor_params, critic_params) from the exported final-weights
    safetensors files (not the mid-training checkpoints/*.npz files)."""
    for key in ("FROZEN_ACTOR_PATH", "FROZEN_CRITIC_PATH"):
        if not config.get(key):
            raise ValueError(f"config['{key}'] is required (e.g. final_actor.safetensors)")
    return (
        _load_params_from_safetensors(config["FROZEN_ACTOR_PATH"]),
        _load_params_from_safetensors(config["FROZEN_CRITIC_PATH"]),
    )


def _load_params_from_safetensors(path):
    flat = load_safetensors(path)
    if not flat:
        raise ValueError(f"{path} contained no tensors")
    sample_key = next(iter(flat))
    # Detect whether the export flattened with ',', '/', or '.' rather than assuming.
    if "," in sample_key:
        sep = ","
    elif "/" in sample_key:
        sep = "/"
    else:
        sep = "."
    nested = unflatten_dict(flat, sep=sep)
    if "params" not in nested:
        nested = {"params": nested}
    top_level = list(nested["params"].keys())
    if not any(k.startswith("Dense") for k in top_level):
        raise ValueError(
            f"Unflattened {path} but top-level keys under 'params' are "
            f"{top_level}, expected something like ['Dense_0', 'Dense_1', "
            "'Dense_2']. The export's key scheme doesn't match what this "
            "loader assumes — print list(load_safetensors(path).keys())[:10] "
            "and send it over so I can fix the unflatten logic."
        )
    return nested


def make_train(config):
    env = build_env(config)
    config = initialize_config(config, env)

    if len(env.agents) != 2:
        raise ValueError("This script only supports exactly 2 agents.")

    def train(rng):
        actor = Actor(env.num_actions, int(config["HIDDEN_SIZE"]))
        critic = Critic(int(config["HIDDEN_SIZE"]))
        comm_module = CommModule(
            hidden_size=int(config.get("COMM_HIDDEN_SIZE", config["HIDDEN_SIZE"])),
            vocab_size=int(config["VOCAB_SIZE"]),
            action_dim=env.num_actions,
            message_embed_dim=int(config.get("MESSAGE_EMBED_DIM", 8)),
        )

        # Frozen macro policy: loaded once, never updated. Closed over below.
        frozen_actor_params, frozen_critic_params = load_frozen_macro_params(config)

        obs_dim = env.observation_space(env.agents[0]).shape[0]
        rng, comm_rng = jax.random.split(rng)
        dummy_obs = jnp.zeros((1, obs_dim))
        dummy_message = jnp.zeros((1,), dtype=jnp.int32)
        comm_params = comm_module.init(comm_rng, dummy_obs, dummy_message)

        # Reuse update_ppo unmodified: comm params go in the "actor" slot
        # (real optimizer, actually trained); critic slot is the frozen
        # Critic wrapped with a zero-gradient optimizer so update_ppo's
        # hardcoded critic update becomes a no-op.
        comm_state = TrainState.create(
            apply_fn=comm_module.apply,
            params=comm_params,
            tx=optax.chain(
                optax.clip_by_global_norm(config.get("MAX_GRAD_NORM", 0.5)),
                optax.adam(config["LR"], eps=1e-5),
            ),
        )
        frozen_critic_state = TrainState.create(
            apply_fn=critic.apply,
            params=frozen_critic_params,
            tx=optax.set_to_zero(),
        )

        rng, reset_rng = jax.random.split(rng)
        reset_keys = jax.random.split(reset_rng, int(config["NUM_ENVS"]))
        obs, env_state = jax.vmap(env.reset)(reset_keys)

        def select_eval_actions(params, eval_obs, action_mask, macro_done, current_macro):
            # deterministic_evaluation gives us no rng, so messages are
            # argmax'd rather than sampled. eval_obs is agent-major
            # (num_eval_envs * 2 actors), same layout as training.
            num_eval_envs = eval_obs.shape[0] // 2
            message_logits = comm_module.apply(
                params, eval_obs, method=comm_module.encode_message
            )
            message = jnp.argmax(message_logits, axis=-1)
            received_message = swap_two_agent_messages(message, num_eval_envs)
            logit_bias = comm_module.apply(
                params, eval_obs, received_message, method=comm_module.correction
            )
            base_logits = actor.apply(frozen_actor_params, eval_obs)
            final_logits = base_logits + logit_bias
            return jnp.argmax(jnp.where(action_mask, final_logits, -1e9), axis=-1)

        def evaluate(params, completed_updates):
            eval_key = jax.random.fold_in(
                jax.random.PRNGKey(int(config.get("EVAL_SEED", 42))),
                completed_updates,
            )
            return deterministic_evaluation(
                env, params, select_eval_actions, config, eval_key
            )

        def update_step(runner, update_index):
            comm_state, frozen_critic_state, env_state, obs, rng = runner

            def env_step(step_runner, step_index):
                env_state, obs, rng = step_runner
                obs_batch = batchify(obs, env.agents, config["NUM_ACTORS"])
                world_state = metadata_batch(
                    obs["world_state"], config["NUM_ACTORS"]
                )
                action_mask = metadata_batch(
                    obs["action_mask"], config["NUM_ACTORS"]
                ).astype(jnp.bool_)

                # --- communication round ---
                message_logits = comm_module.apply(
                    comm_state.params, obs_batch, method=comm_module.encode_message
                )
                rng, message_rng = jax.random.split(rng)
                message_policy = categorical(message_logits)
                message = message_policy.sample(seed=message_rng)
                message_log_prob = message_policy.log_prob(message)
                received_message = swap_two_agent_messages(
                    message, int(config["NUM_ENVS"])
                )

                logit_bias = comm_module.apply(
                    comm_state.params,
                    obs_batch,
                    received_message,
                    method=comm_module.correction,
                )
                base_logits = actor.apply(frozen_actor_params, obs_batch)
                final_logits = base_logits + logit_bias

                policy = masked_categorical(final_logits, action_mask)
                rng, action_rng, step_rng = jax.random.split(rng, 3)
                action = policy.sample(seed=action_rng)
                log_prob = policy.log_prob(action)
                value = critic.apply(frozen_critic_state.params, world_state)

                env_action = unbatchify(
                    action, env.agents, int(config["NUM_ENVS"])
                )
                step_keys = jax.random.split(step_rng, int(config["NUM_ENVS"]))
                next_obs, next_env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0)
                )(step_keys, env_state, env_action)
                primitive_timestep = (
                    update_index * int(config["NUM_STEPS"]) + step_index
                ) * int(config["NUM_ENVS"])
                reward, shaping_coefficient = add_annealed_shaped_reward(
                    reward,
                    info["shaped_reward"],
                    primitive_timestep,
                    float(config.get("REW_SHAPING_HORIZON", 0.0)),
                )
                transition = {
                    "obs": obs_batch,
                    "world_state": world_state,
                    "action": action,
                    "action_mask": action_mask,
                    "old_log_prob": log_prob,
                    "old_value": value,
                    "message": message,
                    "old_message_log_prob": message_log_prob,
                    "received_message": received_message,
                    "reward": batchify(
                        reward, env.agents, config["NUM_ACTORS"]
                    ),
                    "shaped_reward": batchify(
                        info["shaped_reward"], env.agents, config["NUM_ACTORS"]
                    ),
                    "shaping_coefficient": jnp.full(
                        (config["NUM_ACTORS"],), shaping_coefficient
                    ),
                    "done": jnp.tile(done["__all__"], env.num_agents),
                    "returned_episode": metadata_batch(
                        info["returned_episode"], config["NUM_ACTORS"]
                    ),
                    "returned_episode_returns": metadata_batch(
                        info["returned_episode_returns"], config["NUM_ACTORS"]
                    ),
                }
                return (next_env_state, next_obs, rng), transition

            (env_state, obs, rng), trajectory = jax.lax.scan(
                env_step,
                (env_state, obs, rng),
                jnp.arange(int(config["NUM_STEPS"])),
                int(config["NUM_STEPS"]),
            )
            last_world_state = metadata_batch(
                obs["world_state"], config["NUM_ACTORS"]
            )
            last_value = critic.apply(frozen_critic_state.params, last_world_state)
            advantage, target = calculate_gae(
                trajectory["reward"],
                trajectory["done"],
                trajectory["old_value"],
                last_value,
                config["GAMMA"],
                config["GAE_LAMBDA"],
            )
            batch = jax.tree.map(
                lambda x: x.reshape((-1,) + x.shape[2:]),
                {
                    **trajectory,
                    "advantage": advantage,
                    "target": target,
                    "loss_mask": jnp.ones_like(advantage, dtype=jnp.bool_),
                },
            )

            def comm_loss_fn(params, minibatch):
                # Action-loss branch: correction head + frozen backbone.
                logit_bias = comm_module.apply(
                    params,
                    minibatch["obs"],
                    minibatch["received_message"],
                    method=comm_module.correction,
                )
                base_logits = actor.apply(frozen_actor_params, minibatch["obs"])
                policy = masked_categorical(
                    base_logits + logit_bias, minibatch["action_mask"]
                )
                action_loss, action_metrics = clipped_actor_loss(
                    policy.log_prob(minibatch["action"]),
                    minibatch["old_log_prob"],
                    minibatch["advantage"],
                    policy.entropy(),
                    minibatch["loss_mask"],
                    config["CLIP_EPS"],
                    config["ENT_COEF"],
                )

                # Message-loss branch: RIAL-style, same advantage as the
                # action it enabled (credit flows to the sender via the
                # team reward the pair achieved).
                message_logits = comm_module.apply(
                    params, minibatch["obs"], method=comm_module.encode_message
                )
                message_policy = categorical(message_logits)
                message_loss, message_metrics = clipped_actor_loss(
                    message_policy.log_prob(minibatch["message"]),
                    minibatch["old_message_log_prob"],
                    minibatch["advantage"],
                    message_policy.entropy(),
                    minibatch["loss_mask"],
                    config["CLIP_EPS"],
                    config.get("MESSAGE_ENT_COEF", config["ENT_COEF"]),
                )

                total = action_loss + config.get(
                    "MESSAGE_LOSS_COEF", 1.0
                ) * message_loss
                metrics = {
                    **{f"action_{k}": v for k, v in action_metrics.items()},
                    **{f"message_{k}": v for k, v in message_metrics.items()},
                }
                return total, metrics

            rng, comm_state, frozen_critic_state, loss_metrics = update_ppo(
                rng,
                comm_state,
                frozen_critic_state,
                batch,
                comm_loss_fn,
                config,
            )
            episode_mask = trajectory["returned_episode"]
            metrics = {
                **loss_metrics,
                "episode_return": jnp.sum(
                    trajectory["returned_episode_returns"] * episode_mask
                )
                / jnp.maximum(jnp.sum(episode_mask), 1),
                "mean_shaped_reward": jnp.mean(trajectory["shaped_reward"]),
                "shaping_coefficient": jnp.mean(
                    trajectory["shaping_coefficient"]
                ),
            }
            metrics["eval_return"] = maybe_evaluate_and_save_best(
                update_index,
                comm_state,
                frozen_critic_state,
                evaluate,
                config,
            )
            next_runner = (comm_state, frozen_critic_state, env_state, obs, rng)
            emit_live_metrics(
                update_index,
                metrics,
                int(config["NUM_STEPS"]) * int(config["NUM_ENVS"]),
                config,
            )
            maybe_checkpoint(update_index, next_runner, config)
            return next_runner, metrics

        initial_runner = (comm_state, frozen_critic_state, env_state, obs, rng)

        @jax.jit
        def run_updates(runner):
            return jax.lax.scan(
                update_step,
                runner,
                jnp.arange(0, config["NUM_UPDATES"]),
            )

        runner, metrics = run_updates(initial_runner)
        return {"runner_state": runner, "metrics": metrics}

    return train


@hydra.main(
    version_base=None,
    config_path="config",
    config_name="mappo_macro_every_step",
)
def main(config):
    config = OmegaConf.to_container(config, resolve=True)
    if config["ENV_NAME"] != "overcooked_v3_macro_interruptible":
        raise ValueError("Requires the interruptible macro env")
    run_experiment(config, make_train, Path(__file__).stem)


if __name__ == "__main__":
    main()