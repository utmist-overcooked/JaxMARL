"""Discrete learned communication on top of a frozen BOUNDARY macro policy.

Boundary counterpart of mappo_macro_every_step_comm.py. Only the communication
module (message encoder + correction head) is trained; the underlying macro
actor and critic are loaded frozen from a completed mappo_macro_boundary.py run
via FROZEN_ACTOR_PATH / FROZEN_CRITIC_PATH.

Key difference from the every-step comm variant: agents communicate ONCE PER
MACRO BOUNDARY rather than once per primitive step. Between boundaries an
agent's outgoing message is held, so its partner keeps receiving the last thing
it actually said. This matters for credit assignment -- the every-step variant
makes a fresh stochastic message decision every primitive step (up to
max_macro_steps=150 of them per macro), all sharing one diluted advantage
signal, which is a plausible reason that variant collapsed to a constant
symbol. Here the number of message decisions equals the number of macro
decisions, and each one is scored by the SMDP return of the macro it informed.

Reward/advantage handling matches mappo_macro_boundary.py exactly: rewards are
accumulated across a macro's duration with per-step discounting, transitions are
emitted into a masked event buffer at macro completion, and advantages come from
calculate_smdp_gae using the macro duration.
"""

from pathlib import Path

import hydra
import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState
from omegaconf import OmegaConf

from mappo_macro_common import (
    Actor,
    Critic,
    add_annealed_shaped_reward,
    anneal_burn_penalty,
    batchify,
    build_env,
    calculate_smdp_gae,
    categorical,
    clipped_actor_loss,
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
from mappo_macro_every_step_comm import (
    CommModule,
    load_frozen_macro_params,
    swap_two_agent_messages,
)
from jaxmarl.environments.overcooked_v3.settings import REWARD_COMPONENT_KEYS


def make_train(config):
    env = build_env(config)
    config = initialize_config(config, env)
    episode_steps = int(config.get("ENV_KWARGS", {}).get("max_steps", 400))
    if int(config["NUM_STEPS"]) % episode_steps != 0:
        raise ValueError(
            "Boundary MAPPO requires NUM_STEPS to be a multiple of the "
            "environment max_steps so every pending macro is flushed before update"
        )
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

        # Frozen boundary macro policy: loaded once, never updated.
        frozen_actor_params, frozen_critic_params = load_frozen_macro_params(config)

        num_actors = config["NUM_ACTORS"]
        num_envs = int(config["NUM_ENVS"])
        obs_size = env.observation_space(env.agents[0]).shape[0]
        world_state_size = env.world_state_size()

        rng, comm_rng = jax.random.split(rng)
        comm_params = comm_module.init(
            comm_rng,
            jnp.zeros((1, obs_size)),
            jnp.zeros((1,), dtype=jnp.int32),
        )

        # Comm params go in update_ppo's "actor" slot (really trained); the
        # frozen critic goes in the critic slot with a zero-gradient optimizer
        # so update_ppo's built-in critic update is a no-op.
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
        reset_keys = jax.random.split(reset_rng, num_envs)
        obs, env_state = jax.vmap(env.reset)(reset_keys)

        def _biased_logits(params, obs_batch, received_message):
            """Frozen macro logits plus the comm module's correction."""
            logit_bias = comm_module.apply(
                params, obs_batch, received_message, method=comm_module.correction
            )
            return actor.apply(frozen_actor_params, obs_batch) + logit_bias

        def evaluate(params, completed_updates):
            """Deterministic eval that holds messages between macro boundaries.

            Written locally rather than reusing deterministic_evaluation because
            the held-message state has to persist across steps, which that
            helper has no carry for. Messages/actions are argmax'd (no rng).
            """
            eval_key = jax.random.fold_in(
                jax.random.PRNGKey(int(config.get("EVAL_SEED", 42))),
                completed_updates,
            )
            num_eval_envs = int(config.get("NUM_EVAL_ENVS", 8))
            num_eval_actors = num_eval_envs * env.num_agents
            reset_keys = jax.random.split(eval_key, num_eval_envs)
            eval_obs, eval_env_state = jax.vmap(env.reset)(reset_keys)

            def eval_step(carry, _):
                eval_obs, eval_env_state, last_message, rng = carry
                obs_batch = batchify(eval_obs, env.agents, num_eval_actors)
                action_mask = metadata_batch(
                    eval_obs["action_mask"], num_eval_actors
                ).astype(jnp.bool_)
                macro_done = metadata_batch(
                    eval_obs["macro_done"], num_eval_actors
                )
                current_macro = metadata_batch(
                    eval_obs["current_macro"], num_eval_actors
                )

                # Speak only at a boundary; otherwise keep saying the last thing.
                message_logits = comm_module.apply(
                    params, obs_batch, method=comm_module.encode_message
                )
                message = jnp.where(
                    macro_done,
                    jnp.argmax(message_logits, axis=-1),
                    last_message,
                )
                received_message = swap_two_agent_messages(message, num_eval_envs)

                final_logits = _biased_logits(params, obs_batch, received_message)
                proposed = jnp.argmax(
                    jnp.where(action_mask, final_logits, -1e9), axis=-1
                )
                action = jnp.where(macro_done, proposed, current_macro)

                env_action = unbatchify(action, env.agents, num_eval_envs)
                rng, step_rng = jax.random.split(rng)
                step_keys = jax.random.split(step_rng, num_eval_envs)
                next_obs, next_env_state, reward, _, _ = jax.vmap(
                    env.step, in_axes=(0, 0, 0)
                )(step_keys, eval_env_state, env_action)
                mean_team_reward = jnp.mean(
                    jnp.stack([reward[agent] for agent in env.agents], axis=-1),
                    axis=-1,
                )
                return (
                    next_obs,
                    next_env_state,
                    message,
                    rng,
                ), mean_team_reward

            _, rewards = jax.lax.scan(
                eval_step,
                (
                    eval_obs,
                    eval_env_state,
                    jnp.zeros((num_eval_actors,), dtype=jnp.int32),
                    eval_key,
                ),
                None,
                episode_steps,
            )
            return jnp.mean(jnp.sum(rewards, axis=0))

        empty_pending = {
            "obs": jnp.zeros((num_actors, obs_size), dtype=jnp.float32),
            "world_state": jnp.zeros(
                (num_actors, world_state_size), dtype=jnp.float32
            ),
            "action": jnp.zeros((num_actors,), dtype=jnp.int32),
            "action_mask": jnp.ones(
                (num_actors, env.num_actions), dtype=jnp.bool_
            ),
            "old_log_prob": jnp.zeros((num_actors,), dtype=jnp.float32),
            "old_value": jnp.zeros((num_actors,), dtype=jnp.float32),
            # Message state captured at the boundary this macro started from.
            "message": jnp.zeros((num_actors,), dtype=jnp.int32),
            "old_message_log_prob": jnp.zeros((num_actors,), dtype=jnp.float32),
            "received_message": jnp.zeros((num_actors,), dtype=jnp.int32),
            "reward": jnp.zeros((num_actors,), dtype=jnp.float32),
            "discount": jnp.ones((num_actors,), dtype=jnp.float32),
            "duration": jnp.zeros((num_actors,), dtype=jnp.int32),
            "active": jnp.zeros((num_actors,), dtype=jnp.bool_),
        }

        def update_step(runner, update_index):
            comm_state, frozen_critic_state, env_state, obs, last_message, rng = runner

            def env_step(step_runner, step_index):
                env_state, obs, pending, last_message, rng = step_runner
                obs_batch = batchify(obs, env.agents, num_actors)
                world_state = metadata_batch(obs["world_state"], num_actors)
                macro_done = metadata_batch(obs["macro_done"], num_actors)
                current_macro = metadata_batch(obs["current_macro"], num_actors)
                action_mask = metadata_batch(
                    obs["action_mask"], num_actors
                ).astype(jnp.bool_)

                # --- communication round, gated to macro boundaries ---
                # A fresh symbol is sampled only where the agent is choosing a
                # new macro; elsewhere the previous message persists so the
                # partner keeps receiving the last thing actually said.
                message_logits = comm_module.apply(
                    comm_state.params, obs_batch, method=comm_module.encode_message
                )
                rng, message_rng, action_rng, step_rng = jax.random.split(rng, 4)
                message_policy = categorical(message_logits)
                sampled_message = message_policy.sample(seed=message_rng)
                message = jnp.where(macro_done, sampled_message, last_message)
                message_log_prob = message_policy.log_prob(message)
                received_message = swap_two_agent_messages(message, num_envs)

                final_logits = _biased_logits(
                    comm_state.params, obs_batch, received_message
                )
                policy = masked_categorical(final_logits, action_mask)
                proposed_action = policy.sample(seed=action_rng)
                proposed_log_prob = policy.log_prob(proposed_action)
                value = critic.apply(frozen_critic_state.params, world_state)

                def start(new, old):
                    shape = (num_actors,) + (1,) * (new.ndim - 1)
                    return jnp.where(macro_done.reshape(shape), new, old)

                pending = {
                    "obs": start(obs_batch, pending["obs"]),
                    "world_state": start(world_state, pending["world_state"]),
                    "action": start(proposed_action, pending["action"]),
                    "action_mask": start(action_mask, pending["action_mask"]),
                    "old_log_prob": start(
                        proposed_log_prob, pending["old_log_prob"]
                    ),
                    "old_value": start(value, pending["old_value"]),
                    "message": start(message, pending["message"]),
                    "old_message_log_prob": start(
                        message_log_prob, pending["old_message_log_prob"]
                    ),
                    "received_message": start(
                        received_message, pending["received_message"]
                    ),
                    "reward": jnp.where(macro_done, 0.0, pending["reward"]),
                    "discount": jnp.where(macro_done, 1.0, pending["discount"]),
                    "duration": jnp.where(macro_done, 0, pending["duration"]),
                    "active": pending["active"] | macro_done,
                }

                action = jnp.where(macro_done, proposed_action, current_macro)
                env_action = unbatchify(action, env.agents, num_envs)
                step_keys = jax.random.split(step_rng, num_envs)
                next_obs, next_env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0)
                )(step_keys, env_state, env_action)

                primitive_timestep = (
                    update_index * int(config["NUM_STEPS"]) + step_index
                ) * num_envs
                reward, shaping_coefficient = add_annealed_shaped_reward(
                    reward,
                    info["shaped_reward"],
                    primitive_timestep,
                    float(config.get("REW_SHAPING_HORIZON", 0.0)),
                )
                raw_burn_penalty = {
                    agent: info["reward_breakdown"]["BURN_PENALTY"][:, agent_idx]
                    for agent_idx, agent in enumerate(env.agents)
                }
                reward, burn_penalty_coefficient = anneal_burn_penalty(
                    reward,
                    raw_burn_penalty,
                    primitive_timestep,
                    float(config.get("REW_SHAPING_HORIZON", 0.0)),
                )
                reward_breakdown = {
                    key: metadata_batch(info["reward_breakdown"][key], num_actors)
                    for key in REWARD_COMPONENT_KEYS
                }

                reward_batch = batchify(reward, env.agents, num_actors)
                accumulated_reward = (
                    pending["reward"] + pending["discount"] * reward_batch
                )
                duration = pending["duration"] + 1
                completed = metadata_batch(
                    jnp.stack(
                        [info["macro_action_done"][agent] for agent in env.agents],
                        axis=-1,
                    ),
                    num_actors,
                )
                valid = completed & pending["active"]

                transition = {
                    "obs": pending["obs"],
                    "world_state": pending["world_state"],
                    "action": pending["action"],
                    "action_mask": pending["action_mask"],
                    "old_log_prob": pending["old_log_prob"],
                    "old_value": pending["old_value"],
                    "message": pending["message"],
                    "old_message_log_prob": pending["old_message_log_prob"],
                    "received_message": pending["received_message"],
                    "reward": accumulated_reward,
                    "shaped_reward": batchify(
                        info["shaped_reward"], env.agents, num_actors
                    ),
                    "shaping_coefficient": jnp.full(
                        (num_actors,), shaping_coefficient
                    ),
                    "burn_penalty_coefficient": jnp.full(
                        (num_actors,), burn_penalty_coefficient
                    ),
                    "reward_breakdown": reward_breakdown,
                    "duration": duration,
                    "done": jnp.tile(done["__all__"], env.num_agents),
                    "valid": valid,
                    "returned_episode": metadata_batch(
                        info["returned_episode"], num_actors
                    ),
                    "returned_episode_returns": metadata_batch(
                        info["returned_episode_returns"], num_actors
                    ),
                }

                pending = {
                    **pending,
                    "reward": jnp.where(completed, 0.0, accumulated_reward),
                    "discount": jnp.where(
                        completed, 1.0, pending["discount"] * config["GAMMA"]
                    ),
                    "duration": jnp.where(completed, 0, duration),
                    "active": pending["active"] & ~completed,
                }
                return (
                    next_env_state,
                    next_obs,
                    pending,
                    message,
                    rng,
                ), transition

            (
                env_state,
                obs,
                pending,
                last_message,
                rng,
            ), trajectory = jax.lax.scan(
                env_step,
                (env_state, obs, empty_pending, last_message, rng),
                jnp.arange(int(config["NUM_STEPS"])),
                int(config["NUM_STEPS"]),
            )

            advantage, target = calculate_smdp_gae(
                trajectory["reward"],
                trajectory["duration"],
                trajectory["done"],
                trajectory["old_value"],
                trajectory["valid"],
                config["GAMMA"],
                config["GAE_LAMBDA"],
            )
            batch = jax.tree.map(
                lambda x: x.reshape((-1,) + x.shape[2:]),
                {
                    **trajectory,
                    "advantage": advantage,
                    "target": target,
                    "loss_mask": trajectory["valid"],
                },
            )

            def comm_loss_fn(params, minibatch):
                # Action branch: correction head on top of the frozen backbone,
                # scored only at completed macro events (loss_mask == valid).
                policy = masked_categorical(
                    _biased_logits(
                        params, minibatch["obs"], minibatch["received_message"]
                    ),
                    minibatch["action_mask"],
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

                # Message branch: RIAL-style, credited with the same SMDP
                # advantage as the macro that message informed.
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

                total = action_loss + config.get("MESSAGE_LOSS_COEF", 1.0) * message_loss
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

            event_mask = trajectory["valid"]
            episode_mask = trajectory["returned_episode"]
            metrics = {
                **loss_metrics,
                "episode_return": jnp.sum(
                    trajectory["returned_episode_returns"] * episode_mask
                )
                / jnp.maximum(jnp.sum(episode_mask), 1),
                "mean_macro_duration": jnp.sum(
                    trajectory["duration"] * event_mask
                )
                / jnp.maximum(jnp.sum(event_mask), 1),
                "macro_decisions": jnp.sum(event_mask),
                "unfinished_macros": jnp.sum(pending["active"]),
                "mean_shaped_reward": jnp.mean(trajectory["shaped_reward"]),
                "shaping_coefficient": jnp.mean(trajectory["shaping_coefficient"]),
                "burn_penalty_coefficient": jnp.mean(
                    trajectory["burn_penalty_coefficient"]
                ),
                # Fraction of message symbols that are non-zero, at real
                # decision events only -- a quick collapse detector: a channel
                # stuck on one symbol carries no information.
                "message_nonzero_fraction": jnp.sum(
                    (trajectory["message"] != 0) * event_mask
                )
                / jnp.maximum(jnp.sum(event_mask), 1),
                **{
                    f"reward/{key}": jnp.mean(trajectory["reward_breakdown"][key])
                    for key in REWARD_COMPONENT_KEYS
                },
            }
            metrics["eval_return"] = maybe_evaluate_and_save_best(
                update_index, comm_state, frozen_critic_state, evaluate, config
            )
            next_runner = (
                comm_state,
                frozen_critic_state,
                env_state,
                obs,
                last_message,
                rng,
            )
            emit_live_metrics(
                update_index,
                metrics,
                int(config["NUM_STEPS"]) * num_envs,
                config,
            )
            maybe_checkpoint(update_index, next_runner, config)
            return next_runner, metrics

        initial_runner = (
            comm_state,
            frozen_critic_state,
            env_state,
            obs,
            jnp.zeros((num_actors,), dtype=jnp.int32),
            rng,
        )

        @jax.jit
        def run_updates(runner):
            return jax.lax.scan(
                update_step, runner, jnp.arange(0, config["NUM_UPDATES"])
            )

        runner, metrics = run_updates(initial_runner)
        return {"runner_state": runner, "metrics": metrics}

    return train


@hydra.main(
    version_base=None,
    config_path="config",
    config_name="mappo_macro_boundary_comm",
)
def main(config):
    config = OmegaConf.to_container(config, resolve=True)
    if config["ENV_NAME"] != "overcooked_v3_macro":
        raise ValueError(
            "Boundary comm MAPPO requires the committed macro environment "
            "(overcooked_v3_macro), matching the frozen boundary actor it "
            "builds on."
        )
    if config.get("USE_RNN", False):
        raise ValueError(
            "USE_RNN is not supported here -- the frozen boundary actor is an "
            "MLP, and the boundary pending buffer stores a macro-start obs "
            "snapshot rather than the per-step observation stream a GRU needs."
        )
    run_experiment(config, make_train, Path(__file__).stem)


if __name__ == "__main__":
    main()
