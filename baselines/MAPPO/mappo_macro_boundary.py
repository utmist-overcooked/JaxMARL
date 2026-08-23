"""Asynchronous MAPPO that selects actions only at macro boundaries.

Rewards are accumulated between an agent's decision events and transitions are
stored in a fixed-size, masked event buffer. Rollouts end on complete episode
boundaries so no macro selected by an older policy crosses a PPO update.
"""

from pathlib import Path

import hydra
import jax
import jax.numpy as jnp
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
    deterministic_evaluation,
    emit_live_metrics,
    initialize_config,
    initialize_actor_critic,
    make_train_state,
    masked_categorical,
    maybe_checkpoint,
    maybe_evaluate_and_save_best,
    metadata_batch,
    restore_training_checkpoint,
    run_experiment,
    unbatchify,
    update_ppo,
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

    def train(rng):
        actor = Actor(env.num_actions, int(config["HIDDEN_SIZE"]))
        critic = Critic(int(config["HIDDEN_SIZE"]))
        rng, actor_state, critic_state = initialize_actor_critic(
            actor,
            critic,
            jnp.zeros((1, env.observation_space(env.agents[0]).shape[0])),
            jnp.zeros((1, env.world_state_size())),
            rng,
            config,
        )

        rng, reset_rng = jax.random.split(rng)
        reset_keys = jax.random.split(reset_rng, int(config["NUM_ENVS"]))
        obs, env_state = jax.vmap(env.reset)(reset_keys)
        num_actors = config["NUM_ACTORS"]
        obs_size = env.observation_space(env.agents[0]).shape[0]
        world_state_size = env.world_state_size()

        def select_eval_actions(params, eval_obs, action_mask, *_):
            logits = actor.apply(params, eval_obs)
            return jnp.argmax(
                jnp.where(action_mask, logits, -1e9), axis=-1
            )

        def evaluate(params, completed_updates):
            eval_key = jax.random.fold_in(
                jax.random.PRNGKey(int(config.get("EVAL_SEED", 42))),
                completed_updates,
            )
            return deterministic_evaluation(
                env, params, select_eval_actions, config, eval_key
            )

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
            "reward": jnp.zeros((num_actors,), dtype=jnp.float32),
            "discount": jnp.ones((num_actors,), dtype=jnp.float32),
            "duration": jnp.zeros((num_actors,), dtype=jnp.int32),
            "active": jnp.zeros((num_actors,), dtype=jnp.bool_),
        }

        def update_step(runner, update_index):
            actor_state, critic_state, env_state, obs, rng = runner

            def env_step(step_runner, step_index):
                env_state, obs, pending, rng = step_runner
                obs_batch = batchify(obs, env.agents, num_actors)
                world_state = metadata_batch(obs["world_state"], num_actors)
                macro_done = metadata_batch(obs["macro_done"], num_actors)
                current_macro = metadata_batch(obs["current_macro"], num_actors)
                action_mask = metadata_batch(
                    obs["action_mask"], num_actors
                ).astype(jnp.bool_)

                policy = masked_categorical(
                    actor.apply(actor_state.params, obs_batch), action_mask
                )
                value = critic.apply(critic_state.params, world_state)
                rng, action_rng, step_rng = jax.random.split(rng, 3)
                proposed_action = policy.sample(seed=action_rng)
                proposed_log_prob = policy.log_prob(proposed_action)

                def start(new, old):
                    shape = (num_actors,) + (1,) * (new.ndim - 1)
                    return jnp.where(macro_done.reshape(shape), new, old)

                pending = {
                    "obs": start(obs_batch, pending["obs"]),
                    "world_state": start(world_state, pending["world_state"]),
                    "action": start(proposed_action, pending["action"]),
                    "action_mask": start(
                        action_mask, pending["action_mask"]
                    ),
                    "old_log_prob": start(
                        proposed_log_prob, pending["old_log_prob"]
                    ),
                    "old_value": start(value, pending["old_value"]),
                    "reward": jnp.where(macro_done, 0.0, pending["reward"]),
                    "discount": jnp.where(macro_done, 1.0, pending["discount"]),
                    "duration": jnp.where(macro_done, 0, pending["duration"]),
                    "active": pending["active"] | macro_done,
                }
                action = jnp.where(
                    macro_done, proposed_action, current_macro
                )
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
                        [
                            info["macro_action_done"][agent]
                            for agent in env.agents
                        ],
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
                        completed,
                        1.0,
                        pending["discount"] * config["GAMMA"],
                    ),
                    "duration": jnp.where(completed, 0, duration),
                    "active": pending["active"] & ~completed,
                }
                return (next_env_state, next_obs, pending, rng), transition

            (env_state, obs, pending, rng), trajectory = jax.lax.scan(
                env_step,
                (env_state, obs, empty_pending, rng),
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

            def actor_loss_fn(params, minibatch):
                policy = masked_categorical(
                    actor.apply(params, minibatch["obs"]),
                    minibatch["action_mask"],
                )
                return clipped_actor_loss(
                    policy.log_prob(minibatch["action"]),
                    minibatch["old_log_prob"],
                    minibatch["advantage"],
                    policy.entropy(),
                    minibatch["loss_mask"],
                    config["CLIP_EPS"],
                    config["ENT_COEF"],
                )

            rng, actor_state, critic_state, loss_metrics = update_ppo(
                rng,
                actor_state,
                critic_state,
                batch,
                actor_loss_fn,
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
                "shaping_coefficient": jnp.mean(
                    trajectory["shaping_coefficient"]
                ),
                "burn_penalty_coefficient": jnp.mean(
                    trajectory["burn_penalty_coefficient"]
                ),
                **{
                    f"reward/{key}": jnp.mean(trajectory["reward_breakdown"][key])
                    for key in REWARD_COMPONENT_KEYS
                },
            }
            metrics["eval_return"] = maybe_evaluate_and_save_best(
                update_index,
                actor_state,
                critic_state,
                evaluate,
                config,
            )
            next_runner = (actor_state, critic_state, env_state, obs, rng)
            emit_live_metrics(
                update_index,
                metrics,
                int(config["NUM_STEPS"]) * int(config["NUM_ENVS"]),
                config,
            )
            maybe_checkpoint(update_index, next_runner, config)
            return next_runner, metrics

        initial_runner = (actor_state, critic_state, env_state, obs, rng)
        initial_runner, start_update = restore_training_checkpoint(
            initial_runner, config
        )

        @jax.jit
        def run_updates(runner):
            return jax.lax.scan(
                update_step,
                runner,
                jnp.arange(start_update, config["NUM_UPDATES"]),
            )

        runner, metrics = run_updates(initial_runner)
        return {"runner_state": runner, "metrics": metrics}

    return train


@hydra.main(
    version_base=None,
    config_path="config",
    config_name="mappo_macro_boundary",
)
def main(config):
    config = OmegaConf.to_container(config, resolve=True)
    if config["ENV_NAME"] != "overcooked_v3_macro":
        raise ValueError("Boundary MAPPO requires the committed macro environment")
    run_experiment(config, make_train, Path(__file__).stem)


if __name__ == "__main__":
    main()
