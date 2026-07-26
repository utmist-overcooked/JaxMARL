"""MAPPO baseline that selects an interruptible macro every primitive step."""

from pathlib import Path

import hydra
import jax
import jax.numpy as jnp
from omegaconf import OmegaConf

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


def make_train(config):
    env = build_env(config)
    config = initialize_config(config, env)

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

        def update_step(runner, update_index):
            actor_state, critic_state, env_state, obs, rng = runner

            def env_step(step_runner, step_index):
                env_state, obs, rng = step_runner
                obs_batch = batchify(obs, env.agents, config["NUM_ACTORS"])
                world_state = metadata_batch(
                    obs["world_state"], config["NUM_ACTORS"]
                )
                action_mask = metadata_batch(
                    obs["action_mask"], config["NUM_ACTORS"]
                ).astype(jnp.bool_)
                logits = actor.apply(actor_state.params, obs_batch)
                policy = masked_categorical(logits, action_mask)
                rng, action_rng, step_rng = jax.random.split(rng, 3)
                action = policy.sample(seed=action_rng)
                log_prob = policy.log_prob(action)
                value = critic.apply(critic_state.params, world_state)

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
            last_value = critic.apply(critic_state.params, last_world_state)
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
    config_name="mappo_macro_every_step",
)
def main(config):
    config = OmegaConf.to_container(config, resolve=True)
    if config["ENV_NAME"] != "overcooked_v3_macro_interruptible":
        raise ValueError("Every-step MAPPO requires the interruptible macro env")
    run_experiment(config, make_train, Path(__file__).stem)


if __name__ == "__main__":
    main()
