"""Hierarchical MAPPO with learned CONTINUE/REPLAN macro termination."""

from pathlib import Path

import hydra
import jax
import jax.numpy as jnp
from omegaconf import OmegaConf

from baselines.MAPPO.mappo_macro_common import (
    Critic,
    ReplanActor,
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


CONTINUE = 0
REPLAN = 1


def make_train(config):
    env = build_env(config)
    config = initialize_config(config, env)

    def train(rng):
        actor = ReplanActor(env.num_actions, int(config["HIDDEN_SIZE"]))
        critic = Critic(int(config["HIDDEN_SIZE"]))
        dummy_obs = jnp.zeros(
            (1, env.observation_space(env.agents[0]).shape[0])
        )
        rng, actor_state, critic_state = initialize_actor_critic(
            actor,
            critic,
            dummy_obs,
            jnp.zeros((1, env.world_state_size())),
            rng,
            config,
        )

        rng, reset_rng = jax.random.split(rng)
        reset_keys = jax.random.split(reset_rng, int(config["NUM_ENVS"]))
        obs, env_state = jax.vmap(env.reset)(reset_keys)

        def select_eval_actions(
            params, eval_obs, action_mask, macro_done, current_macro
        ):
            macro_logits, replan_logits = actor.apply(params, eval_obs)
            replacement_mask = action_mask & ~(
                (~macro_done)[:, None]
                & (
                    jnp.arange(env.num_actions)[None, :]
                    == current_macro[:, None]
                )
            )
            macro_action = jnp.argmax(
                jnp.where(replacement_mask, macro_logits, -1e9), axis=-1
            )
            replan_action = jnp.argmax(replan_logits, axis=-1)
            replace = macro_done | (replan_action == REPLAN)
            return jnp.where(replace, macro_action, current_macro)

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
                macro_done = metadata_batch(
                    obs["macro_done"], config["NUM_ACTORS"]
                )
                current_macro = metadata_batch(
                    obs["current_macro"], config["NUM_ACTORS"]
                )
                available_macro_actions = metadata_batch(
                    obs["action_mask"], config["NUM_ACTORS"]
                ).astype(jnp.bool_)
                macro_logits, replan_logits = actor.apply(
                    actor_state.params, obs_batch
                )
                replan_policy = categorical(replan_logits)
                rng, macro_rng, replan_rng, step_rng = jax.random.split(rng, 4)
                replan_action = replan_policy.sample(seed=replan_rng)

                # Replanning must choose a genuinely new subgoal. At idle
                # boundaries all macros remain available.
                macro_action_mask = available_macro_actions & ~(
                    (~macro_done)[:, None]
                    & (
                        jnp.arange(env.num_actions)[None, :]
                        == current_macro[:, None]
                    )
                )
                masked_macro_logits = jnp.where(
                    macro_action_mask, macro_logits, -1e9
                )
                macro_policy = masked_categorical(
                    masked_macro_logits, macro_action_mask
                )
                macro_action = macro_policy.sample(seed=macro_rng)

                gate_mask = ~macro_done
                macro_mask = macro_done | (replan_action == REPLAN)
                env_action_batch = jnp.where(
                    macro_mask, macro_action, current_macro
                )
                value = critic.apply(critic_state.params, world_state)
                env_action = unbatchify(
                    env_action_batch, env.agents, int(config["NUM_ENVS"])
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
                    "macro_action": macro_action,
                    "replan_action": replan_action,
                    "old_macro_log_prob": macro_policy.log_prob(macro_action),
                    "old_replan_log_prob": replan_policy.log_prob(replan_action),
                    "macro_mask": macro_mask,
                    "gate_mask": gate_mask,
                    "macro_action_mask": macro_action_mask,
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
            last_value = critic.apply(
                critic_state.params,
                metadata_batch(obs["world_state"], config["NUM_ACTORS"]),
            )
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
                macro_logits, replan_logits = actor.apply(
                    params, minibatch["obs"]
                )
                macro_policy = categorical(
                    jnp.where(
                        minibatch["macro_action_mask"], macro_logits, -1e9
                    )
                )
                replan_policy = categorical(replan_logits)
                macro_loss, macro_metrics = clipped_actor_loss(
                    macro_policy.log_prob(minibatch["macro_action"]),
                    minibatch["old_macro_log_prob"],
                    minibatch["advantage"],
                    macro_policy.entropy(),
                    minibatch["macro_mask"],
                    config["CLIP_EPS"],
                    config["ENT_COEF"],
                )
                replan_loss, replan_metrics = clipped_actor_loss(
                    replan_policy.log_prob(minibatch["replan_action"]),
                    minibatch["old_replan_log_prob"],
                    minibatch["advantage"],
                    replan_policy.entropy(),
                    minibatch["gate_mask"],
                    config["CLIP_EPS"],
                    config["ENT_COEF"],
                )
                metrics = {
                    **{f"macro_{key}": value for key, value in macro_metrics.items()},
                    **{f"replan_{key}": value for key, value in replan_metrics.items()},
                }
                return macro_loss + replan_loss, metrics

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
                "replan_rate": jnp.sum(
                    (trajectory["replan_action"] == REPLAN)
                    * trajectory["gate_mask"]
                )
                / jnp.maximum(jnp.sum(trajectory["gate_mask"]), 1),
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
    config_name="mappo_macro_replan",
)
def main(config):
    config = OmegaConf.to_container(config, resolve=True)
    if config["ENV_NAME"] != "overcooked_v3_macro_interruptible":
        raise ValueError("Learned replanning requires the interruptible macro env")
    run_experiment(config, make_train, Path(__file__).stem)


if __name__ == "__main__":
    main()
