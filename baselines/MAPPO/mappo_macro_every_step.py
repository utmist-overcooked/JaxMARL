"""MAPPO baseline that selects an interruptible macro every primitive step.

Two actor/critic architectures are available, selected by USE_RNN in the config:

  USE_RNN: false -> memoryless MLP Actor/Critic (the original baseline).
  USE_RNN: true  -> recurrent GRU ActorRNN/CriticRNN.

The recurrent path exists because under partial observability
(ENV_KWARGS.agent_view_size) a memoryless policy cannot distinguish states that
require opposite macros -- e.g. from most of a 9x13 layout the 5x5 window shows
no pot at all, so "pot empty" and "pot cooked and ready" produce byte-identical
observations. A GRU lets the policy carry what it saw earlier in the episode.

The two paths differ in more than the network: the RNN needs TIME-MAJOR data
with trajectories kept contiguous (sequence_minibatches, not the flat shuffle in
minibatches) so BPTT sees real sequences, and it threads a hidden state through
the rollout, the PPO update, and evaluation. Checkpoints are NOT interchangeable
between the two -- the parameter trees differ.
"""

from pathlib import Path

import hydra
import jax
import jax.numpy as jnp
from omegaconf import OmegaConf

from mappo_macro_common import (
    Actor,
    ActorRNN,
    Critic,
    CriticRNN,
    ScannedRNN,
    add_annealed_shaped_reward,
    anneal_burn_penalty,
    batchify,
    build_env,
    calculate_gae,
    categorical,
    clipped_actor_loss,
    deterministic_evaluation,
    deterministic_evaluation_rnn,
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
    sequence_minibatches,
    unbatchify,
    update_ppo,
)
from jaxmarl.environments.overcooked_v3.settings import REWARD_COMPONENT_KEYS


def _annealed_step_reward(reward, info, primitive_timestep, config, env, num_actors):
    """Apply shaped-reward decay and burn-penalty ramp; return per-key extras.

    Shared by both architectures so the reward pipeline can never drift between
    them. Returns (reward, shaping_coefficient, burn_coefficient, breakdown).
    """
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
    breakdown = {
        key: metadata_batch(info["reward_breakdown"][key], num_actors)
        for key in REWARD_COMPONENT_KEYS
    }
    return reward, shaping_coefficient, burn_penalty_coefficient, breakdown


def _training_metrics(trajectory, loss_metrics):
    """Metrics common to both architectures (wandb logs these per update)."""
    episode_mask = trajectory["returned_episode"]
    return {
        **loss_metrics,
        "episode_return": jnp.sum(
            trajectory["returned_episode_returns"] * episode_mask
        )
        / jnp.maximum(jnp.sum(episode_mask), 1),
        "mean_shaped_reward": jnp.mean(trajectory["shaped_reward"]),
        "shaping_coefficient": jnp.mean(trajectory["shaping_coefficient"]),
        "burn_penalty_coefficient": jnp.mean(
            trajectory["burn_penalty_coefficient"]
        ),
        **{
            f"reward/{key}": jnp.mean(trajectory["reward_breakdown"][key])
            for key in REWARD_COMPONENT_KEYS
        },
    }


# ---------------------------------------------------------------------------
# Memoryless MLP path (original baseline, unchanged behaviour)
# ---------------------------------------------------------------------------
def _make_train_mlp(config, env):
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
            return jnp.argmax(jnp.where(action_mask, logits, -1e9), axis=-1)

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
                (
                    reward,
                    shaping_coefficient,
                    burn_penalty_coefficient,
                    reward_breakdown,
                ) = _annealed_step_reward(
                    reward, info, primitive_timestep, config, env,
                    config["NUM_ACTORS"],
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
                    "burn_penalty_coefficient": jnp.full(
                        (config["NUM_ACTORS"],), burn_penalty_coefficient
                    ),
                    "reward_breakdown": reward_breakdown,
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
                rng, actor_state, critic_state, batch, actor_loss_fn, config
            )
            metrics = _training_metrics(trajectory, loss_metrics)
            metrics["eval_return"] = maybe_evaluate_and_save_best(
                update_index, actor_state, critic_state, evaluate, config
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


# ---------------------------------------------------------------------------
# Recurrent GRU path
# ---------------------------------------------------------------------------
def _make_train_rnn(config, env):
    hidden_size = int(config["HIDDEN_SIZE"])
    num_actors = int(config["NUM_ACTORS"])
    num_minibatches = int(config["NUM_MINIBATCHES"])
    # Sequence minibatching splits the ACTOR axis (time stays whole), so the
    # divisibility requirement is on NUM_ACTORS rather than BATCH_SIZE.
    if num_actors % num_minibatches != 0:
        raise ValueError(
            f"USE_RNN requires NUM_ACTORS ({num_actors} = num_agents * NUM_ENVS) "
            f"to be divisible by NUM_MINIBATCHES ({num_minibatches}), because "
            "minibatches are formed by splitting actors while keeping each "
            "actor's full trajectory intact for BPTT."
        )

    def train(rng):
        actor = ActorRNN(env.num_actions, hidden_size)
        critic = CriticRNN(hidden_size)

        obs_dim = env.observation_space(env.agents[0]).shape[0]
        world_state_dim = env.world_state_size()
        init_actor_hidden = ScannedRNN.initialize_carry(num_actors, hidden_size)
        init_critic_hidden = ScannedRNN.initialize_carry(num_actors, hidden_size)

        rng, actor_rng, critic_rng = jax.random.split(rng, 3)
        # Leading axis of 1 is the time axis the GRU scans over.
        dummy_dones = jnp.zeros((1, num_actors), dtype=jnp.bool_)
        actor_params = actor.init(
            actor_rng,
            init_actor_hidden,
            (jnp.zeros((1, num_actors, obs_dim)), dummy_dones),
        )
        critic_params = critic.init(
            critic_rng,
            init_critic_hidden,
            (jnp.zeros((1, num_actors, world_state_dim)), dummy_dones),
        )
        actor_state = make_train_state(
            actor, actor_params, config, config["NUM_UPDATES"]
        )
        critic_state = make_train_state(
            critic, critic_params, config, config["NUM_UPDATES"]
        )

        rng, reset_rng = jax.random.split(rng)
        reset_keys = jax.random.split(reset_rng, int(config["NUM_ENVS"]))
        obs, env_state = jax.vmap(env.reset)(reset_keys)

        def select_eval_actions(
            params, hidden, eval_obs, last_done, action_mask, *_
        ):
            new_hidden, logits = actor.apply(
                params, hidden, (eval_obs[None, :], last_done[None, :])
            )
            logits = logits.squeeze(0)
            return new_hidden, jnp.argmax(
                jnp.where(action_mask, logits, -1e9), axis=-1
            )

        def evaluate(params, completed_updates):
            eval_key = jax.random.fold_in(
                jax.random.PRNGKey(int(config.get("EVAL_SEED", 42))),
                completed_updates,
            )
            return deterministic_evaluation_rnn(
                env, params, select_eval_actions, config, eval_key, hidden_size
            )

        def update_step(runner, update_index):
            (
                actor_state,
                critic_state,
                env_state,
                obs,
                last_done,
                hidden_states,
                rng,
            ) = runner
            # Hidden state at the START of this rollout -- replayed during the
            # PPO update so the recomputed sequence matches what was collected.
            rollout_start_hidden = hidden_states

            def env_step(step_runner, step_index):
                env_state, obs, last_done, hidden_states, rng = step_runner
                actor_hidden, critic_hidden = hidden_states
                obs_batch = batchify(obs, env.agents, num_actors)
                world_state = metadata_batch(obs["world_state"], num_actors)
                action_mask = metadata_batch(
                    obs["action_mask"], num_actors
                ).astype(jnp.bool_)

                # One timestep at a time during rollout: leading axis of 1.
                actor_hidden, logits = actor.apply(
                    actor_state.params,
                    actor_hidden,
                    (obs_batch[None, :], last_done[None, :]),
                )
                logits = logits.squeeze(0)
                policy = masked_categorical(logits, action_mask)
                rng, action_rng, step_rng = jax.random.split(rng, 3)
                action = policy.sample(seed=action_rng)
                log_prob = policy.log_prob(action)

                critic_hidden, value = critic.apply(
                    critic_state.params,
                    critic_hidden,
                    (world_state[None, :], last_done[None, :]),
                )
                value = value.squeeze(0)

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
                (
                    reward,
                    shaping_coefficient,
                    burn_penalty_coefficient,
                    reward_breakdown,
                ) = _annealed_step_reward(
                    reward, info, primitive_timestep, config, env, num_actors
                )
                next_done = jnp.tile(done["__all__"], env.num_agents)
                transition = {
                    "obs": obs_batch,
                    "world_state": world_state,
                    "action": action,
                    "action_mask": action_mask,
                    "old_log_prob": log_prob,
                    "old_value": value,
                    # "done" is this step's outcome, used for GAE bootstrapping.
                    # "prev_done" is what was fed to the GRU alongside this obs,
                    # and is what the update must replay to reproduce the carry.
                    "done": next_done,
                    "prev_done": last_done,
                    "reward": batchify(reward, env.agents, num_actors),
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
                    "returned_episode": metadata_batch(
                        info["returned_episode"], num_actors
                    ),
                    "returned_episode_returns": metadata_batch(
                        info["returned_episode_returns"], num_actors
                    ),
                }
                return (
                    next_env_state,
                    next_obs,
                    next_done,
                    (actor_hidden, critic_hidden),
                    rng,
                ), transition

            (
                env_state,
                obs,
                last_done,
                hidden_states,
                rng,
            ), trajectory = jax.lax.scan(
                env_step,
                (env_state, obs, last_done, hidden_states, rng),
                jnp.arange(int(config["NUM_STEPS"])),
                int(config["NUM_STEPS"]),
            )

            last_world_state = metadata_batch(obs["world_state"], num_actors)
            _, last_value = critic.apply(
                critic_state.params,
                hidden_states[1],
                (last_world_state[None, :], last_done[None, :]),
            )
            last_value = last_value.squeeze(0)
            advantage, target = calculate_gae(
                trajectory["reward"],
                trajectory["done"],
                trajectory["old_value"],
                last_value,
                config["GAMMA"],
                config["GAE_LAMBDA"],
            )

            # Time-major throughout: leaves stay (NUM_STEPS, num_actors, ...).
            # The two hidden states carry a dummy time axis of 1 so the same
            # actor-axis shuffle applies to them.
            batch = {
                "obs": trajectory["obs"],
                "world_state": trajectory["world_state"],
                "action": trajectory["action"],
                "action_mask": trajectory["action_mask"],
                "old_log_prob": trajectory["old_log_prob"],
                "old_value": trajectory["old_value"],
                "prev_done": trajectory["prev_done"],
                "advantage": advantage,
                "target": target,
                "loss_mask": jnp.ones_like(advantage, dtype=jnp.bool_),
                "init_actor_hidden": rollout_start_hidden[0][None, :],
                "init_critic_hidden": rollout_start_hidden[1][None, :],
            }

            def actor_loss_fn(params, minibatch):
                _, logits = actor.apply(
                    params,
                    minibatch["init_actor_hidden"][0],
                    (minibatch["obs"], minibatch["prev_done"]),
                )
                policy = masked_categorical(logits, minibatch["action_mask"])
                return clipped_actor_loss(
                    policy.log_prob(minibatch["action"]),
                    minibatch["old_log_prob"],
                    minibatch["advantage"],
                    policy.entropy(),
                    minibatch["loss_mask"],
                    config["CLIP_EPS"],
                    config["ENT_COEF"],
                )

            def critic_predict(params, minibatch):
                _, value = critic.apply(
                    params,
                    minibatch["init_critic_hidden"][0],
                    (minibatch["world_state"], minibatch["prev_done"]),
                )
                return value

            rng, actor_state, critic_state, loss_metrics = update_ppo(
                rng,
                actor_state,
                critic_state,
                batch,
                actor_loss_fn,
                config,
                critic_predict=critic_predict,
                minibatch_fn=lambda shuffle_rng, full_batch: sequence_minibatches(
                    shuffle_rng, full_batch, num_minibatches, num_actors
                ),
            )
            metrics = _training_metrics(trajectory, loss_metrics)
            metrics["eval_return"] = maybe_evaluate_and_save_best(
                update_index, actor_state, critic_state, evaluate, config
            )
            next_runner = (
                actor_state,
                critic_state,
                env_state,
                obs,
                last_done,
                hidden_states,
                rng,
            )
            emit_live_metrics(
                update_index,
                metrics,
                int(config["NUM_STEPS"]) * int(config["NUM_ENVS"]),
                config,
            )
            maybe_checkpoint(update_index, next_runner, config)
            return next_runner, metrics

        initial_runner = (
            actor_state,
            critic_state,
            env_state,
            obs,
            jnp.zeros((num_actors,), dtype=jnp.bool_),
            (init_actor_hidden, init_critic_hidden),
            rng,
        )
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


def make_train(config):
    env = build_env(config)
    config = initialize_config(config, env)
    if config.get("USE_RNN", False):
        return _make_train_rnn(config, env)
    return _make_train_mlp(config, env)


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
