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
    ActorRNN,
    Critic,
    CriticRNN,
    ScannedRNN,
    add_annealed_shaped_reward,
    anneal_burn_penalty,
    batchify,
    build_env,
    calculate_smdp_gae,
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


def make_train(config):
    env = build_env(config)
    config = initialize_config(config, env)
    episode_steps = int(config.get("ENV_KWARGS", {}).get("max_steps", 400))
    if int(config["NUM_STEPS"]) % episode_steps != 0:
        raise ValueError(
            "Boundary MAPPO requires NUM_STEPS to be a multiple of the "
            "environment max_steps so every pending macro is flushed before update"
        )
    if config.get("USE_RNN", False):
        return _make_train_rnn(config, env)
    return _make_train_mlp(config, env)


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


def _make_train_rnn(config, env):
    """Boundary MAPPO with a recurrent actor/critic over the REAL obs stream.

    The MLP path stores pending["obs"] -- a snapshot held from the step a macro
    started -- and scores the loss at macro-COMPLETION rows. That layout is
    useless to a GRU: along the time axis it is a step function of repeated
    stale observations, so replaying it would build hidden states that never
    occurred during rollout and the PPO ratio would be wrong.

    So this path is organised differently:
      * The trajectory keeps the genuine per-primitive-step observation stream
        (step_obs / step_world_state / step_prev_done), which is what the GRU
        scans, exactly as in mappo_macro_every_step.py.
      * The policy loss is evaluated at macro-START rows, where the per-step
        obs/action/log_prob/value are already the ones the decision used -- no
        snapshot needed, and naturally aligned with the RNN outputs.
      * SMDP advantages are still computed at completion rows (unchanged
        semantics), then SCATTERED back to the start row of the macro they
        belong to, using a start_index recorded in the pending buffer.

    Net effect: real BPTT through the observation stream, with macro-level SMDP
    credit assignment preserved.
    """
    hidden_size = int(config["HIDDEN_SIZE"])
    num_actors = int(config["NUM_ACTORS"])
    num_minibatches = int(config["NUM_MINIBATCHES"])
    num_steps = int(config["NUM_STEPS"])
    # Sequence minibatching splits actors, keeping each actor's whole
    # trajectory intact for BPTT.
    if num_actors % num_minibatches != 0:
        raise ValueError(
            f"USE_RNN requires NUM_ACTORS ({num_actors} = num_agents * NUM_ENVS) "
            f"to be divisible by NUM_MINIBATCHES ({num_minibatches})."
        )

    def train(rng):
        actor = ActorRNN(env.num_actions, hidden_size)
        critic = CriticRNN(hidden_size)

        obs_size = env.observation_space(env.agents[0]).shape[0]
        world_state_size = env.world_state_size()
        init_actor_hidden = ScannedRNN.initialize_carry(num_actors, hidden_size)
        init_critic_hidden = ScannedRNN.initialize_carry(num_actors, hidden_size)

        rng, actor_rng, critic_rng = jax.random.split(rng, 3)
        dummy_dones = jnp.zeros((1, num_actors), dtype=jnp.bool_)
        actor_params = actor.init(
            actor_rng,
            init_actor_hidden,
            (jnp.zeros((1, num_actors, obs_size)), dummy_dones),
        )
        critic_params = critic.init(
            critic_rng,
            init_critic_hidden,
            (jnp.zeros((1, num_actors, world_state_size)), dummy_dones),
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
            params, hidden, eval_obs, last_done, action_mask, macro_done, current_macro
        ):
            new_hidden, logits = actor.apply(
                params, hidden, (eval_obs[None, :], last_done[None, :])
            )
            logits = logits.squeeze(0)
            proposed = jnp.argmax(jnp.where(action_mask, logits, -1e9), axis=-1)
            # Boundary semantics: commit a new macro only at a boundary.
            return new_hidden, jnp.where(macro_done, proposed, current_macro)

        def evaluate(params, completed_updates):
            eval_key = jax.random.fold_in(
                jax.random.PRNGKey(int(config.get("EVAL_SEED", 42))),
                completed_updates,
            )
            return deterministic_evaluation_rnn(
                env, params, select_eval_actions, config, eval_key, hidden_size
            )

        # Only the accumulators need holding now; obs/action/value come from the
        # per-step arrays, so the snapshot fields the MLP path needs are gone.
        empty_pending = {
            "old_value": jnp.zeros((num_actors,), dtype=jnp.float32),
            "reward": jnp.zeros((num_actors,), dtype=jnp.float32),
            "discount": jnp.ones((num_actors,), dtype=jnp.float32),
            "duration": jnp.zeros((num_actors,), dtype=jnp.int32),
            "active": jnp.zeros((num_actors,), dtype=jnp.bool_),
            "start_index": jnp.zeros((num_actors,), dtype=jnp.int32),
        }

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
            rollout_start_hidden = hidden_states

            def env_step(step_runner, step_index):
                env_state, obs, pending, last_done, hidden_states, rng = step_runner
                actor_hidden, critic_hidden = hidden_states
                obs_batch = batchify(obs, env.agents, num_actors)
                world_state = metadata_batch(obs["world_state"], num_actors)
                macro_done = metadata_batch(obs["macro_done"], num_actors)
                current_macro = metadata_batch(obs["current_macro"], num_actors)
                action_mask = metadata_batch(
                    obs["action_mask"], num_actors
                ).astype(jnp.bool_)

                actor_hidden, logits = actor.apply(
                    actor_state.params,
                    actor_hidden,
                    (obs_batch[None, :], last_done[None, :]),
                )
                policy = masked_categorical(logits.squeeze(0), action_mask)
                rng, action_rng, step_rng = jax.random.split(rng, 3)
                proposed_action = policy.sample(seed=action_rng)
                proposed_log_prob = policy.log_prob(proposed_action)

                critic_hidden, value = critic.apply(
                    critic_state.params,
                    critic_hidden,
                    (world_state[None, :], last_done[None, :]),
                )
                value = value.squeeze(0)

                pending = {
                    # Value at macro start: still needed, because SMDP GAE is
                    # indexed by completion row but bootstraps from the value
                    # at the row the macro began.
                    "old_value": jnp.where(
                        macro_done, value, pending["old_value"]
                    ),
                    "reward": jnp.where(macro_done, 0.0, pending["reward"]),
                    "discount": jnp.where(macro_done, 1.0, pending["discount"]),
                    "duration": jnp.where(macro_done, 0, pending["duration"]),
                    "active": pending["active"] | macro_done,
                    "start_index": jnp.where(
                        macro_done, step_index, pending["start_index"]
                    ),
                }

                action = jnp.where(macro_done, proposed_action, current_macro)
                env_action = unbatchify(
                    action, env.agents, int(config["NUM_ENVS"])
                )
                step_keys = jax.random.split(step_rng, int(config["NUM_ENVS"]))
                next_obs, next_env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0)
                )(step_keys, env_state, env_action)
                primitive_timestep = (
                    update_index * num_steps + step_index
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
                        [info["macro_action_done"][agent] for agent in env.agents],
                        axis=-1,
                    ),
                    num_actors,
                )
                valid = completed & pending["active"]
                next_done = jnp.tile(done["__all__"], env.num_agents)

                transition = {
                    # --- per-step (macro-START aligned): what the GRU scans and
                    # what the policy loss is evaluated on ---
                    "step_obs": obs_batch,
                    "step_world_state": world_state,
                    "step_action": proposed_action,
                    "step_action_mask": action_mask,
                    "step_old_log_prob": proposed_log_prob,
                    "step_old_value": value,
                    "step_prev_done": last_done,
                    "step_macro_started": macro_done,
                    # --- completion aligned: SMDP credit assignment ---
                    "reward": accumulated_reward,
                    "duration": duration,
                    "old_value": pending["old_value"],
                    "done": next_done,
                    "valid": valid,
                    "start_index": pending["start_index"],
                    # --- logging ---
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
                    next_done,
                    (actor_hidden, critic_hidden),
                    rng,
                ), transition

            (
                env_state,
                obs,
                pending,
                last_done,
                hidden_states,
                rng,
            ), trajectory = jax.lax.scan(
                env_step,
                (env_state, obs, empty_pending, last_done, hidden_states, rng),
                jnp.arange(num_steps),
                num_steps,
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

            # Move each macro's advantage/target from the row where it COMPLETED
            # to the row where it STARTED, so they line up with the RNN output
            # for the decision that actually chose it. Invalid rows are pointed
            # out of bounds and dropped. Each (start_index, actor) pair is unique
            # -- a macro starts once and completes once -- so no update collides.
            actor_index = jnp.broadcast_to(
                jnp.arange(num_actors)[None, :], (num_steps, num_actors)
            )
            scatter_rows = jnp.where(
                trajectory["valid"], trajectory["start_index"], num_steps
            )
            zeros = jnp.zeros((num_steps, num_actors), dtype=jnp.float32)
            advantage_at_start = zeros.at[scatter_rows, actor_index].set(
                advantage, mode="drop"
            )
            target_at_start = zeros.at[scatter_rows, actor_index].set(
                target, mode="drop"
            )
            loss_mask_at_start = (
                jnp.zeros((num_steps, num_actors), dtype=jnp.bool_)
                .at[scatter_rows, actor_index]
                .set(True, mode="drop")
            )

            # Time-major, matching mappo_macro_every_step.py's RNN batch.
            batch = {
                "obs": trajectory["step_obs"],
                "world_state": trajectory["step_world_state"],
                "action": trajectory["step_action"],
                "action_mask": trajectory["step_action_mask"],
                "old_log_prob": trajectory["step_old_log_prob"],
                "old_value": trajectory["step_old_value"],
                "prev_done": trajectory["step_prev_done"],
                "advantage": advantage_at_start,
                "target": target_at_start,
                "loss_mask": loss_mask_at_start,
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
                # Sanity check: scattered rows must equal completed macros. A
                # gap would mean advantages were dropped before reaching a loss.
                "scattered_decisions": jnp.sum(loss_mask_at_start),
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
                num_steps * int(config["NUM_ENVS"]),
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
