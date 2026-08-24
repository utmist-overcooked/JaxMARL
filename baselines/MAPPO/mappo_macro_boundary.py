"""Asynchronous MAPPO that selects actions only at macro boundaries.

Rewards are accumulated between an agent's decision events and transitions are
stored in a fixed-size, masked event buffer. Rollouts end on complete episode
boundaries so no macro selected by an older policy crosses a PPO update.

Two actor/critic architectures are available, selected by USE_RNN in the config:

  USE_RNN: false -> memoryless MLP Actor/Critic (the original baseline).
  USE_RNN: true  -> recurrent GRU ActorRNN/CriticRNN with DECISION-GATED
                    recurrence: the hidden state advances once per macro
                    decision (frozen while a macro executes) and resets on
                    episode boundaries. BPTT flows over the decision sequence,
                    reusing the SMDP return machinery unchanged.

The recurrent path exists for the same reason as in mappo_macro_every_step.py:
under partial observability (ENV_KWARGS.agent_view_size) a memoryless policy
cannot distinguish states that require opposite macros -- e.g. "pot empty" and
"pot cooked and ready" produce byte-identical local observations when the pot
is outside the agent's window. A GRU lets the policy carry what it saw earlier
in the episode. Checkpoints are NOT interchangeable between the two paths -- the
parameter trees differ. See docs/rnn_boundary_trainer_plan.md for the design
rationale and references.
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


def _boundary_metrics(trajectory, pending, loss_metrics):
    """Metrics common to both architectures (wandb logs these per update)."""
    event_mask = trajectory["valid"]
    episode_mask = trajectory["returned_episode"]
    return {
        **loss_metrics,
        "episode_return": jnp.sum(
            trajectory["returned_episode_returns"] * episode_mask
        )
        / jnp.maximum(jnp.sum(episode_mask), 1),
        "mean_macro_duration": jnp.sum(trajectory["duration"] * event_mask)
        / jnp.maximum(jnp.sum(event_mask), 1),
        "macro_decisions": jnp.sum(event_mask),
        "unfinished_macros": jnp.sum(pending["active"]),
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
        num_layers = int(config.get("NUM_LAYERS", 2))
        actor = Actor(env.num_actions, int(config["HIDDEN_SIZE"]), num_layers)
        critic = Critic(int(config["HIDDEN_SIZE"]), num_layers)
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
            return jnp.argmax(jnp.where(action_mask, logits, -1e9), axis=-1)

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
                    "action_mask": start(action_mask, pending["action_mask"]),
                    "old_log_prob": start(
                        proposed_log_prob, pending["old_log_prob"]
                    ),
                    "old_value": start(value, pending["old_value"]),
                    "reward": jnp.where(macro_done, 0.0, pending["reward"]),
                    "discount": jnp.where(macro_done, 1.0, pending["discount"]),
                    "duration": jnp.where(macro_done, 0, pending["duration"]),
                    "active": pending["active"] | macro_done,
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
            metrics = _boundary_metrics(trajectory, pending, loss_metrics)
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


# ---------------------------------------------------------------------------
# Recurrent GRU path (decision-gated recurrence over the macro sequence)
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
        num_layers = int(config.get("NUM_LAYERS", 2))
        actor = ActorRNN(env.num_actions, hidden_size, num_layers)
        critic = CriticRNN(hidden_size, num_layers)

        obs_size = env.observation_space(env.agents[0]).shape[0]
        world_state_size = env.world_state_size()
        init_actor_hidden = ScannedRNN.initialize_carry(num_actors, hidden_size)
        init_critic_hidden = ScannedRNN.initialize_carry(num_actors, hidden_size)

        rng, actor_rng, critic_rng = jax.random.split(rng, 3)
        # Leading axis of 1 is the time axis the GRU scans over. Init with a
        # 2-tuple (no advance mask); the module treats that as advance-every-
        # step, and the advance mask adds no parameters, so the param tree is
        # identical whether or not the gate is used at call time.
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
            params, hidden, eval_obs, last_done, action_mask, macro_done,
            current_macro,
        ):
            # Decision-gated at eval too: the hidden advances only where the
            # agent is at a macro boundary (advance=macro_done), resets where
            # the previous step ended an episode (reset=last_done).
            new_hidden, logits = actor.apply(
                params,
                hidden,
                (eval_obs[None, :], last_done[None, :], macro_done[None, :]),
            )
            logits = logits.squeeze(0)
            proposed = jnp.argmax(jnp.where(action_mask, logits, -1e9), axis=-1)
            action = jnp.where(macro_done, proposed, current_macro)
            return new_hidden, action

        def evaluate(params, completed_updates):
            eval_key = jax.random.fold_in(
                jax.random.PRNGKey(int(config.get("EVAL_SEED", 42))),
                completed_updates,
            )
            return deterministic_evaluation_rnn(
                env, params, select_eval_actions, config, eval_key, hidden_size
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
            # Whether the decision currently being accumulated was the first of
            # a new episode; replayed as the GRU reset flag during the update.
            "reset": jnp.zeros((num_actors,), dtype=jnp.bool_),
            "reward": jnp.zeros((num_actors,), dtype=jnp.float32),
            "discount": jnp.ones((num_actors,), dtype=jnp.float32),
            "duration": jnp.zeros((num_actors,), dtype=jnp.int32),
            "active": jnp.zeros((num_actors,), dtype=jnp.bool_),
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
            # Hidden entering this rollout -- replayed during the PPO update so
            # the recomputed decision sequence matches what was collected.
            rollout_start_hidden = hidden_states

            def env_step(step_runner, step_index):
                (
                    env_state,
                    obs,
                    pending,
                    last_done,
                    hidden_states,
                    rng,
                ) = step_runner
                actor_hidden, critic_hidden = hidden_states
                obs_batch = batchify(obs, env.agents, num_actors)
                world_state = metadata_batch(obs["world_state"], num_actors)
                macro_done = metadata_batch(obs["macro_done"], num_actors)
                current_macro = metadata_batch(obs["current_macro"], num_actors)
                action_mask = metadata_batch(
                    obs["action_mask"], num_actors
                ).astype(jnp.bool_)

                # Decision-gated recurrence: advance the hidden only where the
                # agent is at a boundary this step (advance=macro_done); reset
                # where the previous step ended an episode (reset=last_done).
                # The returned hidden is already gated by the module.
                actor_hidden, logits = actor.apply(
                    actor_state.params,
                    actor_hidden,
                    (obs_batch[None, :], last_done[None, :], macro_done[None, :]),
                )
                logits = logits.squeeze(0)
                policy = masked_categorical(logits, action_mask)
                critic_hidden, value = critic.apply(
                    critic_state.params,
                    critic_hidden,
                    (
                        world_state[None, :],
                        last_done[None, :],
                        macro_done[None, :],
                    ),
                )
                value = value.squeeze(0)

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
                    "action_mask": start(action_mask, pending["action_mask"]),
                    "old_log_prob": start(
                        proposed_log_prob, pending["old_log_prob"]
                    ),
                    "old_value": start(value, pending["old_value"]),
                    # last_done at the decision start == "first of a new episode".
                    "reset": jnp.where(macro_done, last_done, pending["reset"]),
                    "reward": jnp.where(macro_done, 0.0, pending["reward"]),
                    "discount": jnp.where(macro_done, 1.0, pending["discount"]),
                    "duration": jnp.where(macro_done, 0, pending["duration"]),
                    "active": pending["active"] | macro_done,
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
                next_done = jnp.tile(done["__all__"], env.num_agents)
                transition = {
                    "obs": pending["obs"],
                    "world_state": pending["world_state"],
                    "action": pending["action"],
                    "action_mask": pending["action_mask"],
                    "old_log_prob": pending["old_log_prob"],
                    "old_value": pending["old_value"],
                    "decision_reset": pending["reset"],
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
                    "done": next_done,
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

            # Time-major throughout: leaves stay (NUM_STEPS, num_actors, ...) so
            # sequence_minibatches can split the actor axis while keeping each
            # trajectory whole for BPTT over the decision sequence.
            batch = {
                "obs": trajectory["obs"],
                "world_state": trajectory["world_state"],
                "action": trajectory["action"],
                "action_mask": trajectory["action_mask"],
                "old_log_prob": trajectory["old_log_prob"],
                "old_value": trajectory["old_value"],
                "decision_reset": trajectory["decision_reset"],
                "valid": trajectory["valid"],
                "advantage": advantage,
                "target": target,
                "loss_mask": trajectory["valid"],
                "init_actor_hidden": rollout_start_hidden[0][None, :],
                "init_critic_hidden": rollout_start_hidden[1][None, :],
            }

            def actor_loss_fn(params, minibatch):
                # Replay with the decision gate: advance the hidden only on
                # valid (completed-decision) steps -- which, for one macro in
                # flight per agent, are exactly the decision sequence in order
                # -- feeding each the stored start-of-macro obs, resetting on
                # the per-decision new-episode flag.
                _, logits = actor.apply(
                    params,
                    minibatch["init_actor_hidden"][0],
                    (
                        minibatch["obs"],
                        minibatch["decision_reset"],
                        minibatch["valid"],
                    ),
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
                    (
                        minibatch["world_state"],
                        minibatch["decision_reset"],
                        minibatch["valid"],
                    ),
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
            metrics = _boundary_metrics(trajectory, pending, loss_metrics)
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
    episode_steps = int(config.get("ENV_KWARGS", {}).get("max_steps", 400))
    if int(config["NUM_STEPS"]) % episode_steps != 0:
        raise ValueError(
            "Boundary MAPPO requires NUM_STEPS to be a multiple of the "
            "environment max_steps so every pending macro is flushed before update"
        )
    if config.get("USE_RNN", False):
        return _make_train_rnn(config, env)
    return _make_train_mlp(config, env)


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
