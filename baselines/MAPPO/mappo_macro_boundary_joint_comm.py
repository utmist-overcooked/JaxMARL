"""Boundary MAPPO where the macro policy and comm module are trained TOGETHER.

Unlike mappo_macro_boundary_comm.py, nothing is frozen. That file trains a comm
module on top of a macro policy that was already trained without communication,
which is self-defeating for an information-asymmetry experiment: a policy
trained on an unobservable recipe converges to marginalising over it ("always
fetch tomato") with a large logit margin, and a small additive correction head
cannot realistically overturn that. Measured on a real checkpoint, the frozen
actor's onion-vs-tomato logit gap was 7.68 while the trained correction head's
swing between the two symbols was 0.037 -- roughly 200x too weak to ever change
an ingredient choice. Training jointly means the policy never gets the chance to
settle into a no-comm solution, because the message is available while it is
still learning what to fetch.

Same config surface as mappo_macro_boundary_comm.py, minus FROZEN_ACTOR_PATH /
FROZEN_CRITIC_PATH (nothing is loaded) and plus COMM_MODE (below). USE_RNN and
COMM_USE_MEMORY behave identically.

COMM_MODE selects how a sent message reaches its recipient. This is what makes
the control experiments a config change rather than a code change -- every mode
has identical architecture, parameter count and training budget, and differs
only in the information the channel carries:

  "normal"   partner's message (real communication)
  "self"     the agent's own message back -- same capacity, zero transfer
  "shuffled" partner's message permuted across envs -- destroys the correlation
             with this env's recipe while preserving message statistics
  "constant" a fixed symbol -- channel fully severed

Reporting "normal" against "self"/"shuffled" attributes any gain to the
information transferred, not to the extra parameters.

COMM_INJECTION selects HOW a received message reaches the policy:

  "concat" (default) the message embedding is concatenated onto the
      observation and fed to the actor's INPUT, so the trunk conditions on it
      from the first layer and the action-loss gradient flows back through the
      actor into the embedding table and the speaker.
  "bias"   the two-stage trainer's additive correction head, which adds to the
      actor's OUTPUT logits. Kept only as an ablation.

"bias" exists because a frozen trunk's input could not be changed. Carrying it
into joint training reproduced the same failure it was meant to fix: measured
on a jointly trained checkpoint, the speaker encoded the recipe well
(mutual information 0.58 bits) while the listener's logit swing between symbols
was 0.023 and it chose get_ingredient_1 on 15/15 decisions -- the additive head
never overcame the trunk's own confident, message-independent output. Under
"concat" the trunk has no message-independent output to overcome.
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
    emit_live_metrics,
    initialize_config,
    make_train_state,
    masked_categorical,
    masked_mean,
    maybe_checkpoint,
    maybe_evaluate_and_save_best,
    metadata_batch,
    run_experiment,
    paired_sequence_minibatches,
    sequence_minibatches,
    unbatchify,
    update_ppo,
)
from mappo_macro_every_step_comm import CommModule, swap_two_agent_messages
from jaxmarl.environments.overcooked_v3.settings import REWARD_COMPONENT_KEYS


COMM_MODES = ("normal", "self", "shuffled", "constant")

# How a received message reaches the policy. "concat" feeds its embedding into
# the actor's INPUT so the trunk conditions on it directly; "bias" adds a
# correction to the actor's OUTPUT logits, which is the two-stage trainer's
# design and is kept only as an ablation. See policy_logits().
COMM_INJECTIONS = ("concat", "bias")

# How the speaker is trained.
#   "reinforce" (RIAL) the symbol is sampled and the message head is trained by
#       policy gradient on the SAME advantage as the action. The channel is not
#       differentiable, so the speaker's only signal is a delayed, high-variance
#       scalar shared across every message it emitted in the episode.
#   "dial"      the symbol is a straight-through Gumbel-softmax sample, so it is
#       still discrete on the forward pass but the LISTENER'S action-loss
#       gradient flows straight back into the speaker's encoder. This is the
#       standard fix for the failure "reinforce" produces here (Foerster et al.
#       2016): with agent_0 hitting a macro boundary nearly every primitive step
#       it emits ~400 messages per episode, only a handful of which precede an
#       ingredient choice, yet all receive the same advantage.
COMM_CHANNELS = ("reinforce", "dial")


def route_messages(message, rng, num_envs: int, mode: str):
    """Map sent messages to received messages according to COMM_MODE.

    Every mode returns the same shape, so switching between them changes only
    what information crosses the channel -- never the architecture.
    """
    if mode == "normal":
        return swap_two_agent_messages(message, num_envs)
    if mode == "self":
        # Same input distribution to the correction head, but an agent hears
        # only itself: capacity is unchanged, transfer is zero.
        return message
    if mode == "constant":
        # For integer symbols this is symbol 0; for one-hot/soft messages it
        # must stay a valid distribution, so pin all the mass on symbol 0.
        if message.ndim == 1:
            return jnp.zeros_like(message)
        return jnp.zeros_like(message).at[..., 0].set(1.0)
    if mode == "shuffled":
        # Partner's message, but from a randomly chosen environment. The batch
        # is agent-major, so permute WITHIN each agent block: the listener
        # still hears its partner's role, and the symbol still has the right
        # marginal distribution -- only the link to THIS env's recipe is cut.
        swapped = swap_two_agent_messages(message, num_envs)
        num_agents = swapped.shape[0] // num_envs
        trailing = swapped.shape[1:]
        per_agent = swapped.reshape((num_agents, num_envs) + trailing)
        permutations = jax.vmap(jax.random.permutation, in_axes=(0, None))(
            jax.random.split(rng, num_agents), num_envs
        )
        # broadcast the gather index over any trailing (e.g. vocab) dimensions
        index = permutations.reshape(permutations.shape + (1,) * len(trailing))
        index = jnp.broadcast_to(index, (num_agents, num_envs) + trailing)
        return jnp.take_along_axis(per_agent, index, axis=1).reshape(swapped.shape)
    raise ValueError(f"Unknown COMM_MODE {mode!r}; expected one of {COMM_MODES}")


def gumbel_straight_through(logits, gumbel_noise, tau: float):
    """Discrete forward, differentiable backward (DIAL's channel).

    The forward value is a hard one-hot, so the channel really does carry one
    discrete symbol. The backward value is the softmax relaxation, so the
    listener's gradient reaches the speaker's logits. `gumbel_noise` is passed
    in rather than drawn here because the PPO update must reproduce the exact
    symbol the rollout sent -- otherwise the importance ratio would compare two
    different messages.
    """
    soft = jax.nn.softmax((logits + gumbel_noise) / tau, axis=-1)
    hard = jax.nn.one_hot(jnp.argmax(soft, axis=-1), logits.shape[-1])
    return hard + (soft - jax.lax.stop_gradient(soft))


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
    mode = config.get("COMM_MODE", "normal")
    if mode not in COMM_MODES:
        raise ValueError(f"COMM_MODE must be one of {COMM_MODES}, got {mode!r}")
    if config.get("COMM_USE_MEMORY", False) and not config.get("USE_RNN", False):
        raise ValueError("COMM_USE_MEMORY=true requires USE_RNN=true.")

    use_rnn = bool(config.get("USE_RNN", False))
    comm_use_memory = bool(config.get("COMM_USE_MEMORY", False))
    num_minibatches_cfg = int(config["NUM_MINIBATCHES"])
    comm_injection = config.get("COMM_INJECTION", "concat")
    if comm_injection not in COMM_INJECTIONS:
        raise ValueError(
            f"COMM_INJECTION must be one of {COMM_INJECTIONS}, got {comm_injection!r}"
        )
    comm_channel = config.get("COMM_CHANNEL", "reinforce")
    if comm_channel not in COMM_CHANNELS:
        raise ValueError(
            f"COMM_CHANNEL must be one of {COMM_CHANNELS}, got {comm_channel!r}"
        )
    if comm_channel == "dial" and comm_injection != "concat":
        raise ValueError(
            "COMM_CHANNEL=dial requires COMM_INJECTION=concat: the gradient "
            "reaches the speaker through the actor's input, which the 'bias' "
            "correction head does not provide."
        )
    if comm_channel == "dial" and int(config["NUM_ENVS"]) % num_minibatches_cfg != 0:
        raise ValueError(
            f"COMM_CHANNEL=dial splits minibatches by ENVIRONMENT to keep agent "
            f"pairs together, so NUM_ENVS ({config['NUM_ENVS']}) must be "
            f"divisible by NUM_MINIBATCHES ({num_minibatches_cfg})."
        )
    gumbel_tau = float(config.get("COMM_GUMBEL_TAU", 1.0))
    hidden_size = int(config["HIDDEN_SIZE"])
    comm_hidden_size = int(config.get("COMM_HIDDEN_SIZE", config["HIDDEN_SIZE"]))
    num_actors = int(config["NUM_ACTORS"])
    num_envs = int(config["NUM_ENVS"])
    num_steps = int(config["NUM_STEPS"])
    num_minibatches = int(config["NUM_MINIBATCHES"])
    if use_rnn and num_actors % num_minibatches != 0:
        raise ValueError(
            f"USE_RNN requires NUM_ACTORS ({num_actors}) divisible by "
            f"NUM_MINIBATCHES ({num_minibatches})."
        )

    def train(rng):
        actor = (
            ActorRNN(env.num_actions, hidden_size)
            if use_rnn
            else Actor(env.num_actions, hidden_size)
        )
        critic = CriticRNN(hidden_size) if use_rnn else Critic(hidden_size)
        comm_module = CommModule(
            hidden_size=comm_hidden_size,
            vocab_size=int(config["VOCAB_SIZE"]),
            action_dim=env.num_actions,
            message_embed_dim=int(config.get("MESSAGE_EMBED_DIM", 8)),
            use_memory=comm_use_memory,
            # Non-zero by default here (unlike the two-stage trainers): a zero
            # message head blocks gradient to the speaker's encoder and GRU
            # entirely, so the agent that can see the recipe cannot learn to
            # talk about it. Small enough that initial messages stay near
            # uniform.
            message_head_scale=float(config.get("MESSAGE_HEAD_INIT_SCALE", 0.01)),
        )

        obs_size = env.observation_space(env.agents[0]).shape[0]
        # Under "concat" the actor consumes [obs, message_embedding], so its
        # first layer is wider than the raw observation. This is why a concat
        # checkpoint and a bias checkpoint are not interchangeable.
        actor_input_size = obs_size + (
            int(config.get("MESSAGE_EMBED_DIM", 8))
            if comm_injection == "concat" else 0
        )
        world_state_size = env.world_state_size()
        init_actor_hidden = ScannedRNN.initialize_carry(num_actors, hidden_size)
        init_critic_hidden = ScannedRNN.initialize_carry(num_actors, hidden_size)
        init_comm_hidden = ScannedRNN.initialize_carry(num_actors, comm_hidden_size)

        rng, actor_rng, critic_rng, comm_rng = jax.random.split(rng, 4)
        if use_rnn:
            dummy_dones = jnp.zeros((1, num_actors), dtype=jnp.bool_)
            actor_params = actor.init(
                actor_rng, init_actor_hidden,
                (jnp.zeros((1, num_actors, actor_input_size)), dummy_dones),
            )
            critic_params = critic.init(
                critic_rng, init_critic_hidden,
                (jnp.zeros((1, num_actors, world_state_size)), dummy_dones),
            )
        else:
            actor_params = actor.init(actor_rng, jnp.zeros((1, actor_input_size)))
            critic_params = critic.init(critic_rng, jnp.zeros((1, world_state_size)))

        if comm_use_memory:
            comm_params = comm_module.init(
                comm_rng,
                jnp.zeros((1, num_actors, obs_size)),
                jnp.zeros((1, num_actors), dtype=jnp.int32),
                init_comm_hidden,
                jnp.zeros((1, num_actors), dtype=jnp.bool_),
            )
        else:
            comm_params = comm_module.init(
                comm_rng, jnp.zeros((1, obs_size)),
                jnp.zeros((1,), dtype=jnp.int32),
            )

        # Actor and comm share one optimiser slot as a single pytree, so PPO
        # updates them together -- this is what "joint" means here. The critic
        # is a normal trained critic (not frozen, unlike the two-stage script).
        policy_state = make_train_state(
            actor, {"actor": actor_params, "comm": comm_params},
            config, config["NUM_UPDATES"],
        )
        critic_state = make_train_state(
            critic, critic_params, config, config["NUM_UPDATES"]
        )

        rng, reset_rng = jax.random.split(rng)
        reset_keys = jax.random.split(reset_rng, num_envs)
        obs, env_state = jax.vmap(env.reset)(reset_keys)

        def encode(params, comm_hidden, obs_seq, dones_seq):
            """-> (comm_carry, summary, message_logits). Sequence-shaped."""
            if comm_use_memory:
                return comm_module.apply(
                    params["comm"], comm_hidden, obs_seq, dones_seq,
                    method=comm_module.encode_message_recurrent,
                )
            logits = comm_module.apply(
                params["comm"], obs_seq, method=comm_module.encode_message
            )
            return comm_hidden, None, logits

        def policy_logits(
            params, actor_hidden, summary, obs_seq, dones_seq, received_seq
        ):
            """Macro logits conditioned on the received message.

            COMM_INJECTION="concat" (default): the trainable embedding of the
            received symbol is concatenated onto the observation and fed to the
            ACTOR'S INPUT, so the trunk conditions on the message from its very
            first layer and the action-loss gradient reaches the embedding
            table through the actor.

            COMM_INJECTION="bias": the legacy additive correction head. It adds
            a separately-computed bias to the actor's OUTPUT logits. That design
            only ever existed because the two-stage trainer could not modify a
            frozen trunk's input; kept here purely as an ablation, since it
            makes the channel fight the trunk's own confident output instead of
            informing it.
            """
            if comm_injection == "concat":
                # Integer symbols during rollout (no gradient needed); under
                # DIAL the loss passes a (..., vocab) straight-through vector
                # instead, which must be embedded differentiably.
                if jnp.issubdtype(received_seq.dtype, jnp.integer):
                    embed = comm_module.apply(
                        params["comm"], received_seq, method=comm_module.embed_message
                    )
                else:
                    embed = comm_module.apply(
                        params["comm"], received_seq,
                        method=comm_module.embed_message_soft,
                    )
                actor_input = jnp.concatenate((obs_seq, embed), axis=-1)
            else:
                actor_input = obs_seq

            if use_rnn:
                actor_hidden, logits = actor.apply(
                    params["actor"], actor_hidden, (actor_input, dones_seq)
                )
            else:
                logits = actor.apply(params["actor"], actor_input)

            if comm_injection == "concat":
                return actor_hidden, logits

            if comm_use_memory:
                logit_bias = comm_module.apply(
                    params["comm"], summary, obs_seq, received_seq,
                    method=comm_module.correction_recurrent,
                )
            else:
                logit_bias = comm_module.apply(
                    params["comm"], obs_seq, received_seq,
                    method=comm_module.correction,
                )
            return actor_hidden, logits + logit_bias

        def evaluate(params, completed_updates):
            """Deterministic eval carrying carries, held messages and dones."""
            eval_key = jax.random.fold_in(
                jax.random.PRNGKey(int(config.get("EVAL_SEED", 42))),
                completed_updates,
            )
            num_eval_envs = int(config.get("NUM_EVAL_ENVS", 8))
            num_eval_actors = num_eval_envs * env.num_agents
            reset_keys = jax.random.split(eval_key, num_eval_envs)
            eval_obs, eval_env_state = jax.vmap(env.reset)(reset_keys)

            def eval_step(carry, _):
                (eval_obs, eval_env_state, actor_hidden, comm_hidden,
                 last_message, last_done, rng) = carry
                obs_batch = batchify(eval_obs, env.agents, num_eval_actors)
                action_mask = metadata_batch(
                    eval_obs["action_mask"], num_eval_actors
                ).astype(jnp.bool_)
                macro_done = metadata_batch(eval_obs["macro_done"], num_eval_actors)
                current_macro = metadata_batch(
                    eval_obs["current_macro"], num_eval_actors
                )

                comm_hidden, summary, message_logits = encode(
                    params, comm_hidden, obs_batch[None, :], last_done[None, :]
                )
                message = jnp.where(
                    macro_done,
                    jnp.argmax(message_logits.squeeze(0), axis=-1),
                    last_message,
                )
                rng, route_rng = jax.random.split(rng)
                received = route_messages(message, route_rng, num_eval_envs, mode)

                actor_hidden, final_logits = policy_logits(
                    params, actor_hidden, summary, obs_batch[None, :],
                    last_done[None, :], received[None, :],
                )
                final_logits = final_logits.squeeze(0)
                proposed = jnp.argmax(
                    jnp.where(action_mask, final_logits, -1e9), axis=-1
                )
                action = jnp.where(macro_done, proposed, current_macro)

                env_action = unbatchify(action, env.agents, num_eval_envs)
                rng, step_rng = jax.random.split(rng)
                step_keys = jax.random.split(step_rng, num_eval_envs)
                next_obs, next_env_state, reward, done, _ = jax.vmap(
                    env.step, in_axes=(0, 0, 0)
                )(step_keys, eval_env_state, env_action)
                mean_team_reward = jnp.mean(
                    jnp.stack([reward[a] for a in env.agents], axis=-1), axis=-1
                )
                next_done = jnp.tile(done["__all__"], env.num_agents)
                return (next_obs, next_env_state, actor_hidden, comm_hidden,
                        message, next_done, rng), mean_team_reward

            _, rewards = jax.lax.scan(
                eval_step,
                (eval_obs, eval_env_state,
                 ScannedRNN.initialize_carry(num_eval_actors, hidden_size),
                 ScannedRNN.initialize_carry(num_eval_actors, comm_hidden_size),
                 jnp.zeros((num_eval_actors,), dtype=jnp.int32),
                 jnp.zeros((num_eval_actors,), dtype=jnp.bool_),
                 eval_key),
                None, episode_steps,
            )
            return jnp.mean(jnp.sum(rewards, axis=0))

        empty_pending = {
            "obs": jnp.zeros((num_actors, obs_size), dtype=jnp.float32),
            "world_state": jnp.zeros((num_actors, world_state_size), dtype=jnp.float32),
            "action": jnp.zeros((num_actors,), dtype=jnp.int32),
            "action_mask": jnp.ones((num_actors, env.num_actions), dtype=jnp.bool_),
            "old_log_prob": jnp.zeros((num_actors,), dtype=jnp.float32),
            "old_value": jnp.zeros((num_actors,), dtype=jnp.float32),
            "message": jnp.zeros((num_actors,), dtype=jnp.int32),
            "old_message_log_prob": jnp.zeros((num_actors,), dtype=jnp.float32),
            "received_message": jnp.zeros((num_actors,), dtype=jnp.int32),
            "reward": jnp.zeros((num_actors,), dtype=jnp.float32),
            "discount": jnp.ones((num_actors,), dtype=jnp.float32),
            "duration": jnp.zeros((num_actors,), dtype=jnp.int32),
            "active": jnp.zeros((num_actors,), dtype=jnp.bool_),
            "start_index": jnp.zeros((num_actors,), dtype=jnp.int32),
        }

        def update_step(runner, update_index):
            (policy_state, critic_state, env_state, obs, last_done,
             last_message, hidden_states, rng) = runner
            rollout_start_hidden = hidden_states

            def env_step(step_runner, step_index):
                (env_state, obs, pending, last_done, last_message,
                 hidden_states, rng) = step_runner
                actor_hidden, critic_hidden, comm_hidden = hidden_states
                obs_batch = batchify(obs, env.agents, num_actors)
                world_state = metadata_batch(obs["world_state"], num_actors)
                macro_done = metadata_batch(obs["macro_done"], num_actors)
                current_macro = metadata_batch(obs["current_macro"], num_actors)
                action_mask = metadata_batch(
                    obs["action_mask"], num_actors
                ).astype(jnp.bool_)

                comm_hidden, summary, message_logits = encode(
                    policy_state.params, comm_hidden,
                    obs_batch[None, :], last_done[None, :],
                )
                message_logits = message_logits.squeeze(0)
                rng, msg_rng, act_rng, route_rng, step_rng = jax.random.split(rng, 5)
                message_policy = categorical(message_logits)
                # Drawn every step even under DIAL so the RNG stream (and hence
                # the rest of the rollout) does not depend on COMM_CHANNEL.
                gumbel_noise = jax.random.gumbel(
                    msg_rng, message_logits.shape, dtype=message_logits.dtype
                )
                if comm_channel == "dial":
                    sampled_message = jnp.argmax(message_logits + gumbel_noise, axis=-1)
                else:
                    sampled_message = message_policy.sample(seed=msg_rng)
                # Speak only at a macro boundary; hold otherwise.
                message = jnp.where(macro_done, sampled_message, last_message)
                message_log_prob = message_policy.log_prob(message)
                received = route_messages(message, route_rng, num_envs, mode)

                actor_hidden, final_logits = policy_logits(
                    policy_state.params, actor_hidden, summary,
                    obs_batch[None, :], last_done[None, :], received[None, :],
                )
                policy = masked_categorical(final_logits.squeeze(0), action_mask)
                proposed_action = policy.sample(seed=act_rng)
                proposed_log_prob = policy.log_prob(proposed_action)

                if use_rnn:
                    critic_hidden, value = critic.apply(
                        critic_state.params, critic_hidden,
                        (world_state[None, :], last_done[None, :]),
                    )
                    value = value.squeeze(0)
                else:
                    value = critic.apply(critic_state.params, world_state)

                def start(new, old):
                    shape = (num_actors,) + (1,) * (new.ndim - 1)
                    return jnp.where(macro_done.reshape(shape), new, old)

                pending = {
                    "obs": start(obs_batch, pending["obs"]),
                    "world_state": start(world_state, pending["world_state"]),
                    "action": start(proposed_action, pending["action"]),
                    "action_mask": start(action_mask, pending["action_mask"]),
                    "old_log_prob": start(proposed_log_prob, pending["old_log_prob"]),
                    "old_value": start(value, pending["old_value"]),
                    "message": start(message, pending["message"]),
                    "old_message_log_prob": start(
                        message_log_prob, pending["old_message_log_prob"]
                    ),
                    "received_message": start(received, pending["received_message"]),
                    "reward": jnp.where(macro_done, 0.0, pending["reward"]),
                    "discount": jnp.where(macro_done, 1.0, pending["discount"]),
                    "duration": jnp.where(macro_done, 0, pending["duration"]),
                    "active": pending["active"] | macro_done,
                    "start_index": jnp.where(
                        macro_done, step_index, pending["start_index"]
                    ),
                }

                action = jnp.where(macro_done, proposed_action, current_macro)
                env_action = unbatchify(action, env.agents, num_envs)
                step_keys = jax.random.split(step_rng, num_envs)
                next_obs, next_env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0)
                )(step_keys, env_state, env_action)

                primitive_timestep = (update_index * num_steps + step_index) * num_envs
                reward, shaping_coefficient = add_annealed_shaped_reward(
                    reward, info["shaped_reward"], primitive_timestep,
                    float(config.get("REW_SHAPING_HORIZON", 0.0)),
                )
                raw_burn = {
                    a: info["reward_breakdown"]["BURN_PENALTY"][:, i]
                    for i, a in enumerate(env.agents)
                }
                reward, burn_coefficient = anneal_burn_penalty(
                    reward, raw_burn, primitive_timestep,
                    float(config.get("REW_SHAPING_HORIZON", 0.0)),
                )
                reward_breakdown = {
                    k: metadata_batch(info["reward_breakdown"][k], num_actors)
                    for k in REWARD_COMPONENT_KEYS
                }

                reward_batch = batchify(reward, env.agents, num_actors)
                accumulated_reward = (
                    pending["reward"] + pending["discount"] * reward_batch
                )
                duration = pending["duration"] + 1
                completed = metadata_batch(
                    jnp.stack([info["macro_action_done"][a] for a in env.agents],
                              axis=-1),
                    num_actors,
                )
                valid = completed & pending["active"]
                next_done = jnp.tile(done["__all__"], env.num_agents)

                transition = {
                    # per-step stream (what a GRU scans; macro-START aligned)
                    "step_obs": obs_batch,
                    "step_world_state": world_state,
                    "step_action": proposed_action,
                    "step_action_mask": action_mask,
                    "step_old_log_prob": proposed_log_prob,
                    "step_old_value": value,
                    "step_prev_done": last_done,
                    "step_message": message,
                    "step_old_message_log_prob": message_log_prob,
                    "step_received_message": received,
                    # DIAL re-derives the message inside the loss; replaying the
                    # same noise makes that reproduce the symbol actually sent.
                    "step_gumbel_noise": gumbel_noise,
                    "step_macro_done": macro_done,
                    "step_last_message": last_message,
                    # macro-start snapshots (used by the MLP path)
                    "pending_obs": pending["obs"],
                    "pending_world_state": pending["world_state"],
                    "pending_action": pending["action"],
                    "pending_action_mask": pending["action_mask"],
                    "pending_old_log_prob": pending["old_log_prob"],
                    "pending_message": pending["message"],
                    "pending_old_message_log_prob": pending["old_message_log_prob"],
                    "pending_received_message": pending["received_message"],
                    # completion aligned
                    "reward": accumulated_reward,
                    "duration": duration,
                    "old_value": pending["old_value"],
                    "done": next_done,
                    "valid": valid,
                    "start_index": pending["start_index"],
                    # logging
                    "shaped_reward": batchify(
                        info["shaped_reward"], env.agents, num_actors
                    ),
                    "shaping_coefficient": jnp.full((num_actors,), shaping_coefficient),
                    "burn_penalty_coefficient": jnp.full((num_actors,), burn_coefficient),
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
                return (next_env_state, next_obs, pending, next_done, message,
                        (actor_hidden, critic_hidden, comm_hidden), rng), transition

            (env_state, obs, pending, last_done, last_message,
             hidden_states, rng), trajectory = jax.lax.scan(
                env_step,
                (env_state, obs, empty_pending, last_done, last_message,
                 hidden_states, rng),
                jnp.arange(num_steps), num_steps,
            )

            advantage, target = calculate_smdp_gae(
                trajectory["reward"], trajectory["duration"], trajectory["done"],
                trajectory["old_value"], trajectory["valid"],
                config["GAMMA"], config["GAE_LAMBDA"],
            )

            if use_rnn:
                # Score at macro-START rows so the RNN outputs line up with the
                # decision; move each advantage from its completion row back.
                actor_index = jnp.broadcast_to(
                    jnp.arange(num_actors)[None, :], (num_steps, num_actors)
                )
                scatter_rows = jnp.where(
                    trajectory["valid"], trajectory["start_index"], num_steps
                )
                zeros = jnp.zeros((num_steps, num_actors), dtype=jnp.float32)
                batch = {
                    "obs": trajectory["step_obs"],
                    "world_state": trajectory["step_world_state"],
                    "action": trajectory["step_action"],
                    "action_mask": trajectory["step_action_mask"],
                    "old_log_prob": trajectory["step_old_log_prob"],
                    "old_value": trajectory["step_old_value"],
                    "prev_done": trajectory["step_prev_done"],
                    "message": trajectory["step_message"],
                    "old_message_log_prob": trajectory["step_old_message_log_prob"],
                    "received_message": trajectory["step_received_message"],
                    "gumbel_noise": trajectory["step_gumbel_noise"],
                    "macro_done": trajectory["step_macro_done"],
                    "last_message": trajectory["step_last_message"],
                    "advantage": zeros.at[scatter_rows, actor_index].set(
                        advantage, mode="drop"),
                    "target": zeros.at[scatter_rows, actor_index].set(
                        target, mode="drop"),
                    "loss_mask": jnp.zeros((num_steps, num_actors), dtype=jnp.bool_)
                        .at[scatter_rows, actor_index].set(True, mode="drop"),
                    "init_actor_hidden": rollout_start_hidden[0][None, :],
                    "init_critic_hidden": rollout_start_hidden[1][None, :],
                    "init_comm_hidden": rollout_start_hidden[2][None, :],
                }
                if comm_channel == "dial":
                    # Routing inside the loss needs both agents of an env in the
                    # same minibatch, so split by environment, not by actor.
                    minibatch_fn = lambda r, b: paired_sequence_minibatches(
                        r, b, num_minibatches, num_envs, env.num_agents
                    )
                else:
                    minibatch_fn = lambda r, b: sequence_minibatches(
                        r, b, num_minibatches, num_actors
                    )
            else:
                batch = jax.tree.map(
                    lambda x: x.reshape((-1,) + x.shape[2:]),
                    {
                        "obs": trajectory["pending_obs"],
                        "world_state": trajectory["pending_world_state"],
                        "action": trajectory["pending_action"],
                        "action_mask": trajectory["pending_action_mask"],
                        "old_log_prob": trajectory["pending_old_log_prob"],
                        "old_value": trajectory["old_value"],
                        "message": trajectory["pending_message"],
                        "old_message_log_prob":
                            trajectory["pending_old_message_log_prob"],
                        "received_message": trajectory["pending_received_message"],
                        "advantage": advantage,
                        "target": target,
                        "loss_mask": trajectory["valid"],
                    },
                )
                minibatch_fn = None

            def policy_loss_fn(params, minibatch):
                prev_done = minibatch.get("prev_done")
                comm_carry = (
                    minibatch["init_comm_hidden"][0] if use_rnn else None
                )
                _, summary, message_logits = encode(
                    params, comm_carry, minibatch["obs"], prev_done
                )

                if comm_channel == "dial":
                    # Re-derive the message HERE, from params, so that the
                    # action loss below differentiates through the channel into
                    # the speaker. Replaying the rollout's gumbel noise makes
                    # this reproduce the symbol that was actually sent, keeping
                    # the PPO importance ratio meaningful.
                    sent = gumbel_straight_through(
                        message_logits, minibatch["gumbel_noise"], gumbel_tau
                    )
                    # Outside a macro boundary the agent holds its previous
                    # message. That was emitted at an earlier boundary, so its
                    # gradient belongs to that step; using the stored symbol
                    # here truncates the path rather than double-counting it.
                    held = jax.nn.one_hot(
                        minibatch["last_message"], int(config["VOCAB_SIZE"])
                    )
                    sent = jnp.where(minibatch["macro_done"][..., None], sent, held)
                    # Route along the ACTOR axis (axis 1 of this time-major
                    # batch); paired_sequence_minibatches guarantees both agents
                    # of an environment are present and agent-major.
                    actors_in_minibatch = sent.shape[1]
                    envs_in_minibatch = actors_in_minibatch // env.num_agents
                    routed = route_messages(
                        jnp.moveaxis(sent, 1, 0),
                        jax.random.PRNGKey(0), envs_in_minibatch, mode,
                    )
                    received = jnp.moveaxis(routed, 0, 1)
                else:
                    received = minibatch["received_message"]

                _, final_logits = policy_logits(
                    params,
                    minibatch["init_actor_hidden"][0] if use_rnn else None,
                    summary, minibatch["obs"], prev_done, received,
                )
                policy = masked_categorical(final_logits, minibatch["action_mask"])
                action_loss, action_metrics = clipped_actor_loss(
                    policy.log_prob(minibatch["action"]),
                    minibatch["old_log_prob"], minibatch["advantage"],
                    policy.entropy(), minibatch["loss_mask"],
                    config["CLIP_EPS"], config["ENT_COEF"],
                )

                message_policy = categorical(message_logits)
                message_entropy = masked_mean(
                    message_policy.entropy(), minibatch["loss_mask"]
                )
                if comm_channel == "dial":
                    # No REINFORCE term: the speaker is already trained by the
                    # gradient flowing back through `received` above. Entropy is
                    # kept as the only pressure against symbol collapse.
                    message_loss = -config.get(
                        "MESSAGE_ENT_COEF", config["ENT_COEF"]
                    ) * message_entropy
                    message_metrics = {"entropy": message_entropy}
                else:
                    message_loss, message_metrics = clipped_actor_loss(
                        message_policy.log_prob(minibatch["message"]),
                        minibatch["old_message_log_prob"], minibatch["advantage"],
                        message_policy.entropy(), minibatch["loss_mask"],
                        config["CLIP_EPS"],
                        config.get("MESSAGE_ENT_COEF", config["ENT_COEF"]),
                    )

                total = action_loss + config.get("MESSAGE_LOSS_COEF", 1.0) * message_loss
                metrics = {
                    **{f"action_{k}": v for k, v in action_metrics.items()},
                    **{f"message_{k}": v for k, v in message_metrics.items()},
                }
                return total, metrics

            def critic_predict(params, minibatch):
                _, value = critic.apply(
                    params, minibatch["init_critic_hidden"][0],
                    (minibatch["world_state"], minibatch["prev_done"]),
                )
                return value

            rng, policy_state, critic_state, loss_metrics = update_ppo(
                rng, policy_state, critic_state, batch, policy_loss_fn, config,
                critic_predict=critic_predict if use_rnn else None,
                minibatch_fn=minibatch_fn,
            )

            event_mask = trajectory["valid"]
            episode_mask = trajectory["returned_episode"]
            decision_mask = (
                batch["loss_mask"] if use_rnn else event_mask
            )
            metrics = {
                **loss_metrics,
                "episode_return": jnp.sum(
                    trajectory["returned_episode_returns"] * episode_mask
                ) / jnp.maximum(jnp.sum(episode_mask), 1),
                "mean_macro_duration": jnp.sum(trajectory["duration"] * event_mask)
                    / jnp.maximum(jnp.sum(event_mask), 1),
                "macro_decisions": jnp.sum(event_mask),
                "unfinished_macros": jnp.sum(pending["active"]),
                "mean_shaped_reward": jnp.mean(trajectory["shaped_reward"]),
                "shaping_coefficient": jnp.mean(trajectory["shaping_coefficient"]),
                "burn_penalty_coefficient": jnp.mean(
                    trajectory["burn_penalty_coefficient"]
                ),
                # Collapse detector: a channel pinned to one symbol carries
                # nothing, whatever the return curve looks like.
                "message_nonzero_fraction": jnp.sum(
                    (trajectory["step_message"] != 0) * decision_mask
                ) / jnp.maximum(jnp.sum(decision_mask), 1),
                **{
                    f"reward/{k}": jnp.mean(trajectory["reward_breakdown"][k])
                    for k in REWARD_COMPONENT_KEYS
                },
            }
            metrics["eval_return"] = maybe_evaluate_and_save_best(
                update_index, policy_state, critic_state, evaluate, config
            )
            next_runner = (policy_state, critic_state, env_state, obs, last_done,
                           last_message, hidden_states, rng)
            emit_live_metrics(
                update_index, metrics, num_steps * num_envs, config
            )
            maybe_checkpoint(update_index, next_runner, config)
            return next_runner, metrics

        initial_runner = (
            policy_state, critic_state, env_state, obs,
            jnp.zeros((num_actors,), dtype=jnp.bool_),
            jnp.zeros((num_actors,), dtype=jnp.int32),
            (init_actor_hidden, init_critic_hidden, init_comm_hidden),
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
    config_name="mappo_macro_boundary_joint_comm",
)
def main(config):
    config = OmegaConf.to_container(config, resolve=True)
    if config["ENV_NAME"] != "overcooked_v3_macro":
        raise ValueError(
            "Boundary joint comm MAPPO requires the committed macro environment "
            "(overcooked_v3_macro)."
        )
    run_experiment(config, make_train, Path(__file__).stem)


if __name__ == "__main__":
    main()
