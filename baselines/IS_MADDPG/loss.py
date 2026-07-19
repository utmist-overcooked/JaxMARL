import jax
import jax.numpy as jnp
import flax.linen as nn
import chex
from typing import Tuple

from networks import ISAgentNet, ISCriticNet
from buffer import Batch
# ISAgentNet contains the action_predictor and obs_predictor definitions

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def received_messages(msgs: jnp.ndarray) -> jnp.ndarray:
    """Build per-agent received message tensors by excluding each agent's own message.

    Args:
        msgs: (B, N, msg_dim)  all agents' outgoing messages

    Returns:
        (B, N, N-1, msg_dim)  for each agent i, the messages from all j != i
    """
    B, N, msg_dim = msgs.shape

    # Build index of "other agents" for each agent i: shape (N, N-1)
    # For agent i, other_idx[i] = [0, 1, ..., i-1, i+1, ..., N-1]
    all_idx   = jnp.arange(N)
    other_idx = jnp.array([[j for j in range(N) if j != i] for i in range(N)])  # (N, N-1)

    # Gather: msgs[:, other_idx[i], :] for each i
    # msgs[:, other_idx, :] -> (B, N, N-1, msg_dim)
    return msgs[:, other_idx, :]  # JAX advanced indexing broadcasts over B


# ---------------------------------------------------------------------------
# Critic loss  (step 5)
# ---------------------------------------------------------------------------

def critic_loss(
    critic_params:        chex.ArrayTree,
    target_critic_params: chex.ArrayTree,
    target_actor_params:  chex.ArrayTree,
    critic:               ISCriticNet,
    target_critic:        ISCriticNet,
    target_actor:         ISAgentNet,
    batch:                Batch,
    agent_idx:            int,
    *,
    gamma:      float,
    gumbel_tau: float,
    rng:        chex.PRNGKey,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Bellman MSE loss for the centralised critic of agent `agent_idx`.

    Target:  y = r_i + γ * (1 - done) * Q_target(s', a'_target)
    Loss:    L = MSE(Q(s, a), y)

    All agents' target actions are computed from the shared target actor,
    then used to evaluate the target critic. Gradients do NOT flow through
    target networks (they are updated via Polyak averaging, not backprop).

    Args:
        critic_params:        parameters of the online critic
        target_critic_params: parameters of the target critic (stop-gradient)
        target_actor_params:  parameters of the target actor  (stop-gradient)
        critic:               ISCriticNet module (apply-only, stateless)
        target_critic:        ISCriticNet module for target Q
        target_actor:         ISAgentNet module for target actions
        batch:                sampled minibatch from the replay buffer
        agent_idx:            which agent's critic loss to compute (0..N-1)
        gamma:                discount factor
        gumbel_tau:           Gumbel temperature for target actor actions
        rng:                  PRNG key (consumed for target actor sampling)

    Returns:
        loss:  scalar MSE loss
        q_val: (B, 1) online Q-values (for logging)
    """
    B  = batch.obs.shape[0]
    N  = batch.obs.shape[1]

    # ------------------------------------------------------------------
    # 1. Compute target actions for all agents at next state (no gradient)
    # ------------------------------------------------------------------
    next_received = received_messages(batch.next_prev_msgs)  # (B, N, N-1, msg_dim)

    # Collect next actions and messages from target actor for all agents
    next_actions_list = []
    next_msgs_list    = []

    for j in range(N):
        rng, subkey = jax.random.split(rng)
        _, next_a_oh, next_m, _ = target_actor.apply(
            target_actor_params,
            batch.next_obs[:, j, :],       # (B, obs_dim)
            next_received[:, j, :, :],     # (B, N-1, msg_dim)
            rng=subkey,
            gumbel_tau=gumbel_tau,
            gumbel_hard=True,
        )
        next_actions_list.append(next_a_oh)
        next_msgs_list.append(next_m)

    next_actions_all = jnp.stack(next_actions_list, axis=1)  # (B, N, act_dim)
    next_msgs_all    = jnp.stack(next_msgs_list,    axis=1)  # (B, N, msg_dim)

    # ------------------------------------------------------------------
    # 2. Compute Bellman target y (no gradient through target networks)
    # ------------------------------------------------------------------
    agent_id_batch = jnp.full((B,), agent_idx, dtype=jnp.int32)

    q_next = target_critic.apply(
        target_critic_params,
        batch.next_obs,
        batch.next_prev_msgs,
        next_actions_all,
        next_msgs_all,
        agent_id_batch,
    )  # (B, 1)

    reward_i = batch.rewards[:, agent_idx : agent_idx + 1]   # (B, 1)
    done     = batch.dones[:, None]                           # (B, 1)

    # Bellman target — stop_gradient ensures no gradients flow into target nets
    y = jax.lax.stop_gradient(
        reward_i + gamma * (1.0 - done) * q_next
    )  # (B, 1)

    # ------------------------------------------------------------------
    # 3. Online Q-value and MSE loss
    # ------------------------------------------------------------------
    q_val = critic.apply(
        critic_params,
        batch.obs,
        batch.prev_msgs,
        batch.actions,
        batch.msgs,
        agent_id_batch,
    )  # (B, 1)

    loss = jnp.mean((q_val - y) ** 2)

    return loss, q_val


# ---------------------------------------------------------------------------
# Actor loss
# ---------------------------------------------------------------------------

def actor_loss(
    actor_params:   chex.ArrayTree,
    critic_params:  chex.ArrayTree,
    actor:          ISAgentNet,
    critic:         ISCriticNet,
    batch:          Batch,
    agent_idx:      int,
    *,
    gumbel_tau:     float,
    gumbel_hard:    bool,
    pred_loss_coef: float,
    rng:            chex.PRNGKey,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """MADDPG actor loss + auxiliary intention-prediction losses.

    Total loss = -Q(s, a_current) + pred_loss_coef * (action_pred + obs_pred)

    The actor loss maximises the critic's Q-value estimate.
    The prediction losses teach the actor's world model to accurately
    predict other agents' actions and the next observation, which
    enables better imagined rollouts and thus better messages.

    Other agents' actions: gradient is STOPPED (detached) — we only
    optimise agent i's own actor, not others'. Each agent's policy is updated independently.

    Args:
        actor_params:   parameters of the online actor (gradient flows here)
        critic_params:  parameters of the online critic (stop-gradient)
        actor:          ISAgentNet module
        critic:         ISCriticNet module
        batch:          sampled minibatch
        agent_idx:      which agent to update (0..N-1)
        gumbel_tau:     Gumbel temperature
        gumbel_hard:    straight-through one-hot if True
        pred_loss_coef: weight on the auxiliary prediction losses
        rng:            PRNG key

    Returns:
        total_loss:       scalar (actor Q-loss + weighted pred losses)
        actor_q_loss:     scalar (-mean Q) for logging
        pred_loss_scalar: scalar (action_pred + obs_pred) for logging
    """
    B = batch.obs.shape[0]
    N = batch.obs.shape[1]

    received = received_messages(batch.prev_msgs)  # (B, N, N-1, msg_dim)

    # ------------------------------------------------------------------
    # 1. Compute current actions + messages for ALL agents
    #
    # Agent i:     gradient flows through (actor_params)
    # Agent j≠i:   stop_gradient — we don't update other agents' policies
    #              here, they get their own separate loss computation.
    # ------------------------------------------------------------------
    cur_actions_list = []
    cur_msgs_list    = []

    for j in range(N):
        rng, subkey = jax.random.split(rng)
        _, a_oh_j, m_j, _ = actor.apply(
            actor_params,
            batch.obs[:, j, :],        # (B, obs_dim)
            received[:, j, :, :],      # (B, N-1, msg_dim)
            rng=subkey,
            gumbel_tau=gumbel_tau,
            gumbel_hard=gumbel_hard,
        )

        if j != agent_idx:
            # Stop gradient for other agents — their actors are not being
            # updated in this loss computation
            a_oh_j = jax.lax.stop_gradient(a_oh_j)
            m_j    = jax.lax.stop_gradient(m_j)

        cur_actions_list.append(a_oh_j)
        cur_msgs_list.append(m_j)

    cur_actions_all = jnp.stack(cur_actions_list, axis=1)  # (B, N, act_dim)
    cur_msgs_all    = jnp.stack(cur_msgs_list,    axis=1)  # (B, N, msg_dim)

    # ------------------------------------------------------------------
    # 2. Actor Q-loss: maximise Q(s, a_current) for agent i
    #    Critic params are stop-gradiented — critic is fixed during actor update
    # ------------------------------------------------------------------
    agent_id_batch = jnp.full((B,), agent_idx, dtype=jnp.int32)

    actor_q = critic.apply(
        jax.lax.stop_gradient(critic_params),
        batch.obs,
        batch.prev_msgs,
        cur_actions_all,
        cur_msgs_all,
        agent_id_batch,
    )  # (B, 1)

    # Negative because we maximise Q (gradient ascent = minimise -Q)
    actor_q_loss = -jnp.mean(actor_q)

    # ------------------------------------------------------------------
    # 3. Auxiliary prediction losses for agent i's world model
    # ------------------------------------------------------------------

    # --- 3a. Other-agent action prediction loss (cross-entropy) ---
    # Predict what other agents did, given agent i's current obs.
    # This trains action_predictor, which feeds into the imagined rollout.
    pred_other_logits = actor.apply(
        actor_params,
        batch.obs[:, agent_idx, :],       # (B, obs_dim)  agent i's obs only
        received[:, agent_idx, :, :],     # (B, N-1, msg_dim)
        rng=jax.random.PRNGKey(0),        # dummy key — we only need the predictor output
        gumbel_tau=gumbel_tau,
        gumbel_hard=False,
    )
    # Re-run just the action_predictor submodule to get its logits directly
    # (the full forward pass above discards them; we need raw predictor output)
    raw_pred_logits = actor.apply(
        actor_params,
        batch.obs[:, agent_idx, :],
        method=lambda module, x: module.action_predictor(x),
    )  # (B, (N-1)*act_dim)

    act_dim = batch.actions.shape[-1]
    raw_pred_logits = raw_pred_logits.reshape(B, N - 1, act_dim)  # (B, N-1, act_dim)

    # Ground-truth: other agents' actual actions (argmax of one-hot)
    other_actions = jnp.concatenate([
        batch.actions[:, :agent_idx, :],
        batch.actions[:, agent_idx + 1:, :],
    ], axis=1)  # (B, N-1, act_dim)

    other_act_idx = jnp.argmax(other_actions, axis=-1)  # (B, N-1)  integer labels

    # Softmax cross-entropy over flattened (B*(N-1), act_dim)
    action_pred_loss = jnp.mean(
        -jnp.sum(
            jax.nn.one_hot(other_act_idx.reshape(-1), act_dim) *
            jax.nn.log_softmax(raw_pred_logits.reshape(-1, act_dim), axis=-1),
            axis=-1,
        )
    )

    # --- 3b. Next-observation prediction loss (MSE on delta) ---
    # Predict the change in agent i's obs given (obs_i, action_i, actions_others).
    # Residual (delta) prediction is more stable than predicting absolute next obs.
    own_action    = batch.actions[:, agent_idx, :]          # (B, act_dim)
    other_actions_flat = jnp.concatenate([
        batch.actions[:, :agent_idx, :],
        batch.actions[:, agent_idx + 1:, :],
    ], axis=1).reshape(B, -1)                               # (B, (N-1)*act_dim)

    obs_pred_input = jnp.concatenate([
        batch.obs[:, agent_idx, :],   # (B, obs_dim)
        own_action,                   # (B, act_dim)
        other_actions_flat,           # (B, (N-1)*act_dim)
    ], axis=-1)

    delta_pred = actor.apply(
        actor_params,
        obs_pred_input,
        method=lambda module, x: module.obs_predictor(x),
    )  # (B, obs_dim)

    delta_true   = batch.next_obs[:, agent_idx, :] - batch.obs[:, agent_idx, :]
    obs_pred_loss = jnp.mean((delta_pred - delta_true) ** 2)

    # ------------------------------------------------------------------
    # 4. Combine losses
    # ------------------------------------------------------------------
    pred_loss_scalar = action_pred_loss + obs_pred_loss
    total_loss       = actor_q_loss + pred_loss_coef * pred_loss_scalar

    return total_loss, actor_q_loss, pred_loss_scalar