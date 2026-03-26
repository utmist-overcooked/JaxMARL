import jax
import jax.numpy as jnp
import optax
import chex
from typing import NamedTuple, Tuple, Dict
from functools import partial

from networks import ISAgentNet, ISCriticNet
from buffer import Batch
from loss import critic_loss, actor_loss, received_messages


# ---------------------------------------------------------------------------
# Training state  (immutable snapshot passed around instead of self.*)
# ---------------------------------------------------------------------------

class TrainState(NamedTuple):
    """Complete mutable state of the IS-MADDPG trainer.

    All arrays are JAX pytrees so this can be passed into jit-compiled
    functions. Flax parameter dicts and optax optimizer states are both
    valid JAX pytrees.

    Fields:
        actor_params:         online actor parameters
        target_actor_params:  target actor parameters (Polyak-lagged copy)
        critic_params:        online critic parameters
        target_critic_params: target critic parameters (Polyak-lagged copy)
        actor_opt_state:      optax optimizer state for the actor
        critic_opt_state:     optax optimizer state for the critic
        rng:                  current PRNG key (advanced each update)
    """
    actor_params:         chex.ArrayTree
    target_actor_params:  chex.ArrayTree
    critic_params:        chex.ArrayTree
    target_critic_params: chex.ArrayTree
    actor_opt_state:      optax.OptState
    critic_opt_state:     optax.OptState
    rng:                  chex.PRNGKey


class UpdateMetrics(NamedTuple):
    """Logged scalars returned from a single train step, matching PyTorch."""
    critic_loss: float
    actor_loss:  float
    pred_loss:   float
    q_mean:      float


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

def init_train_state(
    *,
    actor:       ISAgentNet,
    critic:      ISCriticNet,
    actor_lr:    float,
    critic_lr:   float,
    grad_clip:   float,
    obs_dim:     int,
    num_agents:  int,
    msg_dim:     int,
    act_dim:     int,
    batch_size:  int,
    rng:         chex.PRNGKey,
) -> TrainState:
    """Initialise all parameters and optimizer states.

    Uses dummy inputs of the right shape to trigger Flax's parameter
    initialisation via net.init(). Target networks are initialised as
    exact copies of the online networks (τ=1 Polyak at step 0).

    Args:
        actor:      ISAgentNet module (stateless — holds no params)
        critic:     ISCriticNet module (stateless)
        actor_lr:   Adam learning rate for the actor
        critic_lr:  Adam learning rate for the critic
        obs_dim:    per-agent observation dimension
        num_agents: number of agents N
        msg_dim:    per-agent message dimension
        act_dim:    per-agent action dimension
        batch_size: used for dummy input shapes during init
        rng:        PRNG key (split internally)

    Returns:
        Fully initialised TrainState.
    """
    rng, rng_actor, rng_critic, rng_init = jax.random.split(rng, 4)

    # Dummy inputs — only shapes matter for init, values are irrelevant
    dummy_obs      = jnp.zeros((batch_size, obs_dim))
    dummy_msgs     = jnp.zeros((batch_size, num_agents - 1, msg_dim))
    dummy_obs_all  = jnp.zeros((batch_size, num_agents, obs_dim))
    dummy_msg_all  = jnp.zeros((batch_size, num_agents, msg_dim))
    dummy_act_all  = jnp.zeros((batch_size, num_agents, act_dim))
    dummy_agent_id = jnp.zeros((batch_size,), dtype=jnp.int32)

    actor_params  = actor.init(rng_actor,  dummy_obs, dummy_msgs, rng=rng_init)
    critic_params = critic.init(rng_critic, dummy_obs_all, dummy_msg_all,
                                dummy_act_all, dummy_msg_all, dummy_agent_id)

    # Target networks start as exact copies of online networks
    target_actor_params  = actor_params
    target_critic_params = critic_params

    def make_opt(lr):
        return optax.chain(
            optax.clip_by_global_norm(grad_clip),
            optax.adam(lr),
        )

    actor_opt_state  = make_opt(actor_lr).init(actor_params)
    critic_opt_state = make_opt(critic_lr).init(critic_params)

    return TrainState(
        actor_params=         actor_params,
        target_actor_params=  target_actor_params,
        critic_params=        critic_params,
        target_critic_params= target_critic_params,
        actor_opt_state=      actor_opt_state,
        critic_opt_state=     critic_opt_state,
        rng=                  rng,
    )


# ---------------------------------------------------------------------------
# Polyak (soft) target update  (step 8)
# ---------------------------------------------------------------------------

def polyak_update(
    online_params: chex.ArrayTree,
    target_params: chex.ArrayTree,
    tau:           float,
) -> chex.ArrayTree:
    """Exponential moving average update of target network parameters.

    target = (1 - τ) * target + τ * online

    τ=1.0 → hard copy (used at init)
    τ=0.01 → slow-moving target (typical MADDPG value)

    Uses jax.tree_util.tree_map to apply the update to every leaf in the
    parameter pytree simultaneously, without any explicit loops.

    Args:
        online_params: current online network parameters
        target_params: current target network parameters
        tau:           interpolation coefficient ∈ (0, 1]

    Returns:
        Updated target parameters.
    """
    return jax.tree_util.tree_map(
        lambda online, target: (1.0 - tau) * target + tau * online,
        online_params,
        target_params,
    )


# ---------------------------------------------------------------------------
# Single train step  (step 7)
# ---------------------------------------------------------------------------

def train_step(
    state:    TrainState,
    batch:    Batch,
    actor:    ISAgentNet,
    critic:   ISCriticNet,
    *,
    gamma:          float,
    tau:            float,
    gumbel_tau:     float,
    gumbel_hard:    bool,
    pred_loss_coef: float,
    grad_clip:      float,
    num_agents:     int,
    actor_lr:       float,
    critic_lr:      float,
) -> Tuple[TrainState, UpdateMetrics]:
    """One full IS-MADDPG update.

    Order:
        1. Compute all target actions at next state (no gradient)
        2. Update critic for each agent   (MSE Bellman loss)
        3. Update actor for each agent    (Q-loss + prediction losses)
        4. Polyak update both target networks

    All per-agent critic updates share the same target actions computed
    in step 1, which is more efficient than recomputing per agent and
    matches the PyTorch implementation.

    Args:
        state:          current TrainState
        batch:          minibatch sampled from the replay buffer
        actor:          ISAgentNet module (stateless)
        critic:         ISCriticNet module (stateless)
        gamma:          discount factor
        tau:            Polyak averaging coefficient
        gumbel_tau:     Gumbel temperature
        gumbel_hard:    straight-through one-hot if True
        pred_loss_coef: weight on auxiliary prediction losses
        grad_clip:      max gradient norm (applied per optimizer step)
        num_agents:     number of agents N
        actor_lr:       Adam learning rate for actor (used to rebuild opt)
        critic_lr:      Adam learning rate for critic (used to rebuild opt)

    Returns:
        new_state: updated TrainState
        metrics:   UpdateMetrics for logging
    """
    rng = state.rng
    B   = batch.obs.shape[0]

    # Rebuild optimizer (stateless in optax — state is in TrainState)
    def make_opt(lr):
        return optax.chain(
            optax.clip_by_global_norm(grad_clip),
            optax.adam(lr),
        )
    actor_opt  = make_opt(actor_lr)
    critic_opt = make_opt(critic_lr)

    # ------------------------------------------------------------------
    # 1. Target actions at next state (no gradient — matches torch.no_grad)
    # ------------------------------------------------------------------
    next_received = received_messages(batch.next_prev_msgs)  # (B, N, N-1, msg_dim)

    next_actions_list = []
    next_msgs_list    = []

    for j in range(num_agents):
        rng, subkey = jax.random.split(rng)
        _, next_a_oh, next_m, _ = actor.apply(
            state.target_actor_params,
            batch.next_obs[:, j, :],
            next_received[:, j, :, :],
            rng=subkey,
            gumbel_tau=gumbel_tau,
            gumbel_hard=True,
        )
        # Stop gradient — target actor is never updated by backprop
        next_actions_list.append(jax.lax.stop_gradient(next_a_oh))
        next_msgs_list.append(jax.lax.stop_gradient(next_m))

    next_actions_all = jnp.stack(next_actions_list, axis=1)  # (B, N, act_dim)
    next_msgs_all    = jnp.stack(next_msgs_list,    axis=1)  # (B, N, msg_dim)

    # ------------------------------------------------------------------
    # 2. Critic updates — one per agent, sequential
    #    (agents share one critic network, differentiated by agent_id)
    # ------------------------------------------------------------------
    critic_params     = state.critic_params
    critic_opt_state  = state.critic_opt_state
    critic_loss_sum   = jnp.zeros(())
    q_mean_sum        = jnp.zeros(())

    for i in range(num_agents):
        rng, subkey = jax.random.split(rng)

        def critic_loss_fn(c_params, agent_i=i):
            agent_id_batch = jnp.full((B,), agent_i, dtype=jnp.int32)

            # Bellman target
            q_next = critic.apply(
                state.target_critic_params,   # target critic — stop_gradient below
                batch.next_obs,
                batch.next_prev_msgs,
                next_actions_all,
                next_msgs_all,
                agent_id_batch,
            )

            # Clip q_next before Bellman backup to prevent runaway bootstrap.
            # Shaped rewards are ~0.1-0.5, delivery is 20.
            # Cap at 10 is conservative but prevents critic divergence.
            q_next   = jnp.clip(q_next, -5.0, 10.0)
            reward_i = batch.rewards[:, agent_i : agent_i + 1]
            done     = batch.dones[:, None]
            y = jax.lax.stop_gradient(
                reward_i + gamma * (1.0 - done) * q_next
            )

            q_val = critic.apply(
                c_params,
                batch.obs,
                batch.prev_msgs,
                batch.actions,
                batch.msgs,
                agent_id_batch,
            )
            loss = jnp.mean((q_val - y) ** 2)
            loss = jnp.where(jnp.isnan(loss), jnp.zeros_like(loss), loss)
            return loss, q_val
            # No clip on y needed now — q_next is already bounded

        (c_loss, q_val), c_grads = jax.value_and_grad(
            critic_loss_fn, has_aux=True
        )(critic_params)

        # chain optimizer handles clipping internally
        c_updates, critic_opt_state = critic_opt.update(
            c_grads, critic_opt_state, critic_params
        )
        critic_params = optax.apply_updates(critic_params, c_updates)

        critic_loss_sum += c_loss
        q_mean_sum      += jnp.mean(q_val)

    # ------------------------------------------------------------------
    # 3. Actor updates — one per agent, sequential
    # ------------------------------------------------------------------
    actor_params    = state.actor_params
    actor_opt_state = state.actor_opt_state
    actor_loss_sum  = jnp.zeros(())
    pred_loss_sum   = jnp.zeros(())

    received = received_messages(batch.prev_msgs)  # (B, N, N-1, msg_dim)

    for i in range(num_agents):
        rng, subkey = jax.random.split(rng)

        def actor_loss_fn(a_params, agent_i=i, key=subkey):
            act_dim = batch.actions.shape[-1]

            # Collect actions from all agents; stop_gradient for j != i
            cur_actions_list = []
            cur_msgs_list    = []
            rng_inner = key

            for j in range(num_agents):
                rng_inner, sk = jax.random.split(rng_inner)
                _, a_oh_j, m_j, _ = actor.apply(
                    a_params,
                    batch.obs[:, j, :],
                    received[:, j, :, :],
                    rng=sk,
                    gumbel_tau=gumbel_tau,
                    gumbel_hard=gumbel_hard,
                )
                if j != agent_i:
                    a_oh_j = jax.lax.stop_gradient(a_oh_j)
                    m_j    = jax.lax.stop_gradient(m_j)
                cur_actions_list.append(a_oh_j)
                cur_msgs_list.append(m_j)

            cur_actions_all = jnp.stack(cur_actions_list, axis=1)
            cur_msgs_all    = jnp.stack(cur_msgs_list,    axis=1)

            # Q-loss: maximise Q for agent i
            agent_id_batch = jnp.full((B,), agent_i, dtype=jnp.int32)
            actor_q = critic.apply(
                jax.lax.stop_gradient(critic_params),  # critic fixed during actor update
                batch.obs,
                batch.prev_msgs,
                cur_actions_all,
                cur_msgs_all,
                agent_id_batch,
            )
            q_loss = -jnp.mean(actor_q)

            # Extract submodule param subtrees directly from the full param dict.
            # actor_params has structure:
            #   {"params": {"action_predictor": {...}, "obs_predictor": {...}, ...}}

            # action_predictor_params = {"params": a_params["params"]["action_predictor"]}
            # obs_predictor_params    = {"params": a_params["params"]["obs_predictor"]}            

            # Call submodules via their own apply — correct scope, correct params
            raw_pred_logits = actor.apply(
                a_params,
                batch.obs[:, agent_i, :],
                method=ISAgentNet.apply_action_predictor,
            ).reshape(B, num_agents - 1, act_dim)  # (B, N-1, act_dim)

            other_actions = jnp.concatenate([
                batch.actions[:, :agent_i, :],
                batch.actions[:, agent_i + 1:, :],
            ], axis=1)  # (B, N-1, act_dim)
            other_act_idx = jnp.argmax(other_actions, axis=-1)  # (B, N-1)

            log_probs     = jax.nn.log_softmax(raw_pred_logits, axis=-1)
            # Clamp log_probs to prevent -inf * 0 = nan in the cross-entropy sum
            log_probs = jnp.clip(log_probs, -20.0, 0.0)

            target_oh     = jax.nn.one_hot(other_act_idx, act_dim)
            action_pred_loss = -jnp.mean(jnp.sum(target_oh * log_probs, axis=-1))
            action_pred_loss = jnp.where(
                jnp.isnan(action_pred_loss),
                jnp.zeros_like(action_pred_loss),
                action_pred_loss,
            )            

            # Auxiliary: next-obs delta prediction (MSE)
            other_act_flat = jnp.concatenate([
                batch.actions[:, :agent_i, :],
                batch.actions[:, agent_i + 1:, :],
            ], axis=1).reshape(B, -1)

            obs_pred_input = jnp.concatenate([
                batch.obs[:, agent_i, :],
                batch.actions[:, agent_i, :],
                other_act_flat,
            ], axis=-1)

            delta_pred = actor.apply(
                a_params,
                obs_pred_input,
                method=ISAgentNet.apply_obs_predictor,
            )  # (B, obs_dim)
            delta_true    = batch.next_obs[:, agent_i, :] - batch.obs[:, agent_i, :]
            obs_pred_loss = jnp.mean((delta_pred - delta_true) ** 2)

            pred_loss  = action_pred_loss + obs_pred_loss
            total_loss = q_loss + pred_loss_coef * pred_loss

            # Replace NaN loss with zero so one bad batch doesn't
            # corrupt all subsequent parameter updates
            total_loss = jnp.where(jnp.isnan(total_loss), jnp.zeros_like(total_loss), total_loss)
            return total_loss, (q_loss, pred_loss)

        (a_loss, (q_loss, pred_loss)), a_grads = jax.value_and_grad(
            actor_loss_fn, has_aux=True
        )(actor_params)

        a_updates, actor_opt_state = actor_opt.update(
            a_grads, actor_opt_state, actor_params
        )
        actor_params = optax.apply_updates(actor_params, a_updates)

        actor_loss_sum += q_loss
        pred_loss_sum  += pred_loss

    # ------------------------------------------------------------------
    # 4. Polyak update both target networks
    # ------------------------------------------------------------------
    new_target_actor_params  = polyak_update(actor_params,  state.target_actor_params,  tau)
    new_target_critic_params = polyak_update(critic_params, state.target_critic_params, tau)

    new_state = TrainState(
        actor_params=         actor_params,
        target_actor_params=  new_target_actor_params,
        critic_params=        critic_params,
        target_critic_params= new_target_critic_params,
        actor_opt_state=      actor_opt_state,
        critic_opt_state=     critic_opt_state,
        rng=                  rng,
    )

    metrics = UpdateMetrics(
        critic_loss=(critic_loss_sum / num_agents),
        actor_loss= (actor_loss_sum  / num_agents),
        pred_loss=  (pred_loss_sum   / num_agents),
        q_mean=     (q_mean_sum      / num_agents),
    )

    return new_state, metrics