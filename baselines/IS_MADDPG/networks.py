import jax
import jax.numpy as jnp
import flax.linen as nn
import chex
from typing import Tuple


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def split_other_agent_logits(flat: jnp.ndarray, num_other: int, act_dim: int) -> jnp.ndarray:
    """Reshape flat other-agent logits into per-agent logits.

    Args:
        flat:      (B, (N-1) * act_dim)  — concatenated logits for all other agents
        num_other: number of other agents (N-1)
        act_dim:   number of discrete actions

    Returns:
        (B, N-1, act_dim)
    """
    B = flat.shape[0]
    return flat.reshape(B, num_other, act_dim)


def gumbel_softmax(
    logits: jnp.ndarray,
    key: chex.PRNGKey,
    tau: float = 1.0,
    hard: bool = True,
) -> jnp.ndarray:
    """Gumbel-softmax reparameterisation trick for discrete actions.

    In soft mode (hard=False): returns a differentiable probability vector.
    In hard mode (hard=True):  returns a one-hot vector in the forward pass,
    but the gradient flows through the soft probabilities (straight-through
    estimator), keeping the operation end-to-end differentiable.

    Args:
        logits: (B, act_dim)  unnormalised action scores
        key:    JAX PRNG key consumed for Gumbel noise sampling
        tau:    temperature — lower = more peaked / closer to argmax
        hard:   if True, apply straight-through one-hot in forward pass

    Returns:
        (B, act_dim)  soft probabilities or hard one-hot
    """
    # Sample Gumbel noise: G ~ Gumbel(0,1) via the inverse-CDF trick.
    # minval=tiny prevents log(0) which would give -inf noise.
    u = jax.random.uniform(
        key, logits.shape,
        minval=jnp.finfo(jnp.float32).tiny,
        maxval=1.0,
    )
    gumbel_noise = -jnp.log(-jnp.log(u))

    # Perturb logits and apply temperature-scaled softmax
    y_soft = jax.nn.softmax((logits + gumbel_noise) / tau, axis=-1)

    if hard:
        # Forward pass: hard one-hot at the argmax of y_soft
        y_hard = jax.nn.one_hot(jnp.argmax(y_soft, axis=-1), logits.shape[-1])
        # Straight-through trick:
        #   value   = y_hard          (discrete, non-differentiable)
        #   gradient = d/d(y_soft)    (continuous, differentiable)
        # Written as stop_gradient(y_hard - y_soft) + y_soft so that:
        #   forward:  y_soft cancels → y_hard
        #   backward: stop_gradient term has zero gradient → gradient of y_soft passes through
        return jax.lax.stop_gradient(y_hard - y_soft) + y_soft

    return y_soft


# ---------------------------------------------------------------------------
# Shared constant and Dense factory
# ---------------------------------------------------------------------------

# Orthogonal gain for ReLU networks (√2 ≈ 1.414).
# Defined as a plain Python float to avoid the Flax dataclass error that
# occurs when a JAX array (jnp.sqrt(2)) is used as a field default.
_SQRT2 = float(jnp.sqrt(2))


def orthogonal_dense(features: int, gain: float = _SQRT2) -> nn.Dense:
    """Create a Dense layer with orthogonal weight init and zero bias init.

    Orthogonal initialisation preserves gradient norms through linear layers,
    which stabilises early training. The gain scales the orthogonal matrix:
      - √2  for hidden layers preceded by ReLU  (default)
      - 0.01 for the final actor head (near-uniform initial policy)
      - 1.0  for the critic value head
    """
    return nn.Dense(
        features,
        kernel_init=nn.initializers.orthogonal(gain),
        bias_init=nn.initializers.zeros,
    )


# ---------------------------------------------------------------------------
# Building-block MLPs
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    """Single hidden-layer MLP: Linear -> ReLU -> Linear.

    Used for the action predictor (predicts other agents' actions from obs)
    and the obs predictor (predicts next obs delta).

    Attributes:
        hidden_dim: width of the hidden layer
        out_dim:    output dimension
        out_gain:   orthogonal init gain for the output layer (default √2)
    """
    hidden_dim: int
    out_dim: int
    out_gain: float = _SQRT2

    def setup(self):
        # setup() (not @nn.compact) is required here because this module
        # is called inside a Python for-loop that JAX traces. Using setup()
        # ensures parameters are created once before tracing begins.
        self.hidden = orthogonal_dense(self.hidden_dim)
        self.out    = orthogonal_dense(self.out_dim, self.out_gain)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.out(nn.relu(self.hidden(x)))


class DeepMLP(nn.Module):
    """Two hidden-layer MLP: Linear -> ReLU -> Linear -> ReLU -> Linear.

    Used for the actor network. Deeper than MLP to give the policy more
    capacity to combine observations and received messages.

    Attributes:
        hidden_dim: width of both hidden layers
        out_dim:    output dimension (number of discrete actions)
        out_gain:   orthogonal init gain for the final layer.
                    Default 0.01 keeps initial logits near-uniform,
                    encouraging exploration at the start of training.
    """
    hidden_dim: int
    out_dim: int
    out_gain: float = 0.01

    def setup(self):
        self.h1  = orthogonal_dense(self.hidden_dim)
        self.h2  = orthogonal_dense(self.hidden_dim)
        self.out = orthogonal_dense(self.out_dim, self.out_gain)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = nn.relu(self.h1(x))
        x = nn.relu(self.h2(x))
        return self.out(x)


# ---------------------------------------------------------------------------
# Attention module
# ---------------------------------------------------------------------------

class AttentionProjection(nn.Module):
    """Scaled dot-product attention over an imagined trajectory.

    Computes a single-query attention where:
      - Query  = projection of the received messages (what the agent cares about)
      - Keys   = projection of each imagined (obs, action) step
      - Values = projection of each imagined (obs, action) step

    The output is a weighted sum of values — a message summarising the parts
    of the imagined future most relevant to what other agents communicated.

    Attributes:
        hidden_dim: dimension of query and key projections
        msg_dim:    dimension of the output message (= value projection dim)
    """
    hidden_dim: int
    msg_dim: int

    def setup(self):
        self.W_Q = orthogonal_dense(self.hidden_dim)   # query projection
        self.W_K = orthogonal_dense(self.hidden_dim)   # key   projection
        self.W_V = orthogonal_dense(self.msg_dim)      # value projection

    def __call__(
        self,
        msgs_flat: jnp.ndarray,            # (B, (N-1)*msg_dim)
        imagined_trajectory: jnp.ndarray,  # (B, H, obs_dim + act_dim)
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Returns:
            msg_out: (B, msg_dim)   attention-weighted message
            alpha:   (B, 1, H)     attention weights over horizon steps
        """
        # Project received messages to a single query vector
        query = self.W_Q(msgs_flat)[:, None, :]        # (B, 1, hidden_dim)

        # Project each imagined step to keys and values
        keys   = self.W_K(imagined_trajectory)         # (B, H, hidden_dim)
        values = self.W_V(imagined_trajectory)         # (B, H, msg_dim)

        # Scaled dot-product attention scores
        d_k    = keys.shape[-1]
        scores = jnp.matmul(query, jnp.swapaxes(keys, 1, 2)) / jnp.sqrt(float(d_k))
        # (B, 1, H) — divide by √d_k to prevent vanishing softmax gradients

        alpha   = nn.softmax(scores, axis=-1)          # (B, 1, H)
        msg_out = jnp.matmul(alpha, values).squeeze(1) # (B, msg_dim)

        return msg_out, alpha


# ---------------------------------------------------------------------------
# Agent network (decentralised actor + message generator)
# ---------------------------------------------------------------------------

class ISAgentNet(nn.Module):
    """Intention-Sharing agent network (decentralised execution).

    Each agent runs its own copy of this network with shared parameters
    (joint policy). At each timestep it:
      1. Computes env action logits from (obs, received_messages).
      2. Rolls out an imagined H-step future trajectory using learned
         predictors for its own next obs and other agents' actions.
      3. Attends over the imagined trajectory to produce a message
         that summarises its intentions to other agents.

    The imagined rollout uses a Python for-loop (not lax.scan) so that
    Flax sub-module calls happen in normal tracing context. JAX unrolls
    the loop at compile time, producing the same XLA graph as scan for
    small horizon_H.

    Attributes:
        obs_dim:    dimension of the observation vector
        act_dim:    number of discrete environment actions
        msg_dim:    dimension of the continuous communication message
        hidden_dim: width of all hidden layers
        num_agents: total number of agents (including self), must be >= 2
        horizon_H:  imagination horizon (number of rollout steps), must be >= 1
    """
    obs_dim:    int
    act_dim:    int
    msg_dim:    int
    hidden_dim: int
    num_agents: int
    horizon_H:  int

    def setup(self):
        # Validation runs once at module construction time
        if self.num_agents < 2:
            raise ValueError("num_agents >= 2 required")
        if self.horizon_H < 1:
            raise ValueError("horizon_H >= 1 required")

        # Predicts the discrete action probability for each other agent
        # given the current imagined observation.
        # Input:  (B, obs_dim)
        # Output: (B, (N-1) * act_dim)
        self.action_predictor = MLP(
            hidden_dim=self.hidden_dim,
            out_dim=(self.num_agents - 1) * self.act_dim,
        )

        # Predicts the change in observation (residual / delta) given
        # current obs, own action, and other agents' predicted actions.
        # Input:  (B, obs_dim + act_dim + (N-1)*act_dim)
        # Output: (B, obs_dim)   — added to current obs as a residual
        self.obs_predictor = MLP(
            hidden_dim=self.hidden_dim,
            out_dim=self.obs_dim,
        )

        # Policy network: maps (obs, received_msgs) -> action logits.
        # Used both for the real action and inside the imagined rollout
        # (soft probabilities, no Gumbel noise) so weights are shared.
        # Input:  (B, obs_dim + (N-1)*msg_dim)
        # Output: (B, act_dim)
        self.actor_net = DeepMLP(
            hidden_dim=self.hidden_dim,
            out_dim=self.act_dim,
            out_gain=0.01,   # near-zero init → near-uniform policy at start
        )

        # Attention module that reads the imagined trajectory and produces
        # the outgoing communication message.
        self.attention = AttentionProjection(
            hidden_dim=self.hidden_dim,
            msg_dim=self.msg_dim,
        )

    def __call__(
        self,
        obs: jnp.ndarray,            # (B, obs_dim)
        received_msgs: jnp.ndarray,  # (B, N-1, msg_dim)
        *,
        rng: chex.PRNGKey,
        gumbel_tau: float = 1.0,
        gumbel_hard: bool = True,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Forward pass: action selection + message generation.

        Args:
            obs:           current observation for this agent
            received_msgs: messages received from all other agents last step
            rng:           PRNG key (consumed for Gumbel sampling)
            gumbel_tau:    Gumbel temperature (lower = harder / more greedy)
            gumbel_hard:   if True, straight-through one-hot action

        Returns:
            action_logits: (B, act_dim)   raw logits (used for loss)
            action_onehot: (B, act_dim)   hard or soft action sample
            message_out:   (B, msg_dim)   outgoing communication message
            alpha:         (B, 1, H)      attention weights over horizon
        """
        B = obs.shape[0]
        chex.assert_shape(obs,           (B, self.obs_dim))
        chex.assert_shape(received_msgs, (B, self.num_agents - 1, self.msg_dim))

        # Flatten received messages for use as actor input
        total_msg_dim = (self.num_agents - 1) * self.msg_dim
        msgs_flat = received_msgs.reshape(B, total_msg_dim)  # (B, (N-1)*msg_dim)

        # ------------------------------------------------------------------
        # Step 1: Compute current env action
        # ------------------------------------------------------------------

        # Concatenate obs with flattened received messages, then get logits
        action_logits = self.actor_net(jnp.concatenate([obs, msgs_flat], axis=-1))

        # Split the RNG: one key for the real Gumbel action, one for rollout
        key_action, key_rollout = jax.random.split(rng)

        # Sample action via Gumbel-softmax (hard one-hot in forward pass)
        action_onehot = gumbel_softmax(
            action_logits, key_action, tau=gumbel_tau, hard=gumbel_hard
        )

        # ------------------------------------------------------------------
        # Step 2: Imagined H-step rollout to build the trajectory τ^i
        #
        # τ^i = [(ô_t, â_t), (ô_{t+1}, â_{t+1}), ..., (ô_{t+H-1}, â_{t+H-1})]
        #
        # Each future step is predicted using:
        #   - action_predictor:  other agents' next actions given ô
        #   - obs_predictor:     next obs delta given (ô, â, â_others)
        #   - actor_net:         own soft action at the predicted next obs
        #
        # msgs_flat is kept fixed throughout (messages from the *current*
        # timestep are used as context for the whole rollout, matching the
        # original paper's formulation).
        #
        # We use a Python for-loop rather than lax.scan because Flax sub-
        # module calls inside lax.scan trigger scope/parameter allocation
        # errors. JAX unrolls the loop at trace time, producing an identical
        # XLA computation graph for small horizon_H.
        # ------------------------------------------------------------------

        # Initialise the trajectory list with the current (real) step
        imagined_steps = [jnp.concatenate([obs, action_onehot], axis=-1)]  # [(B, obs+act)]

        hat_o = obs           # imagined obs  at step h
        hat_a = action_onehot # imagined action at step h
        key   = key_rollout    

        for _ in range(self.horizon_H - 1):
            # Advance the PRNG (subkey unused here but kept for reproducibility
            # if stochastic elements are added to the rollout later)
            key, subkey = jax.random.split(key)

            # --- Predict other agents' actions from current imagined obs ---
            pred_other_logits = self.action_predictor(hat_o)   # (B, (N-1)*act_dim)
            pred_other_probs  = nn.softmax(
                pred_other_logits.reshape(B, self.num_agents - 1, self.act_dim),
                axis=-1,
            ).reshape(B, -1)                                    # (B, (N-1)*act_dim)

            # --- Predict next obs as current obs + residual delta ---
            obs_in     = jnp.concatenate([hat_o, hat_a, pred_other_probs], axis=-1)
            delta_o    = self.obs_predictor(obs_in)             # (B, obs_dim)

            # Clip delta to prevent runaway imagined trajectories.
            # Without this, obs_predictor errors compound over H steps
            # and produce inf/nan that propagates into the attention output.
            delta_o    = jnp.clip(delta_o, -1.0, 1.0)            
            hat_o_next = hat_o + delta_o                        # residual connection

            # --- Compute own soft action at predicted next obs ---
            # No Gumbel noise here: we want smooth, differentiable probabilities
            # so that gradients flow back through the entire rollout.
            actor_in    = jnp.concatenate([hat_o_next, msgs_flat], axis=-1)
            next_logits = self.actor_net(actor_in)
            hat_a_next  = nn.softmax(next_logits, axis=-1)      # (B, act_dim)

            imagined_steps.append(jnp.concatenate([hat_o_next, hat_a_next], axis=-1))
            hat_o = hat_o_next
            hat_a = hat_a_next

        # Stack list of H tensors (B, obs+act) into (B, H, obs+act)
        imagined_trajectory = jnp.stack(imagined_steps, axis=1)

        # ------------------------------------------------------------------
        # Step 3: Attend over imagined trajectory to produce message
        #
        # The query is derived from received messages (what others said),
        # so the agent selects which parts of its imagined future are most
        # relevant to the information it has received — and communicates that.
        # ------------------------------------------------------------------
        message_out, alpha = self.attention(msgs_flat, imagined_trajectory)

        return action_logits, action_onehot, message_out, alpha
    
    
    def apply_action_predictor(self, obs: jnp.ndarray) -> jnp.ndarray:
        """Call action_predictor submodule directly.

        Used in update.py to compute other-agent action prediction loss
        without running the full forward pass.

        Args:
            obs: (B, obs_dim)

        Returns:
            (B, (N-1) * act_dim)  flat logits for all other agents
        """
        return self.action_predictor(obs)
    

    def apply_obs_predictor(self, obs_pred_input: jnp.ndarray) -> jnp.ndarray:
        """Call obs_predictor submodule directly.

        Used in update.py to compute next-obs delta prediction loss.

        Args:
            obs_pred_input: (B, obs_dim + act_dim + (N-1)*act_dim)

        Returns:
            (B, obs_dim)  predicted observation delta
        """
        return self.obs_predictor(obs_pred_input)


# ---------------------------------------------------------------------------
# Centralised critic (used only during training, not execution)
# ---------------------------------------------------------------------------

class ISCriticNet(nn.Module):
    """Centralised critic shared across all agents (CTDE).

    A single critic network estimates Q-values for any agent by conditioning
    on a one-hot agent identity vector appended to the global state. This is
    more parameter-efficient than one critic per agent and ensures a
    consistent value function across the team.

    Global state concatenation order:
        [obs_all | prev_msgs_all | actions_all | msgs_all | agent_id_onehot]
         (N*obs)   (N*msg)         (N*act)       (N*msg)    (N,)

    Attributes:
        num_agents: total number of agents N
        obs_dim:    per-agent observation dimension
        act_dim:    per-agent action dimension (one-hot)
        msg_dim:    per-agent message dimension
        hidden_dim: width of hidden layers (default 256)
    """
    num_agents: int
    obs_dim:    int
    act_dim:    int
    msg_dim:    int
    hidden_dim: int = 256

    @nn.compact
    def __call__(
        self,
        obs_all:       jnp.ndarray,  # (B, N, obs_dim)
        prev_msgs_all: jnp.ndarray,  # (B, N, msg_dim)
        actions_all:   jnp.ndarray,  # (B, N, act_dim)
        msgs_all:      jnp.ndarray,  # (B, N, msg_dim)
        agent_id:      jnp.ndarray,  # (B,)             integer agent index in [0, N)
    ) -> jnp.ndarray:                # (B, 1)            scalar Q-value for that agent
        B = obs_all.shape[0]

        chex.assert_shape(obs_all,       (B, self.num_agents, self.obs_dim))
        chex.assert_shape(prev_msgs_all, (B, self.num_agents, self.msg_dim))
        chex.assert_shape(actions_all,   (B, self.num_agents, self.act_dim))
        chex.assert_shape(msgs_all,      (B, self.num_agents, self.msg_dim))
        chex.assert_shape(agent_id,      (B,))

        # One-hot encode the agent index so the critic can condition on identity.
        # This is what lets one network serve all N agents without ambiguity.
        agent_onehot = jax.nn.one_hot(agent_id, self.num_agents)  # (B, N)

        # Build global state vector, same for all agents, plus identity
        x = jnp.concatenate([
            obs_all.reshape(B, -1),        # (B, N*obs_dim)
            prev_msgs_all.reshape(B, -1),  # (B, N*msg_dim)
            actions_all.reshape(B, -1),    # (B, N*act_dim)
            msgs_all.reshape(B, -1),       # (B, N*msg_dim)
            agent_onehot,                  # (B, N)
        ], axis=-1)

        for _ in range(2):
            x = nn.relu(nn.Dense(
                self.hidden_dim,
                kernel_init=nn.initializers.orthogonal(jnp.sqrt(2)),
                bias_init=nn.initializers.zeros,
            )(x))

        q = nn.Dense(
            1,
            kernel_init=nn.initializers.orthogonal(1.0),
            bias_init=nn.initializers.zeros,
        )(x)

        return q  # (B, 1)
    


# import jax
# import jax.numpy as jnp
# import flax.linen as nn
# import chex
# from jax import lax


# def split_other_agent_logits(flat, num_other, act_dim):
#     # (B, (N-1)*A) -> (B, N-1, A)
#     B = flat.shape[0]
#     return flat.reshape(B, num_other, act_dim)


# def gumbel_softmax(logits, key, tau=1.0, hard=True):
#     gumbel_noise = -jnp.log(-jnp.log(jax.random.uniform(key, logits.shape) + 1e-8) + 1e-8)
#     y = nn.softmax((logits + gumbel_noise) / tau, axis=-1)

#     if hard:
#         y_hard = jax.nn.one_hot(jnp.argmax(y, axis=-1), y.shape[-1])
#         y = y_hard + jax.lax.stop_gradient(y - y_hard)

#     return y


# class ISAgentNet(nn.Module):
#     obs_dim: int
#     act_dim: int
#     msg_dim: int
#     hidden_dim: int
#     num_agents: int
#     horizon_H: int


#     def setup(self):
#         self.action_predictor = nn.Sequential([
#             nn.Dense(self.hidden_dim),
#             nn.relu,
#             nn.Dense((self.num_agents - 1) * self.act_dim),
#         ])

#         self.obs_predictor = nn.Sequential([
#             nn.Dense(self.hidden_dim),
#             nn.relu,
#             nn.Dense(self.obs_dim),
#         ])

#         self.actor_net = nn.Sequential([
#             nn.Dense(self.hidden_dim),
#             nn.relu,
#             nn.Dense(self.hidden_dim),
#             nn.relu,
#             nn.Dense(self.act_dim,
#                      kernel_init=nn.initializers.orthogonal(0.01)),
#         ])

#         self.W_Q = nn.Dense(self.hidden_dim)
#         self.W_K = nn.Dense(self.hidden_dim)
#         self.W_V = nn.Dense(self.msg_dim)

#     # @nn.compact
#     def __call__(self, obs, received_msgs, *, rng, gumbel_tau=1.0, gumbel_hard=True):
#         """
#         obs: (B, obs_dim)
#         received_msgs: (B, N-1, msg_dim)
#         """

#         B = obs.shape[0]
#         chex.assert_shape(obs, (B, self.obs_dim))

#         total_msg_dim = (self.num_agents - 1) * self.msg_dim
#         msgs_flat = received_msgs.reshape(B, total_msg_dim)

#         # ==========================================================
#         # 1️⃣ Current action
#         # ==========================================================

#         actor_input = jnp.concatenate([obs, msgs_flat], axis=-1)
#         action_logits = self.actor_net(actor_input)

#         key_action, key_rollout = jax.random.split(rng)
#         action_onehot = gumbel_softmax(
#             action_logits, key_action,
#             tau=gumbel_tau,
#             hard=gumbel_hard
#         )

#         # ==========================================================
#         # 2️⃣ Imagined Rollout (scan-based)
#         # ==========================================================

#         def rollout_step(carry, _):
#             hat_o, hat_a, key = carry

#             key, subkey = jax.random.split(key)

#             # Predict other agents
#             pred_other_logits = self.action_predictor(hat_o)
#             pred_other_logits = split_other_agent_logits(
#                 pred_other_logits,
#                 self.num_agents - 1,
#                 self.act_dim
#             )

#             pred_other_probs = nn.softmax(pred_other_logits, axis=-1)
#             pred_other_probs = pred_other_probs.reshape(B, -1)

#             # Predict next obs
#             obs_pred_input = jnp.concatenate(
#                 [hat_o, hat_a, pred_other_probs],
#                 axis=-1
#             )
#             delta_o = self.obs_predictor(obs_pred_input)
#             hat_o_next = hat_o + delta_o

#             # Next action (soft)
#             next_actor_input = jnp.concatenate(
#                 [hat_o_next, msgs_flat],
#                 axis=-1
#             )
#             next_logits = self.actor_net(next_actor_input)
#             hat_a_next = nn.softmax(next_logits, axis=-1)

#             new_carry = (hat_o_next, hat_a_next, key)
#             step_output = jnp.concatenate([hat_o_next, hat_a_next], axis=-1)

#             return new_carry, step_output

#         # Initial step
#         initial_step = jnp.concatenate([obs, action_onehot], axis=-1)

#         carry = (obs, action_onehot, key_rollout)

#         carry, imagined_future = lax.scan(
#             rollout_step,
#             carry,
#             xs=None,
#             length=self.horizon_H - 1
#         )

#         imagined_future = jnp.swapaxes(imagined_future, 0, 1)
#         # (H-1, B, D) -> (B, H-1, D)

#         imagined_trajectory = jnp.concatenate(
#             [
#                 initial_step[:, None, :],   # (B,1,D)
#                 imagined_future             # (B,H-1,D)
#             ],
#             axis=1
#         ) # (B, H, obs+act)

#         # ==========================================================
#         # 3️⃣ Attention
#         # ==========================================================

#         query = self.W_Q(msgs_flat)[:, None, :]  # (B,1,Hd)
#         keys = self.W_K(imagined_trajectory)     # (B,H,Hd)
#         values = self.W_V(imagined_trajectory)   # (B,H,msg)

#         d_k = keys.shape[-1]
#         scores = jnp.matmul(query, jnp.swapaxes(keys, 1, 2)) / jnp.sqrt(d_k)
#         alpha = nn.softmax(scores, axis=-1)   # (B,1,H)

#         message_out = jnp.matmul(alpha, values).squeeze(1)  # (B,msg)

#         return action_logits, action_onehot, message_out, alpha
    


# class ISCriticNet(nn.Module):
#     num_agents: int
#     obs_dim: int
#     act_dim: int
#     msg_dim: int
#     hidden_dim: int = 256

#     @nn.compact
#     def __call__(
#         self,
#         obs_all,
#         prev_msgs_all,
#         actions_all,
#         msgs_all,
#     ):
#         """
#         obs_all: (B, N, obs_dim)
#         prev_msgs_all: (B, N, msg_dim)
#         actions_all: (B, N, act_dim)
#         msgs_all: (B, N, msg_dim)

#         Returns:
#             q_values: (B, 1)
#         """

#         B = obs_all.shape[0]
#         N = self.num_agents

#         # Shape checks
#         chex.assert_shape(obs_all, (B, N, self.obs_dim))
#         chex.assert_shape(prev_msgs_all, (B, N, self.msg_dim))
#         chex.assert_shape(actions_all, (B, N, self.act_dim))
#         chex.assert_shape(msgs_all, (B, N, self.msg_dim))

#         # Flatten each (B, N, D) → (B, N*D)
#         obs_flat = obs_all.reshape(B, -1)
#         prev_msgs_flat = prev_msgs_all.reshape(B, -1)
#         actions_flat = actions_all.reshape(B, -1)
#         msgs_flat = msgs_all.reshape(B, -1)

#         # Concatenate
#         x = jnp.concatenate(
#             [obs_flat, prev_msgs_flat, actions_flat, msgs_flat],
#             axis=-1,
#         )

#         # MLP
#         x = nn.Dense(
#             self.hidden_dim,
#             kernel_init=nn.initializers.orthogonal(jnp.sqrt(2.0)),
#         )(x)
#         x = nn.relu(x)

#         x = nn.Dense(
#             self.hidden_dim,
#             kernel_init=nn.initializers.orthogonal(jnp.sqrt(2.0)),
#         )(x)
#         x = nn.relu(x)

#         q = nn.Dense(
#             1,
#             kernel_init=nn.initializers.orthogonal(1.0),
#         )(x)

#         return q


# B = 32
# N = 3
# obs_dim = 10
# act_dim = 5
# msg_dim = 8
# H = 4

# rng = jax.random.PRNGKey(0)

# dummy_obs = jnp.zeros((B, obs_dim))
# dummy_received_msgs = jnp.zeros((B, N-1, msg_dim))

# dummy_obs_all = jnp.zeros((B, N, obs_dim))
# dummy_prev_msgs_all = jnp.zeros((B, N, msg_dim))
# dummy_actions_all = jnp.zeros((B, N, act_dim))
# dummy_msgs_all = jnp.zeros((B, N, msg_dim))



# actor = ISAgentNet(
#     obs_dim=obs_dim,
#     act_dim=act_dim,
#     msg_dim=msg_dim,
#     hidden_dim=256,
#     num_agents=N,
#     horizon_H=H,
# )

# rng, actor_key = jax.random.split(rng)

# actor_params = actor.init(
#     actor_key,
#     dummy_obs,
#     dummy_received_msgs,
#     rng=actor_key,   # required for gumbel
# )

# critic = ISCriticNet(
#     num_agents=N,
#     obs_dim=obs_dim,
#     act_dim=act_dim,
#     msg_dim=msg_dim,
#     hidden_dim=256,
# )

# rng, critic_key = jax.random.split(rng)

# critic_params = critic.init(
#     critic_key,
#     dummy_obs_all,
#     dummy_prev_msgs_all,
#     dummy_actions_all,
#     dummy_msgs_all,
# )


# rng, forward_key = jax.random.split(rng)

# action_logits, action_onehot, message_out, alpha = actor.apply(
#     actor_params,
#     dummy_obs,
#     dummy_received_msgs,
#     rng=forward_key,
# )

# print("action_logits:", action_logits.shape)   # (B, act_dim)
# print("action_onehot:", action_onehot.shape)   # (B, act_dim)
# print("message_out:", message_out.shape)       # (B, msg_dim)
# print("alpha:", alpha.shape)                   # (B, 1, H)

# q_values = critic.apply(
#     critic_params,
#     dummy_obs_all,
#     dummy_prev_msgs_all,
#     dummy_actions_all,
#     dummy_msgs_all,
# )

# print("q_values:", q_values.shape)




    
# # ---------------------------------------------------------------------------
# # Small reusable MLP (replaces nn.Sequential + bare relu)
# # ---------------------------------------------------------------------------

# class MLP(nn.Module):
#     """Simple MLP with ReLU activations and orthogonal init on all layers."""
#     hidden_dim: int
#     out_dim: int
#     out_gain: float = float(jnp.sqrt(2))  # override to 0.01 for final actor head

#     @nn.compact
#     def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
#         x = nn.Dense(
#             self.hidden_dim,
#             kernel_init=nn.initializers.orthogonal(jnp.sqrt(2)),
#             bias_init=nn.initializers.zeros,
#         )(x)
#         x = nn.relu(x)
#         x = nn.Dense(
#             self.out_dim,
#             kernel_init=nn.initializers.orthogonal(self.out_gain),
#             bias_init=nn.initializers.zeros,
#         )(x)
#         return x


# class DeepMLP(nn.Module):
#     """Two hidden layers + output; used for the actor."""
#     hidden_dim: int
#     out_dim: int
#     out_gain: float = 0.01

#     @nn.compact
#     def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
#         for _ in range(2):
#             x = nn.Dense(
#                 self.hidden_dim,
#                 kernel_init=nn.initializers.orthogonal(jnp.sqrt(2)),
#                 bias_init=nn.initializers.zeros,
#             )(x)
#             x = nn.relu(x)
#         x = nn.Dense(
#             self.out_dim,
#             kernel_init=nn.initializers.orthogonal(self.out_gain),
#             bias_init=nn.initializers.zeros,
#         )(x)
#         return x


# # ---------------------------------------------------------------------------
# # Attention layers (explicit Dense modules — no Sequential)
# # ---------------------------------------------------------------------------

# class AttentionProjection(nn.Module):
#     """W_Q, W_K, W_V projections."""
#     hidden_dim: int
#     msg_dim: int

#     @nn.compact
#     def __call__(
#         self,
#         msgs_flat: jnp.ndarray,           # (B, (N-1)*msg_dim)
#         imagined_trajectory: jnp.ndarray,  # (B, H, obs+act)
#     ) -> Tuple[jnp.ndarray, jnp.ndarray]:
#         """Returns message_out (B, msg_dim) and alpha (B, 1, H)."""
#         query = nn.Dense(
#             self.hidden_dim,
#             kernel_init=nn.initializers.orthogonal(jnp.sqrt(2)),
#         )(msgs_flat)[:, None, :]  # (B, 1, hidden_dim)

#         keys = nn.Dense(
#             self.hidden_dim,
#             kernel_init=nn.initializers.orthogonal(jnp.sqrt(2)),
#         )(imagined_trajectory)  # (B, H, hidden_dim)

#         values = nn.Dense(
#             self.msg_dim,
#             kernel_init=nn.initializers.orthogonal(jnp.sqrt(2)),
#         )(imagined_trajectory)  # (B, H, msg_dim)

#         d_k = keys.shape[-1]
#         scores = jnp.matmul(query, jnp.swapaxes(keys, 1, 2)) / jnp.sqrt(float(d_k))
#         alpha = nn.softmax(scores, axis=-1)          # (B, 1, H)
#         message_out = jnp.matmul(alpha, values).squeeze(1)  # (B, msg_dim)

#         return message_out, alpha


# # ---------------------------------------------------------------------------
# # Main agent network
# # ---------------------------------------------------------------------------

# class ISAgentNet(nn.Module):
#     obs_dim: int
#     act_dim: int
#     msg_dim: int
#     hidden_dim: int
#     num_agents: int
#     horizon_H: int

#     def setup(self):
#         if self.num_agents < 2:
#             raise ValueError("Intention Sharing requires num_agents >= 2")
#         if self.horizon_H < 1:
#             raise ValueError("horizon_H must be >= 1")

#         # Predicts other agents' action logits from current obs
#         self.action_predictor = MLP(
#             hidden_dim=self.hidden_dim,
#             out_dim=(self.num_agents - 1) * self.act_dim,
#         )

#         # Predicts delta_obs given (obs, own_action, other_action_probs)
#         self.obs_predictor = MLP(
#             hidden_dim=self.hidden_dim,
#             out_dim=self.obs_dim,
#         )

#         # Actor: two hidden layers, small gain on final head for exploration
#         self.actor_net = DeepMLP(
#             hidden_dim=self.hidden_dim,
#             out_dim=self.act_dim,
#             out_gain=0.01,
#         )

#         # Attention projections
#         self.attention = AttentionProjection(
#             hidden_dim=self.hidden_dim,
#             msg_dim=self.msg_dim,
#         )

#     def __call__(
#         self,
#         obs: jnp.ndarray,
#         received_msgs: jnp.ndarray,
#         *,
#         rng: chex.PRNGKey,
#         gumbel_tau: float = 1.0,
#         gumbel_hard: bool = True,
#     ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
#         """
#         Args:
#             obs:           (B, obs_dim)
#             received_msgs: (B, N-1, msg_dim)
#             rng:           PRNG key
#             gumbel_tau:    Gumbel temperature
#             gumbel_hard:   Straight-through if True

#         Returns:
#             action_logits: (B, act_dim)
#             action_onehot: (B, act_dim)
#             message_out:   (B, msg_dim)
#             alpha:         (B, 1, H)   attention weights
#         """
#         B = obs.shape[0]
#         chex.assert_shape(obs, (B, self.obs_dim))
#         chex.assert_shape(received_msgs, (B, self.num_agents - 1, self.msg_dim))

#         total_msg_dim = (self.num_agents - 1) * self.msg_dim
#         msgs_flat = received_msgs.reshape(B, total_msg_dim)

#         # ------------------------------------------------------------------
#         # 1. Current action
#         # ------------------------------------------------------------------
#         actor_input = jnp.concatenate([obs, msgs_flat], axis=-1)
#         action_logits = self.actor_net(actor_input)

#         key_action, key_rollout = jax.random.split(rng)
#         action_onehot = gumbel_softmax(
#             action_logits, key_action,
#             tau=gumbel_tau,
#             hard=gumbel_hard,
#         )

#         # ------------------------------------------------------------------
#         # 2. Imagined rollout  (lax.scan over H-1 future steps)
#         # ------------------------------------------------------------------

#         def rollout_step(
#             carry: Tuple[jnp.ndarray, jnp.ndarray, chex.PRNGKey],
#             _: None,
#         ) -> Tuple[Tuple, jnp.ndarray]:
#             hat_o, hat_a, key = carry
#             key, subkey = jax.random.split(key)

#             # --- other-agent action predictions ---
#             pred_other_logits = self.action_predictor(hat_o)          # (B, (N-1)*A)
#             pred_other_logits = split_other_agent_logits(
#                 pred_other_logits, self.num_agents - 1, self.act_dim
#             )                                                           # (B, N-1, A)
#             pred_other_probs = nn.softmax(pred_other_logits, axis=-1).reshape(B, -1)  # (B, (N-1)*A)

#             # --- next obs prediction ---
#             obs_pred_input = jnp.concatenate([hat_o, hat_a, pred_other_probs], axis=-1)
#             delta_o = self.obs_predictor(obs_pred_input)
#             hat_o_next = hat_o + delta_o

#             # --- next action (soft, no Gumbel noise — differentiable rollout) ---
#             next_actor_input = jnp.concatenate([hat_o_next, msgs_flat], axis=-1)
#             next_logits = self.actor_net(next_actor_input)
#             hat_a_next = nn.softmax(next_logits, axis=-1)

#             step_output = jnp.concatenate([hat_o_next, hat_a_next], axis=-1)
#             return (hat_o_next, hat_a_next, key), step_output

#         # Initial trajectory step (current obs + Gumbel action)
#         initial_step = jnp.concatenate([obs, action_onehot], axis=-1)  # (B, obs+act)

#         if self.horizon_H > 1:
#             _, imagined_future = lax.scan(
#                 rollout_step,
#                 init=(obs, action_onehot, key_rollout),
#                 xs=None,
#                 length=self.horizon_H - 1,
#             )
#             # scan output: (H-1, B, obs+act) -> (B, H-1, obs+act)
#             imagined_future = jnp.swapaxes(imagined_future, 0, 1)
#             imagined_trajectory = jnp.concatenate(
#                 [initial_step[:, None, :], imagined_future], axis=1
#             )  # (B, H, obs+act)
#         else:
#             # horizon_H == 1: just the current step
#             imagined_trajectory = initial_step[:, None, :]  # (B, 1, obs+act)

#         # ------------------------------------------------------------------
#         # 3. Attention -> message
#         # ------------------------------------------------------------------
#         message_out, alpha = self.attention(msgs_flat, imagined_trajectory)

#         return action_logits, action_onehot, message_out, alpha