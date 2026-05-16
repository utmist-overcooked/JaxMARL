"""
Implementation of the TarMAC algorithm (Das et. al., 2020).

At each time step, ith agent sees observation and selects continuous communication message received
by other agents at next timestep. 
Also selects discrete environment action.

Policy:
- implemented as 1-layer GRU
- input is local observation and vector aggregating messages sent by all agents
- updates hidden state of GRU, then policy outputs action given hidden state
- and output head produces outgoing message vector
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import chex
from typing import Tuple, Dict

@chex.dataclass
class TarMACConfig:
    """
    Configuration for TarMAC architecture/

    Attributes:
        hidden_dim: size of agent's internal GRU hidden state
        msg_dim: size of communication vector (Value in attention mechanism)
        key_dim: size of the Query/Key vectors for attention scoring
        num_rounds: number of communication rounds per timestep
    """
    hidden_dim: int = 128
    msg_dim: int = 32
    key_dim: int = 16
    num_rounds: int = 1

class TarMACCell(nn.Module):
    """
    Single timestep logic for TarMAC.
    Processes all agents at once to allow communication.
    """
    action_dim: int
    config: TarMACConfig

    @nn.compact
    def __call__(
        self, 
        carry: Tuple[chex.Array, chex.Array],
        inputs: Tuple[chex.Array, chex.Array]
    ) -> Tuple[Tuple[chex.Array, chex.Array], Tuple[chex.Array, chex.Array, chex.Array]]:
        
        hidden_state, prev_msgs = carry
        obs, dones = inputs

        # Dimensions: B=batch, N=num_agents, D=obs_dim
        sh = obs.shape
        B, N = sh[0], sh[1]

        # Ensure dones is broadcastable [B, N, 1]
        if dones.ndim == 2:
            dones = jnp.expand_dims(dones, axis=-1)

        # Mask out hidden states for agents that just finished/reset
        hidden_state = jnp.where(dones, 0.0, hidden_state)
        prev_msgs = jnp.where(dones, 0.0, prev_msgs)

        if obs.ndim > 3:
            # Overcooked Grid Processing [B, N, H, W, C]
            flat_obs = obs.reshape(B * N, *sh[2:])
            x = nn.Conv(features=32, kernel_size=(3, 3), name="enc_conv1")(flat_obs)
            x = nn.relu(x)
            x = nn.Conv(features=32, kernel_size=(3, 3), name="enc_conv2")(x)
            x = nn.relu(x)
            x = x.reshape((B * N, -1)) 
            enc_obs = nn.Dense(self.config.hidden_dim, name="obs_dense")(x)
            enc_obs = enc_obs.reshape(B, N, -1)
        else:
            # Standard Vector Processing [B, N, D]
            enc_obs = nn.Dense(self.config.hidden_dim, name="obs_encoder_1")(obs)
            enc_obs = nn.relu(enc_obs)
            enc_obs = nn.Dense(self.config.hidden_dim, name="obs_encoder_2")(enc_obs)
            enc_obs = nn.relu(enc_obs)

        # GRU updates agent internal state based on its own obs and previous message
        gru_input = jnp.concatenate([enc_obs, prev_msgs], axis=-1)

        # Flax GRU expects [Batch, Features]
        flat_h = hidden_state.reshape((B * N, -1))
        flat_input = gru_input.reshape((B * N, -1))

        # Apply GRU
        flat_h, _ = nn.GRUCell(self.config.hidden_dim, name="agent_gru")(flat_h, flat_input)
        h = flat_h.reshape(B, N, self.config.hidden_dim)   # [B, N, hidden_dim]

        # Multiple communication rounds
        h_round = h
        final_msgs = prev_msgs

        # Communication (Attention Mechanism)
        Q_layer = nn.Dense(self.config.key_dim, name="query")
        K_layer = nn.Dense(self.config.key_dim, name="key")
        V_layer = nn.Dense(self.config.msg_dim, name="value")

        # shared projection layer
        proj_layer = nn.Dense(self.config.hidden_dim, name="inter_round_proj")

        # masks for dead agents
        sender_mask = (1.0 - dones).reshape(B, 1, N)
        receiver_mask = (1.0 - dones).reshape(B, N, 1)
        
        for r in range(self.config.num_rounds):
            Q = Q_layer(h_round)
            K = K_layer(h_round)
            V = V_layer(h_round)

            # scores: [B, N, N] - attention scores between agents
            scores = jnp.einsum('bik,bjk->bij', Q, K)
            scores = scores / jnp.sqrt(self.config.key_dim)

            # mask out 'done' agents so they don't send messages
            scores = jnp.where(sender_mask > 0, scores, -1e9)

            # Softmax over Senders to get attention weights
            weights = nn.softmax(scores, axis=-1)
            messages = jnp.einsum('bij,bjm->bim', weights, V)
            final_msgs = messages * receiver_mask

            concat_h_msg = jnp.concatenate([final_msgs, h_round], axis=-1)
            h_round = proj_layer(concat_h_msg)
            h_round = nn.tanh(h_round)
            h_round = h_round * receiver_mask
        
        # Action Head
        # Predict action based solely on the updated hidden state
        logits = nn.Dense(self.action_dim, name="action_head")(h_round)
        
        new_carry = (h_round, final_msgs)

        return new_carry, (logits, final_msgs, h_round)

    @staticmethod
    def initialize_carry(hidden_dim, msg_dim, batch_size, num_agents):
        """Initialize the GRU carry (hidden state and messages) to zeros."""
        hidden_state = jnp.zeros((batch_size, num_agents, hidden_dim))
        prev_msgs = jnp.zeros((batch_size, num_agents, msg_dim))
        return (hidden_state, prev_msgs)
    

class CentralizedCritic(nn.Module):
    """
    Centralized Critic for TarMAC.

    Takes hidden states and actions of all agents, flattens into global context vector and predicts the value.
    """

    @nn.compact
    def __call__(
        self,
        hidden_states: chex.Array,
        actions: chex.Array
    ) -> chex.Array:
        """
        Forward pass for Centralized Critic.

        Args:
            hidden_states: [batch, num_agents, hidden_dim]
            actions: [batch, num_agents, action_dim]

        Returns:
            values: [batch, 1]
        """
        # Flatten inputs
        B = hidden_states.shape[0]

        flat_hidden = hidden_states.reshape(B, -1)
        flat_actions = actions.reshape(B, -1)
        
        # Concatenate hidden states and actions]
        critic_input = jnp.concatenate([flat_hidden, flat_actions], axis=-1)

        # MLP for value prediction
        x = nn.Dense(256, name="critic_fc1")(critic_input)
        x = nn.relu(x)

        x = nn.Dense(128, name="critic_fc2")(x)
        x = nn.relu(x)

        value = nn.Dense(1, name="value_head")(x)

        return value


class TarMAC(nn.Module):
    """
    TarMAC Sequence Network.
    
    This module wraps TarMACCell to process an entire sequence of observations using jax.lax.scan.
    To be used during training to forward-pass through a batch of trajectories.
    """
    action_dim: int
    config: TarMACConfig

    @nn.compact
    def __call__(
        self,
        carry: Tuple[chex.Array, chex.Array],
        obs_seq: chex.Array,
        dones_seq: chex.Array
    ) -> Tuple[Tuple[chex.Array, chex.Array], Tuple[chex.Array, chex.Array]]:
        """
        Scans TarMACCell over the time dimension.

        Args:
            carry: (hidden_states, prev_messages)
                hidden_states: [batch, num_agents, hidden_dim]
                prev_messages: [batch, num_agents, msg_dim]
            obs_seq: [time, batch, num_agents, obs_dim]
            dones_seq: [time, batch, num_agents] OR [time, batch, num_agents, 1]

        Returns:
            final_carry: (final_hidden_states, final_messages)
            outputs: (logits_seq, weights_seq)
                logits_seq: [time, batch, num_agents, action_dim]
                weights_seq: [time, batch, num_agents, msg_dim]
        """

        scan_module = nn.scan(
            TarMACCell,
            variable_broadcast="params",
            split_rngs={"params": False},
            in_axes=0,  # scan over axis 0 (time) of inputs
            out_axes=0, # stack outputs along axis 0 (time)
        )(action_dim=self.action_dim, config=self.config)

        final_carry, (logits_seq, weights_seq, hidden_seq) = scan_module(carry, (obs_seq, dones_seq))

        return final_carry, (logits_seq, weights_seq, hidden_seq)

    def initialize_carry(self, batch_size, num_agents):
        """
        Creates initial hidden state and message tensors filled with zeros.

        Returns:
            (hidden_states, messages): initialized to zeros
                hidden_states: [batch, num_agents, hidden_dim]
                messages: [batch, num_agents, msg_dim]
        """
        hidden_state = jnp.zeros((batch_size, num_agents, self.config.hidden_dim))
        prev_msgs = jnp.zeros((batch_size, num_agents, self.config.msg_dim))
        return (hidden_state, prev_msgs)