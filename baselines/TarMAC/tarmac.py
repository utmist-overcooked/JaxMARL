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
    ) -> Tuple[Tuple[chex.Array, chex.Array], chex.Array]:
        """
        Forward pass for one timestep.

        Args:
            carry: (hidden_states, prev_messages)
                hidden_states: [batch, num_agents, hidden_dim]
                prev_messages: [batch, num_agents, msg_dim]
            inputs: (obs, dones)
                obs: [batch, num_agents, obs_dim]
                dones: [batch, num_agents] OR [batch, num_agents, 1] indicating reset

        Returns:
            new_carry: (new_hidden_states, new_messages)
                new_hidden_states: [batch, num_agents, hidden_dim]
                new_messages: [batch, num_agents, msg_dim]
            outputs: (logits, weights)
                logits: [batch, num_agents, action_dim]
                weights: [batch, num_agents, msg_dim] (aggregated incoming message)
        """
        hidden_state, prev_msgs = carry
        obs, dones = inputs

        # Dimensions: B=batch, N=num_agents, D=obs_dim
        B, N, _ = obs.shape

        # Ensure dones is broadcastable [B, N, 1]
        if dones.ndim == 2:
            dones = jnp.expand_dims(dones, axis=-1)

        # Encode observation
        # Project observations into hidden_dim
        # Dense layers broadcast over [batch, num_agents], no need for flattening yet
        enc_obs = nn.Dense(self.config.hidden_dim, name="obs_encoder_1")(obs)
        enc_obs = nn.relu(enc_obs)
        enc_obs = nn.Dense(self.config.hidden_dim, name="obs_encoder_2")(enc_obs)
        enc_obs = nn.relu(enc_obs)

        # GRU update (pre-communication)
        # Reset hidden state and messages for done agents to prevent them from influencing others
        hidden_state = jnp.where(dones, 0.0, hidden_state)
        prev_msgs = jnp.where(dones, 0.0, prev_msgs)

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
        # Agents query messages from others to update their message/state
        Q_layer = nn.Dense(self.config.key_dim, name="query")
        K_layer = nn.Dense(self.config.key_dim, name="key")
        V_layer = nn.Dense(self.config.msg_dim, name="value")
        
        comm_gru = nn.GRUCell(self.config.hidden_dim, name="comm_gru") if self.config.num_rounds > 1 else None
        for r in range(self.config.num_rounds):
            Q = Q_layer(h_round)
            K = K_layer(h_round)
            V = V_layer(h_round)

            # scores: [B, N, N] - attention scores between agents
            # scores[b, i, j] = how much agent i (receiver) cares about agent j (sender)
            scores = jnp.einsum('bik,bjk->bij', Q, K)
            scores = scores / jnp.sqrt(self.config.key_dim)

            # mask out 'done' agents so they don't send messages
            # mask shape [B, 1, N] to broadcast over receivers
            mask = (1.0 - dones).reshape(B, 1, N)
            scores = jnp.where(mask > 0, scores, -1e9)

            # Softmax over Senders to get attention weights
            weights = nn.softmax(scores, axis=-1)
            messages = jnp.einsum('bij,bjm->bim', weights, V)
            final_msgs = messages
            
            # inter round state update
            if r < self.config.num_rounds - 1:
                flat_h_round = h_round.reshape(B * N, -1)
                flat_msgs = messages.reshape(B * N, -1)
                
                flat_h_round, _ = comm_gru(flat_h_round, flat_msgs)
                h_round = flat_h_round.reshape(B, N, self.config.hidden_dim)
        
        # Action Head
        # Predict action based on updated hidden state and aggregated message
        out_input = jnp.concatenate([h_round, final_msgs], axis=-1)
        logits = nn.Dense(self.action_dim, name="action_head")(out_input)
        new_carry = (h, final_msgs) # for next timestep

        return new_carry, (logits, final_msgs, h)
    

    @staticmethod
    def initialize_carry(hidden_dim, batch_size):
        """Initialize the GRU carry (hidden state) to zeros."""
        return jnp.zeros((batch_size, hidden_dim))
    

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