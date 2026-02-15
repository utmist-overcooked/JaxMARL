"""IC3Net-family model cores (JAX/Flax implementation).

This module implements the core architectures used in IC3Net:
- IndependentMLP: IC/IRIC baseline without communication (feedforward)
- IndependentLSTM: IC/IRIC baseline with LSTM recurrence
- CommNetDiscrete: CommNet/IC3Net with message passing (feedforward)
- CommNetLSTM: CommNet/IC3Net with LSTM recurrence and message passing
"""
from typing import Optional, Tuple
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal


class IndependentMLP(nn.Module):
    """IC/IRIC network: independent controller without communication.
    
    Applies the same weights to each agent's observation.
    
    Args:
        action_dim: Number of discrete actions
        hidden_dim: Size of hidden layers
    
    Input:
        obs: (B, N, obs_dim) batched observations
    
    Output:
        logits: (B, N, action_dim) action logits
        values: (B, N) state values
    """
    action_dim: int
    hidden_dim: int = 64
    
    @nn.compact
    def __call__(self, obs: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # obs shape: (B, N, obs_dim)
        x = nn.Dense(
            self.hidden_dim, 
            kernel_init=orthogonal(jnp.sqrt(2)), 
            bias_init=constant(0.0)
        )(obs)
        x = nn.tanh(x)
        
        h = nn.Dense(
            self.hidden_dim,
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0)
        )(x)
        h = nn.tanh(h + x)  # skip connection
        
        # Action logits
        logits = nn.Dense(
            self.action_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0)
        )(h)
        
        # Value head
        values = nn.Dense(
            1,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0)
        )(h)
        
        return logits, jnp.squeeze(values, axis=-1)


class CommNetDiscrete(nn.Module):
    """CommNet-style discrete policy with optional IC3Net hard-attention gating.
    
    - Encodes observations into hidden vectors
    - Performs message passing between agents
    - Optionally gates communication with talk/silent actions (IC3Net)
    
    Args:
        num_agents: Number of agents
        action_dim: Number of discrete actions
        hidden_dim: Size of hidden layers
        comm_passes: Number of communication rounds
        comm_mode: 'avg' or 'sum' for message aggregation
        hard_attn: Enable IC3Net talk/silent gating
        comm_mask_zero: Debug option to disable all communication
        share_weights: Share weights across communication passes
    
    Input:
        obs: (B, N, obs_dim)
        alive_mask: (N,) optional binary mask for alive agents
        comm_action: (N,) optional talk/silent actions (1=talk, 0=silent)
    
    Output:
        logits: (B, N, action_dim) environment action logits
        values: (B, N) state values
        talk_logits: (B, N, 2) talk/silent logits (only if hard_attn=True)
    """
    num_agents: int
    action_dim: int
    hidden_dim: int = 64
    comm_passes: int = 1
    comm_mode: str = "avg"
    hard_attn: bool = False
    comm_mask_zero: bool = False
    share_weights: bool = False
    
    def setup(self):
        # Create communication mask (no self-communication)
        if self.comm_mask_zero:
            self.comm_mask = jnp.zeros((self.num_agents, self.num_agents))
        else:
            self.comm_mask = jnp.ones((self.num_agents, self.num_agents)) - jnp.eye(self.num_agents)
        
        # Encoder
        self.encoder = nn.Dense(
            self.hidden_dim,
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0)
        )
        
        # Communication modules
        # Note: Flax doesn't support nn.ModuleList in the same way as PyTorch
        # We'll create individual modules and reference them by index
        if self.share_weights:
            # Create single set of weights, will reuse
            self.f_module = nn.Dense(
                self.hidden_dim,
                kernel_init=orthogonal(jnp.sqrt(2)),
                bias_init=constant(0.0),
                name='f_shared'
            )
            self.c_module = nn.Dense(
                self.hidden_dim,
                kernel_init=orthogonal(jnp.sqrt(2)),
                bias_init=constant(0.0),
                name='c_shared'
            )
        else:
            # Create separate modules for each pass
            # In Flax, we define them dynamically in __call__
            pass
        
        # Output heads
        self.action_head = nn.Dense(
            self.action_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0)
        )
        self.value_head = nn.Dense(
            1,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0)
        )
        
        if self.hard_attn:
            self.talk_head = nn.Dense(
                2,
                kernel_init=orthogonal(0.01),
                bias_init=constant(0.0)
            )
    
    def _agent_masks(
        self,
        batch_size: int,
        alive_mask: Optional[jnp.ndarray],
        comm_action: Optional[jnp.ndarray],
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Build communication masks matching IC3Net semantics.
        
        Returns:
            agent_mask: (B, sender, receiver, 1) mask for communication
            num_alive: scalar number of alive agents (as jnp.ndarray, not int)
        """
        n = self.num_agents
        
        if alive_mask is None:
            alive = jnp.ones(n)
        else:
            alive = alive_mask.astype(jnp.float32).reshape(-1)
        
        num_alive = jnp.sum(alive)  # Keep as JAX array, don't convert to int
        
        # Receiver mask: (1, 1, N) -> (B, N, N, 1)
        agent_mask = alive.reshape(1, 1, n)
        agent_mask = jnp.broadcast_to(agent_mask, (batch_size, n, n))
        agent_mask = jnp.expand_dims(agent_mask, -1)
        
        if self.hard_attn:
            if comm_action is None:
                ca = jnp.zeros(n)
            else:
                ca = comm_action.astype(jnp.float32).reshape(-1)
            ca_mask = ca.reshape(1, 1, n)
            ca_mask = jnp.broadcast_to(ca_mask, (batch_size, n, n))
            ca_mask = jnp.expand_dims(ca_mask, -1)
            agent_mask = agent_mask * ca_mask
        
        return agent_mask, num_alive
    
    @nn.compact
    def __call__(
        self,
        obs: jnp.ndarray,
        alive_mask: Optional[jnp.ndarray] = None,
        comm_action: Optional[jnp.ndarray] = None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, Optional[jnp.ndarray]]:
        """Forward pass through CommNet/IC3Net.
        
        Args:
            obs: (B, N, obs_dim) observations
            alive_mask: (N,) optional binary mask for alive agents
            comm_action: (N,) optional talk/silent actions (only used if hard_attn=True)
        
        Returns:
            logits: (B, N, action_dim) environment action logits
            values: (B, N) state values  
            talk_logits: (B, N, 2) talk/silent logits (None if hard_attn=False)
        """
        b, n, _ = obs.shape
        hdim = self.hidden_dim
        
        # Encode observations: (B, N, obs_dim) -> (B, N, H)
        x = self.encoder(obs)
        x = nn.tanh(x)
        
        # Initial hidden state
        h = x
        
        # Build agent masks for communication
        agent_mask, num_alive = self._agent_masks(b, alive_mask, comm_action)
        agent_mask_t = jnp.transpose(agent_mask, (0, 2, 1, 3))
        
        # Mask out self-communication
        comm_self_mask = self.comm_mask.reshape(1, n, n, 1)
        comm_self_mask = jnp.broadcast_to(comm_self_mask, (b, n, n, 1))
        
        # Communication passes
        for hop in range(self.comm_passes):
            # Expand hidden states: (B, N, H) -> (B, sender, receiver, H)
            comm = jnp.expand_dims(h, -2)  # (B, N, 1, H)
            comm = jnp.broadcast_to(comm, (b, n, n, hdim))
            
            # Mask self-communication
            comm = comm * comm_self_mask
            
            # Average mode: divide by (num_alive - 1)
            # Use jnp.where to avoid division by zero
            if self.comm_mode == "avg":
                denom = jnp.maximum(num_alive - 1.0, 1.0)
                comm = comm / denom
            
            # Mask dead/silent agents
            comm = comm * agent_mask
            comm = comm * agent_mask_t
            
            # Sum over senders: (B, sender, receiver, H) -> (B, receiver, H)
            comm_sum = jnp.sum(comm, axis=1)
            
            # Apply communication transform
            if self.share_weights:
                c = self.c_module(comm_sum)
                f_h = self.f_module(h)
            else:
                # Create hop-specific layers
                c_name = f'c_{hop}'
                f_name = f'f_{hop}'
                c = nn.Dense(
                    hdim,
                    kernel_init=orthogonal(jnp.sqrt(2)),
                    bias_init=constant(0.0),
                    name=c_name
                )(comm_sum)
                f_h = nn.Dense(
                    hdim,
                    kernel_init=orthogonal(jnp.sqrt(2)),
                    bias_init=constant(0.0),
                    name=f_name
                )(h)
            
            # Update hidden state with skip connection
            h = nn.tanh(x + f_h + c)
        
        # Output heads
        logits = self.action_head(h)
        values = self.value_head(h)
        values = jnp.squeeze(values, axis=-1)
        
        talk_logits = None
        if self.hard_attn:
            talk_logits = self.talk_head(h)
        
        return logits, values, talk_logits


class IndependentLSTM(nn.Module):
    """IC/IRIC network with LSTM recurrence and skip connections.
    
    Args:
        action_dim: Number of discrete actions
        hidden_dim: Size of hidden layers
    
    Input:
        obs: (B, N, obs_dim) batched observations
        carry: Optional tuple of (h, c) where each is (B, N, hidden_dim)
    
    Output:
        logits: (B, N, action_dim) action logits
        values: (B, N) state values
        carry: Tuple of (h_next, c_next) where each is (B, N, hidden_dim)
    """
    action_dim: int
    hidden_dim: int = 64
    
    @nn.compact
    def __call__(
        self, 
        obs: jnp.ndarray,
        carry: Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None
    ) -> Tuple[jnp.ndarray, jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
        b, n, _ = obs.shape
        
        # Initialize carry if not provided
        if carry is None:
            h = jnp.zeros((b, n, self.hidden_dim))
            c = jnp.zeros((b, n, self.hidden_dim))
        else:
            h, c = carry
        
        # Feedforward encoding
        x = nn.Dense(
            self.hidden_dim,
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0)
        )(obs)
        x = nn.tanh(x)
        
        x = nn.Dense(
            self.hidden_dim,
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0)
        )(x)
        x = nn.tanh(x + nn.Dense(
            self.hidden_dim,
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
            name='skip'
        )(obs))  # skip connection
        
        # LSTM cell - process each agent
        # Reshape to (B*N, H) for LSTM processing
        x_flat = x.reshape(b * n, self.hidden_dim)
        h_flat = h.reshape(b * n, self.hidden_dim)
        c_flat = c.reshape(b * n, self.hidden_dim)
        
        lstm_cell = nn.OptimizedLSTMCell(features=self.hidden_dim)
        (h1_flat, c1_flat), _ = lstm_cell((h_flat, c_flat), x_flat)
        
        # Reshape back to (B, N, H)
        h1 = h1_flat.reshape(b, n, self.hidden_dim)
        c1 = c1_flat.reshape(b, n, self.hidden_dim)
        
        # Skip connection around recurrence
        h_out = nn.tanh(h1 + x)
        
        # Output heads
        logits = nn.Dense(
            self.action_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0)
        )(h_out)
        
        values = nn.Dense(
            1,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0)
        )(h_out)
        
        return logits, jnp.squeeze(values, axis=-1), (h1, c1)


class CommNetLSTM(nn.Module):
    """CommNet/IC3Net with LSTM recurrence and message passing.
    
    This combines CommNet-style communication with LSTM recurrence,
    keeping the same hard-attention semantics for IC3Net.
    
    Args:
        num_agents: Number of agents
        action_dim: Number of discrete actions
        hidden_dim: Size of hidden layers
        comm_passes: Number of communication rounds
        comm_mode: 'avg' or 'sum' for message aggregation
        hard_attn: Enable IC3Net talk/silent gating
        comm_mask_zero: Debug option to disable all communication
        share_weights: Share weights across communication passes
    
    Input:
        obs: (B, N, obs_dim) observations
        carry: Optional tuple of (h, c) where each is (B, N, hidden_dim)
        alive_mask: (N,) optional binary mask for alive agents
        comm_action: (N,) optional talk/silent actions (only used if hard_attn=True)
    
    Output:
        logits: (B, N, action_dim) environment action logits
        values: (B, N) state values
        talk_logits: (B, N, 2) talk/silent logits (None if hard_attn=False)
        carry: Tuple of (h_next, c_next) where each is (B, N, hidden_dim)
    """
    num_agents: int
    action_dim: int
    hidden_dim: int = 64
    comm_passes: int = 1
    comm_mode: str = "avg"
    hard_attn: bool = False
    comm_mask_zero: bool = False
    share_weights: bool = False
    
    def setup(self):
        # Create communication mask (no self-communication)
        if self.comm_mask_zero:
            self.comm_mask = jnp.zeros((self.num_agents, self.num_agents))
        else:
            self.comm_mask = jnp.ones((self.num_agents, self.num_agents)) - jnp.eye(self.num_agents)
        
        # Encoder
        self.encoder = nn.Dense(
            self.hidden_dim,
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0)
        )
        
        # Output heads
        self.action_head = nn.Dense(
            self.action_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0)
        )
        self.value_head = nn.Dense(
            1,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0)
        )
        
        if self.hard_attn:
            self.talk_head = nn.Dense(
                2,
                kernel_init=orthogonal(0.01),
                bias_init=constant(0.0)
            )
    
    def _agent_masks(
        self,
        batch_size: int,
        alive_mask: Optional[jnp.ndarray],
        comm_action: Optional[jnp.ndarray],
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Build communication masks matching IC3Net semantics."""
        n = self.num_agents
        
        if alive_mask is None:
            alive = jnp.ones(n)
        else:
            alive = alive_mask.astype(jnp.float32).reshape(-1)
        
        num_alive = jnp.sum(alive)
        
        agent_mask = alive.reshape(1, 1, n)
        agent_mask = jnp.broadcast_to(agent_mask, (batch_size, n, n))
        agent_mask = jnp.expand_dims(agent_mask, -1)
        
        if self.hard_attn:
            if comm_action is None:
                ca = jnp.zeros(n)
            else:
                ca = comm_action.astype(jnp.float32).reshape(-1)
            ca_mask = ca.reshape(1, 1, n)
            ca_mask = jnp.broadcast_to(ca_mask, (batch_size, n, n))
            ca_mask = jnp.expand_dims(ca_mask, -1)
            agent_mask = agent_mask * ca_mask
        
        return agent_mask, num_alive
    
    @nn.compact
    def __call__(
        self,
        obs: jnp.ndarray,
        carry: Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None,
        alive_mask: Optional[jnp.ndarray] = None,
        comm_action: Optional[jnp.ndarray] = None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, Optional[jnp.ndarray], Tuple[jnp.ndarray, jnp.ndarray]]:
        """Forward pass through CommNet/IC3Net with LSTM."""
        b, n, _ = obs.shape
        hdim = self.hidden_dim
        
        # Initialize carry if not provided
        if carry is None:
            h_t = jnp.zeros((b, n, hdim))
            c_t = jnp.zeros((b, n, hdim))
        else:
            h_t, c_t = carry
        
        # Encode observations
        x = self.encoder(obs)
        x = nn.tanh(x)
        
        # Build agent masks
        agent_mask, num_alive = self._agent_masks(b, alive_mask, comm_action)
        agent_mask_t = jnp.transpose(agent_mask, (0, 2, 1, 3))
        comm_self_mask = self.comm_mask.reshape(1, n, n, 1)
        comm_self_mask = jnp.broadcast_to(comm_self_mask, (b, n, n, 1))
        
        h_round = h_t
        c_round = c_t
        
        # Communication passes with LSTM updates
        for hop in range(self.comm_passes):
            # Message passing
            comm = jnp.expand_dims(h_round, -2)
            comm = jnp.broadcast_to(comm, (b, n, n, hdim))
            comm = comm * comm_self_mask
            
            if self.comm_mode == "avg":
                denom = jnp.maximum(num_alive - 1.0, 1.0)
                comm = comm / denom
            
            comm = comm * agent_mask * agent_mask_t
            comm_sum = jnp.sum(comm, axis=1)
            
            # Apply communication transform
            if self.share_weights:
                msg = nn.Dense(
                    hdim,
                    kernel_init=orthogonal(jnp.sqrt(2)),
                    bias_init=constant(0.0),
                    name='c_shared'
                )(comm_sum)
            else:
                msg = nn.Dense(
                    hdim,
                    kernel_init=orthogonal(jnp.sqrt(2)),
                    bias_init=constant(0.0),
                    name=f'c_{hop}'
                )(comm_sum)
            
            # Combine input encoding with message
            z = x + msg
            
            # LSTM update - process each agent
            z_flat = z.reshape(b * n, hdim)
            h_flat = h_round.reshape(b * n, hdim)
            c_flat = c_round.reshape(b * n, hdim)
            
            lstm_cell = nn.OptimizedLSTMCell(
                features=hdim,
                name=f'lstm_{hop}' if not self.share_weights else 'lstm_shared'
            )
            (h_new_flat, c_new_flat), _ = lstm_cell((h_flat, c_flat), z_flat)
            
            h_new = h_new_flat.reshape(b, n, hdim)
            c_new = c_new_flat.reshape(b, n, hdim)
            
            # Skip connection
            h_round = nn.tanh(h_new + z)
            c_round = c_new
        
        # Output heads
        logits = self.action_head(h_round)
        values = self.value_head(h_round)
        values = jnp.squeeze(values, axis=-1)
        
        talk_logits = None
        if self.hard_attn:
            talk_logits = self.talk_head(h_round)
        
        return logits, values, talk_logits, (h_round, c_round)
