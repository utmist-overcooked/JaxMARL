import jax
import jax.numpy as jnp
import numpy as np
import chex
from typing import Tuple, NamedTuple


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class BufferState(NamedTuple):
    """Immutable snapshot of the replay buffer.

    All arrays live on CPU as numpy (not JAX arrays) since the buffer is
    written to frequently from the environment loop, and JAX arrays are
    immutable — numpy is more efficient for incremental index-based writes.

    Fields:
        obs:            (capacity, N, obs_dim)
        prev_msgs:      (capacity, N, msg_dim)   messages from *previous* step
        actions:        (capacity, N, act_dim)   one-hot actions taken
        msgs:           (capacity, N, msg_dim)   messages sent this step
        rewards:        (capacity, N)            per-agent rewards
        next_obs:       (capacity, N, obs_dim)
        next_prev_msgs: (capacity, N, msg_dim)   = msgs (next step's prev_msgs)
        dones:          (capacity,)              episode termination flag
        idx:            scalar  write pointer (next slot to overwrite)
        size:           scalar  number of valid transitions stored
    """
    obs:            np.ndarray
    prev_msgs:      np.ndarray
    actions:        np.ndarray
    msgs:           np.ndarray
    rewards:        np.ndarray
    next_obs:       np.ndarray
    next_prev_msgs: np.ndarray
    dones:          np.ndarray
    idx:            int
    size:           int


class Batch(NamedTuple):
    """A sampled minibatch ready for JAX training.

    All arrays are JAX arrays with a leading batch dimension B.

    Fields:
        obs:            (B, N, obs_dim)
        prev_msgs:      (B, N, msg_dim)
        actions:        (B, N, act_dim)
        msgs:           (B, N, msg_dim)
        rewards:        (B, N)
        next_obs:       (B, N, obs_dim)
        next_prev_msgs: (B, N, msg_dim)
        dones:          (B,)
    """
    obs:            jnp.ndarray
    prev_msgs:      jnp.ndarray
    actions:        jnp.ndarray
    msgs:           jnp.ndarray
    rewards:        jnp.ndarray
    next_obs:       jnp.ndarray
    next_prev_msgs: jnp.ndarray
    dones:          jnp.ndarray


# ---------------------------------------------------------------------------
# Buffer functions
# ---------------------------------------------------------------------------

def buffer_init(
    *,
    capacity: int,
    num_agents: int,
    obs_dim: int,
    act_dim: int,
    msg_dim: int,
) -> BufferState:
    """Allocate and return an empty BufferState.

    All storage arrays are pre-allocated as numpy arrays of zeros.
    No JAX arrays are used here — the buffer is mutated in-place during
    environment interaction, which is more efficient than functional JAX
    updates for sequential single-step writes.

    Args:
        capacity:   maximum number of transitions stored (circular overwrite)
        num_agents: number of agents N
        obs_dim:    per-agent observation dimension
        act_dim:    per-agent action dimension
        msg_dim:    per-agent message dimension

    Returns:
        An empty BufferState with idx=0 and size=0.
    """
    return BufferState(
        obs=           np.zeros((capacity, num_agents, obs_dim),  dtype=np.float32),
        prev_msgs=     np.zeros((capacity, num_agents, msg_dim),  dtype=np.float32),
        actions=       np.zeros((capacity, num_agents, act_dim),  dtype=np.float32),
        msgs=          np.zeros((capacity, num_agents, msg_dim),  dtype=np.float32),
        rewards=       np.zeros((capacity, num_agents),           dtype=np.float32),
        next_obs=      np.zeros((capacity, num_agents, obs_dim),  dtype=np.float32),
        next_prev_msgs=np.zeros((capacity, num_agents, msg_dim),  dtype=np.float32),
        dones=         np.zeros((capacity,),                      dtype=np.float32),
        idx=0,
        size=0,
    )


def buffer_add(
    state: BufferState,
    *,
    obs:            np.ndarray,  # (N, obs_dim)
    prev_msgs:      np.ndarray,  # (N, msg_dim)
    actions:        np.ndarray,  # (N, act_dim)
    msgs:           np.ndarray,  # (N, msg_dim)
    rewards:        np.ndarray,  # (N,)
    next_obs:       np.ndarray,  # (N, obs_dim)
    next_prev_msgs: np.ndarray,  # (N, msg_dim)
    done:           bool,
) -> BufferState:
    """Write one transition into the buffer and return the updated state.

    The buffer is circular: once full, new transitions overwrite the oldest.
    Arrays inside BufferState are mutated in-place for efficiency (numpy),
    but a new NamedTuple is returned to maintain the functional interface
    expected by the rest of the JAX codebase.

    Args:
        state:          current BufferState
        obs:            current observations for all agents
        prev_msgs:      messages received at the previous step
        actions:        one-hot actions taken this step
        msgs:           messages sent this step
        rewards:        scalar reward per agent
        next_obs:       observations after the transition
        next_prev_msgs: msgs carried forward as prev_msgs for the next step
        done:           True if the episode ended after this transition

    Returns:
        Updated BufferState with the new transition written and idx/size advanced.
    """
    i = state.idx

    # Mutate numpy arrays in-place (safe — NamedTuple holds a reference)
    state.obs[i]            = obs
    state.prev_msgs[i]      = prev_msgs
    state.actions[i]        = actions
    state.msgs[i]           = msgs
    state.rewards[i]        = rewards
    state.next_obs[i]       = next_obs
    state.next_prev_msgs[i] = next_prev_msgs
    state.dones[i]          = float(done)

    # Advance write pointer (circular) and clamp size to capacity
    new_idx  = (i + 1) % state.obs.shape[0]
    new_size = min(state.size + 1, state.obs.shape[0])

    # Return a new NamedTuple with updated scalars (arrays mutated in-place)
    return state._replace(idx=new_idx, size=new_size)


def buffer_sample(
    state:      BufferState,
    batch_size: int,
    rng:        chex.PRNGKey,
) -> Tuple[Batch, chex.PRNGKey]:
    """Sample a random minibatch of transitions and convert to JAX arrays.

    Sampling uses JAX's PRNG for reproducibility and compatibility with
    jit-compiled training steps. The returned key is the consumed/split key
    so the caller's RNG state advances correctly.

    Args:
        state:      current BufferState (must have size >= batch_size)
        batch_size: number of transitions to sample
        rng:        JAX PRNG key

    Returns:
        batch:   Batch NamedTuple of JAX arrays with leading dim B
        new_rng: updated PRNG key for the caller to use next
    """
    if state.size < batch_size:
        raise ValueError(
            f"Buffer has {state.size} transitions, need {batch_size} to sample."
        )

    # Split key so the caller's RNG advances and we don't reuse keys
    rng, sample_key = jax.random.split(rng)

    # Sample without replacement not needed for RL — sample with replacement
    # is standard and avoids sorting/masking overhead
    idxs = jax.random.randint(
        sample_key,
        shape=(batch_size,),
        minval=0,
        maxval=state.size,
    )

    # Convert numpy index array to Python for numpy fancy indexing
    idxs_np = np.array(idxs)

    # Slice numpy arrays, then convert to JAX arrays for training
    # jnp.array() copies from CPU numpy to the JAX default device
    batch = Batch(
        obs=            jnp.array(state.obs[idxs_np]),
        prev_msgs=      jnp.array(state.prev_msgs[idxs_np]),
        actions=        jnp.array(state.actions[idxs_np]),
        msgs=           jnp.array(state.msgs[idxs_np]),
        rewards=        jnp.array(state.rewards[idxs_np]),
        next_obs=       jnp.array(state.next_obs[idxs_np]),
        next_prev_msgs= jnp.array(state.next_prev_msgs[idxs_np]),
        dones=          jnp.array(state.dones[idxs_np]),
    )

    return batch, rng


def buffer_size(state: BufferState) -> int:
    """Return the number of valid transitions currently stored."""
    return state.size


def buffer_is_ready(state: BufferState, batch_size: int) -> bool:
    """Return True if the buffer has enough transitions to sample a batch."""
    return state.size >= batch_size