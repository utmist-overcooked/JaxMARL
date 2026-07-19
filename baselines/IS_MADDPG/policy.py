# used for inference
import jax
import jax.numpy as jnp
import numpy as np
import chex
from typing import Dict, Optional, NamedTuple

from networks import ISAgentNet
from loss import received_messages  # reuse the helper


# ---------------------------------------------------------------------------
# Policy state  (replaces self._prev_msgs and self.actors)
# ---------------------------------------------------------------------------

class PolicyState(NamedTuple):
    """Immutable inference state for the joint policy.

    Kept separate from TrainState so that evaluation can run from a
    checkpoint without carrying optimizer state or critic params.

    Fields:
        actor_params: trained actor parameters (loaded from checkpoint)
        prev_msgs:    (N, msg_dim)  messages from the previous step,
                      initialised to zeros at episode start
    """
    actor_params: chex.ArrayTree
    prev_msgs:    jnp.ndarray   # (N, msg_dim)


# ---------------------------------------------------------------------------
# Initialisation and checkpoint loading
# ---------------------------------------------------------------------------

def policy_init(
    actor:      ISAgentNet,
    num_agents: int,
    msg_dim:    int,
    rng:        chex.PRNGKey,
    *,
    obs_dim:    int,
    act_dim:    int,
) -> PolicyState:
    """Initialise a fresh PolicyState with random actor parameters.

    In practice you will almost always follow this with policy_load() to
    replace the random params with trained ones from a checkpoint.

    Args:
        actor:      ISAgentNet module (stateless)
        num_agents: number of agents N
        msg_dim:    per-agent message dimension
        rng:        PRNG key for parameter initialisation
        obs_dim:    per-agent observation dimension (for dummy init input)
        act_dim:    per-agent action dimension      (for dummy init input)

    Returns:
        PolicyState with random params and zero prev_msgs.
    """
    rng, rng_init, rng_call = jax.random.split(rng, 3)

    dummy_obs  = jnp.zeros((1, obs_dim))
    dummy_msgs = jnp.zeros((1, num_agents - 1, msg_dim))

    actor_params = actor.init(rng_init, dummy_obs, dummy_msgs, rng=rng_call)

    return PolicyState(
        actor_params=actor_params,
        prev_msgs=jnp.zeros((num_agents, msg_dim)),
    )


def policy_load(
    policy_state: PolicyState,
    checkpoint_path: str,
) -> PolicyState:
    """Replace actor_params with weights loaded from a checkpoint file.

    Checkpoint format (saved by the trainer):
        {
            "actor_params":  <pytree of jax arrays>,
            ... (other keys like critic_params are ignored here)
        }

    Args:
        policy_state:     existing PolicyState (params will be replaced)
        checkpoint_path:  path to a .pkl or .msgpack checkpoint file

    Returns:
        PolicyState with loaded params and reset prev_msgs.
    """
    import pickle

    with open(checkpoint_path, "rb") as f:
        payload = pickle.load(f)

    if "actor_params" not in payload:
        raise ValueError(f"Checkpoint at {checkpoint_path!r} missing 'actor_params' key.")

    return PolicyState(
        actor_params=payload["actor_params"],
        prev_msgs=jnp.zeros_like(policy_state.prev_msgs),  # reset messages on load
    )


def policy_reset(policy_state: PolicyState) -> PolicyState:
    """Reset prev_msgs to zero at the start of a new episode.

    Call this at every env.reset() during evaluation.

    Args:
        policy_state: current PolicyState

    Returns:
        PolicyState with prev_msgs zeroed, params unchanged.
    """
    return policy_state._replace(
        prev_msgs=jnp.zeros_like(policy_state.prev_msgs)
    )


# ---------------------------------------------------------------------------
# Joint action inference
# ---------------------------------------------------------------------------

def get_joint_action(
    policy_state:  PolicyState,
    observations:  Dict[str, np.ndarray],  # agent_id -> (obs_dim,)
    actor:         ISAgentNet,
    rng:           chex.PRNGKey,
    *,
    num_agents:        int,
    deterministic:     bool  = True,
    gumbel_tau:        float = 1.0,
    delay_messages:    bool  = True,
    disable_interact:  bool  = False,
    interact_indices:  tuple = (4, 5),
) -> tuple[Dict[str, int], PolicyState, chex.PRNGKey]:
    """Compute joint env actions for all agents given their observations.

    Mirrors ISMADDPGPolicy.get_joint_action() from the PyTorch version.

    Message flow:
        delay_messages=True  (default, matches training):
            Use prev_msgs from the previous step as communication context,
            then update prev_msgs with the newly generated messages.
        delay_messages=False:
            Zero out messages each step (ablation / no-communication baseline).

    Args:
        policy_state:      current PolicyState (params + prev_msgs)
        observations:      dict mapping agent_id -> flat obs array (obs_dim,)
        actor:             ISAgentNet module (stateless)
        rng:               PRNG key (consumed for Gumbel sampling)
        num_agents:        number of agents N
        deterministic:     argmax over logits if True, else categorical sample
        gumbel_tau:        Gumbel temperature for message generation
        delay_messages:    use previous step's messages as context
        disable_interact:  mask out interact action indices if True
        interact_indices:  which action indices to mask (default (4, 5))

    Returns:
        actions:           dict mapping agent_id -> int action
        new_policy_state:  updated PolicyState with new prev_msgs
        new_rng:           advanced PRNG key for the caller
    """
    # Sort agent ids for consistent ordering (matches training)
    agent_ids = sorted(observations.keys(), key=lambda x: str(x))
    if len(agent_ids) != num_agents:
        raise ValueError(f"Expected {num_agents} agents, got {len(agent_ids)}")

    # Stack observations: (N, obs_dim)
    obs_all = jnp.array(
        np.stack([np.asarray(observations[aid], dtype=np.float32).reshape(-1)
                  for aid in agent_ids], axis=0)
    )

    # Use zeroed messages if delay_messages=False (no-communication ablation)
    prev_msgs = (policy_state.prev_msgs
                 if delay_messages
                 else jnp.zeros_like(policy_state.prev_msgs))

    # Build received messages: (1, N, msg_dim) -> (1, N, N-1, msg_dim)
    # Add batch dim of 1 since the actor expects (B, ...) inputs
    prev_msgs_batched = prev_msgs[None, :, :]                   # (1, N, msg_dim)
    received = received_messages(prev_msgs_batched)             # (1, N, N-1, msg_dim)

    # ------------------------------------------------------------------
    # Forward pass: run actor for each agent
    # ------------------------------------------------------------------
    actions:   Dict[str, int]  = {}
    next_msgs: list[jnp.ndarray] = []

    for i, aid in enumerate(agent_ids):
        rng, subkey = jax.random.split(rng)

        logits, _, msg, _ = actor.apply(
            policy_state.actor_params,
            obs_all[i : i + 1],         # (1, obs_dim)
            received[:, i, :, :],       # (1, N-1, msg_dim)
            rng=subkey,
            gumbel_tau=gumbel_tau,
            gumbel_hard=True,
        )
        logits = logits[0]  # (act_dim,)  remove batch dim

        # Optionally mask interact actions (e.g. during training-free eval)
        if disable_interact and logits.shape[-1] > max(interact_indices):
            mask = jnp.zeros_like(logits)
            mask = mask.at[list(interact_indices)].set(-1e9)
            logits = logits + mask

        # Action selection
        if deterministic:
            act = int(jnp.argmax(logits))
        else:
            # Categorical sample via Gumbel-argmax (no straight-through needed)
            rng, sample_key = jax.random.split(rng)
            gumbel = -jnp.log(-jnp.log(
                jax.random.uniform(sample_key, logits.shape,
                                   minval=jnp.finfo(jnp.float32).tiny)
            ))
            act = int(jnp.argmax(logits + gumbel))

        actions[aid] = act
        next_msgs.append(msg[0])  # (msg_dim,)

    # Stack new messages: list of N (msg_dim,) -> (N, msg_dim)
    new_prev_msgs = jnp.stack(next_msgs, axis=0)

    new_policy_state = policy_state._replace(prev_msgs=new_prev_msgs)

    return actions, new_policy_state, rng