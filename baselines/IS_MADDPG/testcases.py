# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
import pytest
import jax
import jax.numpy as jnp
import chex
from networks import ISCriticNet, ISAgentNet

# Shared test config
NUM_AGENTS = 3
OBS_DIM    = 16
ACT_DIM    = 5
MSG_DIM    = 8
HIDDEN_DIM = 32
HORIZON_H  = 4
BATCH      = 6


def make_actor():
    return ISAgentNet(
        obs_dim=OBS_DIM,
        act_dim=ACT_DIM,
        msg_dim=MSG_DIM,
        hidden_dim=HIDDEN_DIM,
        num_agents=NUM_AGENTS,
        horizon_H=HORIZON_H,
    )

def make_critic():
    return ISCriticNet(
        num_agents=NUM_AGENTS,
        obs_dim=OBS_DIM,
        act_dim=ACT_DIM,
        msg_dim=MSG_DIM,
        hidden_dim=HIDDEN_DIM,
    )

def init_actor(key=jax.random.PRNGKey(0)):
    net = make_actor()
    obs      = jnp.ones((BATCH, OBS_DIM))
    msgs     = jnp.ones((BATCH, NUM_AGENTS - 1, MSG_DIM))
    params   = net.init(key, obs, msgs, rng=jax.random.PRNGKey(1))
    return net, params

def init_critic(key=jax.random.PRNGKey(0)):
    net = make_critic()
    obs      = jnp.ones((BATCH, NUM_AGENTS, OBS_DIM))
    prev_msg = jnp.ones((BATCH, NUM_AGENTS, MSG_DIM))
    acts     = jnp.ones((BATCH, NUM_AGENTS, ACT_DIM))
    msgs     = jnp.ones((BATCH, NUM_AGENTS, MSG_DIM))
    params   = net.init(key, obs, prev_msg, acts, msgs)
    return net, params


# ===========================================================================
# Actor tests
# ===========================================================================

def test_actor_output_shapes():
    """All four outputs have the right shapes."""
    net, params = init_actor()
    obs  = jax.random.normal(jax.random.PRNGKey(2), (BATCH, OBS_DIM))
    msgs = jax.random.normal(jax.random.PRNGKey(3), (BATCH, NUM_AGENTS - 1, MSG_DIM))

    logits, onehot, msg_out, alpha = net.apply(
        params, obs, msgs, rng=jax.random.PRNGKey(4)
    )

    assert logits.shape  == (BATCH, ACT_DIM),              f"logits: {logits.shape}"
    assert onehot.shape  == (BATCH, ACT_DIM),              f"onehot: {onehot.shape}"
    assert msg_out.shape == (BATCH, MSG_DIM),              f"msg_out: {msg_out.shape}"
    assert alpha.shape   == (BATCH, 1, HORIZON_H),         f"alpha: {alpha.shape}"
    print("✅ test_actor_output_shapes")


def test_actor_action_onehot_is_valid():
    """With gumbel_hard=True each row must be exactly one-hot."""
    net, params = init_actor()
    obs  = jax.random.normal(jax.random.PRNGKey(5), (BATCH, OBS_DIM))
    msgs = jax.random.normal(jax.random.PRNGKey(6), (BATCH, NUM_AGENTS - 1, MSG_DIM))

    _, onehot, _, _ = net.apply(params, obs, msgs, rng=jax.random.PRNGKey(7))

    # Each row sums to 1
    chex.assert_trees_all_close(onehot.sum(axis=-1), jnp.ones(BATCH), atol=1e-5)

    # Each row has exactly one non-zero entry equal to 1
    chex.assert_trees_all_close(onehot.max(axis=-1), jnp.ones(BATCH), atol=1e-5)

    print("✅ test_actor_action_onehot_is_valid")


def test_actor_soft_action_when_hard_false():
    """With gumbel_hard=False action_onehot should be a probability vector (no hard 0/1)."""
    net, params = init_actor()
    obs  = jax.random.normal(jax.random.PRNGKey(8), (BATCH, OBS_DIM))
    msgs = jax.random.normal(jax.random.PRNGKey(9), (BATCH, NUM_AGENTS - 1, MSG_DIM))

    _, soft, _, _ = net.apply(
        params, obs, msgs, rng=jax.random.PRNGKey(10),
        gumbel_hard=False,
    )

    chex.assert_trees_all_close(soft.sum(axis=-1), jnp.ones(BATCH), atol=1e-5)
    # Should NOT be all-binary
    assert not jnp.all((soft == 0) | (soft == 1)), "Expected soft probs, got hard one-hots"
    assert jnp.all(soft >= 0) and jnp.all(soft <= 1)
    print("✅ test_actor_soft_action_when_hard_false")


def test_actor_attention_weights_sum_to_one():
    """Attention weights alpha should sum to 1 over the horizon dimension."""
    net, params = init_actor()
    obs  = jax.random.normal(jax.random.PRNGKey(11), (BATCH, OBS_DIM))
    msgs = jax.random.normal(jax.random.PRNGKey(12), (BATCH, NUM_AGENTS - 1, MSG_DIM))

    _, _, _, alpha = net.apply(params, obs, msgs, rng=jax.random.PRNGKey(13))

    # alpha: (B, 1, H) — should sum to 1 over H
    chex.assert_trees_all_close(alpha.sum(axis=-1), jnp.ones((BATCH, 1)), atol=1e-5)
    print("✅ test_actor_attention_weights_sum_to_one")


def test_actor_deterministic_given_same_rng():
    """Same RNG key -> identical outputs (no hidden state)."""
    net, params = init_actor()
    obs  = jax.random.normal(jax.random.PRNGKey(14), (BATCH, OBS_DIM))
    msgs = jax.random.normal(jax.random.PRNGKey(15), (BATCH, NUM_AGENTS - 1, MSG_DIM))

    out1 = net.apply(params, obs, msgs, rng=jax.random.PRNGKey(99))
    out2 = net.apply(params, obs, msgs, rng=jax.random.PRNGKey(99)) # apply same RNG key as used for out1; expect same output

    for a, b in zip(out1, out2):
        chex.assert_trees_all_close(a, b, atol=0.0)
    print("✅ test_actor_deterministic_given_same_rng")


def test_actor_different_rng_gives_different_actions():
    """Different RNG keys should (very likely) produce different hard actions."""
    net, params = init_actor()
    obs  = jax.random.normal(jax.random.PRNGKey(16), (BATCH, OBS_DIM))
    msgs = jax.random.normal(jax.random.PRNGKey(17), (BATCH, NUM_AGENTS - 1, MSG_DIM))

    _, onehot1, _, _ = net.apply(params, obs, msgs, rng=jax.random.PRNGKey(0))
    _, onehot2, _, _ = net.apply(params, obs, msgs, rng=jax.random.PRNGKey(1)) # apply a different key for onehot2; expect different outputs

    assert not jnp.all(onehot1 == onehot2), "Different RNG keys gave identical hard actions"
    print("✅ test_actor_different_rng_gives_different_actions")


def test_actor_horizon_1():
    """horizon_H=1 (no rollout steps) should still produce valid outputs."""
    net = ISAgentNet(
        obs_dim=OBS_DIM, act_dim=ACT_DIM, msg_dim=MSG_DIM,
        hidden_dim=HIDDEN_DIM, num_agents=NUM_AGENTS, horizon_H=1,
    )
    obs  = jax.random.normal(jax.random.PRNGKey(18), (BATCH, OBS_DIM))
    msgs = jax.random.normal(jax.random.PRNGKey(19), (BATCH, NUM_AGENTS - 1, MSG_DIM))
    params = net.init(jax.random.PRNGKey(0), obs, msgs, rng=jax.random.PRNGKey(1))

    logits, onehot, msg_out, alpha = net.apply(params, obs, msgs, rng=jax.random.PRNGKey(2))

    assert alpha.shape == (BATCH, 1, 1), f"Expected (B,1,1) alpha for H=1, got {alpha.shape}"
    chex.assert_trees_all_close(alpha.sum(axis=-1), jnp.ones((BATCH, 1)), atol=1e-5)
    print("✅ test_actor_horizon_1")


def test_actor_no_nan_or_inf():
    """Outputs must be finite for random inputs."""
    net, params = init_actor()
    obs  = jax.random.normal(jax.random.PRNGKey(20), (BATCH, OBS_DIM))
    msgs = jax.random.normal(jax.random.PRNGKey(21), (BATCH, NUM_AGENTS - 1, MSG_DIM))

    outputs = net.apply(params, obs, msgs, rng=jax.random.PRNGKey(22))
    for name, arr in zip(["logits", "onehot", "msg_out", "alpha"], outputs):
        assert jnp.all(jnp.isfinite(arr)), f"{name} contains NaN or Inf"
    print("✅ test_actor_no_nan_or_inf")


def test_actor_vmap_over_agents():
    """vmap over a batch of agents (joint policy usage pattern)."""
    net, params = init_actor()

    obs_all  = jax.random.normal(jax.random.PRNGKey(23), (NUM_AGENTS, BATCH, OBS_DIM))
    msgs_all = jax.random.normal(jax.random.PRNGKey(24), (NUM_AGENTS, BATCH, NUM_AGENTS - 1, MSG_DIM))
    keys     = jax.random.split(jax.random.PRNGKey(25), NUM_AGENTS)

    batched = jax.vmap(
        lambda o, m, k: net.apply(params, o, m, rng=k),
        in_axes=(0, 0, 0),
    )
    logits_all, onehot_all, msgs_out, alphas = batched(obs_all, msgs_all, keys)

    assert logits_all.shape == (NUM_AGENTS, BATCH, ACT_DIM)
    assert msgs_out.shape   == (NUM_AGENTS, BATCH, MSG_DIM)
    print("✅ test_actor_vmap_over_agents")


# ===========================================================================
# Critic tests
# ===========================================================================

def init_critic(key=jax.random.PRNGKey(0)):
    net = make_critic()
    obs      = jnp.ones((BATCH, NUM_AGENTS, OBS_DIM))
    prev_msg = jnp.ones((BATCH, NUM_AGENTS, MSG_DIM))
    acts     = jnp.ones((BATCH, NUM_AGENTS, ACT_DIM))
    msgs     = jnp.ones((BATCH, NUM_AGENTS, MSG_DIM))
    agent_id = jnp.zeros((BATCH,), dtype=jnp.int32)   # agent 0 for init
    params   = net.init(key, obs, prev_msg, acts, msgs, agent_id)
    return net, params


def _random_critic_inputs(base_key=0):
    """Helper to generate random critic inputs including agent_id."""
    obs      = jax.random.normal(jax.random.PRNGKey(base_key + 0), (BATCH, NUM_AGENTS, OBS_DIM))
    prev_msg = jax.random.normal(jax.random.PRNGKey(base_key + 1), (BATCH, NUM_AGENTS, MSG_DIM))
    acts     = jax.random.normal(jax.random.PRNGKey(base_key + 2), (BATCH, NUM_AGENTS, ACT_DIM))
    msgs     = jax.random.normal(jax.random.PRNGKey(base_key + 3), (BATCH, NUM_AGENTS, MSG_DIM))
    agent_id = jnp.zeros((BATCH,), dtype=jnp.int32)
    return obs, prev_msg, acts, msgs, agent_id


def test_critic_output_shape():
    net, params = init_critic()
    obs, prev_msg, acts, msgs, agent_id = _random_critic_inputs(30)
    q = net.apply(params, obs, prev_msg, acts, msgs, agent_id)
    assert q.shape == (BATCH, 1), f"Expected (B,1), got {q.shape}"
    print("✅ test_critic_output_shape")


def test_critic_no_nan_or_inf():
    net, params = init_critic()
    obs, prev_msg, acts, msgs, agent_id = _random_critic_inputs(34)
    q = net.apply(params, obs, prev_msg, acts, msgs, agent_id)
    assert jnp.all(jnp.isfinite(q)), "Critic output contains NaN or Inf"
    print("✅ test_critic_no_nan_or_inf")


def test_critic_different_agents_give_different_q():
    """Same global state but different agent_id should give different Q-values."""
    net, params = init_critic()
    obs, prev_msg, acts, msgs, _ = _random_critic_inputs(50)

    agent_0 = jnp.zeros((BATCH,), dtype=jnp.int32)
    agent_1 = jnp.ones((BATCH,),  dtype=jnp.int32)

    q0 = net.apply(params, obs, prev_msg, acts, msgs, agent_0)
    q1 = net.apply(params, obs, prev_msg, acts, msgs, agent_1)

    assert not jnp.allclose(q0, q1), "Different agent IDs gave identical Q-values"
    print("✅ test_critic_different_agents_give_different_q")


def test_critic_sensitive_to_action_change():
    net, params = init_critic()
    obs, prev_msg, acts, msgs, agent_id = _random_critic_inputs(38)
    q1 = net.apply(params, obs, prev_msg, acts, msgs, agent_id)
    acts2 = acts.at[:, 0, :].set(jax.random.normal(jax.random.PRNGKey(99), (BATCH, ACT_DIM)))
    q2 = net.apply(params, obs, prev_msg, acts2, msgs, agent_id)
    assert not jnp.allclose(q1, q2), "Q-value unchanged after action perturbation"
    print("✅ test_critic_sensitive_to_action_change")


def test_critic_deterministic():
    net, params = init_critic()
    obs, prev_msg, acts, msgs, agent_id = _random_critic_inputs(42)
    q1 = net.apply(params, obs, prev_msg, acts, msgs, agent_id)
    q2 = net.apply(params, obs, prev_msg, acts, msgs, agent_id)
    chex.assert_trees_all_close(q1, q2, atol=0.0)
    print("✅ test_critic_deterministic")


# ===========================================================================
# Run all
# ===========================================================================

if __name__ == "__main__":
    test_actor_output_shapes()
    test_actor_action_onehot_is_valid()
    test_actor_soft_action_when_hard_false()
    test_actor_attention_weights_sum_to_one()
    test_actor_deterministic_given_same_rng()
    test_actor_different_rng_gives_different_actions()
    test_actor_horizon_1()
    test_actor_no_nan_or_inf()
    test_actor_vmap_over_agents()
    test_critic_output_shape()
    test_critic_no_nan_or_inf()
    test_critic_different_agents_give_different_q()
    test_critic_sensitive_to_action_change()
    test_critic_deterministic()
    print("\nAll tests passed.")


# # Quick env sanity check — run separately before training
# from jaxmarl.environments.overcooked_v3.overcooked import OvercookedV3
# import jax
# import jax.numpy as jnp

# env = OvercookedV3(layout="cramped_room", shaped_rewards=True)
# rng = jax.random.PRNGKey(0)
# obs, state = env.reset(rng)

# total_shaped = 0.0
# for step in range(400):
#     if step % 50 == 0:
#         print("hi")
#     rng, k1, k2 = jax.random.split(rng, 3)
#     # Random actions
#     actions = {"agent_0": int(jax.random.randint(k1, (), 0, 6)),
#                "agent_1": int(jax.random.randint(k2, (), 0, 6))}
#     obs, state, rewards, dones, info = env.step_env(rng, state, actions)
#     shaped = info.get("shaped_reward", {})
#     total_shaped += sum(float(v) for v in shaped.values())
#     if dones.get("__all__", False):
#         break

# print(f"Total shaped reward over episode (random policy): {total_shaped:.3f}")
# # Should be > 0 if shaped rewards are working