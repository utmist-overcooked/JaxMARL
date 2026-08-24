"""Focused tests for shared macro-action MAPPO calculations."""

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest


MAPPO_DIR = Path(__file__).parents[2] / "baselines" / "MAPPO"
sys.path.insert(0, str(MAPPO_DIR))

from mappo_macro_common import (  # noqa: E402
    _initial_best_eval_return,
    Actor,
    ScannedRNN,
    add_annealed_shaped_reward,
    build_env,
    calculate_smdp_gae,
    initialize_config,
    metadata_batch,
)


def test_resumed_run_restores_previous_best_eval_return(tmp_path):
    (tmp_path / "best_eval.json").write_text('{"eval_return": 12.5}')

    assert _initial_best_eval_return(tmp_path, "checkpoints/latest.json") == 12.5


def test_fresh_run_does_not_reuse_previous_best_eval_return(tmp_path):
    (tmp_path / "best_eval.json").write_text('{"eval_return": 12.5}')

    assert _initial_best_eval_return(tmp_path, None) == -jnp.inf


def test_shaped_reward_anneals_over_primitive_step_horizon():
    sparse = {"agent_0": jnp.array([1.0])}
    shaped = {"agent_0": jnp.array([2.0])}

    start_reward, start_coefficient = add_annealed_shaped_reward(
        sparse, shaped, 0, 2_500_000
    )
    halfway_reward, halfway_coefficient = add_annealed_shaped_reward(
        sparse, shaped, 1_250_000, 2_500_000
    )
    end_reward, end_coefficient = add_annealed_shaped_reward(
        sparse, shaped, 2_500_000, 2_500_000
    )

    assert jnp.allclose(start_reward["agent_0"], 3.0)
    assert jnp.allclose(start_coefficient, 1.0)
    assert jnp.allclose(halfway_reward["agent_0"], 2.0)
    assert jnp.allclose(halfway_coefficient, 0.5)
    assert jnp.allclose(end_reward["agent_0"], 1.0)
    assert jnp.allclose(end_coefficient, 0.0)


def test_metadata_batch_uses_agent_major_order():
    values = jnp.array([[0, 1], [2, 3], [4, 5]])
    assert jnp.array_equal(
        metadata_batch(values, 6), jnp.array([0, 2, 4, 1, 3, 5])
    )


def test_smdp_gae_discounts_by_macro_duration_and_skips_empty_slots():
    reward = jnp.array([[1.0], [99.0], [2.0]])
    duration = jnp.array([[2], [1], [1]])
    done = jnp.array([[0.0], [0.0], [1.0]])
    value = jnp.zeros_like(reward)
    valid = jnp.array([[True], [False], [True]])

    advantage, target = calculate_smdp_gae(
        reward, duration, done, value, valid, gamma=0.5, gae_lambda=1.0
    )

    assert jnp.allclose(advantage[:, 0], jnp.array([1.5, 0.0, 2.0]))
    assert jnp.allclose(target, advantage)


def test_macro_world_state_contains_actor_and_centralized_features():
    env = build_env(
        {
            "ENV_NAME": "overcooked_v3_macro",
            "ENV_KWARGS": {"layout": "cramped_room"},
        }
    )
    actor_size = env.observation_space(env.agents[0]).shape[0]

    assert actor_size > env._env.base_obs_size
    assert env.world_state_size() == actor_size * env.num_agents + env.num_agents


def test_macro_actor_uses_expanded_environment_action_space():
    env = build_env(
        {
            "ENV_NAME": "overcooked_v3_macro",
            "ENV_KWARGS": {"layout": "cramped_room"},
        }
    )
    actor = Actor(env.num_actions, hidden_size=16)
    obs = jnp.zeros((1, env.observation_space(env.agents[0]).shape[0]))

    logits = actor.apply(actor.init(jax.random.PRNGKey(0), obs), obs)

    assert logits.shape == (1, env.num_actions)
    assert env.num_actions == 18


def test_scanned_rnn_without_advance_mask_advances_every_step():
    """A 2-tuple call (no gate) must behave exactly as before: advance always.

    This is what guarantees the every_step trainer is unaffected by the gate.
    """
    rng = jax.random.PRNGKey(0)
    time_steps, batch, features = 5, 3, 4
    ins = jax.random.normal(rng, (time_steps, batch, features))
    resets = jnp.zeros((time_steps, batch), dtype=jnp.bool_)
    carry = ScannedRNN.initialize_carry(batch, features)

    params = ScannedRNN().init(rng, carry, (ins, resets))
    ungated_carry, ungated_out = ScannedRNN().apply(params, carry, (ins, resets))
    all_advance = jnp.ones((time_steps, batch), dtype=jnp.bool_)
    gated_carry, gated_out = ScannedRNN().apply(
        params, carry, (ins, resets, all_advance)
    )

    assert jnp.allclose(ungated_out, gated_out, atol=1e-6)
    assert jnp.allclose(ungated_carry, gated_carry, atol=1e-6)


def test_scanned_rnn_advance_false_freezes_the_carry():
    """With advance all-False the hidden must never move off its initial value."""
    rng = jax.random.PRNGKey(1)
    time_steps, batch, features = 4, 2, 3
    ins = jax.random.normal(rng, (time_steps, batch, features))
    resets = jnp.zeros((time_steps, batch), dtype=jnp.bool_)
    no_advance = jnp.zeros((time_steps, batch), dtype=jnp.bool_)
    carry = ScannedRNN.initialize_carry(batch, features)

    params = ScannedRNN().init(rng, carry, (ins, resets, no_advance))
    final_carry, _ = ScannedRNN().apply(params, carry, (ins, resets, no_advance))

    assert jnp.allclose(final_carry, carry, atol=1e-7)


def test_scanned_rnn_gate_matches_compacted_decision_sequence():
    """The core §5.2 property behind the boundary RNN trainer.

    Replaying the decision-gated GRU over the fixed-size buffer -- advancing
    only on the (in-order) decision steps and carrying the hidden through the
    frozen steps in between -- must produce exactly the same hidden sequence as
    running an ordinary GRU over just the compacted decision sequence. This is
    what lets the update reuse the fixed-shape masked buffer instead of a
    dynamic squeeze, while still doing true BPTT over the decision sequence.
    """
    rng = jax.random.PRNGKey(2)
    time_steps, batch, features = 6, 1, 4
    ins = jax.random.normal(rng, (time_steps, batch, features))
    # Decisions complete at steps 0, 2, 3, 5; the buffer is frozen elsewhere.
    advance = jnp.array([True, False, True, True, False, True]).reshape(
        time_steps, batch
    )
    # Episode-first decisions reset the hidden; must be a subset of `advance`
    # (non-decision slots always carry decision_reset=False in the trainer).
    resets = jnp.array([True, False, False, True, False, False]).reshape(
        time_steps, batch
    )
    carry = ScannedRNN.initialize_carry(batch, features)

    params = ScannedRNN().init(rng, carry, (ins, resets, advance))
    gated_carry, gated_out = ScannedRNN().apply(
        params, carry, (ins, resets, advance)
    )

    # Compact to just the decision steps, in order, and run without the gate.
    decision_idx = np.where(np.asarray(advance)[:, 0])[0]
    compact_carry, compact_out = ScannedRNN().apply(
        params, carry, (ins[decision_idx], resets[decision_idx])
    )

    # Output at each decision step matches the compacted run, and the final
    # carry (seed for the next rollout) is identical too.
    assert jnp.allclose(gated_out[decision_idx], compact_out, atol=1e-5)
    assert jnp.allclose(gated_carry, compact_carry, atol=1e-5)


def test_config_rejects_a_silently_truncated_timestep_budget():
    env = build_env(
        {
            "ENV_NAME": "overcooked_v3_macro",
            "ENV_KWARGS": {"layout": "cramped_room"},
        }
    )
    with pytest.raises(ValueError, match="silently truncate"):
        initialize_config(
            {
                "NUM_ENVS": 16,
                "NUM_STEPS": 128,
                "TOTAL_TIMESTEPS": 5_000_000,
                "NUM_MINIBATCHES": 4,
            },
            env,
        )
