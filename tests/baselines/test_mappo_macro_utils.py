"""Focused tests for shared macro-action MAPPO calculations."""

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest


MAPPO_DIR = Path(__file__).parents[2] / "baselines" / "MAPPO"
sys.path.insert(0, str(MAPPO_DIR))

from mappo_macro_common import (  # noqa: E402
    _initial_best_eval_return,
    Actor,
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
    assert env.num_actions == 17


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
