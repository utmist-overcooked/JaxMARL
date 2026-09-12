"""Tests for model-agnostic Overcooked V3 rollout GIF logging."""

from pathlib import Path

import imageio.v2 as imageio
import distrax
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxmarl
from baselines.IC3Net.ic3net_train import _build_network
from baselines.IPPO.ippo_cnn_overcooked_v3 import _make_network
from baselines.IPPO.ippo_rnn_overcooked_v3 import ActorCriticRNN
from baselines.overcooked_v3.hooks import RolloutGifHook, select_seed_params
from baselines.overcooked_v3.models.ic3net import IC3NetRolloutPolicy
from baselines.overcooked_v3.models.ippo import (
    IPPOCNNRolloutPolicy,
    IPPORNNRolloutPolicy,
)
from baselines.overcooked_v3.models.mappo_macro import (
    MacroMAPPORolloutPolicy,
    make_macro_rollout_environment,
)
from baselines.overcooked_v3.rollout import resolve_policy_actions, rollout_episode


def _config(tmp_path: Path) -> dict:
    """Return a tiny V3 configuration suitable for adapter smoke tests."""

    return {
        "ENV_NAME": "overcooked_v3",
        "ENV_KWARGS": {
            "layout": "single_file",
            "max_steps": 3,
            "shaped_rewards": True,
        },
        "ACTIVATION": "relu",
        "CNN_CHANNELS": 8,
        "CNN_EMBED_DIM": 8,
        "FC_DIM_SIZE": 8,
        "GRU_HIDDEN_DIM": 8,
        "HIDDEN_DIM": 8,
        "BASELINE": "ic",
        "RECURRENT": False,
        "SEED": 1,
        "NUM_SEEDS": 1,
        "NUM_UPDATES": 1,
        "NUM_CHECKPOINTS": 1,
        "ROLLOUT_GIF_ENABLED": True,
        "ROLLOUT_GIF_COUNT": 1,
        "ROLLOUT_GIF_MAX_STEPS": 3,
        "ROLLOUT_GIF_SEED_INDEX": 0,
        "ROLLOUT_GIF_ENV_SEED": 9,
        "ROLLOUT_GIF_DIR": str(tmp_path),
        "RUN_NAME": "random-ippo",
        "WANDB_MODE": "disabled",
    }


def _cnn_params(env, policy):
    """Initialize random CNN parameters with the training input shape."""

    obs_shape = env.observation_space(env.agents[0]).shape
    return policy.network.init(
        jax.random.PRNGKey(4),
        jnp.zeros((env.num_agents, *obs_shape)),
    )


def _rnn_params(env, policy):
    """Initialize random RNN parameters with the training input shape."""

    obs_shape = env.observation_space(env.agents[0]).shape
    actor_input = (
        jnp.zeros((1, env.num_agents, *obs_shape)),
        jnp.zeros((1, env.num_agents), dtype=bool),
    )
    return policy.network.init(
        jax.random.PRNGKey(5),
        policy.initial_state(env),
        actor_input,
    )


def _ic_params(env, policy):
    """Initialize random feed-forward independent-policy parameters."""

    obs_dim = int(np.prod(env.observation_space(env.agents[0]).shape))
    return policy.network.init(
        jax.random.PRNGKey(6),
        jnp.zeros((1, env.num_agents, obs_dim)),
    )


def test_current_v3_policy_adapters_run_terminal_episodes(tmp_path):
    """Exercise CNN, RNN, and IC3Net-family adapters on the real V3 env."""

    config = _config(tmp_path)
    env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    policies_and_params = (
        (IPPOCNNRolloutPolicy(env, config, _make_network), _cnn_params),
        (IPPORNNRolloutPolicy(env, config, ActorCriticRNN), _rnn_params),
        (IC3NetRolloutPolicy(env, config, _build_network), _ic_params),
    )

    for policy, init_params in policies_and_params:
        episode = rollout_episode(
            env,
            policy,
            init_params(env, policy),
            seed=config["ROLLOUT_GIF_ENV_SEED"],
            max_steps=config["ROLLOUT_GIF_MAX_STEPS"],
        )

        assert episode.length == 3
        assert len(episode.states) == 4
        assert episode.terminated is True
        assert int(episode.states[-1].time) == 3
        assert set(episode.actions[-1]) == set(env.agents)


def test_direct_policy_actions_are_used_unchanged():
    """Preserve explicit actions rather than applying another selection rule."""

    direct = {"agent_0": jnp.array(4), "agent_1": jnp.array(2)}

    resolved = resolve_policy_actions(direct, ("agent_0", "agent_1"))

    assert int(resolved["agent_0"]) == 4
    assert int(resolved["agent_1"]) == 2


def test_distribution_policy_output_uses_highest_probability_action():
    """Use distribution mode only when the policy returns a distribution."""

    distribution = distrax.Categorical(
        logits=jnp.array([[[0.0, 3.0], [4.0, 1.0]]])
    )

    resolved = resolve_policy_actions(
        distribution,
        ("agent_0", "agent_1"),
    )

    assert int(resolved["agent_0"]) == 1
    assert int(resolved["agent_1"]) == 0


def test_hook_records_random_ippo_cnn_policy_gif(tmp_path):
    """Render a valid GIF through the same hook used during training."""

    config = _config(tmp_path)
    env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    policy = IPPOCNNRolloutPolicy(env, config, _make_network)
    hook = RolloutGifHook(
        config,
        lambda rollout_env: IPPOCNNRolloutPolicy(
            rollout_env,
            config,
            _make_network,
        ),
    )

    result = hook.record(
        _cnn_params(env, policy),
        checkpoint_index=1,
        update_step=1,
        env_step=6,
        training_seed=123,
    )

    assert result.path.exists()
    assert result.path.stat().st_size > 0
    assert result.episode_length == 3
    assert result.uploaded is False
    assert len(imageio.mimread(result.path)) >= 1
    assert "trainseed123_rolloutseed9" in result.path.name


@pytest.mark.parametrize(
    ("variant", "env_name"),
    (
        ("boundary", "overcooked_v3_macro"),
        ("every_step", "overcooked_v3_macro_interruptible"),
        ("replan", "overcooked_v3_macro_interruptible"),
    ),
)
def test_macro_mappo_policy_adapters_run_real_v3_episodes(
    variant,
    env_name,
):
    """Exercise all three macro MAPPO action rules in the shared runner."""

    config = {
        "ENV_NAME": env_name,
        "ENV_KWARGS": {
            "layout": "cramped_room",
            "max_steps": 3,
            "max_macro_steps": 3,
        },
        "HIDDEN_SIZE": 8,
    }
    env = make_macro_rollout_environment(config)
    policy = MacroMAPPORolloutPolicy.create(env, config, variant)
    obs, _ = env.reset(jax.random.PRNGKey(20))
    actor_obs = jnp.stack([obs[agent] for agent in env.agents])
    params = policy.actor.init(jax.random.PRNGKey(21), actor_obs)

    episode = rollout_episode(
        env,
        policy,
        params,
        seed=22,
        max_steps=3,
    )

    assert episode.length == 3
    assert episode.terminated is True
    assert set(episode.actions[-1]) == set(env.agents)


def test_macro_checkpoint_hook_renders_base_v3_state(tmp_path):
    """Render macro actions through the same base-state visualizer as V3."""

    config = {
        "ENV_NAME": "overcooked_v3_macro",
        "ENV_KWARGS": {
            "layout": "cramped_room",
            "max_steps": 3,
            "max_macro_steps": 3,
        },
        "HIDDEN_SIZE": 8,
        "SEED": 2,
        "NUM_SEEDS": 1,
        "NUM_CHECKPOINTS": 1,
        "ROLLOUT_GIF_ENABLED": True,
        "ROLLOUT_GIF_COUNT": 1,
        "ROLLOUT_GIF_ENV_SEED": 5,
        "ROLLOUT_GIF_DIR": str(tmp_path),
        "WANDB_MODE": "disabled",
    }
    env = make_macro_rollout_environment(config)
    policy = MacroMAPPORolloutPolicy.create(env, config, "boundary")
    obs, _ = env.reset(jax.random.PRNGKey(23))
    params = policy.actor.init(
        jax.random.PRNGKey(24),
        jnp.stack([obs[agent] for agent in env.agents]),
    )
    hook = RolloutGifHook(
        config,
        lambda rollout_env: MacroMAPPORolloutPolicy.create(
            rollout_env,
            config,
            "boundary",
        ),
        make_macro_rollout_environment,
    )

    result = hook.record(
        params,
        checkpoint_index=1,
        update_step=4,
        env_step=12,
        training_seed=2,
    )

    assert result.path.exists()
    # A random boundary policy may wait for the whole episode, and GIF encoders
    # are allowed to collapse identical frames into one stored image.
    assert len(imageio.mimread(result.path)) >= 1


def test_select_seed_params_handles_vmapped_parameters():
    """Select the configured parameter slice from a vectorized seed batch."""

    params = {"weights": jnp.arange(12).reshape(2, 2, 3)}

    selected, seed = select_seed_params(params, np.array([101, 202]), seed_index=1)

    np.testing.assert_array_equal(selected["weights"], params["weights"][1])
    assert seed == 202
