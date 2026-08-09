"""Focused tests for the CommNet-MAPPO actor and its env-major minibatching.

Two things can silently go wrong in a communicating MAPPO and still appear to
train: messages may never actually reach the other agent, and the PPO update may
shuffle an environment's agents into different minibatches so the CommNet pass
mixes agents that were never together in the rollout. Both are pinned here.
"""

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[2]))

import jaxmarl  # noqa: E402
from baselines.MAPPO.mappo_rnn_overcooked_v3_commnet import (  # noqa: E402
    CommNetActorRNN,
    env_major_minibatches,
)

NUM_AGENTS = 2
NUM_ENVS = 8
NUM_MINIBATCHES = 4
HIDDEN = 32

CONFIG = {
    "GRU_HIDDEN_DIM": HIDDEN,
    "FC_DIM_SIZE": HIDDEN,
    "ACTIVATION": "relu",
    "COMM_PASSES": 2,
    "COMM_MODE": "avg",
}


def _actor(config=CONFIG, action_dim=6):
    return CommNetActorRNN(action_dim, config=config, num_agents=NUM_AGENTS)


def _obs_shape():
    env = jaxmarl.make("overcooked_v3", layout="prep_kitchen", max_steps=800)
    return env.observation_space(env.agents[0]).shape


class TestEnvMajorMinibatching:
    """Agents that communicate must stay in the same minibatch."""

    def test_shape_and_agent_grouping(self):
        # Encode each actor as agent_id * 100 + env_id so we can trace it.
        ids = jnp.array(
            [[a * 100 + e for a in range(NUM_AGENTS) for e in range(NUM_ENVS)]]
        )
        perm = jax.random.permutation(jax.random.PRNGKey(0), NUM_ENVS)
        out = env_major_minibatches(
            ids[..., None], perm, NUM_AGENTS, NUM_ENVS, NUM_MINIBATCHES
        )[..., 0]

        envs_per_mb = NUM_ENVS // NUM_MINIBATCHES
        assert out.shape == (NUM_MINIBATCHES, 1, NUM_AGENTS * envs_per_mb)

        for m in range(NUM_MINIBATCHES):
            grouped = np.array(out[m, 0]).reshape(NUM_AGENTS, envs_per_mb)
            # Row a must contain only agent a...
            for a in range(NUM_AGENTS):
                assert {int(v) // 100 for v in grouped[a]} == {a}
            # ...over exactly the same set of environments as every other agent.
            env_sets = [{int(v) % 100 for v in grouped[a]} for a in range(NUM_AGENTS)]
            assert all(s == env_sets[0] for s in env_sets), env_sets

    def test_every_environment_appears_exactly_once(self):
        ids = jnp.arange(NUM_AGENTS * NUM_ENVS).reshape(1, -1)
        perm = jax.random.permutation(jax.random.PRNGKey(3), NUM_ENVS)
        out = env_major_minibatches(
            ids[..., None], perm, NUM_AGENTS, NUM_ENVS, NUM_MINIBATCHES
        )[..., 0]
        assert sorted(np.array(out).ravel().tolist()) == list(
            range(NUM_AGENTS * NUM_ENVS)
        )


class TestCommunicationFlows:
    """A partner's observation must reach the acting agent - and only its own env."""

    def _setup(self, config=CONFIG):
        obs_shape = _obs_shape()
        net = _actor(config)
        key = jax.random.PRNGKey(0)
        num_actors = NUM_AGENTS * NUM_ENVS
        obs = jax.random.normal(key, (1, num_actors, *obs_shape))
        dones = jnp.zeros((1, num_actors))
        hstate = jnp.zeros((num_actors, HIDDEN))
        params = net.init(key, hstate, (obs, dones))
        return net, params, obs, dones, hstate, obs_shape

    def test_partner_observation_changes_own_logits(self):
        net, params, obs, dones, hstate, obs_shape = self._setup()
        _, pi = net.apply(params, hstate, (obs, dones))
        base = pi.logits[0]

        # Perturb only agent 1 in environment 0 (agent-major actor layout).
        perturbed = obs.at[0, NUM_ENVS + 0].set(
            jax.random.normal(jax.random.PRNGKey(1), obs_shape)
        )
        _, pi2 = net.apply(params, hstate, (perturbed, dones))
        after = pi2.logits[0]

        # Agent 0 in env 0 hears about it...
        assert float(jnp.abs(after[0] - base[0]).max()) > 1e-6
        # ...and agent 0 in every other env does not.
        for e in range(1, NUM_ENVS):
            assert float(jnp.abs(after[e] - base[e]).max()) == 0.0

    def test_zero_comm_passes_is_a_no_communication_control(self):
        config = dict(CONFIG, COMM_PASSES=0)
        net, params, obs, dones, hstate, obs_shape = self._setup(config)
        _, pi = net.apply(params, hstate, (obs, dones))
        base = pi.logits[0]

        perturbed = obs.at[0, NUM_ENVS + 0].set(
            jax.random.normal(jax.random.PRNGKey(1), obs_shape)
        )
        _, pi2 = net.apply(params, hstate, (perturbed, dones))
        assert float(jnp.abs(pi2.logits[0][0] - base[0]).max()) == 0.0
