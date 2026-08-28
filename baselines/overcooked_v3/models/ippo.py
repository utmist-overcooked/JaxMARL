"""Rollout adapters for feed-forward and recurrent IPPO V3 models."""

from __future__ import annotations

import jax.numpy as jnp


class IPPOCNNRolloutPolicy:
    """Adapt the IPPO CNN network to the shared V3 rollout contract."""

    def __init__(self, env, config, network_factory):
        """Build the same CNN network shape used by the training loop."""

        self.env = env
        self.network = network_factory(
            env.action_space(env.agents[0]).n,
            config,
        )

    def initial_state(self, env):
        """Return the stateless CNN policy's empty episode state."""

        del env
        return None

    def act(self, params, obs, policy_state, dones, rng):
        """Return the CNN distribution and unchanged policy state."""

        del dones, rng
        obs_batch = jnp.stack([obs[agent] for agent in self.env.agents]).reshape(
            -1,
            *self.env.observation_space(self.env.agents[0]).shape,
        )
        distribution, _ = self.network.apply(params, obs_batch)
        return distribution, policy_state


class IPPORNNRolloutPolicy:
    """Adapt the recurrent IPPO actor-critic to the V3 rollout contract."""

    def __init__(self, env, config, network_factory):
        """Build the recurrent network with its training configuration."""

        self.env = env
        self.hidden_dim = int(config.get("GRU_HIDDEN_DIM", 128))
        self.network = network_factory(
            env.action_space(env.agents[0]).n,
            config=config,
        )

    def initial_state(self, env):
        """Return a zero GRU state for every environment agent."""

        return jnp.zeros((env.num_agents, self.hidden_dim))

    def act(self, params, obs, policy_state, dones, rng):
        """Return the recurrent distribution and next policy state."""

        del rng
        obs_batch = jnp.stack([obs[agent] for agent in self.env.agents]).reshape(
            -1,
            *self.env.observation_space(self.env.agents[0]).shape,
        )
        done_batch = jnp.stack([dones[agent] for agent in self.env.agents])
        actor_input = (
            obs_batch[jnp.newaxis, :],
            done_batch[jnp.newaxis, :],
        )
        next_state, distribution, _ = self.network.apply(
            params,
            policy_state,
            actor_input,
        )
        return distribution, next_state
