"""Rollout adapter for IC3Net, CommNet, IC, and IRIC V3 models."""

from __future__ import annotations

import distrax
import jax.numpy as jnp


class IC3NetRolloutPolicy:
    """Adapt any IC3Net-family network to the shared V3 rollout contract."""

    def __init__(self, env, config, network_factory):
        """Build the configured communication policy for a single environment."""

        self.env = env
        self.config = config
        self.recurrent = bool(config.get("RECURRENT", True))
        self.hidden_dim = int(config.get("HIDDEN_DIM", 64))
        self.is_independent = config.get("BASELINE", "ic3net") in ("ic", "iric")
        self.network, self.has_talk = network_factory(
            config,
            env.num_agents,
            env.action_space(env.agents[0]).n,
        )

    def initial_state(self, env):
        """Return recurrent and communication state for one episode."""

        hidden = None
        cell = None
        if self.recurrent:
            hidden = jnp.zeros((1, env.num_agents, self.hidden_dim))
            cell = jnp.zeros((1, env.num_agents, self.hidden_dim))
        communication = jnp.zeros((1, env.num_agents), dtype=jnp.int32)
        return hidden, cell, communication

    def act(self, params, obs, policy_state, dones, rng):
        """Choose environment and communication actions for one V3 step."""

        del dones, rng
        hidden, cell, communication = policy_state
        obs_batch = jnp.stack(
            [jnp.ravel(obs[agent]) for agent in self.env.agents],
            axis=0,
        )[None, ...]

        talk_logits = None
        if self.recurrent:
            carry = (hidden, cell)
            if self.is_independent:
                logits, _, next_carry = self.network.apply(
                    params,
                    obs_batch,
                    carry=carry,
                )
            elif self.has_talk:
                logits, _, talk_logits, next_carry = self.network.apply(
                    params,
                    obs_batch,
                    carry=carry,
                    comm_action=communication,
                )
            else:
                logits, _, talk_logits, next_carry = self.network.apply(
                    params,
                    obs_batch,
                    carry=carry,
                )
            hidden, cell = next_carry
        elif self.is_independent:
            logits, _ = self.network.apply(params, obs_batch)
        elif self.has_talk:
            logits, _, talk_logits = self.network.apply(
                params,
                obs_batch,
                comm_action=communication,
            )
        else:
            logits, _, talk_logits = self.network.apply(params, obs_batch)

        action = distrax.Categorical(logits=logits).mode()
        if self.has_talk and talk_logits is not None:
            talk_action = distrax.Categorical(logits=talk_logits).mode()
            communication = (
                jnp.ones_like(talk_action)
                if self.config.get("COMM_ACTION_ONE", False)
                else talk_action
            )

        env_action = {
            agent: action[0, index]
            for index, agent in enumerate(self.env.agents)
        }
        return env_action, (hidden, cell, communication)
