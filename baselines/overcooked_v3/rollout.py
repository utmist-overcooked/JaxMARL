"""Shared Overcooked V3 episode execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import chex
import jax
import jax.numpy as jnp

from baselines.overcooked_v3.policy import RolloutPolicy


@dataclass(frozen=True)
class RolloutEpisode:
    """Host-side record of one environment episode."""

    states: tuple[Any, ...]
    actions: tuple[Mapping[str, chex.Array], ...]
    rewards: tuple[Mapping[str, chex.Array], ...]
    infos: tuple[Mapping[str, Any], ...]
    terminated: bool

    @property
    def length(self) -> int:
        """Return the number of environment transitions in the episode."""

        return len(self.actions)


def _distribution_mode(value):
    """Replace a distribution with its highest-probability output."""

    mode = getattr(value, "mode", None)
    return mode() if callable(mode) else value


def resolve_policy_actions(action_output, agents) -> dict[str, chex.Array]:
    """Convert direct actions or a distribution to an agent-keyed dictionary."""

    if isinstance(action_output, Mapping):
        if set(action_output) != set(agents):
            raise ValueError("Policy action keys must match the environment agents")
        return {
            agent: _distribution_mode(action_output[agent])
            for agent in agents
        }

    actions = jnp.asarray(_distribution_mode(action_output))
    while actions.ndim > 1 and actions.shape[0] == 1:
        actions = jnp.squeeze(actions, axis=0)
    if actions.ndim == 0 and len(agents) == 1:
        actions = actions[jnp.newaxis]
    if actions.ndim == 0 or actions.shape[0] != len(agents):
        raise ValueError(
            "Policy action output must contain exactly one action per agent"
        )
    return {
        agent: actions[index]
        for index, agent in enumerate(agents)
    }


def rollout_episode(
    env,
    policy: RolloutPolicy,
    params,
    *,
    seed: int,
    max_steps: int,
) -> RolloutEpisode:
    """Run one Overcooked V3 episode and retain states for rendering.

    ``step_env`` is used instead of the public auto-resetting ``step`` method so
    the final animation frame is the terminal state rather than a freshly reset
    kitchen.
    """

    max_steps = int(max_steps)
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")

    rng = jax.random.PRNGKey(int(seed))
    rng, reset_rng = jax.random.split(rng)
    obs, env_state = env.reset(reset_rng)
    policy_state = policy.initial_state(env)
    dones = {agent: jnp.array(False) for agent in env.agents}
    dones["__all__"] = jnp.array(False)

    states = [env_state]
    actions = []
    rewards = []
    infos = []
    terminated = False

    for _ in range(max_steps):
        rng, action_rng, step_rng = jax.random.split(rng, 3)
        action_output, policy_state = policy.act(
            params,
            obs,
            policy_state,
            dones,
            action_rng,
        )
        env_actions = resolve_policy_actions(action_output, env.agents)
        obs, env_state, reward, dones, info = env.step_env(
            step_rng,
            env_state,
            env_actions,
        )

        states.append(env_state)
        actions.append(env_actions)
        rewards.append(reward)
        infos.append(info)

        terminated = bool(jax.device_get(dones["__all__"]))
        if terminated:
            break

    return RolloutEpisode(
        states=tuple(states),
        actions=tuple(actions),
        rewards=tuple(rewards),
        infos=tuple(infos),
        terminated=terminated,
    )
