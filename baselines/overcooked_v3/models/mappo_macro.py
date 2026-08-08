"""Rollout adapters for the three macro-action MAPPO actors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import distrax
import jax.numpy as jnp

from baselines.MAPPO.mappo_macro_common import Actor, ReplanActor


CONTINUE = 0
REPLAN = 1


class MacroRolloutEnvironment:
    """Expose macro steps while retaining base V3 states for rendering."""

    def __init__(self, wrapped_env):
        """Unwrap logging while preserving observation augmentation."""

        self.wrapped_env = wrapped_env
        self.augmentation_wrapper = wrapped_env._env
        self.raw_env = self.augmentation_wrapper._env
        self.agents = self.raw_env.agents
        self.num_agents = self.raw_env.num_agents

    def __getattr__(self, name):
        """Delegate visualizer metadata to the raw macro environment."""

        return getattr(self.raw_env, name)

    def reset(self, key):
        """Reset through logging and return the underlying macro state."""

        obs, log_state = self.wrapped_env.reset(key)
        return obs, log_state.env_state

    def step_env(self, key, state, actions):
        """Step without automatic reset and rebuild augmented observations."""

        raw_obs, next_state, reward, done, info = self.raw_env.step_env(
            key,
            state,
            actions,
        )
        obs = self.augmentation_wrapper._augment(raw_obs, next_state)
        return obs, next_state, reward, done, info


def make_macro_rollout_environment(config: Mapping[str, Any]):
    """Build the macro trainer's wrapped environment for shared rollouts."""

    from baselines.MAPPO.mappo_macro_common import build_env

    return MacroRolloutEnvironment(build_env(dict(config)))


@dataclass(frozen=True)
class MacroMAPPORolloutPolicy:
    """Adapt boundary, every-step, or replan MAPPO actors for GIF rollouts."""

    actor: Any
    variant: str
    agents: tuple[str, ...]
    num_actions: int

    @classmethod
    def create(cls, env, config, variant: str):
        """Build the actor architecture used by the requested macro trainer."""

        if variant not in {"boundary", "every_step", "replan"}:
            raise ValueError(f"Unknown macro MAPPO variant: {variant}")
        actor = (
            ReplanActor(env.num_actions, int(config["HIDDEN_SIZE"]))
            if variant == "replan"
            else Actor(env.num_actions, int(config["HIDDEN_SIZE"]))
        )
        return cls(actor, variant, tuple(env.agents), env.num_actions)

    def initial_state(self, env):
        """Return the stateless macro actor's empty policy state."""

        del env
        return None

    def act(self, params, obs, policy_state, dones, rng):
        """Apply the same valid-action rules used by the matching trainer."""

        del dones, rng
        actor_obs = jnp.stack([obs[agent] for agent in self.agents])
        action_mask = obs["action_mask"].astype(jnp.bool_)

        if self.variant != "replan":
            logits = self.actor.apply(params, actor_obs)
            distribution = distrax.Categorical(
                logits=jnp.where(action_mask, logits, -1e9)
            )
            if self.variant == "every_step":
                return distribution, policy_state
            proposed_actions = distribution.mode()
            actions = jnp.where(
                obs["macro_done"],
                proposed_actions,
                obs["current_macro"],
            )
            return actions, policy_state

        macro_done = obs["macro_done"]
        current_macro = obs["current_macro"]
        macro_logits, replan_logits = self.actor.apply(params, actor_obs)
        replacement_mask = action_mask & ~(
            (~macro_done)[:, None]
            & (
                jnp.arange(self.num_actions)[None, :]
                == current_macro[:, None]
            )
        )
        macro_action = distrax.Categorical(
            logits=jnp.where(replacement_mask, macro_logits, -1e9)
        ).mode()
        replan_action = distrax.Categorical(logits=replan_logits).mode()
        replace = macro_done | (replan_action == REPLAN)
        return jnp.where(replace, macro_action, current_macro), policy_state
