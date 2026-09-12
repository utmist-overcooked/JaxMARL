"""MAPPO recurrent actor/critic networks and rollout policy adapter."""

from __future__ import annotations

import functools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from baselines.overcooked_v3 import _jax_compat  # noqa: F401

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.linen.initializers import constant, orthogonal


class ScannedRNN(nn.Module):
    """GRU cell scanned across the leading time dimension."""

    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        """Reset finished actors and advance the GRU one timestep."""

        rnn_state = carry
        inputs, resets = x
        rnn_state = jnp.where(
            resets[:, np.newaxis],
            self.initialize_carry(*rnn_state.shape),
            rnn_state,
        )
        new_rnn_state, output = nn.GRUCell(features=inputs.shape[1])(
            rnn_state,
            inputs,
        )
        return new_rnn_state, output

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        """Return a deterministic zero-equivalent GRU carry."""

        cell = nn.GRUCell(features=hidden_size)
        return cell.initialize_carry(
            jax.random.PRNGKey(0),
            (batch_size, hidden_size),
        )


class ActorRNN(nn.Module):
    """Recurrent categorical actor used by primitive-action MAPPO."""

    action_dim: int
    config: Mapping

    @nn.compact
    def __call__(self, hidden, x):
        """Return the next hidden state and categorical action distribution."""

        obs, dones = x
        embedding = nn.Dense(
            self.config["FC_DIM_SIZE"],
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(obs)
        embedding = nn.relu(embedding)
        hidden, embedding = ScannedRNN()(hidden, (embedding, dones))
        actor_mean = nn.Dense(
            self.config["GRU_HIDDEN_DIM"],
            kernel_init=orthogonal(2),
            bias_init=constant(0.0),
        )(embedding)
        actor_mean = nn.relu(actor_mean)
        action_logits = nn.Dense(
            self.action_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
        )(actor_mean)
        return hidden, distrax.Categorical(logits=action_logits)


class CriticRNN(nn.Module):
    """Recurrent centralized critic used by primitive-action MAPPO."""

    config: Mapping

    @nn.compact
    def __call__(self, hidden, x):
        """Return the next hidden state and value estimate."""

        world_state, dones = x
        embedding = nn.Dense(
            self.config["FC_DIM_SIZE"],
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(world_state)
        embedding = nn.relu(embedding)
        hidden, embedding = ScannedRNN()(hidden, (embedding, dones))
        critic = nn.Dense(
            self.config["GRU_HIDDEN_DIM"],
            kernel_init=orthogonal(2),
            bias_init=constant(0.0),
        )(embedding)
        critic = nn.relu(critic)
        critic = nn.Dense(
            1,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
        )(critic)
        return hidden, jnp.squeeze(critic, axis=-1)


def batchify(x: Mapping, agent_list: Sequence[str], num_actors: int):
    """Flatten agent-keyed observations into an actor-major batch."""

    values = jnp.stack([x[agent] for agent in agent_list])
    return values.reshape((num_actors, -1))


def unbatchify(x, agent_list: Sequence[str], num_envs: int, num_actors: int):
    """Convert an actor-major action batch to an agent-keyed dictionary."""

    values = x.reshape((num_actors, num_envs) + x.shape[1:])
    if values.ndim == 3 and values.shape[-1] == 1:
        values = jnp.squeeze(values, axis=-1)
    return {agent: values[index] for index, agent in enumerate(agent_list)}


@dataclass(frozen=True)
class MAPPORNNPolicy:
    """Adapt ``ActorRNN`` to the shared environment-facing policy contract."""

    actor: ActorRNN
    config: Mapping
    agent_list: tuple[str, ...]

    @classmethod
    def create(cls, env, config: Mapping) -> "MAPPORNNPolicy":
        """Build a rollout adapter matching the environment action space."""

        action_dim = env.action_space(env.agents[0]).n
        return cls(ActorRNN(action_dim, config=config), config, tuple(env.agents))

    def initial_state(self, env):
        """Return one recurrent state per environment agent."""

        return ScannedRNN.initialize_carry(
            env.num_agents,
            self.config["GRU_HIDDEN_DIM"],
        )

    def act(self, params, obs, policy_state, dones, rng):
        """Return the actor distribution and next recurrent state."""

        del rng
        obs_batch = batchify(obs, self.agent_list, len(self.agent_list))
        done_batch = jnp.stack(
            [jnp.asarray(dones[agent]) for agent in self.agent_list]
        )
        actor_input = (
            obs_batch[np.newaxis, :],
            done_batch[np.newaxis, :],
        )
        next_state, distribution = self.actor.apply(
            params,
            policy_state,
            actor_input,
        )
        return distribution, next_state
