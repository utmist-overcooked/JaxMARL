"""Functional reset-time sampling and randomization helpers for Overcooked V3."""

import chex
import jax
import jax.numpy as jnp

from jaxmarl.environments.overcooked_v3.common import (
    Direction,
    DynamicObject,
    Position,
)
from jaxmarl.environments.overcooked_v3.config import OvercookedV3Config
from jaxmarl.environments.overcooked_v3.state import State

def sample_recipe(key: chex.PRNGKey, config: OvercookedV3Config) -> int:
    """Sample a recipe from the configured possible recipes."""
    if config.recipe_probs is None:
        recipe_idx = jax.random.randint(key, (), 0, config.possible_recipes.shape[0])
    else:
        recipe_idx = jax.random.choice(
            key, config.possible_recipes.shape[0], p=config.recipe_probs
        )
    recipe = config.possible_recipes[recipe_idx]
    return DynamicObject.get_recipe_encoding(recipe)

def randomize_agent_positions(
    state: State, key: chex.PRNGKey, config: OvercookedV3Config
) -> State:
    """Randomize agent positions within their connected rooms."""
    agents = state.agents

    def _select_agent_position(taken_mask, x):
        pos, key = x

        allowed_positions = (
            config.enclosed_spaces == config.enclosed_spaces[pos.y, pos.x]
        ) & ~taken_mask
        allowed_positions = allowed_positions.flatten()

        p = allowed_positions / jnp.sum(allowed_positions)
        agent_pos_idx = jax.random.choice(key, allowed_positions.size, (), p=p)
        agent_position = Position(
            x=agent_pos_idx % config.width, y=agent_pos_idx // config.width
        )

        new_taken_mask = taken_mask.at[agent_position.y, agent_position.x].set(True)
        return new_taken_mask, agent_position

    taken_mask = jnp.zeros_like(config.enclosed_spaces, dtype=jnp.bool_)
    key, subkey = jax.random.split(key)
    keys = jax.random.split(subkey, config.num_agents)
    _, agent_positions = jax.lax.scan(
        _select_agent_position, taken_mask, (agents.pos, keys)
    )

    key, subkey = jax.random.split(key)
    directions = jax.random.randint(subkey, (config.num_agents,), 0, len(Direction))

    return state.replace(agents=agents.replace(pos=agent_positions, dir=directions))

def randomize_state(
    state: State, key: chex.PRNGKey, config: OvercookedV3Config
) -> State:
    """Randomize all currently supported reset-time state components."""
    key, subkey = jax.random.split(key)
    return randomize_agent_positions(state, subkey, config)
