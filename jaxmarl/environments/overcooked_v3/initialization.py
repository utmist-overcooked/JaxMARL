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


def select_recipe_type(
    key: chex.PRNGKey,
    next_recipe_idx: chex.Array,
    config: OvercookedV3Config,
) -> tuple[chex.Array, chex.Array]:
    """Select a one-based recipe type and advance alternating-mode state."""
    num_recipes = config.possible_recipes.shape[0]

    if config.recipe_mode == "fixed":
        recipe_idx = jnp.array(0, dtype=jnp.int32)
        new_next_recipe_idx = next_recipe_idx
    elif config.recipe_mode == "random":
        recipe_idx = jax.random.choice(
            key,
            num_recipes,
            p=config.recipe_probs,
        )
        new_next_recipe_idx = next_recipe_idx
    else:
        recipe_idx = next_recipe_idx
        new_next_recipe_idx = (next_recipe_idx + 1) % num_recipes

    # Queue slot zero is reserved for "no order", so recipe types are one-based.
    return recipe_idx.astype(jnp.int32) + 1, new_next_recipe_idx.astype(jnp.int32)


def sample_recipe(key: chex.PRNGKey, config: OvercookedV3Config) -> int:
    """Select the first recipe for a fresh fixed, random, or alternating stream."""
    recipe_type, _ = select_recipe_type(
        key,
        jnp.array(0, dtype=jnp.int32),
        config,
    )
    recipe = config.possible_recipes[recipe_type - 1]
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
