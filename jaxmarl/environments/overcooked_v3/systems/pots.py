"""Pot cooking and burning systems for Overcooked V3."""

from typing import Tuple

import chex
import jax
import jax.numpy as jnp

from jaxmarl.environments.overcooked_v3.common import DynamicObject
from jaxmarl.environments.overcooked_v3.config import OvercookedV3Config
from jaxmarl.environments.overcooked_v3.settings import MAX_POTS

def update_pot_timers(
    grid: chex.Array,
    pot_timers: chex.Array,
    pot_positions: chex.Array,
    pot_active_mask: chex.Array,
    config: OvercookedV3Config,
) -> Tuple[chex.Array, chex.Array]:
    """Update pot cooking timers and handle burning."""

    def _update_single_pot(carry, pot_idx):
        grid, timers = carry
        pot_y, pot_x = pot_positions[pot_idx]
        is_active = pot_active_mask[pot_idx]
        current_timer = timers[pot_idx]

        pot_cell = grid[pot_y, pot_x]
        pot_ingredients = pot_cell[1]

        # Check if pot is full (has 3 ingredients)
        ingredient_count = DynamicObject.ingredient_count(pot_ingredients)
        pot_is_full = ingredient_count == 3
        pot_is_cooking = (current_timer > 0) & pot_is_full

        # Decrement timer if cooking
        new_timer = jax.lax.select(
            is_active & pot_is_cooking, current_timer - 1, current_timer
        )

        # Check if just finished cooking (entered burning window)
        just_finished_cooking = pot_is_cooking & (new_timer == config.pot_burn_time)
        # Mark as cooked when timer reaches burn_time
        new_ingredients = jax.lax.select(
            is_active & just_finished_cooking,
            pot_ingredients | DynamicObject.COOKED,
            pot_ingredients,
        )

        # Check if pot burned (timer hit 0 while cooking)
        just_burned = pot_is_cooking & (new_timer == 0)
        # Reset pot if burned
        new_ingredients = jax.lax.select(
            is_active & just_burned,
            jnp.int32(0),  # Clear pot
            new_ingredients,
        )
        new_timer = jax.lax.select(is_active & just_burned, jnp.int32(0), new_timer)

        # Update grid
        new_cell = jnp.array([pot_cell[0], new_ingredients, pot_cell[2]])
        new_grid = grid.at[pot_y, pot_x].set(new_cell)

        # Update timers
        new_timers = timers.at[pot_idx].set(new_timer)

        return (new_grid, new_timers), None

    (new_grid, new_timers), _ = jax.lax.scan(
        _update_single_pot, (grid, pot_timers), jnp.arange(MAX_POTS)
    )

    return new_grid, new_timers
