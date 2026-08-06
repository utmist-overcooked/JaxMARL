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
) -> Tuple[chex.Array, chex.Array, chex.Array]:
    """Update pot cooking timers.

    Timer semantics:
    - When the final ingredient is placed, the timer starts at
      cook_duration + pot_burn_time.
    - The pot becomes COOKED when the timer reaches pot_burn_time.
    - If pot_burn_time > 0, cooked soup burns/clears when the timer hits 0.
    - If pot_burn_time == 0, cooked soup remains ready indefinitely.

    Returns the updated grid, the updated timers, and the number of pots that
    burned on this step.
    """

    def _update_single_pot(carry, pot_idx):
        grid, timers, burn_count = carry
        pot_y, pot_x = pot_positions[pot_idx]
        is_active = pot_active_mask[pot_idx]
        current_timer = timers[pot_idx]

        pot_cell = grid[pot_y, pot_x]
        pot_ingredients = pot_cell[1]

        # Check if pot is full (has 3 ingredients)
        ingredient_count = DynamicObject.ingredient_count(pot_ingredients)
        pot_is_full = ingredient_count == 3
        pot_already_cooked = (pot_ingredients & DynamicObject.COOKED) != 0
        pot_has_contents = pot_is_full | pot_already_cooked
        timer_is_active = (current_timer > 0) & pot_has_contents

        new_timer = jax.lax.select(
            is_active & timer_is_active, current_timer - 1, current_timer
        )

        burn_enabled = config.pot_burn_time > 0
        cooked_threshold_reached = (
            new_timer <= config.pot_burn_time if burn_enabled else new_timer == 0
        )
        just_finished_cooking = (
            is_active
            & pot_is_full
            & ~pot_already_cooked
            & cooked_threshold_reached
        )
        cooked_ingredients = jax.lax.select(
            just_finished_cooking,
            pot_ingredients | DynamicObject.COOKED,
            pot_ingredients,
        )
        pot_is_cooked_after_update = (cooked_ingredients & DynamicObject.COOKED) != 0
        just_burned = (
            is_active
            & burn_enabled
            & pot_is_cooked_after_update
            & timer_is_active
            & (new_timer == 0)
        )
        new_ingredients = jax.lax.select(just_burned, 0, cooked_ingredients)
        new_timer = jax.lax.select(just_burned, 0, new_timer)

        # Update grid
        new_cell = jnp.array([pot_cell[0], new_ingredients, pot_cell[2]])
        new_grid = grid.at[pot_y, pot_x].set(new_cell)

        # Update timers
        new_timers = timers.at[pot_idx].set(new_timer)
        new_burn_count = burn_count + just_burned.astype(jnp.float32)

        return (new_grid, new_timers, new_burn_count), None

    (new_grid, new_timers, burn_count), _ = jax.lax.scan(
        _update_single_pot,
        (grid, pot_timers, jnp.array(0.0, dtype=jnp.float32)),
        jnp.arange(MAX_POTS),
    )

    return new_grid, new_timers, burn_count
