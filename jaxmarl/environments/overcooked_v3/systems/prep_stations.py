"""Prep station timers for Overcooked V3 (grill cooking/burning, blending)."""

from typing import Tuple

import chex
import jax.numpy as jnp

from jaxmarl.environments.overcooked_v3.common import (
    DynamicObject,
    StaticObject,
    PREP_RAW_START,
    PREP_PROCESSED_SHIFT,
)
from jaxmarl.environments.overcooked_v3.config import OvercookedV3Config


def update_prep_stations(
    grid: chex.Array,
    config: OvercookedV3Config,
) -> Tuple[chex.Array, chex.Array]:
    """Advance grill and blender timers stored in the grid's extra channel.

    Grill semantics mirror the pot: placement sets the timer to
    grill_cook_time + grill_burn_time, the item is done once the timer
    reaches grill_burn_time, and it burns (contents destroyed) when the
    timer hits 0 - unless grill_burn_time is 0, in which case the item is
    done at 0 and never burns. Blenders count down from blend_time after a
    manual start and convert at 0; they never burn. Cutting boards are
    purely interact-driven and need no per-step update.
    """
    static = grid[:, :, 0]
    items = grid[:, :, 1]
    extra = grid[:, :, 2]

    is_grill = static == StaticObject.GRILL
    is_blender = static == StaticObject.BLENDER
    timer_running = extra > 0
    ticking = (is_grill | is_blender) & timer_running
    new_extra = jnp.where(ticking, extra - 1, extra)

    raw_meat = DynamicObject.ingredient(PREP_RAW_START + 1)
    grilled_meat = raw_meat << PREP_PROCESSED_SHIFT
    burn_enabled = config.grill_burn_time > 0
    cooked_threshold = config.grill_burn_time if burn_enabled else 0
    grill_done = (
        is_grill
        & timer_running
        & (items == raw_meat)
        & (new_extra <= cooked_threshold)
    )
    new_items = jnp.where(grill_done, grilled_meat, items)

    just_burned = (
        is_grill
        & burn_enabled
        & timer_running
        & (new_items == grilled_meat)
        & (new_extra == 0)
    )
    new_items = jnp.where(just_burned, 0, new_items)
    burn_count = jnp.sum(just_burned).astype(jnp.float32)

    raw_carrot = DynamicObject.ingredient(PREP_RAW_START + 2)
    blend_done = (
        is_blender & timer_running & (items == raw_carrot) & (new_extra == 0)
    )
    new_items = jnp.where(
        blend_done, raw_carrot << PREP_PROCESSED_SHIFT, new_items
    )

    new_grid = jnp.stack([static, new_items, new_extra], axis=-1)
    return new_grid, burn_count
