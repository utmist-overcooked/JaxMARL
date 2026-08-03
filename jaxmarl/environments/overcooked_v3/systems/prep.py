"""Timed multistage preparation systems."""

from typing import Tuple

import chex
import jax.numpy as jnp

from jaxmarl.environments.overcooked_v3.common import (
    PREP_PROCESSED_SHIFT,
    PREP_RAW_START,
    DynamicObject,
    StaticObject,
)
from jaxmarl.environments.overcooked_v3.config import OvercookedV3Config


def update_prep_stations(
    grid: chex.Array, config: OvercookedV3Config
) -> Tuple[chex.Array, chex.Array]:
    static, items, extra = grid[:, :, 0], grid[:, :, 1], grid[:, :, 2]
    is_grill = static == StaticObject.GRILL
    is_blender = static == StaticObject.BLENDER
    running = extra > 0
    new_extra = jnp.where((is_grill | is_blender) & running, extra - 1, extra)

    raw_meat = DynamicObject.ingredient(PREP_RAW_START + 1)
    grilled_meat = raw_meat << PREP_PROCESSED_SHIFT
    burn_enabled = config.grill_burn_time > 0
    threshold = config.grill_burn_time if burn_enabled else 0
    grill_done = (
        is_grill & running & (items == raw_meat) & (new_extra <= threshold)
    )
    new_items = jnp.where(grill_done, grilled_meat, items)
    just_burned = (
        is_grill
        & burn_enabled
        & running
        & (new_items == grilled_meat)
        & (new_extra == 0)
    )
    new_items = jnp.where(just_burned, 0, new_items)

    raw_carrot = DynamicObject.ingredient(PREP_RAW_START + 2)
    blend_done = is_blender & running & (items == raw_carrot) & (new_extra == 0)
    new_items = jnp.where(blend_done, raw_carrot << PREP_PROCESSED_SHIFT, new_items)
    return (
        jnp.stack([static, new_items, new_extra], axis=-1),
        jnp.sum(just_burned).astype(jnp.float32),
    )
