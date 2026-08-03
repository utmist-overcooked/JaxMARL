"""Order queue system for Overcooked V3."""

from typing import Tuple

import chex
import jax
import jax.numpy as jnp

from jaxmarl.environments.overcooked_v3.config import OvercookedV3Config
from jaxmarl.environments.overcooked_v3.settings import ORDER_EXPIRED_PENALTY
from jaxmarl.environments.overcooked_v3.state import State

def process_order_queue(
    state: State, key: chex.PRNGKey, config: OvercookedV3Config
) -> Tuple[State, float]:
    """Process order queue: generate new orders, check expirations."""
    if not config.enable_order_queue:
        return state, 0.0

    order_types = state.order_types
    order_expirations = state.order_expirations
    order_active_mask = state.order_active_mask

    front_idx = jnp.argmax(order_active_mask)
    clear_front = state.new_correct_delivery & jnp.any(order_active_mask)
    order_types = jax.lax.select(
        clear_front, order_types.at[front_idx].set(0), order_types
    )
    order_expirations = jax.lax.select(
        clear_front, order_expirations.at[front_idx].set(0), order_expirations
    )
    order_active_mask = jax.lax.select(
        clear_front, order_active_mask.at[front_idx].set(False), order_active_mask
    )

    expiration_enabled = config.order_expiration_time > 0
    new_expirations = jnp.where(
        expiration_enabled & order_active_mask,
        order_expirations - 1,
        order_expirations,
    )

    # Check for expired orders
    expired_mask = expiration_enabled & order_active_mask & (new_expirations <= 0)
    num_expired = jnp.sum(expired_mask)
    reward = num_expired * ORDER_EXPIRED_PENALTY

    # Deactivate expired orders
    new_active_mask = order_active_mask & ~expired_mask
    new_order_types = jnp.where(new_active_mask, order_types, 0)
    new_expirations = jnp.where(new_active_mask, new_expirations, 0)

    slot_order = jnp.where(
        new_active_mask,
        jnp.arange(config.max_orders),
        config.max_orders + jnp.arange(config.max_orders),
    )
    compact_indices = jnp.argsort(slot_order)
    num_active = jnp.sum(new_active_mask)
    new_active_mask = jnp.arange(config.max_orders) < num_active
    new_order_types = jnp.where(
        new_active_mask, new_order_types[compact_indices], 0
    )
    new_expirations = jnp.where(
        new_active_mask, new_expirations[compact_indices], 0
    )

    # Maybe generate new order
    key, subkey = jax.random.split(key)
    should_generate = jax.random.uniform(subkey) < config.order_generation_rate

    # Find first empty slot
    empty_slots = ~new_active_mask
    first_empty_idx = jnp.argmax(empty_slots)
    has_empty_slot = jnp.any(empty_slots)

    key, subkey = jax.random.split(key)
    if config.order_queue_mode == "alternating":
        newest_idx = jnp.maximum(jnp.sum(new_active_mask) - 1, 0)
        newest_type = new_order_types[newest_idx]
        new_order_type = jnp.where(
            jnp.any(new_active_mask),
            (newest_type % config.num_order_types) + 1,
            jnp.array(1, dtype=jnp.int32),
        )
    else:
        new_order_type = jax.random.randint(
            subkey, (), 1, config.num_order_types + 1
        )

    should_add = should_generate & has_empty_slot
    new_order_types = jax.lax.select(
        should_add,
        new_order_types.at[first_empty_idx].set(new_order_type),
        new_order_types,
    )
    new_expirations = jax.lax.select(
        should_add,
        new_expirations.at[first_empty_idx].set(config.order_expiration_time),
        new_expirations,
    )
    new_active_mask = jax.lax.select(
        should_add, new_active_mask.at[first_empty_idx].set(True), new_active_mask
    )

    front_idx = jnp.argmax(new_active_mask)
    front_type = jnp.where(jnp.any(new_active_mask), new_order_types[front_idx], 0)
    new_recipe = jnp.where(
        front_type != 0,
        config.order_recipe_encodings[jnp.clip(front_type, 0, config.num_order_types)],
        state.recipe,
    )
    return state.replace(
        order_types=new_order_types,
        order_expirations=new_expirations,
        order_active_mask=new_active_mask,
        recipe=new_recipe,
    ), reward
