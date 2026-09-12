"""Order queue system for Overcooked V3."""

from typing import Tuple

import chex
import jax
import jax.numpy as jnp

from jaxmarl.environments.overcooked_v3.common import SoupType
from jaxmarl.environments.overcooked_v3.config import OvercookedV3Config
from jaxmarl.environments.overcooked_v3.initialization import select_recipe_type
from jaxmarl.environments.overcooked_v3.settings import ORDER_EXPIRED_PENALTY
from jaxmarl.environments.overcooked_v3.state import State


def order_type_to_recipe(
    order_type: chex.Array, config: OvercookedV3Config
) -> chex.Array:
    """Map order type IDs to the recipe encoding shown to the agents."""
    encodings = config.order_recipe_encodings
    safe_order_type = jnp.clip(order_type, 0, encodings.shape[0] - 1)
    return encodings[safe_order_type]


def front_order_type(
    order_types: chex.Array, order_active_mask: chex.Array
) -> chex.Array:
    """Return the oldest active order type, or NONE if the queue is empty."""
    first_active_idx = jnp.argmax(order_active_mask)
    has_active_order = jnp.any(order_active_mask)
    return jnp.where(
        has_active_order,
        order_types[first_active_idx],
        jnp.array(SoupType.NONE, dtype=jnp.int32),
    )


def compact_order_queue(
    order_types: chex.Array,
    order_expirations: chex.Array,
    order_active_mask: chex.Array,
    config: OvercookedV3Config,
) -> Tuple[chex.Array, chex.Array, chex.Array]:
    """Pack active orders toward the front while preserving slot order."""
    max_orders = config.max_orders
    slot_order = jnp.where(
        order_active_mask,
        jnp.arange(max_orders),
        max_orders + jnp.arange(max_orders),
    )
    compact_indices = jnp.argsort(slot_order)
    num_active = jnp.sum(order_active_mask)
    compact_active_mask = jnp.arange(max_orders) < num_active
    compact_types = jnp.where(compact_active_mask, order_types[compact_indices], 0)
    compact_expirations = jnp.where(
        compact_active_mask, order_expirations[compact_indices], 0
    )
    return compact_types, compact_expirations, compact_active_mask


def process_order_queue(
    state: State, key: chex.PRNGKey, config: OvercookedV3Config
) -> Tuple[State, float, chex.Array]:
    """Process order queue: generate new orders, check expirations."""
    if not config.enable_order_queue:
        return state, 0.0, jnp.zeros((2,), dtype=jnp.float32)

    order_types = state.order_types
    order_expirations = state.order_expirations
    order_active_mask = state.order_active_mask

    # Each correct delivery consumes the oldest active slot with that recipe.
    # Scanning per-agent delivery types also handles multiple valid deliveries
    # during one environment step without clearing unrelated duplicate orders.
    def _fulfill_oldest_matching_order(carry, delivered_recipe_type):
        """Remove the first active slot matching one delivered recipe type."""
        types, expirations, active_mask = carry
        matching_slots = (
            active_mask
            & (types == delivered_recipe_type)
            & (delivered_recipe_type > 0)
        )
        matching_slot_idx = jnp.argmax(matching_slots)
        should_fulfill = jnp.any(matching_slots)

        types = jax.lax.select(
            should_fulfill,
            types.at[matching_slot_idx].set(0),
            types,
        )
        expirations = jax.lax.select(
            should_fulfill,
            expirations.at[matching_slot_idx].set(0),
            expirations,
        )
        active_mask = jax.lax.select(
            should_fulfill,
            active_mask.at[matching_slot_idx].set(False),
            active_mask,
        )
        return (types, expirations, active_mask), None

    (order_types, order_expirations, order_active_mask), _ = jax.lax.scan(
        _fulfill_oldest_matching_order,
        (order_types, order_expirations, order_active_mask),
        state.new_correct_delivery_types,
    )

    order_expiration_enabled = config.order_expiration_time > 0

    # Decrement expirations. An expiration time <= 0 means orders stay in
    # the queue until delivered; this lets us train the queue/recipe logic
    # without an unavoidable dense penalty before deliveries are learned.
    new_expirations = jnp.where(
        order_expiration_enabled & order_active_mask,
        order_expirations - 1,
        order_expirations,
    )

    # Check for expired orders
    expired_mask = (
        order_expiration_enabled & order_active_mask & (new_expirations <= 0)
    )
    num_expired = jnp.sum(expired_mask)
    reward = num_expired * ORDER_EXPIRED_PENALTY

    # Deactivate expired orders
    new_active_mask = order_active_mask & ~expired_mask
    new_order_types = jnp.where(new_active_mask, order_types, 0)
    new_expirations = jnp.where(new_active_mask, new_expirations, 0)
    new_order_types, new_expirations, new_active_mask = compact_order_queue(
        new_order_types, new_expirations, new_active_mask, config
    )

    # Maybe generate new order
    key, subkey = jax.random.split(key)
    should_generate = jax.random.uniform(subkey) < config.order_generation_rate

    # Find first empty slot
    empty_slots = ~new_active_mask
    first_empty_idx = jnp.argmax(empty_slots)
    has_empty_slot = jnp.any(empty_slots)

    # Generate from the same fixed/random/alternating recipe stream used by
    # queue-off environments. Alternating state advances only when an order is
    # actually inserted, so failed generation attempts do not skip recipes.
    key, subkey = jax.random.split(key)
    new_order_type, candidate_next_recipe_idx = select_recipe_type(
        subkey,
        state.next_recipe_idx,
        config,
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
    new_next_recipe_idx = jnp.where(
        should_add,
        candidate_next_recipe_idx,
        state.next_recipe_idx,
    )

    current_front_order = front_order_type(new_order_types, new_active_mask)
    new_recipe = jnp.where(
        current_front_order != SoupType.NONE,
        order_type_to_recipe(current_front_order, config),
        state.recipe,
    )
    order_events = jnp.array(
        [num_expired.astype(jnp.float32), should_add.astype(jnp.float32)],
        dtype=jnp.float32,
    )

    return (
        state.replace(
            order_types=new_order_types,
            order_expirations=new_expirations,
            order_active_mask=new_active_mask,
            recipe=new_recipe,
            next_recipe_idx=new_next_recipe_idx,
        ),
        reward,
        order_events,
    )
