"""Order queue system for Overcooked V3."""

from typing import Dict, Tuple

import chex
import jax
import jax.numpy as jnp

from jaxmarl.environments.overcooked_v3.common import SoupType
from jaxmarl.environments.overcooked_v3.config import OvercookedV3Config
from jaxmarl.environments.overcooked_v3.interactions import (
    compute_order_expired_penalty,
    zero_reward_breakdown,
)
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
) -> Tuple[State, float, Dict[str, chex.Array], chex.Array]:
    """Process order queue: generate new orders, check expirations.

    Returns the updated state, the queue reward, the REWARD_COMPONENT_KEYS
    breakdown of that reward (consumed by the macro/comm trainers), and the
    (num_expired, order_added) event pair surfaced in step_env's info dict.
    """
    if not config.enable_order_queue:
        return (
            state,
            0.0,
            zero_reward_breakdown(),
            jnp.zeros((2,), dtype=jnp.float32),
        )

    order_types = state.order_types
    order_expirations = state.order_expirations
    order_active_mask = state.order_active_mask

    # A successful delivery fulfills the oldest active order. The queue is
    # compacted afterward so the recipe indicator always points at the
    # current front order.
    front_idx = jnp.argmax(order_active_mask)
    has_front_order = jnp.any(order_active_mask)
    should_clear_front = state.new_correct_delivery & has_front_order
    order_types = jax.lax.select(
        should_clear_front, order_types.at[front_idx].set(0), order_types
    )
    order_expirations = jax.lax.select(
        should_clear_front,
        order_expirations.at[front_idx].set(0),
        order_expirations,
    )
    order_active_mask = jax.lax.select(
        should_clear_front,
        order_active_mask.at[front_idx].set(False),
        order_active_mask,
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
    # Orders only expire when expiration is enabled; otherwise they sit in the
    # queue until delivered.
    expired_mask = (
        order_expiration_enabled & order_active_mask & (new_expirations <= 0)
    )
    num_expired = jnp.sum(expired_mask)
    reward, reward_breakdown = compute_order_expired_penalty(expired_mask)

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

    # Generate either a random order or a deterministic rotation through every
    # orderable dish. For the alternating mode, step on from the current newest
    # visible order so the queue keeps its cycle even after deliveries or
    # expirations compact the front.
    key, subkey = jax.random.split(key)
    if config.order_queue_mode == "alternating":
        num_active_orders = jnp.sum(new_active_mask)
        newest_order_idx = jnp.maximum(num_active_orders - 1, 0)
        newest_order_type = new_order_types[newest_order_idx]
        has_active_order = num_active_orders > 0
        # Step to the next dish in the rotation, wrapping back to the first.
        next_after_newest = (newest_order_type % config.num_order_types) + 1
        new_order_type = jnp.where(
            has_active_order,
            next_after_newest,
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

    # Keep the recipe indicator pinned to the current front order.
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
        ),
        reward,
        reward_breakdown,
        order_events,
    )
