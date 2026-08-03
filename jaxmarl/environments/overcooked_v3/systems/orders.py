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

    # Decrement expirations
    new_expirations = jnp.where(
        order_active_mask, order_expirations - 1, order_expirations
    )

    # Check for expired orders
    expired_mask = order_active_mask & (new_expirations <= 0)
    num_expired = jnp.sum(expired_mask)
    reward = num_expired * ORDER_EXPIRED_PENALTY

    # Deactivate expired orders
    new_active_mask = order_active_mask & ~expired_mask

    # Maybe generate new order
    key, subkey = jax.random.split(key)
    should_generate = jax.random.uniform(subkey) < config.order_generation_rate

    # Find first empty slot
    empty_slots = ~new_active_mask
    first_empty_idx = jnp.argmax(empty_slots)
    has_empty_slot = jnp.any(empty_slots)

    # Generate random order type (1 = onion soup, 2 = tomato soup if num_ingredients > 1)
    key, subkey = jax.random.split(key)
    new_order_type = jax.random.randint(
        subkey, (), 1, min(config.layout.num_ingredients + 1, 3)
    )

    should_add = should_generate & has_empty_slot
    new_order_types = jax.lax.select(
        should_add, order_types.at[first_empty_idx].set(new_order_type), order_types
    )
    new_expirations = jax.lax.select(
        should_add,
        new_expirations.at[first_empty_idx].set(config.order_expiration_time),
        new_expirations,
    )
    new_active_mask = jax.lax.select(
        should_add, new_active_mask.at[first_empty_idx].set(True), new_active_mask
    )

    return state.replace(
        order_types=new_order_types,
        order_expirations=new_expirations,
        order_active_mask=new_active_mask,
    ), reward
