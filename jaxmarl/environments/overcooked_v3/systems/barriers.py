"""Barrier and pressure plate systems for Overcooked V3."""

import jax
import jax.numpy as jnp

from jaxmarl.environments.overcooked_v3.agent_step import barriers_occupied
from jaxmarl.environments.overcooked_v3.common import ButtonAction
from jaxmarl.environments.overcooked_v3.config import OvercookedV3Config
from jaxmarl.environments.overcooked_v3.settings import MAX_BARRIERS
from jaxmarl.environments.overcooked_v3.state import State

def update_barrier_timers(state: State, config: OvercookedV3Config) -> State:
    """Decrement barrier timers and reactivate barriers when timers reach zero."""

    # A barrier is never reactivated on top of an agent; the timer is held
    # until the tile clears so the agent can't be trapped (see
    # _barriers_occupied).
    barrier_occupied = barriers_occupied(
        state.agents.pos.y,
        state.agents.pos.x,
        state.barrier_positions,
        state.barrier_active_mask,
    )
    held_open_by_plate = jnp.any(
        state.pressure_plate_linked_barrier
        & state.pressure_plate_active_mask[:, None]
        & state.pressure_plate_toggled[:, None]
        & (
            state.pressure_plate_action_type
            == ButtonAction.TOGGLE_BARRIER
        )[:, None],
        axis=0,
    )

    def _update_single_barrier(i):
        is_active_slot = state.barrier_active_mask[i]
        timer = state.barrier_timer[i]
        barrier_active = state.barrier_active[i]
        occupied = barrier_occupied[i]
        plate_open = held_open_by_plate[i]

        # The barrier would reactivate when its timer ticks from 1 to 0, but
        # if an agent is on the tile we hold the timer at 1 and defer.
        would_reactivate = timer == 1
        hold = would_reactivate & occupied

        # Timer only decrements when > 0 (and not held by an occupant)
        has_timer = timer > 0
        new_timer = jax.lax.select(has_timer & ~hold, timer - 1, timer)

        should_reactivate = (
            is_active_slot & would_reactivate & ~occupied & ~plate_open
        )
        new_active = jax.lax.select(should_reactivate, True, barrier_active)

        return new_timer, new_active

    # Process all barriers
    new_timers, new_active_states = jax.vmap(_update_single_barrier)(
        jnp.arange(MAX_BARRIERS)
    )

    return state.replace(
        barrier_timer=new_timers,
        barrier_active=new_active_states,
    )

def update_pressure_plates(state: State, config: OvercookedV3Config) -> State:
    """Open or re-arm barriers controlled by currently pressed pressure plates."""
    if not config.enable_pressure_plates:
        return state

    agent_xs = state.agents.pos.x  # [num_agents]
    agent_ys = state.agents.pos.y  # [num_agents]

    # Which plates currently have an agent standing on them? [num_plates]
    plate_py = state.pressure_plate_positions[:, 0]
    plate_px = state.pressure_plate_positions[:, 1]
    agent_on_plate = jnp.any(
        (agent_xs[None, :] == plate_px[:, None])
        & (agent_ys[None, :] == plate_py[:, None]),
        axis=1,
    )
    pressed = agent_on_plate & state.pressure_plate_active_mask  # [num_plates]

    action = state.pressure_plate_action_type  # [num_plates]
    is_toggle = action == ButtonAction.TOGGLE_BARRIER
    is_timed = action == ButtonAction.TIMED_BARRIER

    # Valid (plate, barrier) links. [num_plates, num_barriers]
    linked = state.pressure_plate_linked_barrier & state.barrier_active_mask[None, :]

    # A barrier is never re-closed on top of an agent; it stays open until
    # the tile is clear, so an agent crossing as the plate releases can't be
    # trapped (see _barriers_occupied).
    barrier_occupied = barriers_occupied(
        agent_ys, agent_xs, state.barrier_positions, state.barrier_active_mask
    )

    # TOGGLE_BARRIER: a linked barrier is open while one of its plates is
    # pressed, a timed control is still armed, or an agent stands on it. It
    # closes once all three conditions clear.
    toggle_links = linked & is_toggle[:, None]
    toggle_controlled = jnp.any(toggle_links, axis=0)  # [num_barriers]
    toggle_open = jnp.any(toggle_links & pressed[:, None], axis=0)  # [num_barriers]
    timer_open = state.barrier_timer > 0
    new_barrier_active = jnp.where(
        toggle_controlled,
        ~toggle_open & ~timer_open & ~barrier_occupied,
        state.barrier_active,
    )

    # TIMED_BARRIER: pressing opens the barrier and (re)arms its timer; the
    # barrier reactivates on its own in _process_barrier_timers.
    timed_open = jnp.any(linked & is_timed[:, None] & pressed[:, None], axis=0)
    new_barrier_active = jnp.where(timed_open, False, new_barrier_active)
    new_barrier_timer = jnp.where(
        timed_open, state.barrier_duration, state.barrier_timer
    )

    return state.replace(
        barrier_active=new_barrier_active,
        barrier_timer=new_barrier_timer,
        pressure_plate_toggled=pressed,
    )
