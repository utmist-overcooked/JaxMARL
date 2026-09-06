"""Tests for macro-action Overcooked V3."""

import jax
import jax.numpy as jnp

from jaxmarl import make
from jaxmarl.environments.overcooked_v3.common import Actions, DynamicObject
from jaxmarl.environments.overcooked_v3.layouts import Layout
from jaxmarl.environments.overcooked_v3.overcooked import OvercookedV3
from jaxmarl.environments.overcooked_v3_macro import (
    MACRO_ACTION_NAMES,
    MacroActions,
    OvercookedV3Macro,
    OvercookedV3MacroInterruptible,
)


def _barrier_macro_env(rows):
    """Create a one-agent macro environment from a barrier test grid."""
    width = len(rows[0])
    service_objects = "WBPX" + "W" * (width - 4)
    service_access = "W   " + "W" * (width - 4)
    layout = Layout.from_string(
        "\n".join([service_objects, service_access, *rows]),
        possible_recipes=[[0, 0, 0]],
        barrier_config=[True],
    )
    return OvercookedV3Macro(layout=layout, max_macro_steps=10)


def test_macro_env_registration_and_action_space():
    env = make("overcooked_v3_macro")

    assert isinstance(env, OvercookedV3Macro)
    assert env.num_actions == len(MacroActions)
    assert env.action_space("agent_0").n == len(MacroActions)
    assert env.action_space("agent_0").n > OvercookedV3().action_space("agent_0").n
    assert MACRO_ACTION_NAMES[-4:] == ("up", "down", "left", "right")
    assert "interact" not in MACRO_ACTION_NAMES


def test_reset_adds_macro_state_fields():
    env = OvercookedV3Macro()
    _, state = env.reset(jax.random.PRNGKey(0))

    assert hasattr(state, "current_macro_actions")
    assert hasattr(state, "macro_action_done")
    assert hasattr(state, "macro_step_count")
    assert jnp.all(state.macro_action_done)
    assert jnp.all(state.current_macro_actions == MacroActions.wait)


def test_available_actions_mask_obviously_invalid_macros():
    env = OvercookedV3Macro(layout="cramped_room")
    _, state = env.reset(jax.random.PRNGKey(0))

    empty_mask = env.get_avail_actions(state)["agent_0"]
    assert empty_mask[MacroActions.wait]
    assert empty_mask[MacroActions.up]
    assert empty_mask[MacroActions.down]
    assert empty_mask[MacroActions.left]
    assert empty_mask[MacroActions.right]
    assert empty_mask[MacroActions.get_ingredient_0]
    assert empty_mask[MacroActions.get_plate]
    assert not empty_mask[MacroActions.put_ingredient_in_nearest_pot]
    assert not empty_mask[MacroActions.deliver]
    assert not empty_mask[MacroActions.drop_on_nearest_counter]
    assert not empty_mask[MacroActions.press_nearest_button]

    ingredient_state = state.replace(
        agents=state.agents.replace(
            inventory=state.agents.inventory.at[0].set(
                DynamicObject.ingredient(0)
            )
        )
    )
    ingredient_mask = env.get_avail_actions(ingredient_state)["agent_0"]
    assert not ingredient_mask[MacroActions.get_ingredient_0]
    assert ingredient_mask[MacroActions.put_ingredient_in_nearest_pot]
    assert ingredient_mask[MacroActions.drop_on_nearest_counter]


def test_wait_macro_emits_primitive_stay():
    key = jax.random.PRNGKey(0)
    macro_env = OvercookedV3Macro()

    _, macro_state = macro_env.reset(key)

    actions = {"agent_0": int(MacroActions.wait), "agent_1": int(MacroActions.wait)}
    _, next_macro_state, _, _, info = macro_env.step_env(key, macro_state, actions)

    assert jnp.array_equal(macro_state.agents.pos.x, next_macro_state.agents.pos.x)
    assert jnp.array_equal(macro_state.agents.pos.y, next_macro_state.agents.pos.y)
    assert jnp.array_equal(macro_state.agents.dir, next_macro_state.agents.dir)
    assert info["primitive_action"]["agent_0"] == int(Actions.stay)
    assert next_macro_state.macro_action_done[0]


def test_primitive_movement_actions_move_one_step_and_finish():
    env = _barrier_macro_env(
        [
            "W0WW#",
            "W   W",
            "W A W",
            "W   W",
            "WWWWW",
        ]
    )
    expected_actions = (
        (MacroActions.up, Actions.up, 0, -1),
        (MacroActions.down, Actions.down, 0, 1),
        (MacroActions.left, Actions.left, -1, 0),
        (MacroActions.right, Actions.right, 1, 0),
    )

    for macro_action, primitive_action, dx, dy in expected_actions:
        key = jax.random.PRNGKey(0)
        _, state = env.reset(key)
        _, next_state, _, _, info = env.step_env(
            key,
            state,
            {"agent_0": int(macro_action)},
        )

        assert info["primitive_action"]["agent_0"] == primitive_action
        assert next_state.agents.pos.x[0] == state.agents.pos.x[0] + dx
        assert next_state.agents.pos.y[0] == state.agents.pos.y[0] + dy
        assert next_state.macro_action_done[0]
        assert next_state.macro_step_count[0] == 0


def test_get_ingredient_macro_persists_until_pickup():
    env = OvercookedV3Macro(layout="cramped_room", max_macro_steps=10)
    key = jax.random.PRNGKey(0)
    _, state = env.reset(key)

    actions = {
        "agent_0": int(MacroActions.get_ingredient_0),
        "agent_1": int(MacroActions.wait),
    }

    for step in range(6):
        key, subkey = jax.random.split(key)
        _, state, _, _, _ = env.step_env(subkey, state, actions)
        if bool(state.macro_action_done[0]):
            break

    assert step < 5
    assert state.agents.inventory[0] == DynamicObject.ingredient(0)
    assert state.macro_action_done[0]


def test_missing_ingredient_macro_noops_and_terminates():
    env = OvercookedV3Macro(layout="cramped_room")
    key = jax.random.PRNGKey(0)
    _, state = env.reset(key)

    actions = {
        "agent_0": int(MacroActions.get_ingredient_2),
        "agent_1": int(MacroActions.wait),
    }
    _, next_state, _, _, info = env.step_env(key, state, actions)

    assert next_state.agents.inventory[0] == DynamicObject.EMPTY
    assert jnp.array_equal(state.agents.pos.x, next_state.agents.pos.x)
    assert jnp.array_equal(state.agents.pos.y, next_state.agents.pos.y)
    assert info["primitive_action"]["agent_0"] == int(Actions.stay)
    assert next_state.macro_action_done[0]


def test_macro_navigation_avoids_waiting_teammate():
    env = OvercookedV3Macro(layout="cramped_room", max_macro_steps=10)
    key = jax.random.PRNGKey(0)
    _, state = env.reset(key)

    program = [
        MacroActions.get_ingredient_0,
        MacroActions.put_ingredient_in_nearest_pot,
    ]
    program_idx = 0

    for step in range(5):
        if step > 0 and bool(state.macro_action_done[0]):
            program_idx += 1
        key, subkey = jax.random.split(key)
        _, state, _, _, info = env.step_env(
            subkey,
            state,
            {"agent_0": int(program[program_idx]), "agent_1": int(MacroActions.wait)},
        )

    assert state.agents.inventory[0] == DynamicObject.EMPTY
    assert state.agents.pos.x[0] == 2
    assert state.agents.pos.y[0] == 1

    key, subkey = jax.random.split(key)
    _, state, _, _, info = env.step_env(
        subkey,
        state,
        {"agent_0": int(MacroActions.get_ingredient_0), "agent_1": int(MacroActions.wait)},
    )

    assert info["primitive_action"]["agent_0"] != int(Actions.left)
    assert bool((state.agents.pos.x[0] != 2) | (state.agents.pos.y[0] != 1))


def test_macro_step_env_jittable():
    env = OvercookedV3Macro(layout="cramped_room", max_macro_steps=10)
    key = jax.random.PRNGKey(0)
    _, state = env.reset(key)
    actions = {
        "agent_0": jnp.array(MacroActions.get_ingredient_0, dtype=jnp.int32),
        "agent_1": jnp.array(MacroActions.wait, dtype=jnp.int32),
    }

    @jax.jit
    def step_fn(step_key, step_state):
        return env.step_env(step_key, step_state, actions)

    _, next_state, _, _, info = step_fn(jax.random.PRNGKey(1), state)

    assert next_state.time == 1
    assert info["current_macro_action"]["agent_0"] == MacroActions.get_ingredient_0


def test_macro_navigation_replans_when_barrier_state_changes_under_jit():
    env = _barrier_macro_env(
        [
            "WWWWWWW",
            "W     W",
            "WA#  0W",
            "W     W",
            "WWWWWWW",
        ]
    )
    _, state = env.reset(jax.random.PRNGKey(0))
    macros = jnp.array([MacroActions.get_ingredient_0], dtype=jnp.int32)

    @jax.jit
    def plan(current_state):
        """Plan one compiled primitive action for the supplied barrier state."""
        return env._macro_to_primitive_actions(current_state, macros)

    closed_actions, closed_reachable = plan(state)
    open_state = state.replace(
        barrier_active=jnp.zeros_like(state.barrier_active)
    )
    open_actions, open_reachable = plan(open_state)

    assert closed_actions[0] == Actions.down
    assert open_actions[0] == Actions.right
    assert closed_reachable[0]
    assert open_reachable[0]


def test_macro_navigation_retargets_when_nearest_target_is_blocked():
    env = _barrier_macro_env(
        [
            "WWWWWWWW",
            "W0  A#0W",
            "WWWWWWWW",
        ]
    )
    key = jax.random.PRNGKey(0)
    _, state = env.reset(key)

    _, next_state, _, _, info = env.step_env(
        key,
        state,
        {"agent_0": int(MacroActions.get_ingredient_0)},
    )

    assert info["primitive_action"]["agent_0"] == Actions.left
    assert next_state.agents.pos.x[0] == state.agents.pos.x[0] - 1
    assert not next_state.macro_action_done[0]


def test_macro_gives_up_when_stuck_at_block():
    """Stuck at a block right away: emit stay and hand control back at once."""
    env = _barrier_macro_env(
        [
            "WWWWWWW",
            "WA#  0W",
            "WWWWWWW",
        ]
    )
    key = jax.random.PRNGKey(0)
    _, state = env.reset(key)

    _, next_state, _, _, info = env.step_env(
        key,
        state,
        {"agent_0": int(MacroActions.get_ingredient_0)},
    )

    # The barrier sits directly beside the agent, so it cannot advance at all: it
    # is stuck at the block, so the macro ends (hands control back) rather than
    # waiting for the barrier to open.
    assert info["primitive_action"]["agent_0"] == Actions.stay
    assert next_state.macro_action_done[0]
    assert jnp.array_equal(next_state.agents.pos.x, state.agents.pos.x)
    assert jnp.array_equal(next_state.agents.pos.y, state.agents.pos.y)


def test_macro_walks_up_to_block_then_gives_up():
    """Agent walks toward a closed barrier, then gives up once stuck at it."""
    env = _barrier_macro_env(
        [
            "W    WWW",
            "WA   #0W",
            "W    WWW",
            "WWWWWWWW",
        ]
    )
    key = jax.random.PRNGKey(0)
    _, state = env.reset(key)
    # Freeze the timer so the (closed) barrier stays closed for the whole test.
    state = state.replace(barrier_timer=jnp.zeros_like(state.barrier_timer))
    actions = {"agent_0": int(MacroActions.get_ingredient_0)}
    start_x = int(state.agents.pos.x[0])
    barrier_x = int(state.barrier_positions[0, 1])

    positions = []
    primitives = []
    dones = []
    for _ in range(8):
        key, subkey = jax.random.split(key)
        _, state, _, _, info = env.step_env(subkey, state, actions)
        positions.append(int(state.agents.pos.x[0]))
        primitives.append(int(info["primitive_action"]["agent_0"]))
        dones.append(bool(state.macro_action_done[0]))
        if dones[-1]:
            break

    # It advances toward the barrier first (moves right, macro still running)...
    assert primitives[0] == Actions.right
    assert positions[0] == start_x + 1
    assert not dones[0]
    # ...then, once it is stuck one tile short of the closed barrier, it emits
    # stay and the macro ends, handing control back to the policy.
    assert dones[-1]
    assert primitives[-1] == Actions.stay
    assert positions[-1] == barrier_x - 1
    assert state.agents.inventory[0] == DynamicObject.EMPTY


def test_macro_terminates_when_permanent_wall_makes_target_unreachable():
    """A permanent wall seals the target off: stuck immediately, so it gives up."""
    env = _barrier_macro_env(
        [
            "WWWWWWW",
            "WA#W 0W",
            "WWWWWWW",
        ]
    )
    key = jax.random.PRNGKey(0)
    _, state = env.reset(key)

    _, next_state, _, _, info = env.step_env(
        key,
        state,
        {"agent_0": int(MacroActions.get_ingredient_0)},
    )

    # Even ignoring the transient barrier, the wall seals the target off, so the
    # agent is stuck at the block from the first tick and the macro terminates.
    assert info["primitive_action"]["agent_0"] == Actions.stay
    assert next_state.macro_action_done[0]
    assert jnp.array_equal(next_state.agents.pos.x, state.agents.pos.x)
    assert jnp.array_equal(next_state.agents.pos.y, state.agents.pos.y)


def test_macro_walkability_honors_pressed_pressure_plate_override():
    env = OvercookedV3Macro(layout="pressure_plate_demo")
    _, state = env.reset(jax.random.PRNGKey(0))
    plate_y, plate_x = state.pressure_plate_positions[0]
    linked_barriers = state.pressure_plate_linked_barrier[0]
    state = state.replace(
        agents=state.agents.replace(
            pos=state.agents.pos.replace(
                x=state.agents.pos.x.at[0].set(plate_x),
                y=state.agents.pos.y.at[0].set(plate_y),
            )
        ),
        barrier_active=jnp.where(
            linked_barriers, True, state.barrier_active
        ),
    )

    walkable = env._current_walkable_mask(state)
    barrier_y = state.barrier_positions[:, 0]
    barrier_x = state.barrier_positions[:, 1]

    assert jnp.all(walkable[barrier_y, barrier_x][linked_barriers])


def test_macro_walkability_tracks_moving_wall_positions_under_jit():
    env = OvercookedV3Macro(layout="moving_wall_bounce_demo")
    _, state = env.reset(jax.random.PRNGKey(0))
    active_walls = state.moving_wall_active_mask
    initial_positions = state.moving_wall_positions[active_walls]
    initial_walkable = env._current_walkable_mask(state)

    @jax.jit
    def move_walls_and_get_walkability(current_state):
        next_state = env._process_moving_walls(current_state)
        return next_state, env._current_walkable_mask(next_state)

    next_state, next_walkable = move_walls_and_get_walkability(state)
    next_positions = next_state.moving_wall_positions[active_walls]

    assert jnp.all(~initial_walkable[initial_positions[:, 0], initial_positions[:, 1]])
    assert jnp.all(next_walkable[initial_positions[:, 0], initial_positions[:, 1]])
    assert jnp.all(~next_walkable[next_positions[:, 0], next_positions[:, 1]])


def test_committed_interface_ignores_replacement_while_macro_is_running():
    env = OvercookedV3Macro(layout="cramped_room", max_macro_steps=10)
    key = jax.random.PRNGKey(0)
    _, state = env.reset(key)
    first = {
        "agent_0": int(MacroActions.get_ingredient_0),
        "agent_1": int(MacroActions.wait),
    }
    _, state, _, _, _ = env.step_env(key, state, first)
    assert not state.macro_action_done[0]

    replacement = {
        "agent_0": int(MacroActions.deliver),
        "agent_1": int(MacroActions.wait),
    }
    _, state, _, _, info = env.step_env(key, state, replacement)

    assert state.current_macro_actions[0] == MacroActions.get_ingredient_0
    assert not info["macro_action_started"]["agent_0"]


def test_interruptible_interface_replaces_running_macro():
    env = OvercookedV3MacroInterruptible(
        layout="cramped_room", max_macro_steps=10
    )
    key = jax.random.PRNGKey(0)
    _, state = env.reset(key)
    first = {
        "agent_0": int(MacroActions.get_ingredient_0),
        "agent_1": int(MacroActions.wait),
    }
    _, state, _, _, _ = env.step_env(key, state, first)
    assert not state.macro_action_done[0]

    replacement = {
        "agent_0": int(MacroActions.deliver),
        "agent_1": int(MacroActions.wait),
    }
    _, state, _, _, info = env.step_env(key, state, replacement)

    assert state.current_macro_actions[0] == MacroActions.deliver
    assert info["macro_action_started"]["agent_0"]


def test_interruptible_interface_repeating_macro_continues_without_reset():
    env = OvercookedV3MacroInterruptible(
        layout="cramped_room", max_macro_steps=10
    )
    key = jax.random.PRNGKey(0)
    _, state = env.reset(key)
    actions = {
        "agent_0": int(MacroActions.get_ingredient_0),
        "agent_1": int(MacroActions.wait),
    }
    _, state, _, _, _ = env.step_env(key, state, actions)
    first_count = state.macro_step_count[0]
    _, state, _, _, info = env.step_env(key, state, actions)

    assert not info["macro_action_started"]["agent_0"]
    expected_count = jnp.where(
        state.macro_action_done[0], 0, first_count + 1
    )
    assert state.macro_step_count[0] == expected_count


def test_available_actions_mask_ignores_dynamic_state():
    """Availability uses inventory + static existence only, never dynamic state.

    At reset no pot is cooking and every counter is empty, yet the macros that
    used to read that (possibly out-of-view) dynamic state are still available.
    """
    env = OvercookedV3Macro(layout="cramped_room")
    _, state = env.reset(jax.random.PRNGKey(0))

    empty_inv_mask = env.get_avail_actions(state)["agent_0"]
    assert empty_inv_mask[MacroActions.wait_for_nearest_pot]  # a pot exists
    assert empty_inv_mask[MacroActions.pickup_from_nearest_counter]  # counters exist

    plate_mask = env.get_avail_actions(
        state.replace(
            agents=state.agents.replace(
                inventory=state.agents.inventory.at[0].set(DynamicObject.PLATE)
            )
        )
    )["agent_0"]
    # No pot is ready, but holding a plate while a pot exists is enough.
    assert plate_mask[MacroActions.get_soup_from_nearest_pot]

    # A finished (cooked) pot would fail the old valid-placement check; holding an
    # ingredient while a pot exists still makes the placement macro available.
    pot_y, pot_x = int(state.pot_positions[0, 0]), int(state.pot_positions[0, 1])
    full_pot_mask = env.get_avail_actions(
        state.replace(
            agents=state.agents.replace(
                inventory=state.agents.inventory.at[0].set(
                    DynamicObject.ingredient(0)
                )
            ),
            grid=state.grid.at[pot_y, pot_x, 1].set(int(DynamicObject.COOKED)),
        )
    )["agent_0"]
    assert full_pot_mask[MacroActions.put_ingredient_in_nearest_pot]
    assert full_pot_mask[MacroActions.drop_on_nearest_counter]


def test_macro_arrives_and_finishes_at_invalid_object():
    """Nearest counter is empty: the agent goes, attempts once, macro ends."""
    env = OvercookedV3Macro(layout="cramped_room", max_macro_steps=20)
    key = jax.random.PRNGKey(0)
    _, state = env.reset(key)
    actions = {
        "agent_0": int(MacroActions.pickup_from_nearest_counter),
        "agent_1": int(MacroActions.wait),
    }

    interacted = False
    for _ in range(20):
        key, subkey = jax.random.split(key)
        _, state, _, _, info = env.step_env(subkey, state, actions)
        if int(info["primitive_action"]["agent_0"]) == int(Actions.interact):
            interacted = True
        if bool(state.macro_action_done[0]):
            break

    # It attempted the pickup once (found the counter empty) and terminated,
    # still empty-handed rather than looping forever.
    assert interacted
    assert state.macro_action_done[0]
    assert state.agents.inventory[0] == DynamicObject.EMPTY


def test_wait_for_nearest_pot_only_reacts_to_pots_in_view():
    """With a view radius, the wait macro only reacts to cooking pots in view."""
    env = OvercookedV3Macro(layout="cramped_room", agent_view_size=1)
    _, state = env.reset(jax.random.PRNGKey(0))
    state = state.replace(
        pot_active_mask=state.pot_active_mask.at[0].set(True),
        pot_cooking_timer=state.pot_cooking_timer.at[0].set(5),
    )
    pot_y, pot_x = int(state.pot_positions[0, 0]), int(state.pot_positions[0, 1])

    def wait_done(agent_state):
        """Whether wait_for_nearest_pot would complete for agent 0 right now."""
        return bool(
            env._macro_done_for_agent(
                agent_state,
                0,
                jnp.array(MacroActions.wait_for_nearest_pot, dtype=jnp.int32),
                jnp.array(Actions.stay, dtype=jnp.int32),
                jnp.array(True),
            )
        )

    # Adjacent to the cooking pot (Chebyshev radius 1 -> in view): keep waiting.
    near = state.replace(
        agents=state.agents.replace(
            pos=state.agents.pos.replace(
                y=state.agents.pos.y.at[0].set(pot_y + 1),
                x=state.agents.pos.x.at[0].set(pot_x),
            )
        )
    )
    assert not wait_done(near)

    # Far corner (outside the view window): nothing worth waiting for -> noop.
    far = state.replace(
        agents=state.agents.replace(
            pos=state.agents.pos.replace(
                y=state.agents.pos.y.at[0].set(env.height - 1),
                x=state.agents.pos.x.at[0].set(env.width - 1),
            )
        )
    )
    assert wait_done(far)
