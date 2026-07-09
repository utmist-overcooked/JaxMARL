"""Tests for macro-action Overcooked V3."""

import jax
import jax.numpy as jnp

from jaxmarl import make
from jaxmarl.environments.overcooked_v3.common import Actions, DynamicObject
from jaxmarl.environments.overcooked_v3.overcooked import OvercookedV3
from jaxmarl.environments.overcooked_v3_macro import (
    MACRO_ACTION_NAMES,
    MacroActions,
    OvercookedV3Macro,
)


def test_macro_env_registration_and_action_space():
    env = make("overcooked_v3_macro")

    assert isinstance(env, OvercookedV3Macro)
    assert env.num_actions == len(MacroActions)
    assert env.action_space("agent_0").n == len(MacroActions)
    assert env.action_space("agent_0").n > OvercookedV3().action_space("agent_0").n
    assert "right" not in MACRO_ACTION_NAMES
    assert "down" not in MACRO_ACTION_NAMES
    assert "left" not in MACRO_ACTION_NAMES
    assert "up" not in MACRO_ACTION_NAMES
    assert "interact" not in MACRO_ACTION_NAMES


def test_reset_adds_macro_state_fields():
    env = OvercookedV3Macro()
    _, state = env.reset(jax.random.PRNGKey(0))

    assert hasattr(state, "current_macro_actions")
    assert hasattr(state, "macro_action_done")
    assert hasattr(state, "macro_step_count")
    assert jnp.all(state.macro_action_done)
    assert jnp.all(state.current_macro_actions == MacroActions.wait)


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
