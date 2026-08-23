"""Tests for the manual macro-control tool ``scripts/play_overcooked_v3_macro.py``.

These cover the importable helpers (labels, inventory text, geometry) and,
crucially, the boundary-vs-interruptible command semantics the three UI modes
rely on: a boundary-mode click is deferred to the next macro boundary while an
interruptible click takes effect on the next step.
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import jax
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import play_overcooked_v3_macro as ui  # noqa: E402
from jaxmarl.environments.overcooked_v3_macro import MacroActions  # noqa: E402


def test_mode_to_env_mapping():
    """Boundary uses the committed env; every_step/replan the interruptible one."""
    assert ui.MODE_TO_ENV["boundary"] == "overcooked_v3_macro"
    assert ui.MODE_TO_ENV["every_step"] == "overcooked_v3_macro_interruptible"
    assert ui.MODE_TO_ENV["replan"] == "overcooked_v3_macro_interruptible"


def test_every_macro_action_has_a_label():
    """The button grid needs a label for all 17 macro actions."""
    assert set(ui.MACRO_LABELS) == set(MacroActions)
    assert all(ui.MACRO_LABELS[a] for a in MacroActions)


def test_describe_inventory():
    """Inventory integers render to short human-readable strings."""
    from jaxmarl.environments.overcooked_v3.common import DynamicObject

    assert ui.describe_inventory(DynamicObject.EMPTY) == "empty"
    assert ui.describe_inventory(DynamicObject.PLATE) == "plate"
    onion = int(DynamicObject.ingredient(0))
    assert "ing0" in ui.describe_inventory(onion)


def test_geometry_and_button_rects():
    """Geometry yields one button per macro per agent, non-overlapping in a row."""
    import pygame

    pygame.init()
    env, _ = ui.build_env("overcooked_v3_macro", "cramped_room")
    geom = ui.compute_geometry(env)
    rects = ui.button_rects(env, geom)
    assert len(rects) == env.num_agents * ui.NUM_MACROS
    # Buttons in agent 0's row are left-to-right ordered and disjoint.
    row0 = [rects[(0, m)] for m in range(ui.NUM_MACROS)]
    for left, right in zip(row0, row0[1:]):
        assert right.x > left.right - 1
    assert geom["win_w"] >= geom["render_w"]
    pygame.quit()


def _run_until_running(env, macro, max_steps=30):
    """Command ``macro`` for agent_0 until it is actively running (not done)."""
    key = jax.random.PRNGKey(0)
    key, rk = jax.random.split(key)
    _, state = env.reset(rk)
    for _ in range(max_steps):
        actions = {a: np.int32(macro) for a in env.agents}
        key, sk = jax.random.split(key)
        _, state, _, _, _ = env.step(sk, state, actions)
        if int(state.current_macro_actions[0]) == int(macro) and not bool(
            state.macro_action_done[0]
        ):
            return state, key
    raise AssertionError("macro never entered a running, not-done state")


def test_boundary_defers_but_interruptible_interrupts():
    """A new command mid-macro is deferred (boundary) vs adopted (interruptible)."""
    running_macro = MacroActions.get_ingredient_0
    new_macro = MacroActions.get_plate

    committed = ui.build_env("overcooked_v3_macro", "cramped_room")[0]
    state, key = _run_until_running(committed, running_macro)
    actions = {a: np.int32(new_macro) for a in committed.agents}
    key, sk = jax.random.split(key)
    _, state, _, _, _ = committed.step(sk, state, actions)
    # Committed: the running macro is unchanged despite the new request.
    assert int(state.current_macro_actions[0]) == int(running_macro)

    interruptible = ui.build_env(
        "overcooked_v3_macro_interruptible", "cramped_room"
    )[0]
    state, key = _run_until_running(interruptible, running_macro)
    actions = {a: np.int32(new_macro) for a in interruptible.agents}
    key, sk = jax.random.split(key)
    _, state, _, _, _ = interruptible.step(sk, state, actions)
    # Interruptible: the new macro is adopted immediately.
    assert int(state.current_macro_actions[0]) == int(new_macro)
