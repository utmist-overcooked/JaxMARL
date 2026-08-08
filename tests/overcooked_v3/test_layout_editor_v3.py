"""Regression tests for the Overcooked V3 layout editor export path."""

import numpy as np
import pygame
import pytest

from jaxmarl.environments.overcooked_v3.common import ButtonAction, StaticObject
from jaxmarl.environments.overcooked_v3.layouts import Layout
from jaxmarl.tools import layout_editor_v3 as editor


@pytest.fixture(autouse=True)
def _use_real_static_object(monkeypatch):
    monkeypatch.setattr(editor, "DEPENDENCIES_AVAILABLE", True)
    monkeypatch.setattr(editor, "StaticObject", StaticObject)


def _editor_from_state(state):
    layout_editor = editor.LevelEditor.__new__(editor.LevelEditor)
    layout_editor.state = state
    return layout_editor


def test_export_uses_parser_symbols_and_config_order():
    state = editor.EditorState(
        width=4,
        height=3,
        static_objects=np.zeros((3, 4), dtype=int),
    )
    state.static_objects[0, 2] = StaticObject.BUTTON
    state.static_objects[0, 3] = StaticObject.BARRIER
    state.static_objects[1, 0] = StaticObject.PRESSURE_PLATE
    state.static_objects[2, 1] = StaticObject.BARRIER
    state.agent_positions = [(0, 2)]
    state.buttons = [
        editor.ButtonInfo(
            y=0,
            x=2,
            action_type=ButtonAction.TOGGLE_BARRIER,
            target_barrier_idxs=[0],
        )
    ]
    state.pressure_plates = [
        editor.PressurePlateInfo(
            y=1,
            x=0,
            action_type=ButtonAction.TIMED_BARRIER,
            target_barrier_idxs=[1],
        )
    ]
    state.barriers = [
        editor.BarrierInfo(y=2, x=1, initially_active=True),
        editor.BarrierInfo(y=0, x=3, initially_active=False),
    ]

    layout_editor = _editor_from_state(state)

    layout_str = layout_editor._layout_string_from_state()
    assert layout_str.strip("\n").split("\n") == ["  !#", "_   ", "A#  "]
    assert "Q" not in layout_str
    assert "K" not in layout_str
    assert "L" not in layout_str

    assert layout_editor._barrier_config() == [False, True]
    assert layout_editor._button_config() == [
        ((1,), int(ButtonAction.TOGGLE_BARRIER))
    ]
    assert layout_editor._pressure_plate_config() == [
        ((0,), int(ButtonAction.TIMED_BARRIER))
    ]

    reconstructed = Layout.from_string(
        layout_str,
        possible_recipes=[[0, 0, 0]],
        barrier_config=layout_editor._barrier_config(),
        button_config=layout_editor._button_config(),
        pressure_plate_config=layout_editor._pressure_plate_config(),
    )

    assert reconstructed.barrier_info == [(0, 3, False), (2, 1, True)]
    assert reconstructed.button_info[0] == (
        0,
        2,
        (1,),
        int(ButtonAction.TOGGLE_BARRIER),
    )
    assert reconstructed.pressure_plate_info[0] == (
        1,
        0,
        (0,),
        int(ButtonAction.TIMED_BARRIER),
    )


def test_wiring_defaults_and_clicks_store_button_action_values():
    assert editor.ButtonInfo(y=0, x=0).action_type == int(
        ButtonAction.TOGGLE_BARRIER
    )
    assert editor.PressurePlateInfo(y=0, x=0).action_type == int(
        ButtonAction.TOGGLE_BARRIER
    )

    state = editor.EditorState(
        width=1,
        height=1,
        static_objects=np.array([[StaticObject.BUTTON]], dtype=int),
    )
    state.buttons = [editor.ButtonInfo(y=0, x=0)]
    layout_editor = _editor_from_state(state)
    layout_editor.wiring_cell = (0, 0)
    layout_editor._wiring_action_rects = [
        (pygame.Rect(0, 0, 10, 10), int(ButtonAction.TIMED_BARRIER))
    ]
    layout_editor._wiring_barrier_rects = []
    layout_editor._barrier_toggle_rect = None

    layout_editor._handle_info_panel_click(1, 1, 1)

    assert state.buttons[0].action_type == int(ButtonAction.TIMED_BARRIER)
