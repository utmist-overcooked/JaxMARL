import json

import pytest

import scripts.generate_overcooked_v3_layouts as layout_generator
from jaxmarl.environments.overcooked_v3.layouts import (
    Layout,
    load_layouts_from_json,
    validate_generated_layout,
)
from jaxmarl.environments.overcooked_v3.overcooked import OvercookedV3
from scripts.generate_overcooked_v3_layouts import generate_document
from scripts.play_overcooked_v3 import register_json_layouts


def _config(**overrides):
    generator = {
        "seed": 11,
        "count": 2,
        "name_prefix": "test_kitchen",
        "width": 8,
        "height": 6,
        "ingredient_piles": [2, 1],
        "possible_recipes": [[0, 0, 0], [1, 1, 1]],
        "pots": 1,
        "plate_piles": 1,
        "depots": 1,
        "object_placement": "boundary",
        "counter_density": 0.1,
        "max_attempts": 1000,
    }
    generator.update(overrides)
    return {"generator": generator, "layouts": {}}


def test_generator_is_deterministic_and_produces_valid_exact_size_layouts():
    first = generate_document(_config())
    second = generate_document(_config())

    assert first == second
    assert len(first["layouts"]) == 2
    for entry in first["layouts"].values():
        grid = entry["ascii"]
        rows = grid.splitlines()
        assert len(rows) == 6
        assert all(len(row) == 8 for row in rows)
        assert grid.count("A") == 2
        assert grid.count("0") == 2
        assert grid.count("1") == 1
        assert grid.count("P") == 1
        assert grid.count("B") == 1
        assert grid.count("X") == 1

        layout = Layout.from_string(
            grid,
            possible_recipes=entry["possible_recipes"],
        )
        assert validate_generated_layout(layout) == (True, [])
        assert entry["validation"] == {"valid": True, "errors": []}


def test_json_loader_reads_and_runs_generated_layout(tmp_path):
    document = generate_document(_config(count=1))
    path = tmp_path / "layouts.json"
    path.write_text(json.dumps(document), encoding="utf-8")

    loaded = load_layouts_from_json(path)
    layout = loaded["test_kitchen_0"]

    assert layout.height == 6
    assert layout.width == 8
    assert len(layout.agent_positions) == 2
    assert layout.get_info()["num_ingredient_piles"] == {0: 2, 1: 1}
    assert OvercookedV3(layout=layout).layout is layout


def test_accessibility_rejects_disconnected_floor_and_workstations():
    grid = "\n".join(
        [
            "WWPWW",
            "0A AW",
            "WWWWW",
            "W   X",
            "WWBWW",
        ]
    )
    layout = Layout.from_string(grid, possible_recipes=[[0, 0, 0]])

    valid, errors = validate_generated_layout(layout)

    assert not valid
    assert any("walkable tile" in error for error in errors)
    assert any("inaccessible" in error for error in errors)
    assert any("cannot be completed" in error for error in errors)


def test_loader_rejects_invalid_map_even_if_validation_metadata_says_valid(tmp_path):
    document = {
        "layouts": {
            "bad": {
                "ascii": "WWPWW\n0A AW\nWWWWW\nW   X\nWWBWW",
                "possible_recipes": [[0, 0, 0]],
                "validation": {"valid": True, "errors": []},
            }
        }
    }
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid layout 'bad'"):
        load_layouts_from_json(path)


def test_generator_rejects_mixed_recipes():
    with pytest.raises(ValueError, match="currently supports same-ingredient"):
        generate_document(
            _config(possible_recipes=[[0, 0, 1]])
        )


def test_generator_can_place_all_workstations_in_the_interior():
    document = generate_document(
        _config(
            count=1,
            object_placement="interior",
            counter_density=0.1,
        )
    )
    grid = document["layouts"]["test_kitchen_0"]["ascii"]
    rows = grid.splitlines()
    workstation_symbols = set("012PBX")
    workstation_positions = [
        (row, col)
        for row, line in enumerate(rows)
        for col, symbol in enumerate(line)
        if symbol in workstation_symbols
    ]

    assert workstation_positions
    assert all(
        0 < row < len(rows) - 1 and 0 < col < len(rows[0]) - 1
        for row, col in workstation_positions
    )
    layout = Layout.from_string(
        grid,
        possible_recipes=document["layouts"]["test_kitchen_0"][
            "possible_recipes"
        ],
    )
    assert validate_generated_layout(layout) == (True, [])


def test_anywhere_mode_generates_valid_interior_and_boundary_workstations():
    document = generate_document(
        _config(
            count=4,
            object_placement="anywhere",
            counter_density=0.1,
        )
    )
    saw_interior = False
    saw_boundary = False

    for entry in document["layouts"].values():
        rows = entry["ascii"].splitlines()
        for row, line in enumerate(rows):
            for col, symbol in enumerate(line):
                if symbol not in set("012PBX"):
                    continue
                if row in {0, len(rows) - 1} or col in {0, len(line) - 1}:
                    saw_boundary = True
                else:
                    saw_interior = True

        layout = Layout.from_string(
            entry["ascii"],
            possible_recipes=entry["possible_recipes"],
        )
        assert validate_generated_layout(layout) == (True, [])

    assert saw_interior
    assert saw_boundary


def test_interactive_player_registers_json_layouts(tmp_path):
    document = generate_document(
        _config(count=1, name_prefix="interactive_test_kitchen")
    )
    path = tmp_path / "interactive-layouts.json"
    path.write_text(json.dumps(document), encoding="utf-8")

    from jaxmarl.environments.overcooked_v3.layouts import overcooked_v3_layouts

    try:
        names = register_json_layouts(path)
        assert names == ["interactive_test_kitchen_0"]
        assert names[0] in overcooked_v3_layouts
    finally:
        overcooked_v3_layouts.pop("interactive_test_kitchen_0", None)


def test_incremental_generation_checkpoints_and_continues_after_failure(
    tmp_path,
    monkeypatch,
):
    output_path = tmp_path / "checkpointed-layouts.json"
    original_generate_layout = layout_generator.generate_layout
    call_count = 0

    def fail_second_layout(config, rng):
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise RuntimeError("simulated generation failure")
        return original_generate_layout(config, rng)

    monkeypatch.setattr(
        layout_generator,
        "generate_layout",
        fail_second_layout,
    )
    messages = []
    result, failures = layout_generator.generate_to_file(
        _config(count=3, name_prefix="checkpoint_test"),
        output_path,
        emit=messages.append,
    )
    checkpoint = json.loads(output_path.read_text(encoding="utf-8"))

    assert result == checkpoint
    assert list(checkpoint["layouts"]) == [
        "checkpoint_test_0",
        "checkpoint_test_2",
    ]
    assert checkpoint["generation_errors"] == {
        "checkpoint_test_1": "simulated generation failure"
    }
    assert checkpoint["generation_progress"] == {
        "requested": 3,
        "completed": 2,
        "failed": 1,
        "status": "completed_with_errors",
    }
    assert any("[1/3] Generating checkpoint_test_0" in message for message in messages)
    assert any("[2/3] FAILED checkpoint_test_1" in message for message in messages)
    assert any("2 map(s) complete" in message for message in messages)
