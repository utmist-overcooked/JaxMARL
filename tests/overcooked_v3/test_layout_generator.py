import json
import random
from collections import deque
from pathlib import Path

import pytest

import scripts.generate_overcooked_v3_layouts as layout_generator
from jaxmarl.environments.overcooked_v3.layouts import (
    Layout,
    load_layouts_from_json,
    validate_generated_layout,
)
from jaxmarl.environments.overcooked_v3.overcooked import OvercookedV3
from scripts.generate_overcooked_v3_layouts import (
    generate_document,
    generate_layout,
    validate_config,
)
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


def _floor_components(grid):
    rows = grid.splitlines()
    walkable = {
        (row, col)
        for row, line in enumerate(rows)
        for col, symbol in enumerate(line)
        if symbol in {" ", "A"}
    }
    components = []
    while walkable:
        start = next(iter(walkable))
        component = {start}
        queue = deque([start])
        walkable.remove(start)
        while queue:
            row, col = queue.popleft()
            for adjacent in (
                (row - 1, col),
                (row + 1, col),
                (row, col - 1),
                (row, col + 1),
            ):
                if adjacent in walkable:
                    walkable.remove(adjacent)
                    component.add(adjacent)
                    queue.append(adjacent)
        components.append(component)
    return rows, components


def _accessible_symbols(rows, component):
    symbols = set()
    for row, col in component:
        for station_row, station_col in (
            (row - 1, col),
            (row + 1, col),
            (row, col - 1),
            (row, col + 1),
        ):
            symbol = rows[station_row][station_col]
            if symbol in set("012PBX"):
                symbols.add(symbol)
    return symbols


def _shared_tiles(grid):
    rows, components = _floor_components(grid)
    component_by_position = {
        position: component_idx
        for component_idx, component in enumerate(components)
        for position in component
    }
    return {
        (row, col)
        for row in range(1, len(rows) - 1)
        for col in range(1, len(rows[0]) - 1)
        if rows[row][col] == "W"
        and {
            component_by_position[position]
            for position in (
                (row - 1, col),
                (row + 1, col),
                (row, col - 1),
                (row, col + 1),
            )
            if position in component_by_position
        }
        == {0, 1}
    }


def _barrier_positions(grid):
    return {
        (row, col)
        for row, line in enumerate(grid.splitlines())
        for col, symbol in enumerate(line)
        if symbol == "#"
    }


def _button_positions(grid):
    return {
        (row, col)
        for row, line in enumerate(grid.splitlines())
        for col, symbol in enumerate(line)
        if symbol == "!"
    }


def _assert_pressure_plates_are_agent_accessible(grid):
    rows, components = _floor_components(grid)
    assert grid.count("_") > 0
    for component in components:
        if any(rows[row][col] == "_" for row, col in component):
            assert any(rows[row][col] == "A" for row, col in component)


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
        assert grid.count("R") == 1

        layout = Layout.from_string(
            grid,
            possible_recipes=entry["possible_recipes"],
        )
        assert validate_generated_layout(layout) == (True, [])
        assert entry["validation"] == {"valid": True, "errors": []}


def test_shipped_example_is_supported_and_generates_a_layout():
    example_path = (
        Path(__file__).parents[2]
        / "scripts"
        / "overcooked_v3_layouts.example.json"
    )
    document = json.loads(example_path.read_text(encoding="utf-8"))
    config = validate_config(document["generator"])

    grid, layout, _ = generate_layout(
        config,
        random.Random(config["seed"]),
    )

    assert config["num_regions"] == 2
    assert grid.count("A") == 2
    assert validate_generated_layout(layout) == (True, [])


def test_frontier_generation_constructs_dense_connected_map_on_first_attempt():
    config = validate_config(
        _config(
            count=1,
            width=10,
            height=10,
            counter_density=0.4,
        )["generator"]
    )

    grid, layout, attempts = generate_layout(
        config,
        random.Random(config["seed"]),
    )

    _, components = _floor_components(grid)
    assert attempts == 1
    assert len(components) == 1
    assert validate_generated_layout(layout) == (True, [])


def test_complete_each_constructs_one_complete_workflow_per_agent_region():
    config = validate_config(
        _config(
            count=1,
            width=10,
            height=10,
            ingredient_piles=[2, 2],
            pots=2,
            plate_piles=2,
            depots=2,
            counter_density=0.4,
            num_regions=2,
            workflow_mode="complete_each",
        )["generator"]
    )

    grid, layout, _ = generate_layout(
        config,
        random.Random(config["seed"]),
    )
    rows, components = _floor_components(grid)

    assert len(components) == 2
    for component in components:
        assert sum(rows[row][col] == "A" for row, col in component) == 1
        assert set("01PBX") <= _accessible_symbols(rows, component)
    assert validate_generated_layout(layout) == (True, [])


def test_shared_workflow_constructs_two_regions_with_counter_handoff():
    config = validate_config(
        _config(
            count=1,
            width=10,
            height=10,
            counter_density=0.4,
            num_regions=2,
            workflow_mode="shared",
        )["generator"]
    )

    grid, layout, _ = generate_layout(
        config,
        random.Random(config["seed"]),
    )
    rows, components = _floor_components(grid)
    component_by_position = {
        position: component_idx
        for component_idx, component in enumerate(components)
        for position in component
    }

    assert len(components) == 2
    assert grid.count("R") == 1
    accessible = [_accessible_symbols(rows, component) for component in components]
    for stations in accessible:
        for recipe in config["possible_recipes"]:
            required = {str(ingredient_idx) for ingredient_idx in recipe}
            required.update({"P", "B", "X"})
            assert not required <= stations

    handoff_counters = []
    for row in range(1, len(rows) - 1):
        for col in range(1, len(rows[0]) - 1):
            if rows[row][col] != "W":
                continue
            adjacent_components = {
                component_by_position[position]
                for position in (
                    (row - 1, col),
                    (row + 1, col),
                    (row, col - 1),
                    (row, col + 1),
                )
                if position in component_by_position
            }
            if adjacent_components == {0, 1}:
                handoff_counters.append((row, col))

    assert handoff_counters
    assert validate_generated_layout(layout) == (True, [])


def test_two_region_generation_enforces_exact_shared_tile_count():
    config = validate_config(
        _config(
            count=1,
            width=10,
            height=10,
            counter_density=0.4,
            num_regions=2,
            num_shared_tiles=3,
            workflow_mode="complete_each",
            ingredient_piles=[2, 2],
            pots=2,
            plate_piles=2,
            depots=2,
        )["generator"]
    )

    grid, layout, _ = generate_layout(
        config,
        random.Random(config["seed"]),
    )

    assert len(_shared_tiles(grid)) == 3
    assert validate_generated_layout(layout) == (True, [])


@pytest.mark.parametrize("value", [-1, 1.5, True, "2"])
def test_generator_rejects_invalid_shared_tile_count(value):
    with pytest.raises(ValueError, match="num_shared_tiles"):
        generate_document(_config(num_regions=2, num_shared_tiles=value))


def test_shared_tile_count_requires_two_regions():
    with pytest.raises(ValueError, match="requires generator.num_regions = 2"):
        generate_document(_config(num_shared_tiles=1))


def test_shared_workflow_rejects_zero_shared_tiles():
    with pytest.raises(ValueError, match="requires at least one shared tile"):
        generate_document(
            _config(
                num_regions=2,
                workflow_mode="shared",
                num_shared_tiles=0,
            )
        )


@pytest.mark.parametrize("value", [-1, 1.5, True, "2"])
def test_generator_rejects_invalid_barrier_count(value):
    with pytest.raises(ValueError, match="barriers"):
        generate_document(_config(barriers=value))


@pytest.mark.parametrize("value", [-1, 3, 1.5, True, "2"])
def test_generator_rejects_invalid_pressure_plate_multiplicity(value):
    with pytest.raises(ValueError, match="pressure_plates_per_barrier"):
        generate_document(_config(pressure_plates_per_barrier=value))


@pytest.mark.parametrize("value", [-1, 1.5, True, "2"])
def test_generator_rejects_invalid_button_multiplicity(value):
    with pytest.raises(ValueError, match="buttons_per_barrier"):
        generate_document(_config(buttons_per_barrier=value))


def test_generator_requires_at_least_one_control_per_barrier():
    with pytest.raises(ValueError, match="at least one pressure plate or button"):
        generate_document(
            _config(
                barriers=1,
                pressure_plates_per_barrier=0,
                buttons_per_barrier=0,
            )
        )


def test_generator_rejects_button_count_above_environment_capacity():
    with pytest.raises(ValueError, match="MAX_BUTTONS"):
        generate_document(_config(barriers=16, buttons_per_barrier=2))

    with pytest.raises(ValueError, match="buttons_per_barrier"):
        generate_document(_config(buttons_per_barrier=17))


def test_shared_barrier_placement_requires_two_regions():
    with pytest.raises(ValueError, match="requires generator.num_regions = 2"):
        generate_document(
            _config(
                barriers=1,
                barrier_placement="shared",
            )
        )


@pytest.mark.parametrize(
    "barrier_placement",
    ["anywhere", "action_adjacent", "shared_or_action_adjacent"],
)
def test_generator_rejects_barriers_that_leave_too_few_floor_tiles(
    barrier_placement,
):
    """Validation accounts for barriers forced onto otherwise walkable tiles."""
    with pytest.raises(ValueError, match="pressure plates and 2 agent spawns"):
        validate_config(
            _config(
                width=5,
                height=5,
                ingredient_piles=[9],
                possible_recipes=[[0, 0, 0]],
                counter_density=0,
                barriers=4,
                barrier_placement=barrier_placement,
                pressure_plates_per_barrier=1,
            )["generator"]
        )


def test_single_region_barrier_placement_requires_two_spawn_tiles():
    """A single floor component still reserves space for both generated agents."""
    config = validate_config(
        _config(
            width=5,
            height=5,
            ingredient_piles=[1],
            possible_recipes=[[0, 0, 0]],
            barriers=1,
            barrier_placement="action_adjacent",
        )["generator"]
    )
    grid = [
        list("W0WWW"),
        list("W   W"),
        list("WWWWW"),
        list("WWWWW"),
        list("WWWWW"),
    ]
    region = {(1, 1), (1, 2), (1, 3)}

    with pytest.raises(
        layout_generator.CandidateGenerationError,
        match="two agent spawns",
    ):
        layout_generator._place_barriers_and_controls(
            grid,
            config,
            [region],
            [],
            random.Random(0),
        )


@pytest.mark.parametrize("plates_per_barrier", [1, 2])
def test_generator_spawns_exact_barriers_with_single_or_paired_reachable_plates(
    plates_per_barrier,
):
    document = generate_document(
        _config(
            count=1,
            width=10,
            height=10,
            counter_density=0.4,
            barriers=3,
            pressure_plates_per_barrier=plates_per_barrier,
            max_attempts=5000,
        )
    )
    entry = next(iter(document["layouts"].values()))

    assert entry["ascii"].count("#") == 3
    assert entry["ascii"].count("_") == 3 * plates_per_barrier
    assert entry["barrier_config"] == [True, True, True]
    target_counts = [0, 0, 0]
    for targets, _ in entry["pressure_plate_config"]:
        assert len(targets) == 1
        target_counts[targets[0]] += 1
    assert target_counts == [plates_per_barrier] * 3
    _assert_pressure_plates_are_agent_accessible(entry["ascii"])


@pytest.mark.parametrize("buttons_per_barrier", [0, 1, 2])
def test_generator_spawns_and_wires_exact_timed_buttons_per_barrier(
    buttons_per_barrier,
):
    document = generate_document(
        _config(
            count=1,
            width=10,
            height=10,
            counter_density=0.2,
            barriers=3,
            pressure_plates_per_barrier=0 if buttons_per_barrier else 1,
            buttons_per_barrier=buttons_per_barrier,
            max_attempts=5000,
        )
    )
    entry = next(iter(document["layouts"].values()))

    assert entry["ascii"].count("!") == 3 * buttons_per_barrier
    assert entry["ascii"].count("_") == (0 if buttons_per_barrier else 3)
    target_counts = [0, 0, 0]
    for targets, action_type in entry["button_config"]:
        assert len(targets) == 1
        assert action_type == int(layout_generator.ButtonAction.TIMED_BARRIER)
        target_counts[targets[0]] += 1
    assert target_counts == [buttons_per_barrier] * 3


@pytest.mark.parametrize("placement", ["boundary", "interior"])
def test_generated_buttons_follow_workstation_placement(placement):
    document = generate_document(
        _config(
            count=1,
            width=10,
            height=10,
            object_placement=placement,
            barriers=2,
            buttons_per_barrier=1,
            max_attempts=5000,
        )
    )
    grid = next(iter(document["layouts"].values()))["ascii"]
    rows = grid.splitlines()

    assert len(_button_positions(grid)) == 2
    for row, col in _button_positions(grid):
        is_boundary = row in {0, len(rows) - 1} or col in {
            0,
            len(rows[0]) - 1,
        }
        assert is_boundary == (placement == "boundary")


def test_shared_barrier_placement_uses_only_two_region_interface_tiles():
    document = generate_document(
        _config(
            count=1,
            width=10,
            height=10,
            counter_density=0.4,
            num_regions=2,
            num_shared_tiles=4,
            workflow_mode="shared",
            barriers=2,
            barrier_placement="shared",
            max_attempts=5000,
        )
    )
    grid = next(iter(document["layouts"].values()))["ascii"]
    rows, components = _floor_components(grid)
    component_by_position = {
        position: component_idx
        for component_idx, component in enumerate(components)
        for position in component
    }

    assert len(components) == 2
    for row, col in _barrier_positions(grid):
        adjacent_components = {
            component_by_position[position]
            for position in (
                (row - 1, col),
                (row + 1, col),
                (row, col - 1),
                (row, col + 1),
            )
            if position in component_by_position
        }
        assert adjacent_components == {0, 1}


@pytest.mark.parametrize(
    "placement",
    ["action_adjacent", "shared_or_action_adjacent"],
)
def test_action_barrier_placements_are_adjacent_to_action_items(placement):
    document = generate_document(
        _config(
            count=1,
            width=10,
            height=10,
            counter_density=0.4,
            num_regions=2,
            workflow_mode="shared",
            barriers=2,
            barrier_placement=placement,
            max_attempts=5000,
        )
    )
    grid = next(iter(document["layouts"].values()))["ascii"]
    rows = grid.splitlines()
    action_symbols = set("0123456789PBX")

    for row, col in _barrier_positions(grid):
        is_shared = len(
            {
                component_idx
                for component_idx, component in enumerate(_floor_components(grid)[1])
                for position in (
                    (row - 1, col),
                    (row + 1, col),
                    (row, col - 1),
                    (row, col + 1),
                )
                if position in component
            }
        ) == 2
        is_action_adjacent = any(
            rows[adjacent_row][adjacent_col] in action_symbols
            for adjacent_row, adjacent_col in (
                (row - 1, col),
                (row + 1, col),
                (row, col - 1),
                (row, col + 1),
            )
        )
        assert is_action_adjacent if placement == "action_adjacent" else (
            is_shared or is_action_adjacent
        )


def test_json_loader_preserves_generated_barrier_controls(tmp_path):
    document = generate_document(
        _config(
            count=1,
            barriers=2,
            buttons_per_barrier=2,
            pressure_plates_per_barrier=2,
            max_attempts=5000,
        )
    )
    path = tmp_path / "barrier-layouts.json"
    path.write_text(json.dumps(document), encoding="utf-8")

    layout = load_layouts_from_json(path)["test_kitchen_0"]

    assert [active for _, _, active in layout.barrier_info] == [True, True]
    button_targets = [
        target
        for _, _, target_idxs, _ in layout.button_info
        for target in target_idxs
    ]
    assert button_targets.count(0) == 2
    assert button_targets.count(1) == 2
    assert all(
        action_type == int(layout_generator.ButtonAction.TIMED_BARRIER)
        for _, _, _, action_type in layout.button_info
    )
    targets = [
        target
        for _, _, target_idxs, _ in layout.pressure_plate_info
        for target in target_idxs
    ]
    assert targets.count(0) == 2
    assert targets.count(1) == 2


def test_complete_each_rejects_insufficient_workstation_copies():
    with pytest.raises(ValueError, match="pots >= num_regions"):
        generate_document(
            _config(
                num_regions=2,
                workflow_mode="complete_each",
                ingredient_piles=[2, 2],
                pots=1,
                plate_piles=2,
                depots=2,
            )
        )


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


def test_json_loader_accepts_legacy_grid_key(tmp_path):
    document = generate_document(_config(count=1))
    entry = document["layouts"]["test_kitchen_0"]
    entry["grid"] = entry.pop("ascii")
    path = tmp_path / "legacy-layouts.json"
    path.write_text(json.dumps(document), encoding="utf-8")

    loaded = load_layouts_from_json(path)

    assert loaded["test_kitchen_0"].width == 8


def test_json_loader_error_mentions_both_supported_grid_keys(tmp_path):
    document = {
        "layouts": {
            "bad": {
                "grid": ["not", "a", "string"],
                "possible_recipes": [[0, 0, 0]],
            }
        }
    }
    path = tmp_path / "bad-grid.json"
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="must contain an 'ascii' or 'grid' string",
    ):
        load_layouts_from_json(path)


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


def test_accessibility_accepts_floor_reachable_through_controlled_barrier():
    """A reachable pressure plate makes the floor beyond its barrier reachable."""
    grid = "\n".join(
        [
            "WWPWWWW",
            "0A_#  X",
            "WA WWWW",
            "WWBWWWW",
        ]
    )
    layout = Layout.from_string(
        grid,
        possible_recipes=[[0, 0, 0]],
        barrier_config=[True],
        pressure_plate_config=[(0, layout_generator.ButtonAction.TOGGLE_BARRIER)],
    )

    assert validate_generated_layout(layout) == (True, [])


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


def test_generator_omits_recipe_indicator_for_one_fixed_recipe():
    document = generate_document(
        _config(
            count=1,
            ingredient_piles=[1],
            possible_recipes=[[0, 0, 0]],
        )
    )

    assert "R" not in document["layouts"]["test_kitchen_0"]["ascii"]


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
    workstation_symbols = set("012PBXR")
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
                if symbol not in set("012PBXR"):
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
