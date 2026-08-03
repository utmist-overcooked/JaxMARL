import json
import random
from collections import deque
from pathlib import Path

import jax
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
            if symbol in set("01234PBXCGMSD"):
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


def test_generator_places_all_new_workstations_and_runs_combined_environment():
    config = validate_config(
        _config(
            count=1,
            width=12,
            height=10,
            ingredient_piles=[0, 0, 1, 1, 1],
            possible_recipes=[[5, 5, 5], [6, 6, 6], [7, 7, 7]],
            cutting_boards=1,
            grills=1,
            blenders=1,
            sinks=1,
            dirty_plate_piles=1,
            object_placement="interior",
            counter_density=0.1,
        )["generator"]
    )

    grid, layout, _ = generate_layout(config, random.Random(config["seed"]))
    rows = grid.splitlines()

    for symbol in "234CGMSD":
        assert grid.count(symbol) == 1
    for row, line in enumerate(rows):
        for col, symbol in enumerate(line):
            if symbol in set("234CGMSD"):
                assert 0 < row < len(rows) - 1
                assert 0 < col < len(line) - 1

    assert validate_generated_layout(layout) == (True, [])
    env = OvercookedV3(layout=layout, enable_dish_washing=True)
    obs, state = env.reset(jax.random.PRNGKey(0))
    assert set(obs) == set(env.agents)
    assert int(state.plate_stack_count) == env.num_plates


def test_generator_auto_recipes_use_processed_ingredients_when_station_exists():
    config = validate_config(
        _config(
            ingredient_piles=[0, 0, 1, 1, 1],
            possible_recipes=None,
            cutting_boards=1,
            grills=1,
            blenders=1,
        )["generator"]
    )

    assert config["possible_recipes"] == [
        [5, 5, 5],
        [6, 6, 6],
        [7, 7, 7],
    ]


@pytest.mark.parametrize(
    ("ingredient_piles", "recipe", "setting"),
    [
        ([0, 0, 1], [5, 5, 5], "cutting_boards"),
        ([0, 0, 0, 1], [6, 6, 6], "grills"),
        ([0, 0, 0, 0, 1], [7, 7, 7], "blenders"),
    ],
)
def test_processed_recipe_requires_matching_prep_station(
    ingredient_piles,
    recipe,
    setting,
):
    with pytest.raises(ValueError, match=setting):
        validate_config(
            _config(
                ingredient_piles=ingredient_piles,
                possible_recipes=[recipe],
            )["generator"]
        )


@pytest.mark.parametrize(
    ("sinks", "dirty_plate_piles"),
    [(1, 0), (0, 1)],
)
def test_generator_requires_sink_and_dirty_pile_together(
    sinks,
    dirty_plate_piles,
):
    with pytest.raises(ValueError, match="must either both be 0 or both be positive"):
        validate_config(
            _config(
                sinks=sinks,
                dirty_plate_piles=dirty_plate_piles,
            )["generator"]
        )


def test_complete_each_requires_prep_and_dish_workstations_per_region():
    base = _config(
        num_regions=2,
        workflow_mode="complete_each",
        ingredient_piles=[0, 0, 2],
        possible_recipes=[[5, 5, 5]],
        cutting_boards=1,
        pots=2,
        plate_piles=2,
        depots=2,
        sinks=2,
        dirty_plate_piles=2,
    )["generator"]
    with pytest.raises(ValueError, match="prep station"):
        validate_config(base)

    base["cutting_boards"] = 2
    base["sinks"] = 1
    with pytest.raises(ValueError, match="generator.sinks >= num_regions"):
        validate_config(base)


def test_complete_each_places_full_prep_and_dish_workflow_in_every_region():
    config = validate_config(
        _config(
            count=1,
            width=14,
            height=10,
            ingredient_piles=[0, 0, 2],
            possible_recipes=[[5, 5, 5]],
            cutting_boards=2,
            pots=2,
            plate_piles=2,
            depots=2,
            sinks=2,
            dirty_plate_piles=2,
            counter_density=0.4,
            num_regions=2,
            workflow_mode="complete_each",
        )["generator"]
    )

    grid, layout, _ = generate_layout(config, random.Random(config["seed"]))
    rows, components = _floor_components(grid)

    assert len(components) == 2
    for component in components:
        assert set("2CPBXSD") <= _accessible_symbols(rows, component)
    assert validate_generated_layout(layout) == (True, [])


def test_shared_workflow_supports_prep_and_dish_washing_stages():
    config = validate_config(
        _config(
            count=1,
            width=12,
            height=10,
            ingredient_piles=[0, 0, 1],
            possible_recipes=[[5, 5, 5]],
            cutting_boards=1,
            sinks=1,
            dirty_plate_piles=1,
            counter_density=0.4,
            num_regions=2,
            workflow_mode="shared",
        )["generator"]
    )

    grid, layout, _ = generate_layout(config, random.Random(config["seed"]))

    assert grid.count("C") == 1
    assert grid.count("S") == 1
    assert grid.count("D") == 1
    assert _shared_tiles(grid)
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
