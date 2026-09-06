import json
import random
from collections import Counter, deque
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


REGION_KEYS = {
    "ingredient_piles",
    "pots",
    "plate_piles",
    "depots",
    "recipe_indicators",
}
COMMON_CONFIG_KEYS = {
    "seed",
    "count",
    "name_prefix",
    "width",
    "height",
    "possible_recipes",
    "counter_density",
    "map_type",
    "regions",
    "randomize_agents",
    "max_attempts",
}
WORKSTATION_SYMBOLS = set("0123456789PBXR")
WALKABLE_SYMBOLS = {" ", "A", "_"}


def _region(
    ingredient_piles,
    *,
    pots=0,
    plate_piles=0,
    depots=0,
    recipe_indicators=0,
):
    """Build one exact per-region workstation specification."""
    return {
        "ingredient_piles": list(ingredient_piles),
        "pots": pots,
        "plate_piles": plate_piles,
        "depots": depots,
        "recipe_indicators": recipe_indicators,
    }


def _asymmetric_config(**overrides):
    """Return a feasible two-region asymmetric-information configuration."""
    config = {
        "seed": 11,
        "count": 2,
        "name_prefix": "asymmetric_kitchen",
        "width": 14,
        "height": 10,
        "possible_recipes": [[0, 0, 0], [1, 1, 1]],
        "counter_density": 0.38,
        "map_type": "asymmetric_info",
        "regions": [
            _region([2, 0], recipe_indicators=1),
            _region([0, 1], pots=1, plate_piles=1, depots=1),
        ],
        "handoff_tiles": 2,
        "randomize_agents": False,
        "max_attempts": 5000,
    }
    config.update(overrides)
    return config


def _temporal_config(**overrides):
    """Return a Temporal configuration with one pot shared by both regions."""
    config = {
        "seed": 17,
        "count": 1,
        "name_prefix": "temporal_kitchen",
        "width": 14,
        "height": 10,
        "possible_recipes": [[0, 0, 0], [1, 1, 1]],
        "counter_density": 0.45,
        "map_type": "temporal",
        "regions": [
            _region([0, 0], pots=1),
            _region(
                [1, 1],
                plate_piles=1,
                depots=1,
                recipe_indicators=1,
            ),
        ],
        "signal_tiles": 1,
        "randomize_agents": False,
        "max_attempts": 5000,
    }
    config.update(overrides)
    return config


def _selection_config(**overrides):
    """Return a feasible control/main/two-gated-room configuration."""
    config = {
        "seed": 23,
        "count": 1,
        "name_prefix": "selection_kitchen",
        "width": 18,
        "height": 14,
        "possible_recipes": [[0, 0, 0], [1, 1, 1]],
        "counter_density": 0.38,
        "map_type": "selection",
        "regions": [
            _region([0, 0]),
            _region(
                [0, 0],
                pots=1,
                plate_piles=1,
                depots=1,
                recipe_indicators=1,
            ),
            _region([1, 0]),
            _region([0, 1]),
        ],
        "control_handoff_tiles": 1,
        "pressure_plates_per_barrier": 1,
        "buttons_per_barrier": 1,
        "randomize_agents": False,
        "max_attempts": 5000,
    }
    config.update(overrides)
    return config


def _document(config):
    """Wrap generator settings in the public JSON document shape."""
    return {"generator": config, "layouts": {}}


def _neighbours(rows, position):
    """Return the in-bounds four-neighbours of a grid position."""
    row, col = position
    height, width = len(rows), len(rows[0])
    return [
        (next_row, next_col)
        for next_row, next_col in (
            (row - 1, col),
            (row + 1, col),
            (row, col - 1),
            (row, col + 1),
        )
        if 0 <= next_row < height and 0 <= next_col < width
    ]


def _floor_components(grid):
    """Return rows, floor components, and a position-to-component index."""
    rows = grid.splitlines()
    remaining = {
        (row, col)
        for row, line in enumerate(rows)
        for col, symbol in enumerate(line)
        if symbol in WALKABLE_SYMBOLS
    }
    components = []
    while remaining:
        start = min(remaining)
        component = {start}
        queue = deque([start])
        remaining.remove(start)
        while queue:
            position = queue.popleft()
            for adjacent in _neighbours(rows, position):
                if adjacent in remaining:
                    remaining.remove(adjacent)
                    component.add(adjacent)
                    queue.append(adjacent)
        components.append(component)
    component_by_position = {
        position: component_idx
        for component_idx, component in enumerate(components)
        for position in component
    }
    return rows, components, component_by_position


def _adjacent_components(rows, component_by_position, position):
    """Return floor-component indexes adjacent to one static grid tile."""
    return {
        component_by_position[adjacent]
        for adjacent in _neighbours(rows, position)
        if adjacent in component_by_position
    }


def _positions_with_symbol(rows, symbols):
    """Return positions containing any requested ASCII symbol."""
    return {
        (row, col)
        for row, line in enumerate(rows)
        for col, symbol in enumerate(line)
        if symbol in symbols
    }


def _agent_component_indexes(layout, component_by_position):
    """Map Layout agent order to the floor component containing each agent."""
    return [
        component_by_position[(agent_y, agent_x)]
        for agent_x, agent_y in layout.agent_positions
    ]


def _accessible_workstation_positions(rows, component):
    """Return unique workstation tiles interactable from a floor component."""
    return {
        adjacent
        for floor_position in component
        for adjacent in _neighbours(rows, floor_position)
        if rows[adjacent[0]][adjacent[1]] in WORKSTATION_SYMBOLS
    }


def _workstation_signature(rows, component):
    """Count workstation symbols interactable from a floor component."""
    return Counter(
        rows[row][col]
        for row, col in _accessible_workstation_positions(rows, component)
    )


def _expected_workstation_signature(region):
    """Translate a region dictionary into its expected ASCII symbol counts."""
    signature = Counter()
    for ingredient_idx, count in enumerate(region["ingredient_piles"]):
        signature[str(ingredient_idx)] = count
    signature["P"] = region["pots"]
    signature["B"] = region["plate_piles"]
    signature["X"] = region["depots"]
    signature["R"] = region["recipe_indicators"]
    return +signature


def _shared_static_tiles(
    rows,
    component_by_position,
    first_component,
    second_component,
    symbols,
):
    """Find static tiles interactable from exactly two specified regions."""
    expected = {first_component, second_component}
    return {
        position
        for position in _positions_with_symbol(rows, symbols)
        if _adjacent_components(rows, component_by_position, position) == expected
    }


def _assert_exact_region_workstations(rows, components, assignments, regions):
    """Assert exact per-region workstation placement for component indexes."""
    assert len(assignments) == len(regions)
    for component_idx, region in zip(assignments, regions):
        assert _workstation_signature(
            rows,
            components[component_idx],
        ) == _expected_workstation_signature(region)


def _only_entry(document):
    """Return the sole generated layout entry in a one-layout document."""
    assert len(document["layouts"]) == 1
    return next(iter(document["layouts"].values()))


def _assert_exact_interior_counter_density(rows, config):
    """Assert that mandatory and decorative counters share one exact budget."""
    actual = sum(
        rows[row][col] == "W"
        for row in range(1, len(rows) - 1)
        for col in range(1, len(rows[0]) - 1)
    )
    expected = round(
        (config["width"] - 2)
        * (config["height"] - 2)
        * config["counter_density"]
    )
    assert actual == expected


def test_normalized_configs_contain_only_common_and_relevant_mode_fields():
    """Validation does not reintroduce removed or irrelevant schema fields."""
    cases = [
        (_asymmetric_config(), {"handoff_tiles"}),
        (_temporal_config(), {"signal_tiles"}),
        (
            _selection_config(),
            {
                "control_handoff_tiles",
                "pressure_plates_per_barrier",
                "buttons_per_barrier",
            },
        ),
    ]
    for raw_config, mode_keys in cases:
        config = validate_config(raw_config)
        assert set(config) == COMMON_CONFIG_KEYS | mode_keys
        assert all(set(region) == REGION_KEYS for region in config["regions"])


def test_common_defaults_apply_without_injecting_other_mode_fields():
    """Only genuinely common convenience settings receive defaults."""
    config = validate_config(
        {
            "map_type": "asymmetric_info",
            "regions": [
                _region([1]),
                _region([0], pots=1, plate_piles=1, depots=1),
            ],
            "handoff_tiles": 1,
        }
    )

    assert config["seed"] == 0
    assert config["count"] == 1
    assert config["name_prefix"] == "generated"
    assert config["width"] == 8
    assert config["height"] == 6
    assert config["counter_density"] == 0.1
    assert config["randomize_agents"] is False
    assert config["max_attempts"] == 1000
    assert config["possible_recipes"] == [[0, 0, 0]]
    assert set(config) == COMMON_CONFIG_KEYS | {"handoff_tiles"}


@pytest.mark.parametrize(
    ("factory", "required_key"),
    [
        (_asymmetric_config, "map_type"),
        (_asymmetric_config, "regions"),
        (_asymmetric_config, "handoff_tiles"),
        (_temporal_config, "signal_tiles"),
        (_selection_config, "control_handoff_tiles"),
        (_selection_config, "pressure_plates_per_barrier"),
        (_selection_config, "buttons_per_barrier"),
    ],
)
def test_map_identity_and_mode_specific_fields_are_required(factory, required_key):
    """The structural settings cannot silently acquire cross-mode defaults."""
    config = factory()
    config.pop(required_key)

    with pytest.raises(ValueError, match=required_key):
        validate_config(config)


@pytest.mark.parametrize(
    ("old_key", "value"),
    [
        ("ingredient_piles", [1]),
        ("pots", 1),
        ("plate_piles", 1),
        ("depots", 1),
        ("object_placement", "anywhere"),
        ("num_regions", 2),
        ("num_shared_tiles", 1),
        ("workflow_mode", "shared"),
        ("barriers", 1),
        ("barrier_placement", "shared"),
    ],
)
def test_generator_rejects_removed_top_level_settings(old_key, value):
    """The breaking schema rejects every superseded top-level setting."""
    config = _asymmetric_config()
    config[old_key] = value

    with pytest.raises(ValueError, match=old_key):
        validate_config(config)


@pytest.mark.parametrize(
    ("factory", "irrelevant_key", "value"),
    [
        (_asymmetric_config, "signal_tiles", 1),
        (_asymmetric_config, "control_handoff_tiles", 1),
        (_asymmetric_config, "pressure_plates_per_barrier", 1),
        (_asymmetric_config, "buttons_per_barrier", 0),
        (_temporal_config, "handoff_tiles", 1),
        (_temporal_config, "control_handoff_tiles", 1),
        (_temporal_config, "pressure_plates_per_barrier", 1),
        (_temporal_config, "buttons_per_barrier", 0),
        (_selection_config, "handoff_tiles", 1),
        (_selection_config, "signal_tiles", 1),
    ],
)
def test_generator_rejects_fields_from_another_map_type(
    factory,
    irrelevant_key,
    value,
):
    """Mode-specific settings are accepted only by their owning map type."""
    config = factory()
    config[irrelevant_key] = value

    with pytest.raises(ValueError, match=irrelevant_key):
        validate_config(config)


@pytest.mark.parametrize("map_type", ["shared", "complete_each", "unknown", 1])
def test_generator_rejects_invalid_map_type(map_type):
    """Only the three deliberate map types are accepted."""
    config = _asymmetric_config(map_type=map_type)

    with pytest.raises(ValueError, match="map_type"):
        validate_config(config)


@pytest.mark.parametrize("value", [0, 1, None, "false", []])
def test_randomize_agents_must_be_boolean(value):
    """Agent-role randomization does not accept truthy integer substitutes."""
    config = _asymmetric_config(randomize_agents=value)

    with pytest.raises(ValueError, match="randomize_agents"):
        validate_config(config)


@pytest.mark.parametrize("missing_key", sorted(REGION_KEYS))
def test_every_region_requires_every_workstation_count(missing_key):
    """Region dictionaries are strict and cannot omit zero-valued fields."""
    config = _asymmetric_config()
    config["regions"][0].pop(missing_key)

    with pytest.raises(ValueError, match=missing_key):
        validate_config(config)


def test_region_dictionaries_reject_unknown_fields():
    """Typos in a region workstation dictionary fail validation."""
    config = _asymmetric_config()
    config["regions"][0]["pot"] = 1

    with pytest.raises(ValueError, match="pot"):
        validate_config(config)


@pytest.mark.parametrize(
    "field",
    ["pots", "plate_piles", "depots", "recipe_indicators"],
)
@pytest.mark.parametrize("value", [-1, 1.5, True, "1"])
def test_region_scalar_counts_must_be_non_negative_integers(field, value):
    """All scalar workstation amounts are strict non-negative integers."""
    config = _asymmetric_config()
    config["regions"][0][field] = value

    with pytest.raises(ValueError, match=field):
        validate_config(config)


@pytest.mark.parametrize(
    "value",
    [None, [], [1, -1], [1, True], [1, 0.5], ["1", 0]],
)
def test_region_ingredient_counts_are_strict_non_negative_integer_lists(value):
    """Ingredient counts retain their indexed fixed-length list representation."""
    config = _asymmetric_config()
    config["regions"][0]["ingredient_piles"] = value

    with pytest.raises(ValueError, match="ingredient_piles"):
        validate_config(config)


def test_all_regions_must_define_the_same_ingredient_array_length():
    """Ingredient indexes have one consistent meaning across all regions."""
    config = _asymmetric_config()
    config["regions"][1]["ingredient_piles"] = [1]

    with pytest.raises(ValueError, match="ingredient_piles"):
        validate_config(config)


@pytest.mark.parametrize(
    ("factory", "region_count"),
    [
        (_asymmetric_config, 1),
        (_asymmetric_config, 3),
        (_temporal_config, 1),
        (_temporal_config, 3),
        (_selection_config, 2),
    ],
)
def test_map_types_enforce_their_region_count(factory, region_count):
    """Two-region modes and the three-or-more-region mode reject bad arity."""
    config = factory()
    if region_count < len(config["regions"]):
        config["regions"] = config["regions"][:region_count]
        config["possible_recipes"] = [[0, 0, 0]]
    else:
        while len(config["regions"]) < region_count:
            config["regions"].append(_region([0, 0]))

    with pytest.raises(ValueError, match="regions"):
        validate_config(config)


@pytest.mark.parametrize("value", [-1, 0, 1.5, True, "2"])
def test_asymmetric_handoff_count_must_be_a_positive_integer(value):
    """Asymmetric information always has an explicit nonempty handoff."""
    config = _asymmetric_config(handoff_tiles=value)

    with pytest.raises(ValueError, match="handoff_tiles"):
        validate_config(config)


@pytest.mark.parametrize("value", [-1, 0, 2, 1.5, True, "1"])
def test_temporal_signal_count_must_equal_region_zero_pots(value):
    """Each Temporal region-zero pot consumes exactly one shared signal tile."""
    config = _temporal_config(signal_tiles=value)

    with pytest.raises(ValueError, match="signal_tiles"):
        validate_config(config)


def test_temporal_requires_a_region_zero_shared_pot():
    """Region-one-only pots cannot replace the required shared interface pot."""
    config = _temporal_config(
        regions=[
            _region([0, 0]),
            _region(
                [1, 1],
                pots=1,
                plate_piles=1,
                depots=1,
                recipe_indicators=1,
            ),
        ]
    )

    with pytest.raises(ValueError, match="at least one region 0 pot"):
        validate_config(config)


def test_temporal_rejects_non_pot_stations_split_between_regions():
    """A plated soup cannot cross the interface between plate and depot."""
    config = _temporal_config(
        regions=[
            _region([1, 1], pots=1, plate_piles=1),
            _region(
                [0, 0],
                depots=1,
                recipe_indicators=1,
            ),
        ],
        signal_tiles=1,
    )

    with pytest.raises(
        ValueError,
        match=(
            r"cannot complete recipe \[0, 0, 0\].*"
            r"region 0 is missing depot.*"
            r"region 1 is missing plate pile"
        ),
    ):
        validate_config(config)


def test_temporal_example_configuration_generates_successfully():
    """The shipped Temporal example uses its signal pot from region one."""
    example_path = Path(layout_generator.__file__).with_name(
        "overcooked_v3_layouts.example.json"
    )
    document = json.loads(example_path.read_text(encoding="utf-8"))
    config = validate_config(document["generator"])

    grid, layout, _ = generate_layout(config, random.Random(config["seed"]))

    assert grid.count("P") == 1
    assert validate_generated_layout(layout) == (True, [])


def test_temporal_shared_pot_joins_ingredient_and_delivery_workflows():
    """One side may fill a shared pot while the other plates and delivers."""
    config = validate_config(
        _temporal_config(
            possible_recipes=[[0, 0, 0]],
            regions=[
                _region([1], pots=1),
                _region([0], plate_piles=1, depots=1),
            ],
        )
    )

    grid, layout, _ = generate_layout(config, random.Random(config["seed"]))
    rows, components, component_by_position = _floor_components(grid)
    region_zero, region_one = _agent_component_indexes(
        layout, component_by_position
    )

    assert _shared_static_tiles(
        rows,
        component_by_position,
        region_zero,
        region_one,
        {"P"},
    )
    assert validate_generated_layout(layout) == (True, [])


@pytest.mark.parametrize("value", [-1, 0, 1.5, True, "1"])
def test_selection_control_handoff_count_must_be_a_positive_integer(value):
    """The control and main rooms always have an explicit counter handoff."""
    config = _selection_config(control_handoff_tiles=value)

    with pytest.raises(ValueError, match="control_handoff_tiles"):
        validate_config(config)


@pytest.mark.parametrize("value", [-1, 3, 1.5, True, "2"])
def test_selection_rejects_invalid_pressure_plate_multiplicity(value):
    """Pressure-plate multiplicity retains the supported zero/one/two values."""
    config = _selection_config(pressure_plates_per_barrier=value)

    with pytest.raises(ValueError, match="pressure_plates_per_barrier"):
        validate_config(config)


@pytest.mark.parametrize("value", [-1, 1.5, True, "2"])
def test_selection_rejects_invalid_button_multiplicity(value):
    """Button multiplicity must be a strict non-negative integer."""
    config = _selection_config(buttons_per_barrier=value)

    with pytest.raises(ValueError, match="buttons_per_barrier"):
        validate_config(config)


def test_selection_requires_at_least_one_control_per_derived_barrier():
    """Every gated room's derived barrier must have a generated control."""
    config = _selection_config(
        pressure_plates_per_barrier=0,
        buttons_per_barrier=0,
    )

    with pytest.raises(ValueError, match="pressure plate or button"):
        validate_config(config)


def test_generator_is_deterministic_and_produces_exact_valid_layouts():
    """A seed reproduces exact-size maps and exact requested object totals."""
    first = generate_document(_document(_asymmetric_config()))
    second = generate_document(_document(_asymmetric_config()))

    assert first == second
    assert len(first["layouts"]) == 2
    expected = sum(
        (
            _expected_workstation_signature(region)
            for region in first["generator"]["regions"]
        ),
        Counter(),
    )
    for entry in first["layouts"].values():
        grid = entry["ascii"]
        rows = grid.splitlines()
        actual = Counter(symbol for symbol in grid if symbol in WORKSTATION_SYMBOLS)

        assert len(rows) == first["generator"]["height"]
        assert all(len(row) == first["generator"]["width"] for row in rows)
        assert grid.count("A") == 2
        assert actual == expected

        layout = Layout.from_string(
            grid,
            possible_recipes=entry["possible_recipes"],
            swap_agents=entry["swap_agents"],
        )
        assert validate_generated_layout(layout) == (True, [])
        assert entry["validation"] == {"valid": True, "errors": []}


def test_asymmetric_information_has_two_exact_regions_and_handoff_tiles():
    """Asymmetric maps isolate agents but expose exact single-counter handoffs."""
    config = validate_config(_asymmetric_config(count=1))
    grid, layout, _ = generate_layout(config, random.Random(config["seed"]))
    rows, components, component_by_position = _floor_components(grid)
    agent_components = _agent_component_indexes(layout, component_by_position)

    assert len(components) == 2
    assert len(set(agent_components)) == 2
    _assert_exact_region_workstations(
        rows,
        components,
        agent_components,
        config["regions"],
    )
    handoffs = _shared_static_tiles(
        rows,
        component_by_position,
        agent_components[0],
        agent_components[1],
        {"W"},
    )
    assert len(handoffs) == config["handoff_tiles"]
    assert not _positions_with_symbol(rows, {"#", "_", "!"})
    _assert_exact_interior_counter_density(rows, config)
    assert validate_generated_layout(layout) == (True, [])


def test_temporal_pots_are_the_only_shared_tiles_between_regions():
    """Region-zero pots bridge both rooms while every other tile stays private."""
    config = validate_config(
        _temporal_config(
            regions=[
                _region(
                    [1, 0],
                    pots=2,
                    plate_piles=1,
                    depots=1,
                    recipe_indicators=1,
                ),
                _region(
                    [1, 1],
                    plate_piles=1,
                    depots=1,
                    recipe_indicators=1,
                ),
            ],
            signal_tiles=2,
        )
    )
    grid, layout, _ = generate_layout(config, random.Random(config["seed"]))
    rows, components, component_by_position = _floor_components(grid)
    agent_components = _agent_component_indexes(layout, component_by_position)
    region_zero, region_one = agent_components

    assert len(components) == 2
    assert len(set(agent_components)) == 2
    assert _workstation_signature(
        rows,
        components[region_zero],
    ) == _expected_workstation_signature(config["regions"][0])
    assert _workstation_signature(
        rows,
        components[region_one],
    ) == (
        _expected_workstation_signature(config["regions"][1])
        + Counter({"P": config["regions"][0]["pots"]})
    )

    shared_pots = _shared_static_tiles(
        rows,
        component_by_position,
        region_zero,
        region_one,
        {"P"},
    )
    assert len(shared_pots) == config["signal_tiles"]
    assert len(shared_pots) == config["regions"][0]["pots"]
    assert _shared_static_tiles(
        rows,
        component_by_position,
        region_zero,
        region_one,
        WORKSTATION_SYMBOLS,
    ) == shared_pots

    for pot_position in shared_pots:
        neighbours_by_component = {
            component: [
                position
                for position in _neighbours(rows, pot_position)
                if component_by_position.get(position) == component
            ]
            for component in (region_zero, region_one)
        }
        assert all(neighbours_by_component.values())
        assert any(
            first_row + second_row == 2 * pot_position[0]
            and first_col + second_col == 2 * pot_position[1]
            for first_row, first_col in neighbours_by_component[region_zero]
            for second_row, second_col in neighbours_by_component[region_one]
        )
        assert _adjacent_components(
            rows,
            component_by_position,
            pot_position,
        ) == {region_zero, region_one}

    # No ordinary counter may form a one-tile item-passing interface.
    assert not _shared_static_tiles(
        rows,
        component_by_position,
        region_zero,
        region_one,
        {"W"},
    )
    assert not _positions_with_symbol(rows, {"#", "_", "!"})
    _assert_exact_interior_counter_density(rows, config)
    assert validate_generated_layout(layout) == (True, [])


def test_selection_builds_control_main_and_one_barrier_per_gated_room():
    """Selection topology is a controlled star rooted at the main room."""
    config = validate_config(_selection_config())
    grid, layout, _ = generate_layout(config, random.Random(config["seed"]))
    rows, components, component_by_position = _floor_components(grid)
    control_component, main_component = _agent_component_indexes(
        layout,
        component_by_position,
    )
    gated_components = set(range(len(components))) - {
        control_component,
        main_component,
    }

    assert len(components) == len(config["regions"])
    assert len(gated_components) == len(config["regions"]) - 2
    assert _workstation_signature(
        rows,
        components[control_component],
    ) == _expected_workstation_signature(config["regions"][0])
    assert _workstation_signature(
        rows,
        components[main_component],
    ) == _expected_workstation_signature(config["regions"][1])
    actual_gated_signatures = Counter(
        tuple(sorted(_workstation_signature(rows, components[idx]).items()))
        for idx in gated_components
    )
    expected_gated_signatures = Counter(
        tuple(sorted(_expected_workstation_signature(region).items()))
        for region in config["regions"][2:]
    )
    assert actual_gated_signatures == expected_gated_signatures

    barrier_positions = sorted(_positions_with_symbol(rows, {"#"}))
    gated_barrier_counts = Counter()
    for barrier_position in barrier_positions:
        adjacent = _adjacent_components(
            rows,
            component_by_position,
            barrier_position,
        )
        assert main_component in adjacent
        assert control_component not in adjacent
        assert len(adjacent) == 2
        gated_component = next(iter(adjacent - {main_component}))
        assert gated_component in gated_components
        gated_barrier_counts[gated_component] += 1
    assert len(barrier_positions) == len(gated_components)
    assert gated_barrier_counts == Counter({idx: 1 for idx in gated_components})

    handoffs = _shared_static_tiles(
        rows,
        component_by_position,
        control_component,
        main_component,
        {"W"},
    )
    assert len(handoffs) == config["control_handoff_tiles"]
    for first_component in range(len(components)):
        for second_component in range(first_component + 1, len(components)):
            shared = _shared_static_tiles(
                rows,
                component_by_position,
                first_component,
                second_component,
                {"W"},
            )
            if {first_component, second_component} == {
                control_component,
                main_component,
            }:
                assert shared == handoffs
            else:
                assert not shared
    for counter_position in _positions_with_symbol(rows, {"W"}):
        adjacent = _adjacent_components(
            rows,
            component_by_position,
            counter_position,
        )
        if len(adjacent) > 1:
            assert adjacent == {control_component, main_component}
    _assert_exact_interior_counter_density(rows, config)
    assert validate_generated_layout(layout) == (True, [])


def test_selection_controls_are_all_in_control_room_and_wired_one_to_one():
    """Every control is reachable only by agent zero and targets one barrier."""
    config = validate_config(_selection_config())
    grid, layout, _ = generate_layout(config, random.Random(config["seed"]))
    rows, components, component_by_position = _floor_components(grid)
    control_component = _agent_component_indexes(layout, component_by_position)[0]
    barrier_count = len(config["regions"]) - 2

    pressure_plate_positions = _positions_with_symbol(rows, {"_"})
    assert len(pressure_plate_positions) == (
        barrier_count * config["pressure_plates_per_barrier"]
    )
    assert all(
        component_by_position[position] == control_component
        for position in pressure_plate_positions
    )

    button_positions = _positions_with_symbol(rows, {"!"})
    assert len(button_positions) == barrier_count * config["buttons_per_barrier"]
    assert all(
        _adjacent_components(rows, component_by_position, position)
        == {control_component}
        for position in button_positions
    )

    pressure_targets = Counter()
    for _, _, targets, action_type in layout.pressure_plate_info:
        assert len(targets) == 1
        assert action_type == int(layout_generator.ButtonAction.TOGGLE_BARRIER)
        pressure_targets[targets[0]] += 1
    assert pressure_targets == Counter(
        {
            barrier_idx: config["pressure_plates_per_barrier"]
            for barrier_idx in range(barrier_count)
        }
    )

    button_targets = Counter()
    for _, _, targets, action_type in layout.button_info:
        assert len(targets) == 1
        assert action_type == int(layout_generator.ButtonAction.TIMED_BARRIER)
        button_targets[targets[0]] += 1
    assert button_targets == Counter(
        {
            barrier_idx: config["buttons_per_barrier"]
            for barrier_idx in range(barrier_count)
        }
    )
    assert [targets[0] for _, _, targets, _ in layout.pressure_plate_info] == list(
        range(barrier_count)
    )
    assert [targets[0] for _, _, targets, _ in layout.button_info] == list(
        range(barrier_count)
    )
    assert [active for _, _, active in layout.barrier_info] == [True] * barrier_count


def test_temporal_rejects_a_counter_budget_too_small_for_isolation():
    """Double-counter isolation is charged to, and constrained by, density."""
    config = _temporal_config(
        width=8,
        height=6,
        counter_density=0,
        max_attempts=2,
    )

    with pytest.raises((ValueError, RuntimeError), match="counter|mandatory|separat"):
        generate_document(_document(config))


def test_fixed_agent_roles_follow_region_order():
    """Without randomization, agent IDs map to region dictionaries in order."""
    config = validate_config(_asymmetric_config(count=1, randomize_agents=False))
    grid, layout, _ = generate_layout(config, random.Random(config["seed"]))
    rows, components, component_by_position = _floor_components(grid)
    assignments = _agent_component_indexes(layout, component_by_position)

    _assert_exact_region_workstations(
        rows,
        components,
        assignments,
        config["regions"],
    )


def test_randomized_agent_roles_can_swap_between_the_two_regions():
    """Randomization produces both possible agent-zero region assignments."""
    raw_config = _asymmetric_config(count=1, randomize_agents=True)
    expected = {
        tuple(sorted(_expected_workstation_signature(region).items()))
        for region in raw_config["regions"]
    }
    observed = set()

    for seed in range(16):
        config = validate_config({**raw_config, "seed": seed})
        grid, layout, _ = generate_layout(config, random.Random(seed))
        rows, components, component_by_position = _floor_components(grid)
        agent_zero_component = _agent_component_indexes(
            layout,
            component_by_position,
        )[0]
        observed.add(
            tuple(
                sorted(
                    _workstation_signature(
                        rows,
                        components[agent_zero_component],
                    ).items()
                )
            )
        )
        if observed == expected:
            break

    assert observed == expected


def test_shipped_example_is_supported_and_generates_a_layout():
    """The checked-in example stays synchronized with the public schema."""
    example_path = (
        Path(__file__).parents[2]
        / "scripts"
        / "overcooked_v3_layouts.example.json"
    )
    document = json.loads(example_path.read_text(encoding="utf-8"))
    config = validate_config(document["generator"])

    grid, layout, _ = generate_layout(config, random.Random(config["seed"]))

    assert config["map_type"] in {"asymmetric_info", "selection", "temporal"}
    assert grid.count("A") == 2
    assert validate_generated_layout(layout) == (True, [])


def test_json_loader_reads_and_runs_generated_layout(tmp_path):
    """Generated JSON remains directly consumable by Overcooked V3."""
    document = generate_document(_document(_asymmetric_config(count=1)))
    path = tmp_path / "layouts.json"
    path.write_text(json.dumps(document), encoding="utf-8")

    loaded = load_layouts_from_json(path)
    layout = loaded["asymmetric_kitchen_0"]

    assert layout.height == 10
    assert layout.width == 14
    assert len(layout.agent_positions) == 2
    assert layout.get_info()["num_ingredient_piles"] == {0: 2, 1: 1}
    assert OvercookedV3(layout=layout).layout is layout


def test_json_loader_preserves_generated_agent_role_order(tmp_path):
    """The serialized swap flag preserves randomized agent IDs on reload."""
    entry = None
    original_layout = None
    for seed in range(16):
        config = validate_config(
            _asymmetric_config(count=1, randomize_agents=True, seed=seed)
        )
        grid, candidate_layout, _ = generate_layout(config, random.Random(seed))
        candidate_entry = layout_generator._layout_entry(
            grid,
            candidate_layout,
            config,
        )
        if candidate_entry["swap_agents"]:
            entry = candidate_entry
            original_layout = candidate_layout
            break

    assert entry is not None
    assert original_layout is not None
    document = {"layouts": {"roles": entry}}
    path = tmp_path / "agent-roles.json"
    path.write_text(json.dumps(document), encoding="utf-8")

    loaded_layout = load_layouts_from_json(path)["roles"]

    assert "swap_agents" in entry
    assert loaded_layout.agent_positions == original_layout.agent_positions


def test_json_loader_preserves_generated_selection_controls(tmp_path):
    """Barrier and control metadata survive generated JSON round trips."""
    document = generate_document(_document(_selection_config()))
    path = tmp_path / "barrier-layouts.json"
    path.write_text(json.dumps(document), encoding="utf-8")

    layout = load_layouts_from_json(path)["selection_kitchen_0"]
    barrier_count = len(document["generator"]["regions"]) - 2

    assert [active for _, _, active in layout.barrier_info] == [True] * barrier_count
    button_targets = [
        target
        for _, _, target_idxs, _ in layout.button_info
        for target in target_idxs
    ]
    pressure_targets = [
        target
        for _, _, target_idxs, _ in layout.pressure_plate_info
        for target in target_idxs
    ]
    assert Counter(button_targets) == Counter(range(barrier_count))
    assert Counter(pressure_targets) == Counter(range(barrier_count))


def test_json_loader_accepts_legacy_grid_key(tmp_path):
    """The historical grid alias remains accepted for generated layouts."""
    document = generate_document(_document(_asymmetric_config(count=1)))
    entry = document["layouts"]["asymmetric_kitchen_0"]
    entry["grid"] = entry.pop("ascii")
    path = tmp_path / "legacy-layouts.json"
    path.write_text(json.dumps(document), encoding="utf-8")

    loaded = load_layouts_from_json(path)

    assert loaded["asymmetric_kitchen_0"].width == 14


def test_json_loader_error_mentions_both_supported_grid_keys(tmp_path):
    """Malformed JSON reports both current and legacy ASCII field names."""
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
    """The general layout validator still rejects unusable disconnected maps."""
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


def test_accessibility_does_not_combine_unrelated_exclusive_pots():
    """Stations around different private pots cannot form one false workflow."""
    grid = "\n".join(
        [
            "WW0PWWW",
            "W A   W",
            "WWWWWWW",
            "WWWWWWW",
            "W A   W",
            "WWPBXWW",
            "WWWWWWW",
        ]
    )
    layout = Layout.from_string(grid, possible_recipes=[[0, 0, 0]])

    valid, errors = validate_generated_layout(layout)

    assert not valid
    assert errors == [
        "Recipe [0, 0, 0] cannot be completed within one "
        "agent-accessible pot workflow"
    ]


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
        pressure_plate_config=[
            (0, layout_generator.ButtonAction.TOGGLE_BARRIER)
        ],
    )

    assert validate_generated_layout(layout) == (True, [])


def test_loader_rejects_invalid_map_even_if_metadata_says_valid(tmp_path):
    """Loader validation trusts the grid rather than cached validation metadata."""
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
    """PMG retains Overcooked V3's homogeneous three-item recipe constraint."""
    config = _asymmetric_config(possible_recipes=[[0, 0, 1]])

    with pytest.raises(ValueError, match="currently supports same-ingredient"):
        generate_document(_document(config))


def test_generator_derives_recipes_from_global_region_ingredients():
    """Omitted recipes derive once from ingredient availability across regions."""
    config = _asymmetric_config()
    config.pop("possible_recipes")

    validated = validate_config(config)

    assert validated["possible_recipes"] == [[0, 0, 0], [1, 1, 1]]


def test_generator_respects_explicit_zero_recipe_indicators():
    """A fixed recipe permits exact zero recipe-indicator workstations."""
    regions = [
        _region([1]),
        _region([0], pots=1, plate_piles=1, depots=1),
    ]
    document = generate_document(
        _document(
            _asymmetric_config(
                count=1,
                possible_recipes=[[0, 0, 0]],
                regions=regions,
                handoff_tiles=1,
            )
        )
    )

    assert "R" not in _only_entry(document)["ascii"]


def test_interactive_player_registers_json_layouts(tmp_path):
    """The play script can register a generated layout document."""
    document = generate_document(
        _document(
            _asymmetric_config(
                count=1,
                name_prefix="interactive_test_kitchen",
            )
        )
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
    """The CLI backend preserves completed layouts around an isolated failure."""
    output_path = tmp_path / "checkpointed-layouts.json"
    original_generate_layout = layout_generator.generate_layout
    call_count = 0

    def fail_second_layout(config, rng):
        """Raise only for the second requested layout."""
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
        _document(
            _asymmetric_config(count=3, name_prefix="checkpoint_test")
        ),
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
