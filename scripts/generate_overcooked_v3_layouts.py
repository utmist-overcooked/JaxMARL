"""Generate validated Overcooked V3 layouts for deliberate workflow modes.

Run::

    python scripts/generate_overcooked_v3_layouts.py layouts.json

The input file is updated in place by default. Generation settings remain in
``generator`` and generated layouts are written to ``layouts``. Pass
``--output`` to preserve the input file.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from jaxmarl.environments.overcooked_v3.common import ButtonAction, MAX_INGREDIENTS
from jaxmarl.environments.overcooked_v3.layouts import (
    Layout,
    validate_generated_layout,
)
from jaxmarl.environments.overcooked_v3.settings import (
    MAX_BARRIERS,
    MAX_BUTTONS,
    MAX_POTS,
    MAX_PRESSURE_PLATES,
)


DEFAULTS = {
    "seed": 0,
    "count": 1,
    "name_prefix": "generated",
    "width": 8,
    "height": 6,
    "counter_density": 0.1,
    "randomize_agents": False,
    "max_attempts": 1000,
}

MAP_TYPES = {"asymmetric_info", "selection", "temporal"}
WORKSTATION_KEYS = {
    "ingredient_piles",
    "pots",
    "plate_piles",
    "depots",
    "recipe_indicators",
}
MODE_FIELDS = {
    "asymmetric_info": {"handoff_tiles"},
    "temporal": {"signal_tiles"},
    "selection": {
        "control_handoff_tiles",
        "pressure_plates_per_barrier",
        "buttons_per_barrier",
    },
}
COMMON_FIELDS = set(DEFAULTS) | {"map_type", "regions", "possible_recipes"}
NEIGHBOUR_DELTAS = ((-1, 0), (1, 0), (0, -1), (0, 1))
Position = tuple[int, int]


class CandidateGenerationError(RuntimeError):
    """A constructive candidate could not satisfy all requested constraints."""


@dataclass
class CandidateTopology:
    """Hold the logical regions and protected topology of one candidate map."""

    grid: list[list[str]]
    regions: list[set[Position]]
    protected_floor: set[Position]
    reserved_stations: set[Position]
    signal_positions: list[Position]
    barrier_positions: list[Position]
    station_owners: dict[Position, int]


def _integer_value(value: Any, path: str, minimum: int) -> int:
    """Validate and return one non-boolean integer configuration value."""
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{path} must be an integer >= {minimum}")
    return value


def _integer(config: dict[str, Any], key: str, minimum: int) -> int:
    """Validate and return one integer from the normalized generator object."""
    return _integer_value(config[key], f"generator.{key}", minimum)


def _validate_regions(raw_regions: Any) -> list[dict[str, Any]]:
    """Validate exact per-region workstation dictionaries."""
    if not isinstance(raw_regions, list) or not raw_regions:
        raise ValueError("generator.regions must be a non-empty list")

    regions: list[dict[str, Any]] = []
    ingredient_type_count: Optional[int] = None
    for region_idx, raw_region in enumerate(raw_regions):
        path = f"generator.regions[{region_idx}]"
        if not isinstance(raw_region, dict):
            raise ValueError(f"{path} must be an object")

        missing = sorted(WORKSTATION_KEYS - set(raw_region))
        unknown = sorted(set(raw_region) - WORKSTATION_KEYS)
        if missing:
            raise ValueError(f"{path} is missing setting(s): {', '.join(missing)}")
        if unknown:
            raise ValueError(f"{path} has unknown setting(s): {', '.join(unknown)}")

        ingredient_piles = raw_region["ingredient_piles"]
        if (
            not isinstance(ingredient_piles, list)
            or not 1 <= len(ingredient_piles) <= MAX_INGREDIENTS
        ):
            raise ValueError(
                f"{path}.ingredient_piles must contain counts for 1-"
                f"{MAX_INGREDIENTS} ingredient types"
            )
        if ingredient_type_count is None:
            ingredient_type_count = len(ingredient_piles)
        elif len(ingredient_piles) != ingredient_type_count:
            raise ValueError(
                "all generator.regions ingredient_piles lists must have the "
                "same length"
            )

        normalized = {
            "ingredient_piles": [
                _integer_value(value, f"{path}.ingredient_piles[{idx}]", 0)
                for idx, value in enumerate(ingredient_piles)
            ]
        }
        for key in ("pots", "plate_piles", "depots", "recipe_indicators"):
            normalized[key] = _integer_value(raw_region[key], f"{path}.{key}", 0)
        regions.append(normalized)
    return regions


def _workstation_symbols(region: dict[str, Any]) -> list[str]:
    """Expand one exact region specification into ASCII workstation symbols."""
    symbols: list[str] = []
    for ingredient_idx, count in enumerate(region["ingredient_piles"]):
        symbols.extend([str(ingredient_idx)] * count)
    symbols.extend(["P"] * region["pots"])
    symbols.extend(["B"] * region["plate_piles"])
    symbols.extend(["X"] * region["depots"])
    symbols.extend(["R"] * region["recipe_indicators"])
    return symbols


def _region_totals(regions: list[dict[str, Any]]) -> dict[str, Any]:
    """Sum strict workstation counts across all configured regions."""
    ingredient_type_count = len(regions[0]["ingredient_piles"])
    return {
        "ingredient_piles": [
            sum(region["ingredient_piles"][idx] for region in regions)
            for idx in range(ingredient_type_count)
        ],
        "pots": sum(region["pots"] for region in regions),
        "plate_piles": sum(region["plate_piles"] for region in regions),
        "depots": sum(region["depots"] for region in regions),
        "recipe_indicators": sum(
            region["recipe_indicators"] for region in regions
        ),
    }


def _validate_recipes(recipes: Any, ingredient_counts: list[int]) -> list[list[int]]:
    """Validate recipes or derive one homogeneous recipe per present ingredient."""
    if recipes is None:
        return [
            [ingredient_idx] * 3
            for ingredient_idx, count in enumerate(ingredient_counts)
            if count > 0
        ]
    if not isinstance(recipes, list) or not recipes:
        raise ValueError("generator.possible_recipes must be a non-empty list")

    validated: list[list[int]] = []
    for recipe_idx, recipe in enumerate(recipes):
        path = f"generator.possible_recipes[{recipe_idx}]"
        if (
            not isinstance(recipe, list)
            or len(recipe) != 3
            or any(isinstance(item, bool) or not isinstance(item, int) for item in recipe)
        ):
            raise ValueError(f"{path} must contain 3 integer ingredient indices")
        if len(set(recipe)) != 1:
            raise ValueError(
                f"{path} is mixed; Overcooked V3 currently supports "
                "same-ingredient soups"
            )
        ingredient_idx = recipe[0]
        if (
            ingredient_idx < 0
            or ingredient_idx >= len(ingredient_counts)
            or ingredient_counts[ingredient_idx] == 0
        ):
            raise ValueError(
                f"{path} references ingredient {ingredient_idx}, which has no pile"
            )
        validated.append(list(recipe))
    return validated


def _missing_delivery_stations(region: dict[str, Any]) -> list[str]:
    """List stations needed to plate and deliver soup on one side of a pot."""
    missing = []
    for key, label in (("plate_piles", "plate pile"), ("depots", "depot")):
        if region[key] == 0:
            missing.append(label)
    return missing


def _validate_temporal_workflows(
    regions: list[dict[str, Any]], recipes: list[list[int]]
) -> None:
    """Require a plated-delivery path on one side of each shared-pot workflow."""
    missing_by_region = [
        _missing_delivery_stations(region) for region in regions
    ]
    if any(not missing for missing in missing_by_region):
        return

    for recipe in recipes:
        details = "; ".join(
            f"region {region_idx} is missing {', '.join(missing)}"
            for region_idx, missing in enumerate(missing_by_region)
        )
        raise ValueError(
            "generator.map_type 'temporal' cannot complete recipe "
            f"{recipe} within either agent's accessible workflow: {details}. "
            "Because both agents can add ingredients to a shared pot, its soup "
            "can be completed when either region contains both a plate pile "
            "and a depot."
        )


def _axis_sizes(width: int, height: int, vertical: bool) -> tuple[int, int]:
    """Return interior primary and secondary axis lengths for an orientation."""
    if vertical:
        return width - 2, height - 2
    return height - 2, width - 2


def _feasible_orientations(
    width: int,
    height: int,
    minimum_primary: int,
    minimum_secondary: int,
) -> list[bool]:
    """Return vertical/horizontal orientations meeting minimum axis lengths."""
    return [
        vertical
        for vertical in (True, False)
        if _axis_sizes(width, height, vertical)[0] >= minimum_primary
        and _axis_sizes(width, height, vertical)[1] >= minimum_secondary
    ]


def validate_config(raw_config: Any) -> dict[str, Any]:
    """Apply common defaults and validate the strict mode-specific schema."""
    if not isinstance(raw_config, dict):
        raise ValueError("JSON must contain a top-level 'generator' object")

    map_type = raw_config.get("map_type")
    if not isinstance(map_type, str) or map_type not in MAP_TYPES:
        raise ValueError(
            "generator.map_type must be 'asymmetric_info', 'selection', or 'temporal'"
        )

    required = {"map_type", "regions"} | MODE_FIELDS[map_type]
    missing = sorted(required - set(raw_config))
    if missing:
        raise ValueError("Missing generator setting(s): " + ", ".join(missing))

    allowed = COMMON_FIELDS | MODE_FIELDS[map_type]
    unknown = sorted(set(raw_config) - allowed)
    if unknown:
        raise ValueError(
            f"Unknown generator setting(s) for {map_type!r}: " + ", ".join(unknown)
        )

    config = {**DEFAULTS, **raw_config}
    width = _integer(config, "width", 5)
    height = _integer(config, "height", 5)
    _integer(config, "seed", 0)
    _integer(config, "count", 1)
    _integer(config, "max_attempts", 1)
    if not isinstance(config["name_prefix"], str) or not config["name_prefix"]:
        raise ValueError("generator.name_prefix must be a non-empty string")
    if not isinstance(config["randomize_agents"], bool):
        raise ValueError("generator.randomize_agents must be a boolean")

    density = config["counter_density"]
    if isinstance(density, bool) or not isinstance(density, (int, float)):
        raise ValueError("generator.counter_density must be a number")
    if not 0 <= density < 1:
        raise ValueError("generator.counter_density must be in [0, 1)")

    regions = _validate_regions(config["regions"])
    config["regions"] = regions
    if map_type in {"asymmetric_info", "temporal"} and len(regions) != 2:
        raise ValueError(f"generator.map_type {map_type!r} requires exactly 2 regions")
    if map_type == "selection" and len(regions) < 3:
        raise ValueError(
            "generator.map_type 'selection' requires regions for control, main, and at "
            "least one gated region"
        )

    totals = _region_totals(regions)
    if sum(totals["ingredient_piles"]) == 0:
        raise ValueError("generator.regions must contain at least one ingredient pile")
    for key in ("pots", "plate_piles", "depots"):
        if totals[key] == 0:
            raise ValueError(f"generator.regions must contain at least one {key}")
    if totals["pots"] > MAX_POTS:
        raise ValueError(
            f"total configured pots cannot exceed MAX_POTS ({MAX_POTS})"
        )
    config["possible_recipes"] = _validate_recipes(
        config.get("possible_recipes"), totals["ingredient_piles"]
    )

    if map_type == "asymmetric_info":
        interface_count = _integer(config, "handoff_tiles", 1)
        if not _feasible_orientations(width, height, 4, interface_count):
            raise ValueError(
                "width/height cannot fit two regions and the requested handoff_tiles"
            )
    elif map_type == "temporal":
        signal_tiles = _integer(config, "signal_tiles", 1)
        shared_pots = regions[0]["pots"]
        if shared_pots == 0:
            raise ValueError(
                "generator.map_type 'temporal' requires at least one region 0 pot"
            )
        if signal_tiles != shared_pots:
            raise ValueError(
                "generator.signal_tiles must equal generator.regions[0].pots "
                f"({shared_pots})"
            )
        _validate_temporal_workflows(regions, config["possible_recipes"])
        if not _feasible_orientations(width, height, 4, signal_tiles):
            raise ValueError(
                "width/height cannot fit two regions and the requested signal_tiles"
            )
    else:
        barrier_count = len(regions) - 2
        if barrier_count > MAX_BARRIERS:
            raise ValueError(
                f"selection cannot exceed MAX_BARRIERS ({MAX_BARRIERS}) gated rooms"
            )
        handoff_tiles = _integer(config, "control_handoff_tiles", 1)
        plates_per_barrier = _integer(config, "pressure_plates_per_barrier", 0)
        buttons_per_barrier = _integer(config, "buttons_per_barrier", 0)
        if plates_per_barrier not in {0, 1, 2}:
            raise ValueError(
                "generator.pressure_plates_per_barrier must be 0, 1, or 2"
            )
        if plates_per_barrier + buttons_per_barrier == 0:
            raise ValueError(
                "selection barriers require at least one pressure plate or button"
            )
        if barrier_count * plates_per_barrier > MAX_PRESSURE_PLATES:
            raise ValueError(
                f"selection controls exceed MAX_PRESSURE_PLATES ({MAX_PRESSURE_PLATES})"
            )
        if barrier_count * buttons_per_barrier > MAX_BUTTONS:
            raise ValueError(
                f"selection controls exceed MAX_BUTTONS ({MAX_BUTTONS})"
            )
        minimum_secondary = max(3 * barrier_count - 2, handoff_tiles)
        if not _feasible_orientations(width, height, 7, minimum_secondary):
            raise ValueError(
                "width/height cannot fit the selection control/main/gated-room topology"
            )
    return config


def _grid_position(primary: int, secondary: int, vertical: bool) -> Position:
    """Convert oriented interior coordinates into a grid row/column position."""
    if vertical:
        return secondary, primary
    return primary, secondary


def _positive_composition(total: int, parts: int, rng: random.Random) -> list[int]:
    """Randomly split an integer into an ordered list of positive parts."""
    if parts == 1:
        return [total]
    cuts = sorted(rng.sample(range(1, total), parts - 1))
    endpoints = [0, *cuts, total]
    return [endpoints[idx + 1] - endpoints[idx] for idx in range(parts)]


def _new_grid(width: int, height: int) -> list[list[str]]:
    """Create a counter-filled grid of the requested exact dimensions."""
    return [["W"] * width for _ in range(height)]


def _carve_regions(grid: list[list[str]], regions: list[set[Position]]) -> None:
    """Carve every logical region position into ordinary walkable floor."""
    for region in regions:
        for row, col in region:
            grid[row][col] = " "


def _build_two_region_topology(
    config: dict[str, Any], rng: random.Random
) -> CandidateTopology:
    """Build asymmetric or temporal regions around a two-counter interface."""
    mode = config["map_type"]
    interface_count = (
        config["handoff_tiles"] if mode == "asymmetric_info" else config["signal_tiles"]
    )
    orientations = _feasible_orientations(
        config["width"], config["height"], 4, interface_count
    )
    if not orientations:
        raise CandidateGenerationError("no feasible two-region orientation exists")
    vertical = rng.choice(orientations)
    primary_size, secondary_size = _axis_sizes(
        config["width"], config["height"], vertical
    )

    region_zero_width = rng.randint(1, primary_size - 3)
    first_separator = region_zero_width + 1
    second_separator = first_separator + 1
    selected_secondary = sorted(
        rng.sample(range(1, secondary_size + 1), interface_count)
    )

    region_zero = {
        _grid_position(primary, secondary, vertical)
        for primary in range(1, region_zero_width + 1)
        for secondary in range(1, secondary_size + 1)
    }
    region_one = {
        _grid_position(primary, secondary, vertical)
        for primary in range(second_separator + 1, primary_size + 1)
        for secondary in range(1, secondary_size + 1)
    }
    protected_floor: set[Position] = set()
    reserved_stations: set[Position] = set()
    signal_positions: list[Position] = []

    if mode == "asymmetric_info":
        # Carve an alcove through one separator layer at each requested
        # handoff. The remaining W is then adjacent to both floor regions.
        for secondary in selected_secondary:
            region_zero_edge = _grid_position(
                region_zero_width, secondary, vertical
            )
            handoff_counter = _grid_position(first_separator, secondary, vertical)
            region_one_alcove = _grid_position(
                second_separator, secondary, vertical
            )
            region_one.add(region_one_alcove)
            protected_floor.update({region_zero_edge, region_one_alcove})
            reserved_stations.add(handoff_counter)
    else:
        # Both counter layers remain intact except where a shared pot replaces
        # one layer and an R1 alcove replaces the other. The pot is therefore
        # interactable from both sides but still blocks travel and loose items.
        for secondary in range(1, secondary_size + 1):
            reserved_stations.add(
                _grid_position(first_separator, secondary, vertical)
            )
            reserved_stations.add(
                _grid_position(second_separator, secondary, vertical)
            )
        for secondary in selected_secondary:
            signal_position = _grid_position(first_separator, secondary, vertical)
            region_one_alcove = _grid_position(
                second_separator, secondary, vertical
            )
            region_one.add(region_one_alcove)
            signal_positions.append(signal_position)
            protected_floor.update(
                {
                    _grid_position(region_zero_width, secondary, vertical),
                    region_one_alcove,
                }
            )

    grid = _new_grid(config["width"], config["height"])
    regions = [region_zero, region_one]
    _carve_regions(grid, regions)
    return CandidateTopology(
        grid=grid,
        regions=regions,
        protected_floor=protected_floor,
        reserved_stations=reserved_stations,
        signal_positions=signal_positions,
        barrier_positions=[],
        station_owners={},
    )


def _build_selection_topology(
    config: dict[str, Any], rng: random.Random
) -> CandidateTopology:
    """Build a control room, main room, and barrier-gated satellite rooms."""
    gated_count = len(config["regions"]) - 2
    minimum_secondary = max(
        3 * gated_count - 2, config["control_handoff_tiles"]
    )
    orientations = _feasible_orientations(
        config["width"], config["height"], 7, minimum_secondary
    )
    if not orientations:
        raise CandidateGenerationError("no feasible selection orientation exists")
    vertical = rng.choice(orientations)
    primary_size, secondary_size = _axis_sizes(
        config["width"], config["height"], vertical
    )

    control_width, main_width, gated_width = _positive_composition(
        primary_size - 4, 3, rng
    )
    # Along the primary axis: control | WW | main | WW | gated rooms.
    control_separator_zero = control_width + 1
    control_separator_one = control_separator_zero + 1
    main_start = control_separator_one + 1
    main_end = main_start + main_width - 1
    gate_separator_zero = main_end + 1
    gate_separator_one = gate_separator_zero + 1
    gated_start = gate_separator_one + 1

    usable_gated_secondary = secondary_size - 2 * (gated_count - 1)
    gated_heights = _positive_composition(
        usable_gated_secondary, gated_count, rng
    )
    gated_ranges: list[range] = []
    cursor = 1
    # Two counter rows/columns between satellites prevent unintended item
    # handoffs among gated rooms.
    for room_height in gated_heights:
        gated_ranges.append(range(cursor, cursor + room_height))
        cursor += room_height + 2

    control = {
        _grid_position(primary, secondary, vertical)
        for primary in range(1, control_width + 1)
        for secondary in range(1, secondary_size + 1)
    }
    main = {
        _grid_position(primary, secondary, vertical)
        for primary in range(main_start, main_end + 1)
        for secondary in range(1, secondary_size + 1)
    }
    gated_regions = [
        {
            _grid_position(primary, secondary, vertical)
            for primary in range(gated_start, primary_size + 1)
            for secondary in secondary_range
        }
        for secondary_range in gated_ranges
    ]

    protected_floor: set[Position] = set()
    reserved_stations: set[Position] = set()
    handoff_secondary = rng.sample(
        range(1, secondary_size + 1), config["control_handoff_tiles"]
    )
    for secondary in handoff_secondary:
        control_edge = _grid_position(control_width, secondary, vertical)
        handoff_counter = _grid_position(
            control_separator_zero, secondary, vertical
        )
        main_alcove = _grid_position(control_separator_one, secondary, vertical)
        main.add(main_alcove)
        protected_floor.update({control_edge, main_alcove})
        reserved_stations.add(handoff_counter)

    barrier_positions: list[Position] = []
    for region, secondary_range in zip(gated_regions, gated_ranges):
        secondary = rng.choice(list(secondary_range))
        main_edge = _grid_position(main_end, secondary, vertical)
        barrier = _grid_position(gate_separator_zero, secondary, vertical)
        gated_alcove = _grid_position(gate_separator_one, secondary, vertical)
        region.add(gated_alcove)
        protected_floor.update({main_edge, gated_alcove})
        barrier_positions.append(barrier)

    grid = _new_grid(config["width"], config["height"])
    regions = [control, main, *gated_regions]
    _carve_regions(grid, regions)
    for row, col in barrier_positions:
        grid[row][col] = "#"
        reserved_stations.add((row, col))
    return CandidateTopology(
        grid=grid,
        regions=regions,
        protected_floor=protected_floor,
        reserved_stations=reserved_stations,
        signal_positions=[],
        barrier_positions=barrier_positions,
        station_owners={},
    )


def _neighbours(position: Position, height: int, width: int) -> list[Position]:
    """Return in-bounds orthogonal neighbours of one grid position."""
    row, col = position
    return [
        (row + row_delta, col + col_delta)
        for row_delta, col_delta in NEIGHBOUR_DELTAS
        if 0 <= row + row_delta < height and 0 <= col + col_delta < width
    ]


def _is_connected(positions: set[Position], height: int, width: int) -> bool:
    """Return whether a non-empty position set is four-connected."""
    if not positions:
        return False
    reached = {next(iter(positions))}
    frontier = list(reached)
    while frontier:
        position = frontier.pop()
        for adjacent in _neighbours(position, height, width):
            if adjacent in positions and adjacent not in reached:
                reached.add(adjacent)
                frontier.append(adjacent)
    return reached == positions


def _add_counter_clutter(
    topology: CandidateTopology,
    config: dict[str, Any],
    minimum_region_floors: list[int],
    rng: random.Random,
) -> None:
    """Reach the exact interior counter budget without breaking room access."""
    height, width = config["height"], config["width"]
    requested = round((width - 2) * (height - 2) * config["counter_density"])
    current = sum(
        topology.grid[row][col] == "W"
        for row in range(1, height - 1)
        for col in range(1, width - 1)
    )
    if current > requested:
        raise CandidateGenerationError(
            f"mandatory topology requires {current} interior counters, but "
            f"counter_density permits only {requested}"
        )

    # Only erode a connected room at positions that are not required by an
    # interface, doorway, or station access path.
    while current < requested:
        candidates = [
            (region_idx, position)
            for region_idx, region in enumerate(topology.regions)
            if len(region) > minimum_region_floors[region_idx]
            for position in region - topology.protected_floor
        ]
        rng.shuffle(candidates)
        chosen: Optional[tuple[int, Position]] = None
        for region_idx, position in candidates:
            remaining = topology.regions[region_idx] - {position}
            if not _is_connected(remaining, height, width):
                continue
            owned_stations_stay_accessible = all(
                owner != region_idx
                or any(
                    neighbour in remaining
                    for neighbour in _neighbours(station, height, width)
                )
                for station, owner in topology.station_owners.items()
            )
            if owned_stations_stay_accessible:
                chosen = region_idx, position
                break
        if chosen is None:
            raise CandidateGenerationError(
                f"counter_density requires {requested} optional counter(s), but "
                f"only {current} can be retained without disconnecting a room"
            )
        region_idx, position = chosen
        topology.regions[region_idx].remove(position)
        topology.grid[position[0]][position[1]] = "W"
        current += 1

    final_count = sum(
        topology.grid[row][col] == "W"
        for row in range(1, height - 1)
        for col in range(1, width - 1)
    )
    if final_count != requested:
        raise CandidateGenerationError(
            f"layout has {final_count} interior counters; requested {requested}"
        )


def _adjacent_region_indices(
    position: Position,
    regions: list[set[Position]],
    height: int,
    width: int,
) -> set[int]:
    """Return logical regions with floor orthogonally adjacent to a position."""
    adjacent = _neighbours(position, height, width)
    return {
        region_idx
        for region_idx, region in enumerate(regions)
        if any(neighbour in region for neighbour in adjacent)
    }


def _station_candidates(
    topology: CandidateTopology, region_idx: int
) -> list[Position]:
    """Return unused counters interactable exclusively from one logical region."""
    height = len(topology.grid)
    width = len(topology.grid[0])
    return [
        (row, col)
        for row in range(height)
        for col in range(width)
        if topology.grid[row][col] == "W"
        and (row, col) not in topology.reserved_stations
        and _adjacent_region_indices(
            (row, col), topology.regions, height, width
        )
        == {region_idx}
    ]


def _place_symbols_by_region(
    topology: CandidateTopology,
    symbols_by_region: list[list[str]],
    rng: random.Random,
) -> None:
    """Place exact station/control symbols next to their assigned regions."""
    placement_order = sorted(
        range(len(topology.regions)),
        key=lambda idx: len(_station_candidates(topology, idx))
        - len(symbols_by_region[idx]),
    )
    for region_idx in placement_order:
        symbols = list(symbols_by_region[region_idx])
        candidates = _station_candidates(topology, region_idx)
        if len(candidates) < len(symbols):
            raise CandidateGenerationError(
                f"region {region_idx} has {len(candidates)} workstation slot(s), "
                f"but requires {len(symbols)}"
            )
        rng.shuffle(candidates)
        rng.shuffle(symbols)
        for position, symbol in zip(candidates, symbols):
            topology.grid[position[0]][position[1]] = symbol
            topology.station_owners[position] = region_idx


def _place_temporal_signals(
    topology: CandidateTopology,
    config: dict[str, Any],
) -> None:
    """Place every region-zero pot in a shared interface position."""
    pot_count = config["regions"][0]["pots"]
    if pot_count != len(topology.signal_positions):
        raise CandidateGenerationError(
            "temporal signal position count does not match region 0 pots"
        )
    for position in topology.signal_positions:
        topology.grid[position[0]][position[1]] = "P"
        topology.station_owners[position] = 0


def _place_selection_controls(
    topology: CandidateTopology,
    config: dict[str, Any],
    rng: random.Random,
) -> tuple[list[bool], list[tuple[int, int]], list[tuple[int, int]]]:
    """Place all barrier controls in the control room and wire exact links."""
    barrier_count = len(topology.barrier_positions)
    plates_per_barrier = config["pressure_plates_per_barrier"]
    buttons_per_barrier = config["buttons_per_barrier"]

    plate_count = barrier_count * plates_per_barrier
    plate_candidates = [
        position
        for position in topology.regions[0]
        if topology.grid[position[0]][position[1]] == " "
    ]
    if len(plate_candidates) < plate_count + 1:
        raise CandidateGenerationError(
            "control region has too few floor tiles for pressure plates and its agent"
        )
    plate_positions = rng.sample(plate_candidates, plate_count)
    for row, col in plate_positions:
        topology.grid[row][col] = "_"

    button_positions = sorted(
        (row, col)
        for row, line in enumerate(topology.grid)
        for col, symbol in enumerate(line)
        if symbol == "!"
    )
    if len(button_positions) != barrier_count * buttons_per_barrier:
        raise CandidateGenerationError("selection button placement count is inconsistent")

    button_targets = [
        barrier_idx
        for barrier_idx in range(barrier_count)
        for _ in range(buttons_per_barrier)
    ]
    plate_targets = [
        barrier_idx
        for barrier_idx in range(barrier_count)
        for _ in range(plates_per_barrier)
    ]
    button_config = [
        (target, int(ButtonAction.TIMED_BARRIER)) for target in button_targets
    ]
    pressure_plate_config = [
        (target, int(ButtonAction.TOGGLE_BARRIER))
        for target in plate_targets
    ]
    return [True] * barrier_count, button_config, pressure_plate_config


def _place_agents(
    topology: CandidateTopology,
    config: dict[str, Any],
    rng: random.Random,
) -> bool:
    """Place two agents and return whether row-major parsing must be reversed."""
    spawn_positions: list[Position] = []
    for region_idx in (0, 1):
        candidates = [
            position
            for position in topology.regions[region_idx]
            if topology.grid[position[0]][position[1]] == " "
        ]
        if not candidates:
            raise CandidateGenerationError(
                f"region {region_idx} has no ordinary floor tile for its agent"
            )
        spawn_positions.append(rng.choice(candidates))

    for row, col in spawn_positions:
        topology.grid[row][col] = "A"

    agent_zero_region = rng.randrange(2) if config["randomize_agents"] else 0
    desired = [
        spawn_positions[agent_zero_region],
        spawn_positions[1 - agent_zero_region],
    ]
    row_major = sorted(spawn_positions)
    if desired == row_major:
        return False
    if desired == list(reversed(row_major)):
        return True
    raise CandidateGenerationError("could not encode the requested agent ordering")


def _generate_candidate(
    config: dict[str, Any], rng: random.Random
) -> tuple[
    str,
    bool,
    list[bool],
    list[tuple[int, int]],
    list[tuple[int, int]],
]:
    """Construct one candidate grid and its control/agent metadata."""
    if config["map_type"] == "selection":
        topology = _build_selection_topology(config, rng)
        plate_count = (
            len(topology.barrier_positions)
            * config["pressure_plates_per_barrier"]
        )
        minimum_floors = [plate_count + 1, 1] + [
            1 for _ in topology.regions[2:]
        ]
    else:
        topology = _build_two_region_topology(config, rng)
        minimum_floors = [1, 1]

    symbols_by_region = [
        _workstation_symbols(region) for region in config["regions"]
    ]
    if config["map_type"] == "temporal":
        _place_temporal_signals(topology, config)
        symbols_by_region[0] = [
            symbol for symbol in symbols_by_region[0] if symbol != "P"
        ]
    elif config["map_type"] == "selection":
        button_count = (
            len(topology.barrier_positions) * config["buttons_per_barrier"]
        )
        symbols_by_region[0].extend(["!"] * button_count)

    # Stations consume W cells, so place them before filling the remaining
    # exact interior-counter budget.
    _place_symbols_by_region(topology, symbols_by_region, rng)
    _add_counter_clutter(topology, config, minimum_floors, rng)

    if config["map_type"] == "selection":
        barrier_config, button_config, pressure_plate_config = (
            _place_selection_controls(topology, config, rng)
        )
    else:
        barrier_config, button_config, pressure_plate_config = [], [], []

    swap_agents = _place_agents(topology, config, rng)
    grid = "\n".join("".join(row) for row in topology.grid)
    return (
        grid,
        swap_agents,
        barrier_config,
        button_config,
        pressure_plate_config,
    )


def generate_layout(
    config: dict[str, Any], rng: random.Random
) -> tuple[str, Layout, int]:
    """Construct and validate one layout, retrying recoverable placements."""
    last_errors: list[str] = []
    for attempt in range(1, config["max_attempts"] + 1):
        try:
            (
                grid,
                swap_agents,
                barrier_config,
                button_config,
                pressure_plate_config,
            ) = _generate_candidate(config, rng)
        except CandidateGenerationError as exc:
            last_errors = [str(exc)]
            continue

        try:
            layout = Layout.from_string(
                grid,
                possible_recipes=config["possible_recipes"],
                button_config=button_config,
                barrier_config=barrier_config,
                pressure_plate_config=pressure_plate_config,
                swap_agents=swap_agents,
            )
        except (TypeError, ValueError) as exc:
            last_errors = [str(exc)]
            continue
        valid, last_errors = validate_generated_layout(layout)
        if valid:
            return grid, layout, attempt
    raise RuntimeError(
        f"Could not generate a valid layout after {config['max_attempts']} "
        f"attempts. Last validation error(s): {'; '.join(last_errors)}"
    )


def _layout_uses_swap(grid: str, layout: Layout) -> bool:
    """Infer whether a layout's persisted agent ordering reverses ASCII order."""
    row_major = [
        (col, row)
        for row, line in enumerate(grid.splitlines())
        for col, symbol in enumerate(line)
        if symbol == "A"
    ]
    actual = [tuple(position) for position in layout.agent_positions]
    return actual == list(reversed(row_major))


def _layout_entry(grid: str, layout: Layout, config: dict[str, Any]) -> dict[str, Any]:
    """Serialize one generated layout and all runtime-relevant metadata."""
    return {
        "ascii": grid,
        "possible_recipes": config["possible_recipes"],
        "swap_agents": _layout_uses_swap(grid, layout),
        "button_config": [
            [list(target_idxs), int(action_type)]
            for _, _, target_idxs, action_type in layout.button_info
        ],
        "barrier_config": [bool(active) for _, _, active in layout.barrier_info],
        "pressure_plate_config": [
            [list(target_idxs), int(action_type)]
            for _, _, target_idxs, action_type in layout.pressure_plate_info
        ],
        "validation": {"valid": True, "errors": []},
    }


def generate_document(document: Any) -> dict[str, Any]:
    """Generate every layout requested by a parsed JSON document."""
    if not isinstance(document, dict):
        raise ValueError("The JSON document must be an object")
    config = validate_config(document.get("generator"))
    rng = random.Random(config["seed"])
    generated: dict[str, dict[str, Any]] = {}

    digits = max(1, len(str(config["count"] - 1)))
    for index in range(config["count"]):
        name = f"{config['name_prefix']}_{index:0{digits}d}"
        grid, layout, _ = generate_layout(config, rng)
        generated[name] = _layout_entry(grid, layout, config)

    result = dict(document)
    result["generator"] = config
    result["layouts"] = generated
    return result


def _write_json_checkpoint(document: dict[str, Any], output_path: Path) -> None:
    """Atomically write one generation checkpoint."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(document, file, indent=2)
        file.write("\n")
    temporary_path.replace(output_path)


def generate_to_file(
    document: Any,
    output_path: Path,
    *,
    emit: Optional[Callable[[str], None]] = print,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Generate layouts while atomically checkpointing every result."""
    if not isinstance(document, dict):
        raise ValueError("The JSON document must be an object")

    config = validate_config(document.get("generator"))
    result = dict(document)
    result["generator"] = config
    result["layouts"] = {}
    result["generation_errors"] = {}
    result["generation_progress"] = {
        "requested": config["count"],
        "completed": 0,
        "failed": 0,
        "status": "running",
    }
    _write_json_checkpoint(result, output_path)

    rng = random.Random(config["seed"])
    digits = max(1, len(str(config["count"] - 1)))
    failures: dict[str, str] = result["generation_errors"]
    for index in range(config["count"]):
        name = f"{config['name_prefix']}_{index:0{digits}d}"
        if emit is not None:
            emit(f"[{index + 1}/{config['count']}] Generating {name}...")
        try:
            grid, layout, attempts = generate_layout(config, rng)
        except RuntimeError as exc:
            failures[name] = str(exc)
            result["generation_progress"]["failed"] += 1
            _write_json_checkpoint(result, output_path)
            if emit is not None:
                emit(
                    f"[{index + 1}/{config['count']}] FAILED {name}; "
                    f"checkpoint saved to {output_path}"
                )
            continue

        result["layouts"][name] = _layout_entry(grid, layout, config)
        result["generation_progress"]["completed"] += 1
        _write_json_checkpoint(result, output_path)
        if emit is not None:
            emit(
                f"[{index + 1}/{config['count']}] Saved {name} "
                f"after {attempts} attempt(s); "
                f"{result['generation_progress']['completed']} map(s) complete"
            )

    result["generation_progress"]["status"] = (
        "completed_with_errors" if failures else "complete"
    )
    _write_json_checkpoint(result, output_path)
    return result, failures


def parse_args() -> argparse.Namespace:
    """Parse command-line paths for the JSON generator."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "json_file",
        type=Path,
        help="JSON config to read and, by default, update in place",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional output JSON path (defaults to json_file)",
    )
    return parser.parse_args()


def main() -> None:
    """Run layout generation from the command line."""
    args = parse_args()
    with args.json_file.open("r", encoding="utf-8") as file:
        document = json.load(file)

    output_path = args.output or args.json_file
    result, failures = generate_to_file(document, output_path)
    completed = result["generation_progress"]["completed"]
    requested = result["generation_progress"]["requested"]
    print(
        f"Generation finished: {completed}/{requested} layout(s) saved to "
        f"{output_path}"
    )
    if failures:
        print(
            f"{len(failures)} layout(s) failed. See 'generation_errors' in "
            "the output JSON."
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
