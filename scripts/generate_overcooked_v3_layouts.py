"""Generate validated Overcooked V3 ASCII layouts in one JSON file.

Run:
    python scripts/generate_overcooked_v3_layouts.py layouts.json

The input file is updated in place by default: generation settings remain in
``generator`` and generated ASCII maps are written to ``layouts``. Pass
``--output`` to write a different file. Progress is printed for every map and
the JSON is atomically checkpointed after each success or failure.
"""

from __future__ import annotations

import argparse
import json
import random
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
    # List index is the ingredient symbol; values are pile counts.
    "ingredient_piles": [1],
    "pots": 1,
    "plate_piles": 1,
    "depots": 1,
    "object_placement": "boundary",
    "counter_density": 0.1,
    "num_regions": 1,
    # Exact number of counters accessible from both regions. None leaves the
    # number unconstrained (except that shared workflows still require one).
    "num_shared_tiles": None,
    "
  ": "single_region",
    # Barriers start active and are opened by generated plates or buttons.
    "barriers": 0,
    "barrier_placement": "anywhere",
    "pressure_plates_per_barrier": 1,
    "buttons_per_barrier": 0,
    "max_attempts": 1000,
}

WORKFLOW_MODES = {"complete_each", "single_region", "shared"}
BARRIER_PLACEMENTS = {
    "anywhere",
    "shared",
    "action_adjacent",
    "shared_or_action_adjacent",
}
ACTION_ITEM_SYMBOLS = set("0123456789PBX")
NEIGHBOUR_DELTAS = ((-1, 0), (1, 0), (0, -1), (0, 1))


class CandidateGenerationError(RuntimeError):
    """A constructive candidate could not satisfy all requested constraints."""


def _minimum_two_region_blockers(width: int, height: int) -> int:
    """Return the vertex cut needed to split the rectangular interior grid."""
    interior_width = width - 2
    interior_height = height - 2
    return min(2, interior_width, interior_height)


def _integer(config: dict[str, Any], key: str, minimum: int) -> int:
    value = config[key]
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"generator.{key} must be an integer >= {minimum}")
    return value


def _validate_recipes(recipes: Any, ingredient_counts: list[int]) -> list[list[int]]:
    if recipes is None:
        return [
            [ingredient_idx] * 3
            for ingredient_idx, count in enumerate(ingredient_counts)
            if count > 0
        ]
    if not isinstance(recipes, list) or not recipes:
        raise ValueError("generator.possible_recipes must be a non-empty list")

    validated = []
    for index, recipe in enumerate(recipes):
        if (
            not isinstance(recipe, list)
            or len(recipe) != 3
            or any(isinstance(item, bool) or not isinstance(item, int) for item in recipe)
        ):
            raise ValueError(
                f"generator.possible_recipes[{index}] must contain 3 integer "
                "ingredient indices"
            )
        if len(set(recipe)) != 1:
            raise ValueError(
                f"generator.possible_recipes[{index}] is mixed; Overcooked V3 "
                "currently supports same-ingredient soups"
            )
        ingredient_idx = recipe[0]
        if (
            ingredient_idx < 0
            or ingredient_idx >= len(ingredient_counts)
            or ingredient_counts[ingredient_idx] == 0
        ):
            raise ValueError(
                f"generator.possible_recipes[{index}] references ingredient "
                f"{ingredient_idx}, which has no pile"
            )
        validated.append(list(recipe))
    return validated


def validate_config(raw_config: Any) -> dict[str, Any]:
    """Apply defaults and validate a generator configuration."""
    if not isinstance(raw_config, dict):
        raise ValueError("JSON must contain a top-level 'generator' object")

    allowed = set(DEFAULTS) | {"possible_recipes"}
    unknown = sorted(set(raw_config) - allowed)
    if unknown:
        raise ValueError("Unknown generator setting(s): " + ", ".join(unknown))

    config = {**DEFAULTS, **raw_config}
    width = _integer(config, "width", 5)
    height = _integer(config, "height", 5)
    _integer(config, "seed", 0)
    _integer(config, "count", 1)
    _integer(config, "pots", 1)
    _integer(config, "plate_piles", 1)
    _integer(config, "depots", 1)
    num_regions = _integer(config, "num_regions", 1)
    barriers = _integer(config, "barriers", 0)
    pressure_plates_per_barrier = _integer(
        config,
        "pressure_plates_per_barrier",
        0,
    )
    buttons_per_barrier = _integer(config, "buttons_per_barrier", 0)
    _integer(config, "max_attempts", 1)
    num_shared_tiles = config["num_shared_tiles"]
    if num_shared_tiles is not None and (
        isinstance(num_shared_tiles, bool)
        or not isinstance(num_shared_tiles, int)
        or num_shared_tiles < 0
    ):
        raise ValueError(
            "generator.num_shared_tiles must be null or an integer >= 0"
        )

    if num_regions > 2:
        raise ValueError(
            "generator.num_regions cannot exceed 2 because generated layouts "
            "currently contain exactly two agents"
        )
    if barriers > MAX_BARRIERS:
        raise ValueError(
            f"generator.barriers cannot exceed MAX_BARRIERS ({MAX_BARRIERS})"
        )
    if pressure_plates_per_barrier not in {0, 1, 2}:
        raise ValueError(
            "generator.pressure_plates_per_barrier must be 0 (none), "
            "1 (single), or 2 (paired)"
        )
    if buttons_per_barrier > MAX_BUTTONS:
        raise ValueError(
            "generator.buttons_per_barrier cannot exceed "
            f"MAX_BUTTONS ({MAX_BUTTONS})"
        )
    if barriers > 0 and pressure_plates_per_barrier + buttons_per_barrier < 1:
        raise ValueError(
            "each barrier requires at least one pressure plate or button; "
            "set generator.pressure_plates_per_barrier or "
            "generator.buttons_per_barrier to at least 1"
        )
    pressure_plate_count = barriers * pressure_plates_per_barrier
    if pressure_plate_count > MAX_PRESSURE_PLATES:
        raise ValueError(
            f"{barriers} barriers with {pressure_plates_per_barrier} pressure "
            f"plate(s) each require {pressure_plate_count} pressure plates, "
            f"but MAX_PRESSURE_PLATES is {MAX_PRESSURE_PLATES}"
        )
    button_count = barriers * buttons_per_barrier
    if button_count > MAX_BUTTONS:
        raise ValueError(
            f"{barriers} barriers with {buttons_per_barrier} button(s) each "
            f"require {button_count} buttons, but MAX_BUTTONS is {MAX_BUTTONS}"
        )
    if (
        not isinstance(config["barrier_placement"], str)
        or config["barrier_placement"] not in BARRIER_PLACEMENTS
    ):
        raise ValueError(
            "generator.barrier_placement must be 'anywhere', 'shared', "
            "'action_adjacent', or 'shared_or_action_adjacent'"
        )
    if (
        barriers > 0
        and config["barrier_placement"] == "shared"
        and num_regions != 2
    ):
        raise ValueError(
            f"generator.barrier_placement {config['barrier_placement']!r} "
            "requires generator.num_regions = 2"
        )
    if num_shared_tiles is not None and num_regions != 2:
        raise ValueError(
            "generator.num_shared_tiles requires generator.num_regions = 2"
        )
    if (
        not isinstance(config["workflow_mode"], str)
        or config["workflow_mode"] not in WORKFLOW_MODES
    ):
        raise ValueError(
            "generator.workflow_mode must be 'complete_each', "
            "'single_region', or 'shared'"
        )
    if config["workflow_mode"] == "shared" and num_regions != 2:
        raise ValueError(
            "generator.workflow_mode 'shared' requires generator.num_regions = 2"
        )
    if config["workflow_mode"] == "shared" and num_shared_tiles == 0:
        raise ValueError(
            "generator.workflow_mode 'shared' requires at least one shared tile"
        )
    if config["pots"] > MAX_POTS:
        raise ValueError(f"generator.pots cannot exceed MAX_POTS ({MAX_POTS})")
    if not isinstance(config["name_prefix"], str) or not config["name_prefix"]:
        raise ValueError("generator.name_prefix must be a non-empty string")
    if (
        not isinstance(config["object_placement"], str)
        or config["object_placement"] not in {"boundary", "interior", "anywhere"}
    ):
        raise ValueError(
            "generator.object_placement must be 'boundary', 'interior', "
            "or 'anywhere'"
        )

    ingredient_counts = config["ingredient_piles"]
    if (
        not isinstance(ingredient_counts, list)
        or not 1 <= len(ingredient_counts) <= MAX_INGREDIENTS
        or any(
            isinstance(count, bool) or not isinstance(count, int) or count < 0
            for count in ingredient_counts
        )
        or sum(ingredient_counts) == 0
    ):
        raise ValueError(
            "generator.ingredient_piles must be a list of non-negative pile "
            f"counts for 1-{MAX_INGREDIENTS} ingredients, with at least one pile"
        )
    config["ingredient_piles"] = list(ingredient_counts)
    config["possible_recipes"] = _validate_recipes(
        config.get("possible_recipes"),
        ingredient_counts,
    )

    density = config["counter_density"]
    if isinstance(density, bool) or not isinstance(density, (int, float)):
        raise ValueError("generator.counter_density must be a number")
    if not 0 <= density < 1:
        raise ValueError("generator.counter_density must be in [0, 1)")

    interior_tiles = (width - 2) * (height - 2)
    counter_count = round(interior_tiles * density)
    if interior_tiles - counter_count < 2:
        raise ValueError("counter_density leaves fewer than two agent spawn tiles")
    minimum_blocking_tiles = _minimum_two_region_blockers(width, height)
    if num_regions == 2 and counter_count < minimum_blocking_tiles:
        raise ValueError(
            "generator.num_regions = 2 requires at least "
            f"{minimum_blocking_tiles} interior counters to separate the "
            f"{height - 2}x{width - 2} interior grid; counter_density provides "
            f"{counter_count}"
        )
    if num_shared_tiles is not None and num_shared_tiles > counter_count:
        raise ValueError(
            "generator.num_shared_tiles cannot exceed the number of interior "
            f"counters ({counter_count})"
        )
    if num_shared_tiles is not None and num_shared_tiles > counter_count:
        raise ValueError(
            "generator.num_shared_tiles cannot exceed the number of interior "
            f"counters ({counter_count})"
        )
    if (
        barriers > 0
        and config["barrier_placement"] == "shared"
        and num_shared_tiles is not None
        and barriers > num_shared_tiles
    ):
        raise ValueError(
            "generator.barriers cannot exceed generator.num_shared_tiles "
            "when barrier_placement is 'shared'"
        )
    if config["workflow_mode"] == "shared" and counter_count == 0:
        raise ValueError(
            "generator.workflow_mode 'shared' requires an interior counter for "
            "cross-region handoffs"
        )

    if config["workflow_mode"] == "complete_each":
        required_ingredients = {
            ingredient_idx
            for recipe in config["possible_recipes"]
            for ingredient_idx in recipe
        }
        insufficient = [
            str(ingredient_idx)
            for ingredient_idx in sorted(required_ingredients)
            if ingredient_counts[ingredient_idx] < num_regions
        ]
        if insufficient:
            raise ValueError(
                "generator.workflow_mode 'complete_each' needs at least "
                f"num_regions piles for recipe ingredient(s): {', '.join(insufficient)}"
            )
        for key in ("pots", "plate_piles", "depots"):
            if config[key] < num_regions:
                raise ValueError(
                    "generator.workflow_mode 'complete_each' requires "
                    f"generator.{key} >= num_regions"
                )

    workstation_count = (
        sum(ingredient_counts)
        + config["pots"]
        + config["plate_piles"]
        + config["depots"]
        + (len(config["possible_recipes"]) > 1)
        + button_count
    )
    boundary_slots = 2 * (width - 2) + 2 * (height - 2)
    max_interior_workstations = interior_tiles - counter_count - 2
    placement = config["object_placement"]
    if placement == "boundary" and workstation_count > boundary_slots:
        raise ValueError(
            f"{workstation_count} workstations do not fit in the "
            f"{boundary_slots} non-corner boundary slots"
        )
    if placement == "interior" and workstation_count > max_interior_workstations:
        raise ValueError(
            f"{workstation_count} workstations, {counter_count} counters, and "
            f"2 agents do not fit in the {interior_tiles} interior tiles"
        )
    if (
        placement == "anywhere"
        and workstation_count > boundary_slots + max_interior_workstations
    ):
        raise ValueError(
            f"{workstation_count} workstations do not fit in the available "
            "boundary and interior positions"
        )
    if placement == "boundary":
        minimum_interior_workstations = 0
    elif placement == "interior":
        minimum_interior_workstations = workstation_count
    else:
        minimum_interior_workstations = max(
            0,
            workstation_count - boundary_slots,
        )
    maximum_floor_tiles = (
        interior_tiles
        - counter_count
        - minimum_interior_workstations
    )
    if pressure_plate_count + 2 > maximum_floor_tiles:
        raise ValueError(
            f"{pressure_plate_count} pressure plates and 2 agent spawns do not "
            f"fit in at most {maximum_floor_tiles} walkable interior tiles"
        )
    return config


def _boundary_slots(width: int, height: int) -> list[tuple[int, int]]:
    return (
        [(0, col) for col in range(1, width - 1)]
        + [(height - 1, col) for col in range(1, width - 1)]
        + [(row, 0) for row in range(1, height - 1)]
        + [(row, width - 1) for row in range(1, height - 1)]
    )


def _neighbours(
    position: tuple[int, int],
    positions: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    row, col = position
    return [
        (row + row_delta, col + col_delta)
        for row_delta, col_delta in NEIGHBOUR_DELTAS
        if (row + row_delta, col + col_delta) in positions
    ]


def _station_symbols(config: dict[str, Any]) -> list[str]:
    stations = []
    for ingredient_idx, pile_count in enumerate(config["ingredient_piles"]):
        stations.extend([str(ingredient_idx)] * pile_count)
    stations.extend(["P"] * config["pots"])
    stations.extend(["B"] * config["plate_piles"])
    stations.extend(["X"] * config["depots"])
    if len(config["possible_recipes"]) > 1:
        stations.append("R")
    stations.extend(
        ["!"] * (config["barriers"] * config["buttons_per_barrier"])
    )
    return stations


def _allocate_stations_to_regions(
    config: dict[str, Any],
    rng: random.Random,
) -> list[list[str]]:
    """Assign every workstation to the region that must access it."""
    num_regions = config["num_regions"]
    allocations = [[] for _ in range(num_regions)]
    stations = _station_symbols(config)
    mode = config["workflow_mode"]

    if num_regions == 1 or mode == "single_region":
        productive_region = rng.randrange(num_regions)
        allocations[productive_region] = stations
        return allocations

    if mode == "shared":
        # Split the ordered cooking pipeline at a random stage. Keeping every
        # instance of one station type on the same side prevents either region
        # from accidentally becoming independently complete.
        first_region = rng.randrange(2)
        second_region = 1 - first_region
        split_after = rng.randint(1, 3)
        stages = [
            [
                str(ingredient_idx)
                for ingredient_idx, pile_count in enumerate(
                    config["ingredient_piles"]
                )
                for _ in range(pile_count)
            ],
            ["P"] * config["pots"],
            ["B"] * config["plate_piles"],
            ["X"] * config["depots"],
        ]
        for stage_idx, stage in enumerate(stages):
            target = first_region if stage_idx < split_after else second_region
            allocations[target].extend(stage)
        if len(config["possible_recipes"]) > 1:
            allocations[rng.randrange(num_regions)].append("R")
        for _ in range(config["barriers"] * config["buttons_per_barrier"]):
            allocations[rng.randrange(num_regions)].append("!")
        return allocations

    # complete_each: reserve one complete workflow in every region, then
    # distribute duplicate/unused workstations for variety.
    remaining = stations.copy()
    required_ingredients = sorted(
        {
            ingredient_idx
            for recipe in config["possible_recipes"]
            for ingredient_idx in recipe
        }
    )
    for region in range(num_regions):
        required_symbols = [str(idx) for idx in required_ingredients]
        required_symbols.extend(["P", "B", "X"])
        for symbol in required_symbols:
            remaining.remove(symbol)
            allocations[region].append(symbol)
    rng.shuffle(remaining)
    for symbol in remaining:
        allocations[rng.randrange(num_regions)].append(symbol)
    return allocations


def _choose_region_seeds(
    interior: set[tuple[int, int]],
    num_regions: int,
    rng: random.Random,
) -> list[tuple[int, int]]:
    first = rng.choice(tuple(sorted(interior)))
    if num_regions == 1:
        return [first]

    distances = {
        position: abs(position[0] - first[0]) + abs(position[1] - first[1])
        for position in interior
    }
    max_distance = max(distances.values())
    farthest = [
        position
        for position, distance in distances.items()
        if distance == max_distance
    ]
    return [first, rng.choice(farthest)]


def _grow_floor_regions(
    interior: set[tuple[int, int]],
    floor_count: int,
    num_regions: int,
    rng: random.Random,
) -> list[set[tuple[int, int]]]:
    """Grow exact-size, mutually disconnected floor regions from frontiers."""
    seeds = _choose_region_seeds(interior, num_regions, rng)
    regions = [{seed} for seed in seeds]
    owners = {seed: region for region, seed in enumerate(seeds)}

    while len(owners) < floor_count:
        candidates: list[tuple[int, tuple[int, int]]] = []
        region_sizes = [len(region) for region in regions]
        smallest_size = min(region_sizes)

        for region_idx, region in enumerate(regions):
            # Prefer balanced growth when space permits, but allow a larger
            # component to keep growing if another frontier becomes boxed in.
            size_penalty = region_sizes[region_idx] - smallest_size
            for position in region:
                for adjacent in _neighbours(position, interior):
                    if adjacent in owners:
                        continue
                    adjacent_owners = {
                        owners[neighbour]
                        for neighbour in _neighbours(adjacent, interior)
                        if neighbour in owners
                    }
                    if adjacent_owners and adjacent_owners != {region_idx}:
                        continue
                    candidates.append((size_penalty, adjacent))

        if not candidates:
            raise CandidateGenerationError(
                "region frontiers cannot reach the requested floor count "
                "without merging"
            )

        minimum_penalty = min(penalty for penalty, _ in candidates)
        preferred = [
            position
            for penalty, position in candidates
            if penalty == minimum_penalty
        ]
        chosen = rng.choice(preferred)
        adjacent_regions = {
            owners[neighbour]
            for neighbour in _neighbours(chosen, interior)
            if neighbour in owners
        }
        region_idx = next(iter(adjacent_regions))
        owners[chosen] = region_idx
        regions[region_idx].add(chosen)

    return regions


def _adjacent_regions(
    position: tuple[int, int],
    regions: list[set[tuple[int, int]]],
    all_grid_positions: set[tuple[int, int]],
) -> set[int]:
    return {
        region_idx
        for neighbour in _neighbours(position, all_grid_positions)
        for region_idx, region in enumerate(regions)
        if neighbour in region
    }


def _shared_tiles(
    grid: list[list[str]],
    regions: list[set[tuple[int, int]]],
    interior: set[tuple[int, int]],
    all_grid_positions: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    """Return ordinary counters that can be interacted with from both regions."""
    return [
        position
        for position in interior - set().union(*regions)
        if grid[position[0]][position[1]] == "W"
        and _adjacent_regions(position, regions, all_grid_positions) == {0, 1}
    ]


def _connected_floor_components(
    grid: list[list[str]],
) -> list[set[tuple[int, int]]]:
    """Return four-connected walkable floor/pressure-plate components."""
    all_positions = {
        (row, col)
        for row in range(len(grid))
        for col in range(len(grid[0]))
    }
    floor = {
        (row, col)
        for row, line in enumerate(grid)
        for col, symbol in enumerate(line)
        if symbol in {" ", "_"}
    }
    components = []
    while floor:
        start = min(floor)
        floor.remove(start)
        component = {start}
        frontier = [start]
        while frontier:
            position = frontier.pop()
            for adjacent in _neighbours(position, all_positions):
                if adjacent in floor:
                    floor.remove(adjacent)
                    component.add(adjacent)
                    frontier.append(adjacent)
        components.append(component)
    return components


def _place_barriers_and_controls(
    grid: list[list[str]],
    config: dict[str, Any],
    regions: list[set[tuple[int, int]]],
    shared_tiles: list[tuple[int, int]],
    rng: random.Random,
) -> tuple[
    list[bool],
    list[tuple[int, int]],
    list[tuple[int, int]],
]:
    """Place active barriers and wire generated buttons and pressure plates.

    Control targets are returned in row-major tile order, matching
    ``Layout.from_string`` parsing for ``button_config`` and
    ``pressure_plate_config``.
    """
    barrier_count = config["barriers"]
    if barrier_count == 0:
        return [], [], []

    height, width = config["height"], config["width"]
    all_positions = {
        (row, col)
        for row in range(height)
        for col in range(width)
    }
    floor = set().union(*regions)
    shared_candidates = set(shared_tiles)
    action_adjacent_candidates = {
        position
        for position in floor
        if any(
            grid[adjacent[0]][adjacent[1]] in ACTION_ITEM_SYMBOLS
            for adjacent in _neighbours(position, all_positions)
        )
    }
    anywhere_candidates = floor | {
        position
        for position in all_positions
        if grid[position[0]][position[1]] == "W"
        and _adjacent_regions(position, regions, all_positions)
    }

    placement = config["barrier_placement"]
    if placement == "shared":
        candidates = shared_candidates
    elif placement == "action_adjacent":
        candidates = action_adjacent_candidates
    elif placement == "shared_or_action_adjacent":
        candidates = shared_candidates | action_adjacent_candidates
    else:
        candidates = anywhere_candidates

    if len(candidates) < barrier_count:
        raise CandidateGenerationError(
            f"only {len(candidates)} legal {placement!r} barrier position(s) "
            f"exist; requested {barrier_count}"
        )

    barrier_positions = rng.sample(sorted(candidates), barrier_count)
    for row, col in barrier_positions:
        grid[row][col] = "#"

    floor_components = _connected_floor_components(grid)
    if len(floor_components) != config["num_regions"]:
        raise CandidateGenerationError(
            "barrier placement split a walkable region; retrying with "
            "non-separating barrier positions"
        )

    plates_per_barrier = config["pressure_plates_per_barrier"]
    pressure_plate_count = barrier_count * plates_per_barrier
    available_plate_positions = set().union(*floor_components)
    if len(available_plate_positions) < pressure_plate_count + len(floor_components):
        raise CandidateGenerationError(
            "too few floor tiles remain for pressure plates and one agent "
            "spawn per region"
        )

    # Reserve one ordinary floor tile in every component for its agent spawn.
    reserved_spawns = {
        rng.choice(sorted(component))
        for component in floor_components
    }
    plate_positions = rng.sample(
        sorted(available_plate_positions - reserved_spawns),
        pressure_plate_count,
    )

    barrier_positions = sorted(barrier_positions)
    barrier_index = {
        position: index for index, position in enumerate(barrier_positions)
    }
    target_assignments = [
        barrier_index[position]
        for position in barrier_positions
        for _ in range(plates_per_barrier)
    ]
    rng.shuffle(target_assignments)
    target_by_plate_position = dict(zip(plate_positions, target_assignments))
    for row, col in plate_positions:
        grid[row][col] = "_"

    pressure_plate_targets = [
        (target_by_plate_position[position], int(ButtonAction.TOGGLE_BARRIER))
        for position in sorted(plate_positions)
    ]

    buttons_per_barrier = config["buttons_per_barrier"]
    button_positions = sorted(
        (row, col)
        for row, line in enumerate(grid)
        for col, symbol in enumerate(line)
        if symbol == "!"
    )
    expected_button_count = barrier_count * buttons_per_barrier
    if len(button_positions) != expected_button_count:
        raise CandidateGenerationError(
            f"layout has {len(button_positions)} buttons; expected "
            f"{expected_button_count}"
        )
    button_target_assignments = [
        barrier_idx
        for barrier_idx in range(barrier_count)
        for _ in range(buttons_per_barrier)
    ]
    rng.shuffle(button_target_assignments)
    button_targets = [
        (target_idx, int(ButtonAction.TIMED_BARRIER))
        for target_idx in button_target_assignments
    ]
    return [True] * barrier_count, button_targets, pressure_plate_targets


def _place_region_stations(
    grid: list[list[str]],
    config: dict[str, Any],
    regions: list[set[tuple[int, int]]],
    allocations: list[list[str]],
    interior_station_count: int,
    rng: random.Random,
) -> None:
    height, width = config["height"], config["width"]
    interior = {
        (row, col)
        for row in range(1, height - 1)
        for col in range(1, width - 1)
    }
    all_positions = {
        (row, col)
        for row in range(height)
        for col in range(width)
    }
    boundary = set(_boundary_slots(width, height))
    floor = set().union(*regions)
    available_interior = interior - floor

    candidates_by_region: list[dict[str, list[tuple[int, int]]]] = []
    for region_idx in range(len(regions)):
        candidates_by_region.append(
            {
                "boundary": [
                    position
                    for position in boundary
                    if _adjacent_regions(position, regions, all_positions)
                    == {region_idx}
                ],
                "interior": [
                    position
                    for position in available_interior
                    if _adjacent_regions(position, regions, all_positions)
                    == {region_idx}
                ],
            }
        )

    station_assignments = [
        (region_idx, symbol)
        for region_idx, symbols in enumerate(allocations)
        for symbol in symbols
    ]
    rng.shuffle(station_assignments)

    placement = config["object_placement"]
    if placement == "boundary":
        slot_kinds = ["boundary"] * len(station_assignments)
    elif placement == "interior":
        slot_kinds = ["interior"] * len(station_assignments)
    else:
        slot_kinds = (
            ["interior"] * interior_station_count
            + ["boundary"] * (len(station_assignments) - interior_station_count)
        )
        rng.shuffle(slot_kinds)

    # Assign the more constrained region/kind combinations first.
    planned = list(zip(station_assignments, slot_kinds))
    planned.sort(
        key=lambda item: len(candidates_by_region[item[0][0]][item[1]])
    )
    used = set()
    for (region_idx, symbol), kind in planned:
        choices = [
            position
            for position in candidates_by_region[region_idx][kind]
            if position not in used
        ]
        if not choices:
            raise CandidateGenerationError(
                f"region {region_idx} has too few accessible {kind} "
                "workstation slots"
            )
        position = rng.choice(choices)
        used.add(position)
        grid[position[0]][position[1]] = symbol


def _generate_candidate(
    config: dict[str, Any],
    rng: random.Random,
) -> tuple[
    str,
    list[bool],
    list[tuple[int, int]],
    list[tuple[int, int]],
]:
    width, height = config["width"], config["height"]
    grid = [["W"] * width for _ in range(height)]
    interior = {
        (row, col)
        for row in range(1, height - 1)
        for col in range(1, width - 1)
    }
    stations = _station_symbols(config)
    counter_count = round(len(interior) * config["counter_density"])
    max_interior_workstations = len(interior) - counter_count - 2
    placement = config["object_placement"]
    if placement == "boundary":
        interior_station_count = 0
    elif placement == "interior":
        interior_station_count = len(stations)
    else:
        minimum = max(0, len(stations) - len(_boundary_slots(width, height)))
        maximum = min(len(stations), max_interior_workstations)
        interior_station_count = rng.randint(minimum, maximum)

    floor_count = len(interior) - counter_count - interior_station_count
    regions = _grow_floor_regions(
        interior,
        floor_count,
        config["num_regions"],
        rng,
    )
    for region in regions:
        for row, col in region:
            grid[row][col] = " "

    allocations = _allocate_stations_to_regions(config, rng)
    _place_region_stations(
        grid,
        config,
        regions,
        allocations,
        interior_station_count,
        rng,
    )

    all_positions = {
        (row, col)
        for row in range(height)
        for col in range(width)
    }
    shared_tiles = (
        _shared_tiles(
            grid,
            regions,
            interior,
            all_positions,
        )
        if config["num_regions"] == 2
        else []
    )
    if config["num_regions"] == 2:
        requested_shared_tiles = config["num_shared_tiles"]
        if (
            requested_shared_tiles is not None
            and len(shared_tiles) != requested_shared_tiles
        ):
            raise CandidateGenerationError(
                f"layout has {len(shared_tiles)} shared tiles; "
                f"requested exactly {requested_shared_tiles}"
            )
    if config["workflow_mode"] == "shared":
        all_positions = {
            (row, col)
            for row in range(height)
            for col in range(width)
        }
        shared_tiles = _shared_tiles(
            grid,
            regions,
            interior,
            all_positions,
        )
        requested_shared_tiles = config["num_shared_tiles"]
        if (
            requested_shared_tiles is not None
            and len(shared_tiles) != requested_shared_tiles
        ):
            raise CandidateGenerationError(
                f"layout has {len(shared_tiles)} shared tiles; "
                f"requested exactly {requested_shared_tiles}"
            )
        if config["workflow_mode"] == "shared" and not shared_tiles:
            raise CandidateGenerationError(
                "shared workflow has no counter accessible from both regions"
            )

    barrier_config, button_config, pressure_plate_config = (
        _place_barriers_and_controls(
            grid,
            config,
            regions,
            shared_tiles,
            rng,
        )
    )

    final_regions = _connected_floor_components(grid)
    if config["num_regions"] == 1:
        spawn_candidates = sorted(
            position
            for position in final_regions[0]
            if grid[position[0]][position[1]] == " "
        )
        if len(spawn_candidates) < 2:
            raise CandidateGenerationError(
                "too few ordinary floor tiles remain for two agent spawns"
            )
        spawn_positions = rng.sample(spawn_candidates, 2)
    else:
        spawn_positions = [
            rng.choice(
                sorted(
                    position
                    for position in region
                    if grid[position[0]][position[1]] == " "
                )
            )
            for region in final_regions
        ]
    for row, col in spawn_positions:
        grid[row][col] = "A"

    return (
        "\n".join("".join(row) for row in grid),
        barrier_config,
        button_config,
        pressure_plate_config,
    )


def generate_layout(
    config: dict[str, Any],
    rng: random.Random,
) -> tuple[str, Layout, int]:
    """Construct one layout, retrying only recoverable frontier dead ends."""
    last_errors = []
    for attempt in range(1, config["max_attempts"] + 1):
        try:
            (
                grid,
                barrier_config,
                button_config,
                pressure_plate_config,
            ) = _generate_candidate(config, rng)
        except CandidateGenerationError as exc:
            last_errors = [str(exc)]
            continue
        layout = Layout.from_string(
            grid,
            possible_recipes=config["possible_recipes"],
            button_config=button_config,
            barrier_config=barrier_config,
            pressure_plate_config=pressure_plate_config,
        )
        valid, last_errors = validate_generated_layout(layout)
        if valid:
            return grid, layout, attempt
    raise RuntimeError(
        f"Could not generate a valid layout after {config['max_attempts']} "
        f"attempts. Last validation error(s): {'; '.join(last_errors)}"
    )


def _layout_entry(grid: str, layout: Layout, config: dict[str, Any]) -> dict[str, Any]:
    """Serialize a generated layout, including barrier control metadata."""
    return {
        "ascii": grid,
        "possible_recipes": config["possible_recipes"],
        "button_config": [
            [list(target_idxs), int(action_type)]
            for _, _, target_idxs, action_type in layout.button_info
        ],
        "barrier_config": [
            bool(active) for _, _, active in layout.barrier_info
        ],
        "pressure_plate_config": [
            [list(target_idxs), int(action_type)]
            for _, _, target_idxs, action_type in layout.pressure_plate_info
        ],
        "validation": {"valid": True, "errors": []},
    }


def generate_document(document: Any) -> dict[str, Any]:
    """Generate all layouts requested by a parsed JSON document."""
    if not isinstance(document, dict):
        raise ValueError("The JSON document must be an object")
    config = validate_config(document.get("generator"))
    rng = random.Random(config["seed"])
    generated = {}

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
    """Atomically write a generation checkpoint."""
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
    """Generate layouts with an atomic JSON checkpoint after every map.

    A failed map is recorded in ``generation_errors`` and generation continues
    with the next name. Previously completed maps remain usable if generation
    later fails or the process is interrupted.
    """
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
    failures = result["generation_errors"]

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
                f"(constructed and validated after {attempts} attempt(s)); "
                f"{result['generation_progress']['completed']} map(s) complete"
            )

    result["generation_progress"]["status"] = (
        "completed_with_errors" if failures else "complete"
    )
    _write_json_checkpoint(result, output_path)
    return result, failures


def parse_args() -> argparse.Namespace:
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
