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

from jaxmarl.environments.overcooked_v3.common import MAX_INGREDIENTS
from jaxmarl.environments.overcooked_v3.layouts import (
    Layout,
    validate_generated_layout,
)
from jaxmarl.environments.overcooked_v3.settings import MAX_POTS


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
    "max_attempts": 1000,
}


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
    _integer(config, "max_attempts", 1)

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

    workstation_count = (
        sum(ingredient_counts)
        + config["pots"]
        + config["plate_piles"]
        + config["depots"]
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
    return config


def _boundary_slots(width: int, height: int) -> list[tuple[int, int]]:
    return (
        [(0, col) for col in range(1, width - 1)]
        + [(height - 1, col) for col in range(1, width - 1)]
        + [(row, 0) for row in range(1, height - 1)]
        + [(row, width - 1) for row in range(1, height - 1)]
    )


def _generate_candidate(config: dict[str, Any], rng: random.Random) -> str:
    width, height = config["width"], config["height"]
    grid = [["W"] * width for _ in range(height)]
    interior = [
        (row, col)
        for row in range(1, height - 1)
        for col in range(1, width - 1)
    ]
    for row, col in interior:
        grid[row][col] = " "

    stations = []
    for ingredient_idx, pile_count in enumerate(config["ingredient_piles"]):
        stations.extend([str(ingredient_idx)] * pile_count)
    stations.extend(["P"] * config["pots"])
    stations.extend(["B"] * config["plate_piles"])
    stations.extend(["X"] * config["depots"])
    rng.shuffle(stations)

    boundary = _boundary_slots(width, height)
    counter_count = round(len(interior) * config["counter_density"])
    max_interior_workstations = len(interior) - counter_count - 2
    placement = config["object_placement"]
    if placement == "boundary":
        interior_station_count = 0
    elif placement == "interior":
        interior_station_count = len(stations)
    else:
        minimum = max(0, len(stations) - len(boundary))
        maximum = min(len(stations), max_interior_workstations)
        interior_station_count = rng.randint(minimum, maximum)

    station_slots = rng.sample(interior, interior_station_count)
    station_slots.extend(
        rng.sample(boundary, len(stations) - interior_station_count)
    )
    rng.shuffle(station_slots)
    for (row, col), symbol in zip(station_slots, stations):
        grid[row][col] = symbol

    available_counter_slots = [
        (row, col) for row, col in interior if grid[row][col] == " "
    ]
    for row, col in rng.sample(available_counter_slots, counter_count):
        grid[row][col] = "W"

    spawn_candidates = [
        (row, col) for row, col in interior if grid[row][col] == " "
    ]
    for row, col in rng.sample(spawn_candidates, 2):
        grid[row][col] = "A"

    return "\n".join("".join(row) for row in grid)


def generate_layout(
    config: dict[str, Any],
    rng: random.Random,
) -> tuple[str, Layout, int]:
    """Generate one layout, retrying until every validator succeeds."""
    last_errors = []
    for attempt in range(1, config["max_attempts"] + 1):
        grid = _generate_candidate(config, rng)
        layout = Layout.from_string(
            grid,
            possible_recipes=config["possible_recipes"],
        )
        valid, last_errors = validate_generated_layout(layout)
        if valid:
            return grid, layout, attempt
    raise RuntimeError(
        f"Could not generate a valid layout after {config['max_attempts']} "
        f"attempts. Last validation error(s): {'; '.join(last_errors)}"
    )


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
        valid, errors = validate_generated_layout(layout)
        generated[name] = {
            "ascii": grid,
            "possible_recipes": config["possible_recipes"],
            "validation": {"valid": valid, "errors": errors},
        }

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

        valid, errors = validate_generated_layout(layout)
        result["layouts"][name] = {
            "ascii": grid,
            "possible_recipes": config["possible_recipes"],
            "validation": {"valid": valid, "errors": errors},
        }
        result["generation_progress"]["completed"] += 1
        _write_json_checkpoint(result, output_path)
        if emit is not None:
            emit(
                f"[{index + 1}/{config['count']}] Saved {name} "
                f"(valid candidate found after {attempts} attempt(s)); "
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
