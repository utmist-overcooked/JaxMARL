# Overcooked V3 JSON Layout Generation

The Overcooked V3 layout generator creates reproducible ASCII kitchens from a
single JSON file. The same file contains both the generation settings and the
generated layouts, including their recipes and validation results.

The relevant files are:

- `scripts/generate_overcooked_v3_layouts.py`: command-line generator.
- `scripts/overcooked_v3_layouts.example.json`: example configuration.
- `jaxmarl/environments/overcooked_v3/layouts.py`: parsing, loading, and
  validation.
- `scripts/play_overcooked_v3.py`: interactive pygame viewer.

## JSON configuration

The input JSON must contain a top-level `generator` object. The generator writes
the results to the top-level `layouts` object.

```json
{
  "generator": {
    "seed": 7,
    "count": 3,
    "name_prefix": "random_kitchen",
    "width": 8,
    "height": 6,
    "ingredient_piles": [2, 1],
    "possible_recipes": [
      [0, 0, 0],
      [1, 1, 1]
    ],
    "pots": 1,
    "plate_piles": 1,
    "depots": 1,
    "object_placement": "anywhere",
    "counter_density": 0.1,
    "max_attempts": 1000
  },
  "layouts": {}
}
```

### Generator settings

| Setting | Meaning |
| --- | --- |
| `seed` | Non-negative random seed. The same configuration and seed produce the same layouts. |
| `count` | Number of layouts to generate. |
| `name_prefix` | Prefix used for layout names, such as `random_kitchen_0`. |
| `width`, `height` | Exact ASCII dimensions. Both must be at least 5. |
| `ingredient_piles` | Pile counts indexed by ingredient type. `[2, 1]` places two type-`0` piles and one type-`1` pile. |
| `possible_recipes` | Three-ingredient recipes that may be requested by the environment. |
| `pots` | Number of pots, up to the V3 `MAX_POTS` setting. |
| `plate_piles` | Number of plate dispensers. |
| `depots` | Number of delivery zones. |
| `object_placement` | Workstation placement mode: `boundary`, `interior`, or `anywhere`. |
| `counter_density` | Fraction of interior tiles converted to counters, in the range `[0, 1)`. |
| `max_attempts` | Maximum random candidates tried for each requested layout. |

`possible_recipes` may be omitted. In that case, the generator creates one
same-ingredient recipe for every ingredient type with at least one pile.

Overcooked V3 currently supports at most three ingredient types and
same-ingredient soups. Mixed recipes such as `[0, 0, 1]` are rejected by the
generator because the current pot logic cannot complete them.

`object_placement` controls all ingredient piles, pots, plate piles, and
delivery depots:

- `boundary` preserves the original behavior and uses only non-corner border
  cells.
- `interior` places every workstation inside the border.
- `anywhere` randomly chooses a feasible mixture of boundary and interior
  workstation positions for each candidate.

Interior workstations occupy otherwise walkable floor positions, like internal
counters. The capacity checks reserve room for the configured interior
counters and both agents.

## ASCII symbols

| Symbol | Tile |
| --- | --- |
| `W` | Wall or counter |
| `A` | Agent spawn |
| `0`, `1`, `2` | Ingredient pile by ingredient index |
| `P` | Pot |
| `B` | Plate pile |
| `X` | Delivery depot |
| Space | Walkable floor |

Each generated layout contains exactly two `A` spawn positions.

## Generation logic

For every requested layout, the generator:

1. Creates an exact `width` by `height` grid.
2. Fills the boundary with counters and the interior with walkable floor.
3. Randomly places ingredient piles, pots, plate piles, and depots according to
   `object_placement`.
4. Converts a configurable fraction of interior floor tiles into counters.
5. Randomly places two agents on remaining floor tiles.
6. Parses the ASCII through `Layout.from_string`.
7. Runs structural, playability, accessibility, and recipe-solvability checks.
8. Retries with another random candidate if validation fails.

Generation prints the current layout number, its name, the number of attempts
needed, and the number of maps completed. The output JSON is atomically
checkpointed after every success or failure. If the process is interrupted,
all layouts completed before the interruption remain in the file.

If a map cannot be generated within `max_attempts`, its error is recorded and
generation continues with the next requested map. Invalid maps are never added
to `layouts`.

## Validation guarantees

`validate_generated_layout` combines the existing V3 playability checks with
the generated-layout accessibility checks.

A generated map is accepted only when:

- It has exactly two unique agent spawns on walkable tiles.
- It has at least one ingredient pile, pot, plate pile, and delivery depot.
- Every walkable tile can be reached from at least one agent spawn using
  four-direction movement.
- Every ingredient pile, pot, plate pile, and depot can be interacted with from
  a reachable adjacent tile.
- Every configured recipe has its required ingredient piles.
- Every recipe can be completed within one connected agent-accessible region
  containing the required ingredient, a pot, plates, and a depot.
- The layout stays within fixed V3 capacities such as `MAX_POTS`.

Walls and counters are intentionally not treated as walkable tiles.

## Generate layouts

To update the example JSON in place:

```bash
python scripts/generate_overcooked_v3_layouts.py \
  scripts/overcooked_v3_layouts.example.json
```

To preserve the input configuration and write a separate result:

```bash
python scripts/generate_overcooked_v3_layouts.py \
  scripts/overcooked_v3_layouts.example.json \
  --output generated-layouts.json
```

The output contains entries similar to:

```json
{
  "layouts": {
    "random_kitchen_0": {
      "ascii": "WW0PWWWW\nW A    B\nW      W\nW  W A X\nW      W\nWWW1WWWW",
      "possible_recipes": [
        [0, 0, 0],
        [1, 1, 1]
      ],
      "validation": {
        "valid": true,
        "errors": []
      }
    }
  }
}
```

The file also tracks checkpoint progress:

```json
{
  "generation_progress": {
    "requested": 100,
    "completed": 42,
    "failed": 1,
    "status": "running"
  },
  "generation_errors": {
    "random_kitchen_17": "Could not generate a valid layout after 1000 attempts..."
  }
}
```

`status` becomes `complete` when every layout succeeds or
`completed_with_errors` when one or more layouts fail. The command returns a
non-zero exit code when failures occurred, but all successful maps remain
loadable from the checkpoint.

The generator replaces the `layouts` object each time it runs. Change the seed
or copy the output elsewhere before regenerating if previous maps must be kept.

## Load layouts in Python

Load layouts without modifying the global registry:

```python
from jaxmarl.environments.overcooked_v3 import (
    OvercookedV3,
    load_layouts_from_json,
)

layouts = load_layouts_from_json("generated-layouts.json")
env = OvercookedV3(layout=layouts["random_kitchen_0"])
```

Register every JSON layout so it can be selected by name:

```python
from jaxmarl.environments.overcooked_v3 import (
    OvercookedV3,
    load_layouts_from_json,
)

load_layouts_from_json("generated-layouts.json", register=True)
env = OvercookedV3(layout="random_kitchen_0")
```

Registration rejects names that already exist. Use `overwrite=True` only when
replacing a registered layout is intentional:

```python
load_layouts_from_json(
    "generated-layouts.json",
    register=True,
    overwrite=True,
)
```

JSON layouts are independently parsed and validated while loading. The loader
does not trust the stored `validation.valid` field.

## View layouts interactively

Generate the file, then pass it to the pygame player:

```bash
python scripts/play_overcooked_v3.py \
  --layout-json scripts/overcooked_v3_layouts.example.json
```

When `--layout` is omitted, the player opens the first generated layout. Use N
and P to cycle through layouts from the JSON file.

Open a particular generated layout:

```bash
python scripts/play_overcooked_v3.py \
  --layout-json scripts/overcooked_v3_layouts.example.json \
  --layout random_kitchen_2
```

List built-in and loaded layout names without opening pygame:

```bash
python scripts/play_overcooked_v3.py \
  --layout-json scripts/overcooked_v3_layouts.example.json \
  --list
```

Interactive controls:

| Control | Action |
| --- | --- |
| Agent 0: W/A/S/D | Move |
| Agent 0: Space | Interact |
| Agent 1: Arrow keys | Move |
| Agent 1: Enter | Interact |
| N / P | Next or previous generated layout |
| R | Reset |
| Q / Escape | Quit |

## Troubleshooting

- **Workstations do not fit:** Increase the width or height, or reduce the
  station counts. Boundary mode uses non-corner border slots; interior mode
  must also leave space for counters and two agent spawns.
- **Could not generate a valid layout:** Reduce `counter_density`, increase
  `max_attempts`, or increase the map dimensions.
- **Mixed recipe rejected:** Use recipes such as `[0, 0, 0]` and `[1, 1, 1]`.
- **Unknown layout in the player:** Supply `--layout-json` in the same command
  that supplies `--layout`.
- **Layout name already registered:** Rename the generated prefix or load with
  `overwrite=True` when replacement is deliberate.
