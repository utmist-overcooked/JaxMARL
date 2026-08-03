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
    "cutting_boards": 0,
    "grills": 0,
    "blenders": 0,
    "sinks": 0,
    "dirty_plate_piles": 0,
    "object_placement": "anywhere",
    "counter_density": 0.1,
    "num_regions": 1,
    "num_shared_tiles": null,
    "workflow_mode": "single_region",
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
| `cutting_boards` | Number of cutting boards (`C`) for processing raw lettuce `2` into chopped lettuce `5`. |
| `grills` | Number of grills (`G`) for processing raw meat `3` into grilled meat `6`. |
| `blenders` | Number of blenders (`M`) for processing raw carrot `4` into carrot puree `7`. |
| `sinks` | Number of sinks (`S`). Must be positive together with `dirty_plate_piles` to generate a dish-washing layout. |
| `dirty_plate_piles` | Number of dirty-plate piles (`D`). Must be positive together with `sinks`. |
| `object_placement` | Workstation placement mode: `boundary`, `interior`, or `anywhere`. |
| `counter_density` | Fraction of interior tiles converted to counters, in the range `[0, 1)`. |
| `num_regions` | Number of disconnected walkable regions: `1` or `2`. Two regions place one agent in each. |
| `num_shared_tiles` | Exact number of ordinary counters accessible from both regions, or `null` to leave the count unconstrained. Requires two regions. |
| `workflow_mode` | Distribution of the cooking workflow: `single_region`, `complete_each`, or `shared`. |
| `max_attempts` | Maximum constructive retries for recoverable frontier or placement dead ends. |

`possible_recipes` may be omitted. In that case, the generator creates one
same-ingredient recipe for every raw ingredient type with at least one pile.
For prep ingredients, the generated recipe uses the processed result whenever
the matching station is configured: pile `2` plus a cutting board produces
recipe `[5, 5, 5]`, pile `3` plus a grill produces `[6, 6, 6]`, and pile `4`
plus a blender produces `[7, 7, 7]`. Without the station, the raw ingredient
index is used. When more than one recipe is possible, the generator places an
accessible `R` recipe indicator so observations expose the recipe selected for
the current episode.

`ingredient_piles` describes up to five raw pile types (`0` through `4`).
Processed types `5`, `6`, and `7` are never spawned as piles; they are produced
at `C`, `G`, and `M`. An explicit processed recipe is rejected unless its raw
source pile and matching prep station are configured. Mixed recipes such as
`[0, 0, 1]` remain unsupported because the current pot logic only completes
same-ingredient soups.

`object_placement` controls all ingredient piles, prep stations, pots, plate
piles, delivery depots, sinks, dirty-plate piles, and any required recipe
indicator:

- `boundary` preserves the original behavior and uses only non-corner border
  cells.
- `interior` places every workstation inside the border.
- `anywhere` randomly chooses a feasible mixture of boundary and interior
  workstation positions for each candidate.

Interior workstations occupy otherwise walkable floor positions, like internal
counters. The capacity checks reserve room for the configured interior
counters and both agents.

### Prep and dish-washing example

This configuration generates kitchens that can prepare all three processed
recipes and recycle a finite plate supply:

```json
{
  "generator": {
    "seed": 12,
    "count": 10,
    "name_prefix": "prep_dish_kitchen",
    "width": 12,
    "height": 10,
    "ingredient_piles": [0, 0, 1, 1, 1],
    "possible_recipes": [
      [5, 5, 5],
      [6, 6, 6],
      [7, 7, 7]
    ],
    "pots": 1,
    "plate_piles": 1,
    "depots": 1,
    "cutting_boards": 1,
    "grills": 1,
    "blenders": 1,
    "sinks": 1,
    "dirty_plate_piles": 1,
    "object_placement": "anywhere",
    "counter_density": 0.2,
    "num_regions": 1,
    "num_shared_tiles": null,
    "workflow_mode": "single_region",
    "max_attempts": 1000
  },
  "layouts": {}
}
```

Load the result with `enable_dish_washing=True` to activate finite plates,
dirty-plate pickup, and sinks. With dish washing disabled, generated `S` and
`D` tiles retain the environment's existing behavior and become inert
counters.

### Regions and workflow modes

The generator grows one connected floor frontier per region. Different
frontiers are not allowed to merge, and every region receives an agent spawn.
Because generated layouts contain exactly two agents, `num_regions` is limited
to one or two.

For two-region layouts, `num_shared_tiles` can constrain the exact number of
handoff counters between the regions. A tile counts as shared when it is an
ordinary `W` counter with an orthogonally adjacent floor tile in each region.
For example, set `"num_shared_tiles": 3` to require exactly three such
counters. The default, `null`, preserves unconstrained generation. The
`shared` workflow mode still requires at least one shared tile, so it cannot be
combined with a value of `0`.

`workflow_mode` controls which region can access each workstation:

- `single_region` assigns all ingredient piles, prep stations, pots, plate
  piles, delivery zones, and dish-washing stations to one randomly selected
  productive region. With two regions, the second agent is isolated from the
  cooking workflow unless the layout is later extended with auxiliary
  mechanics.
- `complete_each` places a complete copy of every configured recipe workflow
  in every region. Each region must be able to access the required ingredients,
  required prep stations, a pot, plates, a delivery zone, and—when configured—a
  sink and dirty-plate pile without a handoff. Required pile and workstation
  counts must therefore be at least `num_regions`.
- `shared` requires exactly two regions and splits the ordered
  ingredient-to-prep-to-pot-to-plate-to-delivery-to-washing workflow between
  them. Empty stages are omitted. No region can complete the configured
  workflow alone. At least one ordinary counter must be accessible from both
  sides so agents can hand items across it.

For example, a two-region shared workflow can place ingredients and pots in
one region, with plates and delivery in the other:

```text
region 0: ingredient -> pot -> shared counter
                                     |
region 1:                    plate -> delivery
```

The accessibility validator treats floor components joined by shared counters
as one workflow group while still requiring every floor tile to be reachable
from an agent.

## ASCII symbols

| Symbol | Tile |
| --- | --- |
| `W` | Wall or counter |
| `A` | Agent spawn |
| `0` through `4` | Raw ingredient pile by ingredient index |
| `P` | Pot |
| `B` | Plate pile |
| `X` | Delivery depot |
| `C` | Cutting board |
| `G` | Grill |
| `M` | Blender |
| `S` | Sink |
| `D` | Dirty-plate pile |
| `R` | Recipe indicator, included when multiple recipes are possible |
| Space | Walkable floor |

Each generated layout contains exactly two `A` spawn positions.

## Generation logic

For every requested layout, the generator:

1. Creates an exact `width` by `height` counter-filled grid.
2. Calculates the exact number of walkable, workstation, and counter tiles.
3. Selects separated seeds for the requested number of regions.
4. Uses randomized frontier growth to carve the exact number of floor tiles.
   A new tile can join only one region, which preserves connectivity without a
   generate-and-reject connectivity search.
5. Allocates ingredient, prep, cooking, serving, and dish-washing workstations
   to regions according to `workflow_mode` and places each one only where its
   assigned region can interact with it.
6. For two regions, checks the requested exact shared-tile count; in `shared`
   mode, also verifies that at least one counter is accessible from both
   regions.
7. Places both agents on carved floor, one per region when `num_regions` is two.
8. Parses the ASCII through `Layout.from_string` and runs the structural,
   playability, accessibility, handoff, and recipe-solvability checks once.

Constructive generation normally succeeds on the first attempt. A retry can
still occur when a randomized frontier shape leaves too few legal workstation
slots or cannot reach the exact requested floor count without merging regions.

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
- Every ingredient pile, prep station, pot, plate pile, depot, sink, and
  dirty-plate pile can be interacted with from a reachable adjacent tile.
- Every configured recipe has its required source piles and prep stations.
- Sink and dirty-plate-pile settings are either both zero or both positive.
- Every recipe can be completed within one agent-accessible region or a group
  of regions connected through shared handoff counters; configured dish
  washing stations must be in the same workflow group.
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
env = OvercookedV3(
    layout=layouts["random_kitchen_0"],
    enable_dish_washing=True,  # Use when the generated layout contains S and D.
)
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

For a generated layout containing `S` and `D`, activate the dish cycle with:

```bash
python scripts/play_overcooked_v3.py \
  --layout-json generated-prep-dish-layouts.json \
  --enable-dish-washing
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
  `max_attempts`, increase the map dimensions, or reduce `num_regions`. Two
  sizeable regions need enough counters to keep their frontiers apart.
- **`complete_each` rejects the configuration:** Provide at least one required
  source pile, required prep station, pot, plate pile, and depot per region.
  Dish-washing layouts also need a sink and dirty-plate pile per region.
- **`shared` rejects the configuration:** Use `num_regions: 2` and a non-zero
  counter density so the regions can have a shared handoff counter.
- **The requested shared-tile count cannot be generated:** Increase
  `max_attempts`, adjust `counter_density` or the map dimensions, or choose a
  less restrictive `num_shared_tiles` value.
- **Mixed recipe rejected:** Use recipes such as `[0, 0, 0]` and `[1, 1, 1]`.
- **Processed recipe rejected:** Recipe `5` needs raw pile `2` and a cutting
  board, recipe `6` needs raw pile `3` and a grill, and recipe `7` needs raw
  pile `4` and a blender.
- **Dish-washing pair rejected:** Set both `sinks` and `dirty_plate_piles` to a
  positive count, or set both to `0`.
- **Unknown layout in the player:** Supply `--layout-json` in the same command
  that supplies `--layout`.
- **Layout name already registered:** Rename the generated prefix or load with
  `overwrite=True` when replacement is deliberate.
