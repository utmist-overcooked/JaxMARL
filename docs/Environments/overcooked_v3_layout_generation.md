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
    "num_regions": 1,
    "num_shared_tiles": null,
    "workflow_mode": "single_region",
    "barriers": 0,
    "barrier_placement": "anywhere",
    "pressure_plates_per_barrier": 1,
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
| `num_regions` | Number of disconnected walkable regions: `1` or `2`. Two regions place one agent in each. |
| `num_shared_tiles` | Exact number of ordinary counters accessible from both regions, or `null` to leave the count unconstrained. Requires two regions. |
| `workflow_mode` | Distribution of the cooking workflow: `single_region`, `complete_each`, or `shared`. |
| `barriers` | Exact number of active, pressure-plate-controlled barriers. At most 16. |
| `barrier_placement` | Barrier placement mode: `anywhere`, `shared`, `action_adjacent`, or `shared_or_action_adjacent`. |
| `pressure_plates_per_barrier` | `1` for one pressure plate per barrier or `2` for a pair. |
| `max_attempts` | Maximum constructive retries for recoverable frontier or placement dead ends. |

`possible_recipes` may be omitted. In that case, the generator creates one
same-ingredient recipe for every ingredient type with at least one pile.
When more than one recipe is possible, the generator places an accessible
`R` recipe indicator so observations expose the recipe selected for the
current episode.

Overcooked V3 currently supports at most three ingredient types and
same-ingredient soups. Mixed recipes such as `[0, 0, 1]` are rejected by the
generator because the current pot logic cannot complete them.

`object_placement` controls all ingredient piles, pots, plate piles, delivery
depots, and any required recipe indicator:

- `boundary` preserves the original behavior and uses only non-corner border
  cells.
- `interior` places every workstation inside the border.
- `anywhere` randomly chooses a feasible mixture of boundary and interior
  workstation positions for each candidate.

Interior workstations occupy otherwise walkable floor positions, like internal
counters. The capacity checks reserve room for the configured interior
counters and both agents.

### Barriers and pressure plates

Set `barriers` to the exact number of barriers required in each generated
layout. Generated barriers start active. Each pressure plate controls one
barrier with `TOGGLE_BARRIER`; standing on either plate in a configured pair
opens its barrier. The generator never places a pressure plate in a component
without an agent spawn.

`barrier_placement` controls the eligible barrier tiles:

- `anywhere` uses any empty floor or ordinary wall/counter tile accessible
  from a region.
- `shared` uses only interface counters accessible from both regions and
  therefore requires `num_regions` to be `2`.
- `action_adjacent` uses only floor tiles orthogonally adjacent to an
  ingredient pile, pot, plate pile, or delivery depot.
- `shared_or_action_adjacent` uses the union of the previous two candidate
  sets. In a one-region layout, only action-adjacent candidates can exist.

For shared placement, `num_shared_tiles` is checked before selected interface
counters are converted to barriers. A pressure-controlled barrier can connect
the two workflow regions when opened, so the accessibility validator treats it
as a valid dynamic connection.

`pressure_plates_per_barrier` must be `1` (single) or `2` (paired). The total
number of generated pressure plates cannot exceed the V3
`MAX_PRESSURE_PLATES` capacity of 16.

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

- `single_region` assigns all ingredient piles, pots, plate piles, and delivery
  zones to one randomly selected productive region. With two regions, the
  second agent is isolated from the cooking workflow unless the layout is later
  extended with auxiliary mechanics.
- `complete_each` places a complete copy of every configured recipe workflow
  in every region. Each region must be able to access the required ingredients,
  a pot, plates, and a delivery zone without a handoff. The configured pile and
  workstation counts must therefore be at least `num_regions`.
- `shared` requires exactly two regions and splits the ordered
  ingredient-to-pot-to-plate-to-delivery workflow between them. No region can
  complete a recipe alone. At least one ordinary counter must be accessible
  from both sides so agents can hand items across it.

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
| `0`, `1`, `2` | Ingredient pile by ingredient index |
| `P` | Pot |
| `B` | Plate pile |
| `X` | Delivery depot |
| `#` | Active pressure-plate-controlled barrier |
| `_` | Pressure plate |
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
5. Allocates workstations to regions according to `workflow_mode` and places
   each one only where its assigned region can interact with it.
6. For two regions, checks the requested exact shared-tile count; in `shared`
   mode, also verifies that at least one counter is accessible from both
   regions.
7. Places the requested barriers and one or two reachable pressure plates for
   each barrier.
8. Places both agents on carved floor, one per region when `num_regions` is two.
9. Parses the ASCII through `Layout.from_string` and runs the structural,
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
- Every pressure plate can be reached by at least one agent, and every
  generated barrier has exactly the configured number of linked plates.
- Every ingredient pile, pot, plate pile, and depot can be interacted with from
  a reachable adjacent tile.
- Every configured recipe has its required ingredient piles.
- Every recipe can be completed within one agent-accessible region or a group
  of regions connected through shared handoff counters.
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
      "barrier_config": [],
      "pressure_plate_config": [],
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
  `max_attempts`, increase the map dimensions, or reduce `num_regions`. Two
  sizeable regions need enough counters to keep their frontiers apart.
- **`complete_each` rejects the configuration:** Provide at least one required
  ingredient pile, pot, plate pile, and depot per region.
- **`shared` rejects the configuration:** Use `num_regions: 2` and a non-zero
  counter density so the regions can have a shared handoff counter.
- **The requested shared-tile count cannot be generated:** Increase
  `max_attempts`, adjust `counter_density` or the map dimensions, or choose a
  less restrictive `num_shared_tiles` value.
- **Mixed recipe rejected:** Use recipes such as `[0, 0, 0]` and `[1, 1, 1]`.
- **Unknown layout in the player:** Supply `--layout-json` in the same command
  that supplies `--layout`.
- **Layout name already registered:** Rename the generated prefix or load with
  `overwrite=True` when replacement is deliberate.
