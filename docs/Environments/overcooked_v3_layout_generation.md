# Overcooked V3 JSON Layout Generation

The Overcooked V3 procedural map generator (PMG) creates reproducible ASCII
kitchens for three deliberate workflow experiments:

- `asymmetric_info` separates the agents but gives them discrete counters for
  passing resources or information.
- `temporal` prevents loose-resource passing while exposing region-0 pots to
  both agents across an otherwise two-counter-wide interface.
- `selection` gives one agent control of the barriers that determine which
  workstation rooms the other agent can enter.

The input JSON contains both the generation settings and an initially empty
`layouts` object. The generator replaces `layouts` with the layouts it creates,
including their recipes, control wiring, and validation results.

The relevant files are:

- `scripts/generate_overcooked_v3_layouts.py`: command-line generator.
- `scripts/overcooked_v3_layouts.example.json`: example configuration.
- `jaxmarl/environments/overcooked_v3/layouts.py`: parsing, loading, and
  validation.
- `scripts/play_overcooked_v3.py`: interactive pygame viewer.

## JSON configuration

Every configuration has common fields and exactly the extra fields belonging
to its selected `map_type`. This Selection example creates a control room, a
main room, and two gated workstation rooms:

```json
{
  "generator": {
    "seed": 7,
    "count": 3,
    "name_prefix": "selection_kitchen",
    "width": 15,
    "height": 11,
    "possible_recipes": [
      [0, 0, 0],
      [1, 1, 1]
    ],
    "counter_density": 0.45,
    "map_type": "selection",
    "regions": [
      {
        "ingredient_piles": [0, 0],
        "pots": 0,
        "plate_piles": 0,
        "depots": 1,
        "recipe_indicators": 0
      },
      {
        "ingredient_piles": [0, 0],
        "pots": 1,
        "plate_piles": 1,
        "depots": 0,
        "recipe_indicators": 1
      },
      {
        "ingredient_piles": [1, 0],
        "pots": 0,
        "plate_piles": 0,
        "depots": 0,
        "recipe_indicators": 0
      },
      {
        "ingredient_piles": [0, 1],
        "pots": 0,
        "plate_piles": 0,
        "depots": 0,
        "recipe_indicators": 0
      }
    ],
    "randomize_agents": false,
    "control_handoff_tiles": 1,
    "pressure_plates_per_barrier": 1,
    "buttons_per_barrier": 0,
    "max_attempts": 1000
  },
  "layouts": {}
}
```

### Common settings

| Setting | Meaning |
| --- | --- |
| `seed` | Non-negative random seed. The same configuration and seed produce the same layouts. |
| `count` | Positive number of layouts to generate. |
| `name_prefix` | Prefix used for layout names, such as `selection_kitchen_0`. |
| `width`, `height` | Exact ASCII dimensions. Small dimensions may be rejected when the selected topology and exact workstation counts do not fit. |
| `possible_recipes` | Optional non-empty list of three-ingredient recipes which the environment may request. Ingredient indices must fit the common `ingredient_piles` array length and each required type must exist somewhere in the layout. |
| `counter_density` | Target fraction of interior tiles occupied by counters, in `[0, 1)`. Mandatory room boundaries and mode-specific interfaces count toward this target. |
| `map_type` | One of `asymmetric_info`, `temporal`, or `selection`. |
| `regions` | Ordered list of exact per-region workstation dictionaries, described below. The selected map type determines its required length and the meaning of each position. |
| `randomize_agents` | If `false`, agent identities follow region order. If `true`, their assignments to the two agent regions are randomized reproducibly. |
| `max_attempts` | Positive maximum number of constructive retries for each requested layout. |

When `possible_recipes` is omitted, the generator derives one homogeneous
recipe for each ingredient type with at least one pile anywhere in `regions`.
For example, globally present types 0 and 1 produce `[0, 0, 0]` and
`[1, 1, 1]`. The normalized output records the resulting recipe list even
when it was derived.

Workstation placement is always `anywhere`: a station may use any feasible
boundary or interior counter position from which its assigned region can
interact with it. There is no placement-mode setting.

`counter_density` includes counters required to separate regions, form
handoff boundaries, create the Temporal double interface, and enclose gated
rooms. It is not additional clutter placed after those structures. A candidate
is rejected when its density budget cannot accommodate the mandatory geometry.
Within those constraints, the generator randomizes the split orientation and
the dimensions of connected rectangular rooms, then adds connected-preserving
interior counters until the exact density budget is reached.

### Exact region dictionaries

Every entry in `regions` must contain all five fields below, including fields
whose requested value is zero:

```json
{
  "ingredient_piles": [1, 0],
  "pots": 1,
  "plate_piles": 0,
  "depots": 0,
  "recipe_indicators": 0
}
```

| Field | Meaning |
| --- | --- |
| `ingredient_piles` | Exact pile count for each ingredient type. All regions must use arrays of the same length. |
| `pots` | Exact number of pots assigned to the region. |
| `plate_piles` | Exact number of plate dispensers assigned to the region. |
| `depots` | Exact number of delivery depots assigned to the region. |
| `recipe_indicators` | Exact number of recipe indicators assigned to the region. |

Every count must be a non-negative integer. `ingredient_piles` arrays have one
entry per enabled ingredient type and must have equal lengths across all
regions. Overcooked V3 supports at most three ingredient types. Each entry in
`possible_recipes` is instead an ordered three-ingredient recipe; for example,
`[1, 1, 1]` uses ingredient type `1` three times. The current cooking logic
supports homogeneous soups; mixed recipes such as `[0, 0, 1]` are rejected.

Counts are strict. The generator neither moves a requested workstation to a
different region nor silently adds one to make a workflow complete. Controls
created by Selection (`#`, `_`, and `!`) are topology objects and are not
included in these workstation dictionaries. Temporal pots assigned to region
0 are the sole exception to exclusive access: they remain counted as region-0
pots but both agents can interact with them.

## Map types

Only the mode-specific keys shown for the chosen `map_type` are legal. This
keeps configuration files concise and catches misspelled or stale settings.

| `map_type` | Required `regions` length | Additional legal/required keys |
| --- | --- | --- |
| `asymmetric_info` | Exactly 2 | `handoff_tiles` |
| `temporal` | Exactly 2 | `signal_tiles` |
| `selection` | At least 3 | `control_handoff_tiles`, `pressure_plates_per_barrier`, `buttons_per_barrier` |

### Asymmetric information

`asymmetric_info` requires exactly two region dictionaries and one additional
field:

```json
{
  "map_type": "asymmetric_info",
  "regions": [
    {
      "ingredient_piles": [1, 0],
      "pots": 1,
      "plate_piles": 0,
      "depots": 0,
      "recipe_indicators": 1
    },
    {
      "ingredient_piles": [0, 1],
      "pots": 0,
      "plate_piles": 1,
      "depots": 1,
      "recipe_indicators": 0
    }
  ],
  "handoff_tiles": 2
}
```

`handoff_tiles` is the exact number of ordinary one-counter-wide interface
tiles accessible from both regions. Agents can place items on these counters
and retrieve them from the other side, but cannot walk between regions. Each
region contains one agent. With `randomize_agents: false`, agent 0 starts in
region 0 and agent 1 in region 1.

No barriers, pressure plates, or buttons are generated in this mode.

### Temporal

`temporal` requires exactly two region dictionaries and `signal_tiles`:

```json
{
  "map_type": "temporal",
  "regions": [
    {
      "ingredient_piles": [0],
      "pots": 1,
      "plate_piles": 0,
      "depots": 0,
      "recipe_indicators": 0
    },
    {
      "ingredient_piles": [1],
      "pots": 0,
      "plate_piles": 1,
      "depots": 1,
      "recipe_indicators": 1
    }
  ],
  "signal_tiles": 1
}
```

The two walkable regions are permanently separate. Away from a signal pot,
their boundary is exactly two counter layers wide. Each pot assigned to region
0 replaces the full two-counter separation at one position:

```text
ordinary boundary: region 0 floor | W | W | region 1 floor
signal position:   region 0 floor | P | region 1 floor
```

Both agents stand directly adjacent to each shared pot and can add ingredients
or collect soup from it. The pot remains non-walkable and cannot hold a loose
item, so agents still cannot cross the boundary or use it as a resource-passing
counter. Every non-pot workstation assigned to region 0, and every workstation
assigned to region 1, uses normal exclusive placement in its own region.
`signal_tiles` must equal `regions[0].pots`; region-1 pots are private and do
not count as signal tiles.

For recipe completion, ingredients can come from either region because both
agents can add them to the same shared pot. A plate pile and depot must exist
together in at least one region so an agent can collect and deliver the cooked
soup without passing it across the interface. Invalid allocations are rejected
before layout generation.

No barriers, pressure plates, or buttons are generated in this mode.

### Selection

`selection` requires at least three ordered regions:

```text
regions[0]  control room (one agent and every pressure plate/button)
regions[1]  main room (one agent)
regions[2:] gated workstation rooms (no initial agent)
```

It also requires:

| Setting | Meaning |
| --- | --- |
| `control_handoff_tiles` | Exact number of ordinary handoff counters between the control and main rooms. The agents cannot cross this boundary. |
| `pressure_plates_per_barrier` | Exact number of pressure plates generated for each gated-room barrier: `0`, `1`, or `2`. |
| `buttons_per_barrier` | Non-negative exact number of timed buttons generated for each gated-room barrier. |

The number of barriers is derived from the region list: the generator creates
one initially active barrier between the main room and each region in
`regions[2:]`. Opening a barrier lets the main-room agent enter that gated
room. All pressure plates and buttons are reachable from the control-room
agent, and every control targets exactly one corresponding room barrier. The
generator never places a Selection control in the main or gated rooms.

Pressure plates and buttons retain normal Overcooked V3 behavior. A pressure
plate opens its linked barrier while activated; an interacted button opens its
linked barrier for the environment's configured barrier duration. At least one
of `pressure_plates_per_barrier` and `buttons_per_barrier` must be non-zero, so
every gated room can be selected.

With `randomize_agents: false`, agent 0 is the controller in `regions[0]` and
agent 1 starts in the main room. With `true`, the generator reproducibly
chooses per layout whether to swap those agent identities. It does not reorder
the workstation dictionaries or change which room is the control room.

The Selection topology is:

```text
[control room] -- handoff counters -- [main room]
                                         |-- barrier -- [regions[2]]
                                         |-- barrier -- [regions[3]]
                                         `-- barrier -- [...]
```

## Schema migration

The deliberate map types replace the older workflow schema. Remove these keys
from generator JSON files:

- `ingredient_piles`, `pots`, `plate_piles`, and `depots` at generator scope;
  put their exact counts in every `regions` entry instead.
- `workflow_mode` and `num_regions`; use `map_type` and the required number of
  `regions` entries.
- `num_shared_tiles`; use `handoff_tiles`, `signal_tiles`, or
  `control_handoff_tiles` for the corresponding map type.
- `object_placement`; placement is always anywhere.
- `barriers` and `barrier_placement`; Selection derives one barrier per gated
  room and fixes its placement between that room and the main room.

`pressure_plates_per_barrier` and `buttons_per_barrier` now belong only to
Selection configurations. Supplying them for Asymmetric Information or
Temporal maps is an error. Similarly, a handoff/signal field belonging to a
different mode is an error.

## Partial-observation requirement

The generator enforces physical separation and interaction reachability, but
the layouts only test information asymmetry when the environment or policy
uses partial observations. A globally observable policy could see remote
recipe indicators, barrier state, or pot timers directly and bypass the
intended communication problem.

Configure the experiment's observation range so private rooms are hidden while
the intended handoff or signal interface remains observable where appropriate.
The layout JSON does not itself enable partial observation or choose an
observation radius.

## Capacity limits

Generated layouts must fit Overcooked V3's fixed-shape state capacities:

- at most three ingredient types;
- at most four pots in total across all regions;
- at most sixteen barriers;
- at most sixteen pressure plates; and
- at most sixteen buttons.

Selection has `len(regions) - 2` barriers. Its total controls are therefore
that barrier count multiplied by the requested controls per barrier. These
limits are checked before or during generation; counts are never truncated.

Every generated layout contains exactly two agent spawns. Asymmetric
Information and Temporal place one in each of their two regions. Selection
places them in the control and main regions; gated rooms do not receive an
initial spawn.

Each generated layout records the identity assignment as `swap_agents`.
`false` maps agent 0 to region 0 (the Selection control room);
`true` maps agent 1 there. JSON loading preserves this metadata so a generated
layout uses the same assignment when it is loaded later.

## ASCII symbols

| Symbol | Tile |
| --- | --- |
| `W` | Wall or counter |
| `A` | Agent spawn |
| `0`, `1`, `2` | Ingredient pile by ingredient index |
| `P` | Pot |
| `B` | Plate pile |
| `X` | Delivery depot |
| `R` | Recipe indicator |
| `#` | Active pressure-plate/button-controlled barrier |
| `_` | Pressure plate |
| `!` | Interactive barrier button |
| Space | Walkable floor |

## Generation and validation

For each requested layout, the generator constructs randomized connected room
rectangles, adds the selected mode's interfaces and gates, places the exact
per-region workstations, reaches the exact counter budget, places the two
agents, and wires Selection controls. It then
parses the ASCII with `Layout.from_string` and applies structural,
accessibility, topology, workstation-count, control-link, and recipe checks.

A generated candidate is accepted only when, among the checks relevant to its
mode:

- every walkable tile belongs to a region reachable from the appropriate
  agent;
- every configured workstation exists in the exact requested region and can
  be interacted with there;
- agent regions have no unintended walkable connection;
- handoff counters and Temporal signal tiles have the requested count and
  access properties;
- region-0 pots are the only Temporal workstations accessible from both agent
  regions, and ordinary counters never form an unintended handoff;
- every Selection gated room has exactly one main-room barrier and the exact
  requested controls in the control room;
- configured recipes reference available ingredient types;
- each Temporal workflow has a region containing both plates and a depot for
  soup collected from a shared pot; and
- all fixed V3 capacities are respected.

Generation prints progress and atomically checkpoints the output after every
success or failure. If a layout cannot be generated within `max_attempts`, its
error is recorded in `generation_errors`, generation continues with the next
name, and the invalid candidate is not added to `layouts`. The command exits
non-zero if any requested layout fails.

## Generate layouts

Update the example JSON in place:

```bash
uv run python scripts/generate_overcooked_v3_layouts.py \
  scripts/overcooked_v3_layouts.example.json
```

Preserve the input configuration and write a separate result:

```bash
uv run python scripts/generate_overcooked_v3_layouts.py \
  scripts/overcooked_v3_layouts.example.json \
  --output generated-layouts.json
```

The output contains `layouts`, checkpoint progress, and any errors:

```json
{
  "generation_progress": {
    "requested": 3,
    "completed": 2,
    "failed": 1,
    "status": "completed_with_errors"
  },
  "generation_errors": {
    "selection_kitchen_2": "Could not generate a valid layout after 1000 attempts..."
  }
}
```

The generator replaces the `layouts` object each time it runs. Change the seed
or copy the output elsewhere before regenerating if previous maps must be kept.

## Load layouts in Python

Load layouts without modifying the global registry:

```python
from jaxmarl.environments.overcooked_v3 import OvercookedV3, load_layouts_from_json

layouts = load_layouts_from_json("generated-layouts.json")
env = OvercookedV3(layout=layouts["selection_kitchen_0"])
```

Register every generated layout so it can be selected by name:

```python
load_layouts_from_json("generated-layouts.json", register=True)
env = OvercookedV3(layout="selection_kitchen_0")
```

Registration rejects existing names. Pass `overwrite=True` only when replacing
a registered layout is intentional. JSON layouts are parsed and validated
again while loading; the loader does not trust the stored validation result.

## View layouts interactively

Generate the file, then pass it to the pygame player:

```bash
uv run python scripts/play_overcooked_v3.py \
  --layout-json scripts/overcooked_v3_layouts.example.json
```

When `--layout` is omitted, the player opens the first generated layout. Use N
and P to cycle through layouts. To open one layout directly:

```bash
uv run python scripts/play_overcooked_v3.py \
  --layout-json scripts/overcooked_v3_layouts.example.json \
  --layout selection_kitchen_2
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

- **Workstations or rooms do not fit:** Increase `width` or `height`, reduce
  exact workstation counts, or reduce the number of Selection gated rooms.
- **Mandatory boundaries exceed the counter budget:** Increase
  `counter_density` or the map dimensions. Mandatory counters are part of the
  requested density.
- **A handoff or signal count cannot be generated:** Increase the dimensions,
  adjust `counter_density`, or choose a less restrictive mode-specific tile
  count.
- **Selection exceeds a capacity:** Reduce gated rooms or controls per barrier.
- **Temporal does not expose useful information:** Assign the timed pots to
  region 0 and configure partial observations so the other agent can see the
  intended shared-pot positions.
- **Temporal recipe cannot be completed:** Put both a plate pile and depot in
  at least one region. Ingredients may be supplied to a shared pot from either
  side, but plated soup cannot cross the interface.
- **Unknown layout in the player:** Supply `--layout-json` in the same command
  as `--layout`.
- **Layout name already registered:** Rename the prefix or load with
  `overwrite=True` when replacement is intentional.
