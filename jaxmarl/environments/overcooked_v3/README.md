# Overcooked V3 Environment

A GPU-accelerated implementation of the Overcooked cooperative cooking game with additional features like pot burning, order queues, and conveyor belts.

## 🎨 Visual Level Editor

```bash
# If installed via pip
overcooked-editor

# Or run directly
python -m jaxmarl.tools.layout_editor_v3
```

Features:
- ✅ Visual click-to-place grid editor
- ✅ Live preview with proper tile rendering
- ✅ Test play your layouts immediately
- ✅ Automatic validation
- ✅ Export ready-to-use code
- ✅ Load and edit existing layouts
- ✅ Undo/redo support

**[📖 See full Level Editor documentation](../../tools/README_LEVEL_EDITOR.md)**

## File Structure

```
overcooked_v3/
├── __init__.py          # Package exports
├── agent_step.py        # Agent movement and interaction action phase
├── common.py            # Core data structures and enums
├── config.py            # Static functional-core configuration
├── interactions.py      # Item pickup, placement, cooking, and delivery
├── layouts.py           # Layout definitions, parsing, and utilities
├── observations.py      # Observation construction
├── overcooked.py        # Public environment and compatibility wrappers
├── reset.py             # Functional reset pipeline
├── settings.py          # Default constants and fixed array limits
├── state.py             # Environment state dataclass
├── step.py              # Functional timestep pipeline and PRNG partitioning
├── systems/             # Pots, orders, conveyors, walls, and barriers
├── utils.py             # Helper functions
└── README.md            # This file
```

## File Descriptions

### `common.py` - Core Data Structures

Defines all the fundamental types used throughout the environment:

| Class/Enum | Purpose |
|------------|---------|
| `StaticObject` | IntEnum for grid objects (WALL, POT, GOAL, PLATE_PILE, ITEM_CONVEYOR, etc.) |
| `DynamicObject` | IntEnum with bitwise encoding for items (PLATE, COOKED, ingredients) |
| `Direction` | Cardinal directions (UP, DOWN, LEFT, RIGHT) |
| `Position` | Chex dataclass for (x, y) coordinates with movement methods |
| `Agent` | Chex dataclass holding agent state (position, direction, inventory) |
| `Actions` | IntEnum for agent actions (right, down, left, up, stay, interact) |

**Key patterns:**
- `DynamicObject` uses bitwise encoding: bit 0 = plate, bit 1 = cooked, bits 2+ = ingredient counts
- Use `DynamicObject.ingredient(idx)` to get the bit pattern for ingredient type `idx`
- `StaticObject.INGREDIENT_PILE_BASE + idx` gives the pile for ingredient `idx`

### `settings.py` - Configuration Constants

All tunable parameters with defaults:

| Constant | Default | Description |
|----------|---------|-------------|
| `POT_COOK_TIME` | 20 | Steps until a full pot becomes cooked/ready |
| `POT_COOK_TIME_RANGE` | `()` | Optional inclusive `[min, max]` range for random ready times |
| `POT_BURN_TIME` | 40 | Steps in burning window before contents destroyed. Set to `0` to disable burning |
| `DELIVERY_REWARD` | 20.0 | Reward for correct soup delivery |
| `BURN_PENALTY` | -5.0 | Penalty when pot burns |
| `SHAPED_REWARDS` | dict | Intermediate rewards for useful actions |
| `MAX_POTS` | 4 | Maximum pots tracked (fixed array size) |
| `MAX_ITEM_CONVEYORS` | 16 | Maximum item conveyor cells |
| `MAX_PLAYER_CONVEYORS` | 8 | Maximum player conveyor cells |

**Important:** `MAX_*` constants define fixed array sizes for JIT compatibility. Increase these if you need more objects.

#### Variable Pot Cook Times

`pot_cook_time_range` is an optional `OvercookedV3` constructor argument. When set to `[65, 90]`, each cook event samples the time until the soup becomes cooked/ready uniformly from every integer in that inclusive range: `65, 66, 67, ..., 90`.

Omit `pot_cook_time_range` or pass `[]` to keep the fixed `pot_cook_time` behavior.

`pot_burn_time` is separate: after the soup becomes ready, it remains in the burn window for `pot_burn_time` steps before burning. Internally, the pot timer starts at `sampled_cook_time + pot_burn_time`, becomes ready when the timer reaches `pot_burn_time`, and burns when the timer reaches `0`.

Direct Python:
```python
env = make(
    "overcooked_v3",
    layout="cramped_room",
    pot_cook_time_range=[65, 90],
)
```

YAML configs that pass `ENV_KWARGS` into `jaxmarl.make(...)` can specify the same constructor argument:
```yaml
"ENV_NAME": "overcooked_v3"
"ENV_KWARGS":
  "layout": "cramped_room"
  "pot_cook_time_range": [65, 90]
```

`settings.py` provides the default only when `pot_cook_time_range` is not passed:
```python
POT_COOK_TIME_RANGE = ()
```

### `layouts.py` - Layout Definitions

Handles parsing ASCII layouts into `Layout` objects.

**Layout string symbols:**
```
W  - Wall (counter)
A  - Agent starting position
X  - Goal (delivery zone)
B  - Plate (bowl) pile
P  - Pot
R  - Recipe indicator (shows target recipe)
0-9 - Ingredient pile (0=onion, 1=tomato, etc.)
O  - Legacy onion pile (converted to 0)
   - Space = empty walkable cell

Item conveyors (move items):
>  - Push items right
<  - Push items left
^  - Push items up
v  - Push items down

Player conveyors (push agents):
]  - Push agents right
[  - Push agents left
{  - Push agents up
}  - Push agents down

Prep stations (multi-stage preparation):
C  - Cutting board
G  - Grill
M  - Blender (mixer)
```

**Prep chains:** raw ingredients 2-4 never match a recipe directly; they must
be processed at their station first. The processed result is a separate
ingredient type (`raw index + 3`) that recipes reference:

| Raw | Station | Mechanic | Processed |
|-----|---------|----------|-----------|
| 2 lettuce | `C` cutting board | place, then interact `chop_stages` (3) times empty-handed | 5 chopped lettuce |
| 3 meat | `G` grill | place; cooks by itself in `grill_cook_time` steps, **burns** `grill_burn_time` steps later (-5 penalty) | 6 grilled meat |
| 4 carrot | `M` blender | place, interact once to start, ready after `blend_time` steps, never burns | 7 carrot puree |

Each station holds a single unit and only accepts its own raw ingredient.
Empty-handed interact picks up the processed result. Placing a raw ingredient
is a **commitment** — like the pot, a station never hands raw items back (an
earlier version let the grill do so, which made place → pick up → place a
two-step loop that farmed `PREP_PLACEMENT` forever and collapsed grill-map
training into pickup spam with zero deliveries). A recipe such as `[[5, 5, 5]]` then means
"soup of three chopped lettuce": prep 3 units, pot them, plate, deliver.
See the `cutting_board_room`, `grill_room`, `blender_room` and `prep_kitchen`
layouts, plus the `*_handoff` variants (`cutting_board_handoff`,
`grill_handoff`, `blender_handoff`, `prep_kitchen_handoff`) where a counter
wall fully separates the prep side from the cook side so processed
ingredients must be handed over the middle counter (renders in
`docs/imgs/overcooked_v3_prep_layouts/`). Layouts without any station keep
the exact observation schema they
had before prep stations existed; layouts with stations add 3 static layers
plus a prep progress/timer layer.

**Orders and the recipe indicator.** With `enable_order_queue=False` a single
recipe is sampled at reset and stays fixed for the whole episode, so a rollout
only ever shows one dish. To rotate through dishes, enable the queue:
`enable_order_queue=True, order_queue_mode="alternating"` cycles through *every*
orderable dish (`order_generation_rate=1.0` + `order_expiration_time=0` keeps the
queue full and penalty-free). What is orderable is derived from the layout:

- a layout listing **several** `possible_recipes` (the prep kitchens) orders
  exactly those, so a 3-recipe kitchen rotates dish 1 → 2 → 3 → 1 …
- a layout pinning **one** recipe but stocking several ingredient piles
  (`around_the_island`, the conveyor maps) keeps the legacy behaviour: one
  single-ingredient soup per pile, capped at the onion/tomato pair.

`env.order_recipes` shows the resolved list. Alternating needs at least two
orderable dishes.

**Dish washing** (opt-in via `enable_dish_washing=True`): by default the plate
pile is infinite and `S`/`D` tiles collapse into ordinary counters, so a layout
behaves and observes exactly as it did before the feature existed. When enabled
the kitchen owns exactly `num_plates` plates and they cycle:

```
plate pile --take--> clean plate --serve at goal--> dirty pile
dirty pile --take--> dirty plate --sink--> clean plate (--> use, or stack back)
```

| Tile | Empty-handed interact | Holding |
|------|----------------------|---------|
| `B` plate pile | take a clean plate (nothing if the stack is empty) | clean plate → put it back on the stack |
| `D` dirty pile | take a dirty plate (nothing if empty) | – |
| `S` sink | – | dirty plate → becomes clean |

A dirty plate is `PLATE \| DIRTY`, a distinct encoding from a clean plate, so it
can never scoop soup from a pot. **Plates are conserved**: the clean stack, the
dirty pile, every inventory and every plate on the grid always sum to
`num_plates`, and the dirty count starts at 0. Running out of clean plates is
therefore recoverable only by washing, which is what forces the division of
labour. Layouts: `dish_washing_room`, `dish_washing_kitchen`,
`dish_washing_handoff` and `prep_dish_kitchen` (prep chains + washing); renders
in `docs/imgs/overcooked_v3_dish_washing/`. Enabling it adds 7 observation
layers (sink, dirty pile, a plate-count layer, and a dirty-plate bit in each of
the 4 item-encoding blocks).

**Adding a new layout:**
```python
my_layout = """
WWPWW
0A AX
WWBWW
"""

overcooked_v3_layouts["my_layout"] = Layout.from_string(
    my_layout,
    possible_recipes=[[0, 0, 0]],  # Required if no R in layout
    swap_agents=False,              # Reverse agent order if needed
)
```

### `overcooked.py` - Public Environment Wrapper

The `OvercookedV3` class implements `MultiAgentEnv`, validates constructor
arguments, builds `OvercookedV3Config`, and exposes compatibility wrappers. The
JAX-traceable implementation lives in `reset.py`, `step.py`, `agent_step.py`,
`interactions.py`, `observations.py`, and `systems/`.

**Key methods:**
| Method | Description |
|--------|-------------|
| `reset(key) -> Tuple[Dict[str, Array], State]` | Reset the environment and return initial observations and state |
| `step_env(key, state, actions) -> Tuple[obs, State, rewards, dones, info]` | Perform a single timestep: process actions, conveyors, orders, and check termination |
| `step_agents(key, state, actions) -> Tuple[State, float, Array]` | Process agent movement (with collision resolution) and interact actions |
| `process_interact(grid, agent, all_inventories, recipe, pot_timers, pot_positions, pot_active_mask, pot_cook_time=None) -> Tuple[grid, agent, correct_delivery, reward, shaped_reward, event_metrics, pot_timers]` | Handle a single agent's interact action (pickup, drop, cook, deliver) |
| `is_terminal(state) -> bool` | Check whether the episode is done (max steps reached) |
| `get_obs(state) -> Dict[str, Array]` | Get observations for all agents, dispatching by per-agent observation type |
| `get_obs_for_type(state, obs_type) -> Dict[str, Array]` | Get observations for a specific observation type (default or featurized) |
| `get_obs_default(state) -> Array` | Build default grid-based observation tensors for all agents |
| `name` (property) `-> str` | Return the environment name |
| `num_actions` (property) `-> int` | Return the number of possible actions |
| `action_space(agent_id) -> spaces.Discrete` | Return the discrete action space |
| `observation_space(agent_id) -> spaces.Box` | Return the box observation space |

**State dataclass fields:**
```python
@chex.dataclass
class State:
    agents: Agent                    # Agent positions, directions, inventories
    grid: chex.Array                 # [H, W, 3] - static, dynamic, extra channels
    pot_positions: chex.Array        # [MAX_POTS, 2] - pot (y, x) locations
    pot_cooking_timer: chex.Array    # [MAX_POTS] - cooking countdown
    pot_cook_durations: chex.Array   # [MAX_POTS] - sampled steps until ready
    pot_active_mask: chex.Array      # [MAX_POTS] - which pots are valid
    order_types: chex.Array          # [MAX_ORDERS] - order queue
    order_expirations: chex.Array    # [MAX_ORDERS] - time until expiry
    order_active_mask: chex.Array    # [MAX_ORDERS] - active orders
    item_conveyor_*: chex.Array      # Conveyor state arrays
    player_conveyor_*: chex.Array    # Player conveyor state arrays
    time: chex.Array                 # Current timestep
    terminal: bool                   # Episode done flag
    recipe: int                      # Current target recipe (bit-encoded)
```

### `utils.py` - Helper Functions

| Function | Purpose |
|----------|---------|
| `tree_select(pred, a, b)` | Select between pytrees based on predicate |
| `compute_view_box(...)` | Calculate partial observability window |
| `compute_enclosed_spaces(mask)` | Find connected regions for agent placement |
| `mark_adjacent_cells(mask)` | Expand mask to include neighbors |
| `get_closest_true_pos_no_directions(...)` | Find nearest True cell (Manhattan) |
| `move_position_in_direction(...)` | Move position with bounds checking |

## Dependencies

**Required:**
- `jax` - Core array operations and JIT compilation
- `jax.numpy` - NumPy-like API
- `chex` - Dataclass decorators for JAX pytrees
- `flax.struct` - Additional struct utilities
- `numpy` - Layout parsing only (not in JIT paths)

**For visualization (optional):**
- `imageio` - GIF creation
- Window/rendering utilities from `jaxmarl.viz`

## Key Patterns for Development

### 1. JIT Compatibility

All environment logic must be JIT-compilable:

```python
# Good: Use jax.lax for control flow
result = jax.lax.select(condition, value_if_true, value_if_false)

# Bad: Python if statements with traced values
if condition:  # Will fail if condition is a traced value
    result = value_if_true
```

### 2. Fixed Array Sizes

JAX requires static shapes. Use fixed-size arrays with masks:

```python
# Define maximum size
MAX_ITEMS = 10

# Use mask to track valid entries
items = jnp.zeros((MAX_ITEMS,), dtype=jnp.int32)
item_mask = jnp.zeros((MAX_ITEMS,), dtype=jnp.bool_)

# Add item at index 0
items = items.at[0].set(item_value)
item_mask = item_mask.at[0].set(True)
```

### 3. Pytree State Updates

Use `.replace()` for immutable state updates:

```python
# Update state immutably
new_state = state.replace(
    time=state.time + 1,
    agents=state.agents.replace(pos=new_positions)
)
```

### 4. Vectorized Operations

Use `jax.vmap` for batch operations:

```python
# Process all agents in parallel
def process_single_agent(agent, action):
    ...

new_agents = jax.vmap(process_single_agent)(state.agents, actions)
```

### 5. Scan for Sequential Operations

Use `jax.lax.scan` for loops with carried state:

```python
def step_fn(carry, x):
    state, total = carry
    new_state = process(state, x)
    return (new_state, total + 1), output

(final_state, count), outputs = jax.lax.scan(step_fn, (init_state, 0), inputs)
```

## Adding New Features

### New Static Object Type

1. Add to `StaticObject` enum in `common.py`:
   ```python
   class StaticObject(IntEnum):
       ...
       MY_OBJECT = 25  # Pick unused number
   ```

2. Add parsing symbol in `layouts.py`:
   ```python
   char_to_static_item = {
       ...
       "M": StaticObject.MY_OBJECT,
   }
   ```

3. Handle in `process_interact()` in `interactions.py`

4. Add rendering in `overcooked_v3_visualizer.py`

### New Dynamic Item Property

1. Add bit flag to `DynamicObject` in `common.py`:
   ```python
   class DynamicObject(IntEnum):
       ...
       MY_FLAG = 1 << 8  # Pick unused bit
   ```

2. Check/set in game logic:
   ```python
   has_flag = (item & DynamicObject.MY_FLAG) != 0
   item_with_flag = item | DynamicObject.MY_FLAG
   ```

### New Agent Action

1. Add to `Actions` enum in `common.py`
2. Update `ACTION_TO_DIRECTION` if it's a movement
3. Handle in the agent action phase in `agent_step.py`
4. Update action space size if needed

## Testing

Run environment tests:
```bash
pytest tests/overcooked_v3/ -v
```

Run visualization tests:
```bash
pytest tests/overcooked_v3/test_visualization.py -v
```

## Usage Example

```python
import jax
from jaxmarl import make
from jaxmarl.viz.overcooked_v3_visualizer import OvercookedV3Visualizer

# Create environment
env = make("overcooked_v3", layout="cramped_room")

# Or with custom settings
from jaxmarl.environments.overcooked_v3 import OvercookedV3
env = OvercookedV3(
    layout="cramped_room",
    max_steps=400,
    pot_cook_time=90,
    # Optional ready-time range. Omit or pass [] to keep fixed pot_cook_time behavior.
    pot_cook_time_range=[65, 90],
    pot_burn_time=60,
    enable_order_queue=False,
    shaped_rewards=True,
)

# Reset and step
key = jax.random.PRNGKey(0)
obs, state = env.reset(key)

actions = {agent: 0 for agent in env.agents}  # noop
key, subkey = jax.random.split(key)
obs, state, rewards, dones, info = env.step(subkey, state, actions)

# Visualize
viz = OvercookedV3Visualizer(env)
img = viz.render_state(state)
```

## Related Files

- `jaxmarl/viz/overcooked_v3_visualizer.py` - Rendering
- `jaxmarl/registration.py` - Environment registration
- `tests/overcooked_v3/` - Test suite
