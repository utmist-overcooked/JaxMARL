# Migration Guide: CoGrid to Overcooked V3

This document details the migration from the CoGrid Overcooked implementation to JaxMARL's `overcooked_v3` environment.

## Overview

**Source:** `cogrid/cogrid/envs/overcooked/` (PettingZoo-based, NumPy/Python)
**Target:** `jaxmarl/environments/overcooked_v3/` (JaxMARL-based, pure JAX)

The migration converts an object-oriented, mutable implementation to a functional, immutable JAX implementation that supports GPU acceleration and JIT compilation.

---

## Architecture Comparison

| Aspect | CoGrid | Overcooked V3 |
|--------|--------|---------------|
| **Framework** | PettingZoo ParallelEnv | JaxMARL MultiAgentEnv |
| **Compute** | NumPy/Python | JAX (GPU-accelerated) |
| **State** | Mutable objects | Immutable chex dataclass |
| **Grid** | `Grid` class with `GridObj` instances | 3-channel int32 array |
| **Objects** | Class hierarchy with methods | IntEnum + bitwise encoding |
| **Control flow** | Python if/for | `jax.lax.select`, `jax.lax.scan` |
| **Vectorization** | Manual loops | `jax.vmap` |

---

## File Mapping

| CoGrid File | V3 File | Changes |
|-------------|---------|---------|
| `overcooked.py` | `overcooked.py` | Rewritten as pure functions with JIT |
| `overcooked_grid_objects.py` | `common.py` | Classes → IntEnums with bitwise encoding |
| `agent.py` | `common.py` (Agent dataclass) | OOP → chex dataclass |
| `rewards.py` | `overcooked.py` (inline) | Reward classes → pure functions |
| `overcooked_features.py` | `overcooked.py` (get_obs) | Feature system → direct array ops |
| `order_queue.py` | `overcooked.py` (inline) | Queue class → fixed-size arrays |
| `core/layouts.py` | `layouts.py` | Registry → dict with Layout dataclass |
| `core/constants.py` | `settings.py` | Constants preserved with same values |
| N/A | `utils.py` | New JAX-compatible helpers |

---

## Key Transformations

### 1. Grid Objects → Bitwise Encoding

**CoGrid:** Each object is a class instance with methods
```python
class Onion(grid_object.GridObj):
    object_id = "onion"
    color = constants.Colors.Yellow

    def can_pickup(self, agent):
        return True

    def render(self, tile_img):
        fill_coords(tile_img, point_in_circle(0.5, 0.5, 0.3), self.color)
```

**V3:** Objects are integer enums, items use bitwise encoding
```python
class StaticObject(IntEnum):
    EMPTY = 0
    WALL = 1
    POT = 5
    INGREDIENT_PILE_BASE = 10  # 10 = onion pile, 11 = tomato pile, etc.

class DynamicObject(IntEnum):
    PLATE = 1 << 0      # bit 0
    COOKED = 1 << 1     # bit 1
    BASE_INGREDIENT = 1 << 2  # bits 2-3 = ingredient 0 count, etc.
```

**Rationale:** JAX requires static types. Bitwise encoding allows representing complex item states (plate + cooked + 3 onions) as a single int32.

### 2. Grid Representation

**CoGrid:** 2D array of object references
```python
class Grid:
    def get(self, row, col):
        return self.grid[row][col]  # Returns GridObj instance
```

**V3:** 3-channel int32 array
```python
grid: chex.Array  # Shape: [height, width, 3]
# Channel 0: StaticObject enum (wall, pot, goal, etc.)
# Channel 1: DynamicObject bitfield (items, ingredients)
# Channel 2: Extra info (pot timers, conveyor directions)
```

**Rationale:** JAX arrays must have uniform dtype. Three channels separate static layout from dynamic state.

### 3. Pot State Tracking

**CoGrid:** Pot object with internal state
```python
class Pot(grid_object.GridObj):
    cooking_time: int = 90
    burning_time: int = 60

    def __init__(self):
        self.objects_in_pot = []
        self.cooking_timer = self.cooking_time

    def tick(self):
        if len(self.objects_in_pot) == self.capacity:
            if self.cooking_timer > 0:
                self.cooking_timer -= 1
```

**V3:** Separate fixed-size arrays for pot state
```python
# In State dataclass:
pot_positions: chex.Array      # [MAX_POTS, 2] - (y, x)
pot_cooking_timer: chex.Array  # [MAX_POTS] - countdown
pot_active_mask: chex.Array    # [MAX_POTS] - valid pots

# Timer update via jax.lax.scan over pot indices
def _update_single_pot(carry, pot_idx):
    # Pure function, no mutation
    ...
```

**Rationale:** JAX requires fixed array sizes at compile time. Masks track which slots are active.

### 4. Agent State

**CoGrid:** Agent class with methods
```python
class Agent:
    def __init__(self):
        self.pos = (0, 0)
        self.dir = Directions.Right
        self.inventory = []

    @property
    def front_pos(self):
        return self.pos + self.dir_vec
```

**V3:** Chex dataclass with batched arrays
```python
@chex.dataclass
class Agent:
    pos: Position  # Contains batched x, y arrays
    dir: jnp.ndarray  # [num_agents]
    inventory: jnp.ndarray  # [num_agents] - bitwise encoded

    def get_fwd_pos(self):
        return self.pos.move(self.dir)
```

**Rationale:** Dataclass is a JAX pytree, enabling `jax.vmap` over agents.

### 5. Rewards

**CoGrid:** Reward classes with calculate_reward method
```python
class SoupDeliveryReward(reward.Reward):
    def calculate_reward(self, state, agent_actions, new_state):
        for agent_id, action in agent_actions.items():
            if action != Actions.PickupDrop:
                continue
            # Check conditions...
```

**V3:** Inline in step function
```python
def process_interact(self, grid, agent, ...):
    # Delivery check
    successful_delivery = object_is_goal * inventory_is_dish
    is_correct_recipe = inventory == plated_recipe

    reward = successful_delivery * is_correct_recipe * self.delivery_reward
```

**Rationale:** Simpler, avoids class overhead, all computation in one JIT-compiled function.

### 6. Conveyor Belts

**CoGrid:** ConveyorBelt object class
```python
class ConveyorBelt(grid_object.GridObj):
    def __init__(self, direction):
        self.direction = direction
```

**V3:** Static object type + direction in extra channel
```python
# In StaticObject enum:
ITEM_CONVEYOR = 20
PLAYER_CONVEYOR = 21

# Direction stored in grid channel 2
grid[y, x, 2] = Direction.RIGHT  # 0-3

# Conveyor info tracked separately for processing
item_conveyor_positions: chex.Array   # [MAX_ITEM_CONVEYORS, 2]
item_conveyor_directions: chex.Array  # [MAX_ITEM_CONVEYORS]
item_conveyor_active_mask: chex.Array # [MAX_ITEM_CONVEYORS]
```

### 7. Layout Parsing

**CoGrid:** Registry with state encoding
```python
register_layout("cramped_room", layout_string, state_encoding=None)
```

**V3:** Layout dataclass with from_string parser
```python
@dataclass
class Layout:
    agent_positions: List[Tuple[int, int]]
    static_objects: np.ndarray
    num_ingredients: int
    possible_recipes: List[List[int]]
    item_conveyor_info: List[Tuple[int, int, int]]
    player_conveyor_info: List[Tuple[int, int, int]]

    @staticmethod
    def from_string(grid, possible_recipes=None):
        # Parse ASCII to Layout
```

---

## Feature Mapping

| CoGrid Feature | V3 Implementation | Notes |
|----------------|-------------------|-------|
| Onion/Tomato ingredients | Generic ingredients 0-9 | More extensible |
| Pot cooking (90 steps) | `pot_cook_time=90` | Configurable |
| Pot burning (60 step window) | `pot_burn_time=60` | Configurable |
| Delivery reward (20.0) | `delivery_reward=20.0` | Same default |
| Shaped rewards | `SHAPED_REWARDS` dict | Same values |
| Order queue | `enable_order_queue=True` | Optional feature |
| Item conveyors | `enable_item_conveyors=True` | Move items |
| Player conveyors | `enable_player_conveyors=True` | Push agents |
| Recipe indicator | `R` in layout | Shows target recipe |

---

## Constants Preserved

From `cogrid/envs/overcooked/overcooked_grid_objects.py`:
```python
# Pot class
cooking_time: int = 90
burning_time: int = 60
```

To `jaxmarl/environments/overcooked_v3/settings.py`:
```python
POT_COOK_TIME = 90
POT_BURN_TIME = 60
```

Shaped reward values also preserved:
```python
SHAPED_REWARDS = {
    "PLACEMENT_IN_POT": 0.1,
    "SOUP_IN_DISH": 0.3,
    "PLATE_PICKUP": 0.1,
}
```

---

## Layout Symbol Changes

| Symbol | CoGrid | V3 |
|--------|--------|-----|
| `W` | Wall | Wall |
| `@` | DeliveryZone | `X` (Goal) |
| `U` | Pot | `P` (Pot) |
| `=` | PlateStack | `B` (Bowl/Plate pile) |
| `O` | OnionStack | `0` (Ingredient 0) |
| `T` | TomatoStack | `1` (Ingredient 1) |
| `A` | Agent | Agent |
| ` ` | Empty | Empty |
| `>` `<` `^` `v` | N/A | Item conveyors |
| `]` `[` `{` `}` | N/A | Player conveyors |

---

## API Differences

### Environment Creation

**CoGrid:**
```python
from cogrid.envs import registry
env = registry.make("Overcooked-CrampedRoom-V0", render_mode="human")
```

**V3:**
```python
from jaxmarl import make
env = make("overcooked_v3", layout="cramped_room")

# Or directly:
from jaxmarl.environments.overcooked_v3 import OvercookedV3
env = OvercookedV3(layout="cramped_room", max_steps=400)
```

### Step Function

**CoGrid:**
```python
observations, rewards, terminations, truncations, infos = env.step(actions)
```

**V3:**
```python
obs, state, rewards, dones, info = env.step(key, state, actions)
# Note: JAX requires explicit state passing and RNG key
```

### Observations

**CoGrid:** Feature system generates observations
```python
config = {
    "features": ["overcooked_collected_features"]
}
```

**V3:** Direct grid-based observations
```python
env = OvercookedV3(observation_type=ObservationType.DEFAULT)
# Returns [H, W, num_layers] array per agent
```

---

## JIT Compilation Patterns

### Control Flow

**CoGrid (Python):**
```python
if agent.holding_plate and pot.dish_ready:
    soup = pot.pick_up_from(agent)
```

**V3 (JAX):**
```python
successful_dish_pickup = pot_is_ready * inventory_is_plate
new_inventory = jax.lax.select(
    successful_dish_pickup,
    merged_ingredients,
    inventory
)
```

### Loops

**CoGrid (Python):**
```python
for agent_id, action in agent_actions.items():
    if action == Actions.PickupDrop:
        process_pickup(agent_id)
```

**V3 (JAX):**
```python
def _interact_wrapper(carry, x):
    agent, action = x
    return jax.lax.cond(
        action == Actions.interact,
        _interact,
        lambda c, a: (c, (a, 0.0)),
        carry, agent
    )

carry, outputs = jax.lax.scan(_interact_wrapper, init_carry, (agents, actions))
```

---

## Testing Migration

CoGrid tests in `cogrid/testing/unittests/test_overcooked_env.py` were adapted:

| CoGrid Test | V3 Test |
|-------------|---------|
| `test_onion_in_pot` | `test_step_returns_correct_format` |
| `test_tomato_in_pot` | Layout-based integration tests |
| `test_conveyor_belt` | `test_visualizer_with_conveyors` |

New V3-specific tests:
- JIT compilation compatibility
- Vectorized rollouts with `jax.vmap`
- Visualization rendering

---

## Known Differences

1. **Order Queue:** V3 implementation simplified - no complex Order class, just arrays
2. **Tomato Soup:** V3 uses generic recipes, not soup-type-specific objects
3. **Rendering:** V3 uses JIT-compiled rendering, CoGrid uses pygame
4. **Agent Collision:** V3 has explicit collision resolution with swap prevention
5. **Partial Observability:** V3 supports `agent_view_size` parameter

---

## Migration Checklist

When extending V3 or porting additional CoGrid features:

- [ ] Convert class to IntEnum or chex dataclass
- [ ] Replace mutable state with `.replace()` updates
- [ ] Convert Python if/for to `jax.lax.select`/`jax.lax.scan`
- [ ] Use fixed-size arrays with masks for variable-length data
- [ ] Test JIT compilation: `jax.jit(env.step)`
- [ ] Test vectorization: `jax.vmap(env.step)`
- [ ] Add to visualizer if new visual element
- [ ] Update layout parser if new symbol needed
