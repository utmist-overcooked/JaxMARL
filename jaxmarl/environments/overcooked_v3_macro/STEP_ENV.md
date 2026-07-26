# `OvercookedV3Macro.step_env` walkthrough

This document follows one call to `OvercookedV3Macro.step_env`. One call
executes exactly one primitive Overcooked action per agent; a macro normally
continues across several calls.

```text
request macro
  -> start it or retain the active macro
  -> build its valid-target mask
  -> remove currently closed barriers
  -> flood distances from every valid destination
  -> move, face, interact, or stay for one primitive tick
  -> run the base Overcooked V3 transition
  -> mark the macro complete, unreachable, timed out, or still active
```

The macro implementation is [`overcooked.py`](./overcooked.py). Movement,
collision, interaction, reward, and world updates come from the base
[`overcooked_v3/overcooked.py`](../overcooked_v3/overcooked.py).

## Primitive and macro actions

The base environment has six primitive actions:

```python
class Actions(IntEnum):
    right = 0
    down = 1
    left = 2
    up = 3
    stay = 4
    interact = 5
```

Evidence: [`Actions`](../overcooked_v3/common.py#L210-L217).

The policy selects one of seventeen actions: thirteen macros and four
one-step primitive movement actions.

```python
class MacroActions(IntEnum):
    wait = 0
    get_ingredient_0 = 1
    get_ingredient_1 = 2
    get_ingredient_2 = 3
    get_plate = 4
    put_ingredient_in_nearest_pot = 5
    get_soup_from_nearest_pot = 6
    deliver = 7
    drop_on_nearest_counter = 8
    pickup_from_nearest_counter = 9
    press_nearest_button = 10
    stand_on_nearest_pressure_plate = 11
    wait_for_nearest_pot = 12
    up = 13
    down = 14
    left = 15
    right = 16
```

The primitive movement actions emit the corresponding base-environment action
for one step and then complete immediately.

Evidence: [`MacroActions`](./overcooked.py#L43-L62).

## Step 1: read the policy requests

`step_env` collects and clips one requested macro index per agent:

```python
requested_macro_actions = jnp.array(
    [actions[f"agent_{i}"] for i in range(self.num_agents)],
    dtype=jnp.int32,
)
requested_macro_actions = jnp.clip(
    requested_macro_actions, 0, self.num_macro_actions - 1
)
```

Evidence: [`step_env`](./overcooked.py#L191-L204).

The committed interface accepts a request only after the previous macro ends:

```python
def _macro_replacement_mask(self, state, requested_macro_actions):
    del requested_macro_actions
    return state.macro_action_done
```

It then either installs the request or retains the active macro:

```python
current_macro_actions = jnp.where(
    replace_macro,
    requested_macro_actions,
    state.current_macro_actions,
)
```

Evidence: [`_macro_replacement_mask`](./overcooked.py#L206-L211) and macro
latching in [`_step_with_macro_replacements`](./overcooked.py#L222-L229).

`OvercookedV3MacroInterruptible` changes only the replacement mask: requesting
a different macro interrupts the active one.

## Step 2: build the currently walkable grid

Static walkability includes ordinary floor, player conveyors, pressure plates,
and barrier tiles. At runtime, the planner removes barrier tiles that currently
block movement.

If pressure plates are enabled, barriers opened by a currently pressed linked
plate remain walkable:

```python
opened_by_plate = jnp.any(
    state.pressure_plate_linked_barrier & plate_pressed[:, None],
    axis=0,
)

blocked_barriers = (
    state.barrier_active_mask
    & state.barrier_active
    & ~opened_by_plate
)

return self._walkable_mask & (blocked_cells == 0)
```

Evidence: [`_current_walkable_mask`](./overcooked.py#L629-L659). This mirrors
the pressure-plate override and active-barrier check used by base movement in
[`OvercookedV3.step_agents`](../overcooked_v3/overcooked.py#L818-L875).

The walkable mask is rebuilt from the current state on every primitive tick.
Therefore, a barrier opening or closing affects the next planning decision.

## Step 3: build the selected macro's valid-target mask

The planner first creates one boolean grid describing all targets that are valid
for this agent and macro. Only cheap mask construction is repeated for the
different macro types; exactly one flood fill is performed per agent.

Representative target masks are:

```python
# Ingredient pile
static_layer == StaticObject.ingredient_pile(0)

# Plate pile
static_layer == StaticObject.PLATE_PILE

# Pot that can accept the held ingredient
self._valid_pot_placement_mask(state, agent.inventory)

# Pot containing the ready configured recipe
self._ready_recipe_pot_mask(state)

# Empty or occupied counter-like object
counter_mask & (dynamic_layer == DynamicObject.EMPTY)
counter_mask & (dynamic_layer != DynamicObject.EMPTY)

# Delivery and button targets
static_layer == StaticObject.GOAL
static_layer == StaticObject.BUTTON
```

Evidence: target-mask selection in
[`_macro_to_primitive_action`](./overcooked.py#L349-L404), valid-pot rules in
[`_valid_pot_placement_mask`](./overcooked.py#L745-L770), and ready-pot rules in
[`_ready_recipe_pot_mask`](./overcooked.py#L772-L776).

### Per-macro targets and prerequisites

| Macro | Valid targets | Required inventory/action behavior |
| --- | --- | --- |
| `wait` | None | Emit `stay`. |
| `get_ingredient_N` | Every pile for ingredient `N` | Inventory must be empty. |
| `get_plate` | Every plate pile | Base pickup succeeds only with empty inventory. |
| `put_ingredient_in_nearest_pot` | Non-full, same-type or empty, uncooked and unburned pots | Must hold an ingredient. |
| `get_soup_from_nearest_pot` | Pots containing the cooked configured recipe | Must hold a plate. |
| `deliver` | Every delivery goal | Must hold an object with `COOKED` set. |
| `drop_on_nearest_counter` | Empty wall, moving-wall, or conveyor counter-like cells | Inventory must be non-empty. |
| `pickup_from_nearest_counter` | Occupied counter-like cells | Inventory must be empty. |
| `press_nearest_button` | Every button | Navigate beside it and interact. |
| `stand_on_nearest_pressure_plate` | Every pressure-plate cell | Navigate onto it; do not interact. |
| `wait_for_nearest_pot` | None | Emit `stay` until the waiting condition ends. |

Inventory gates are applied in
[`_macro_to_primitive_action`](./overcooked.py#L460-L498). The actual pickup,
placement, and delivery effects are implemented by base
[`process_interact`](../overcooked_v3/overcooked.py#L1242-L1381).

## Step 4: turn object targets into navigation goals

For an interaction macro, the agent must stand in a walkable cell directly
beside a target. The four shifted target masks are combined into one mask of all
valid interaction positions:

```python
interaction_goals = (
    jnp.pad(target_mask[:-1, :], ((1, 0), (0, 0)))
    | jnp.pad(target_mask[1:, :], ((0, 1), (0, 0)))
    | jnp.pad(target_mask[:, :-1], ((0, 0), (1, 0)))
    | jnp.pad(target_mask[:, 1:], ((0, 0), (0, 1)))
) & walkable_mask
```

For `stand_on_nearest_pressure_plate`, the plate cells themselves are goals:

```python
pressure_plate_goals = (
    (static_layer == StaticObject.PRESSURE_PLATE) & walkable_mask
)
```

Evidence: goal construction in
[`_macro_to_primitive_action`](./overcooked.py#L406-L427).

All valid destinations participate simultaneously. There is no permanently
stored target. If one target becomes unreachable while another remains
reachable, the resulting field directs the agent toward the reachable one.

## Step 5: flood a barrier-aware distance field

All goal cells start at distance zero. Each relaxation round assigns every
walkable cell the minimum of its current distance and one plus the smallest
neighbor distance:

```python
distances = jnp.where(goal_mask, 0, INF_DISTANCE).astype(jnp.int32)

def relax(_iteration, current_distances):
    padded = jnp.pad(
        current_distances,
        ((1, 1), (1, 1)),
        constant_values=INF_DISTANCE,
    )
    nearest_neighbor = jnp.minimum(
        jnp.minimum(padded[:-2, 1:-1], padded[2:, 1:-1]),
        jnp.minimum(padded[1:-1, :-2], padded[1:-1, 2:]),
    )
    relaxed = jnp.minimum(current_distances, nearest_neighbor + 1)
    return jnp.where(walkable_mask, relaxed, INF_DISTANCE)

distances = lax.fori_loop(
    0, self.height * self.width, relax, distances
)
```

Evidence: [`_distance_to_goals`](./overcooked.py#L661-L682).

This dense formulation is regular and vectorizable under `jax.jit`, `jax.vmap`,
and batched GPU training. It performs at most `height * width` relaxation rounds.

Reachability is the value at the agent's current tile:

```python
agent_distance = distances[agent.pos.y, agent.pos.x]
has_path = agent_distance < INF_DISTANCE
at_goal = agent_distance == 0
```

Evidence: [`_macro_to_primitive_action`](./overcooked.py#L429-L432).

The possible outcomes are:

1. The original target remains reachable through a detour: follow the detour.
2. That target is cut off but another valid target is reachable: follow the
   other target's distance gradient.
3. No valid target is reachable from the agent's current connected region:
   emit `stay` and mark the macro done.

Other agents are deliberately not removed from the flood-fill grid. Their
occupancy is temporary and is handled only when choosing the immediate step.

## Step 6: emit one primitive action

### Moving

When the agent is not yet at a goal, the planner scores its four neighboring
cells by the new distance field. It excludes out-of-bounds, non-walkable, and
currently occupied cells:

```python
scores = jnp.where(
    candidate_in_bounds & candidate_walkable & candidate_unoccupied,
    candidate_distances,
    INF_DISTANCE,
)
best_idx = jnp.argmin(scores)
action = self._move_actions[best_idx]
return jnp.where(has_step, action, Actions.stay).astype(jnp.int32)
```

Evidence: [`_next_action_avoiding_agents`](./overcooked.py#L684-L717) and the
occupancy check in
[`_cell_unoccupied_by_other_agents`](./overcooked.py#L719-L731).

### Facing and interacting

At an interaction goal, the planner finds an adjacent valid target. If the
agent already faces it, the primitive action is `interact`; otherwise, the
cardinal action turns the agent toward it:

```python
adjacent_targets = candidate_in_bounds & target_mask[safe_y, safe_x]
target_direction = jnp.argmax(adjacent_targets)
face_action = self._dir_to_action[target_direction]
interact_action = jnp.where(
    agent.dir == target_direction, Actions.interact, face_action
)
```

Evidence: [`_macro_to_primitive_action`](./overcooked.py#L438-L458).

At a pressure-plate goal, the agent emits `stay` because merely occupying the
cell activates the plate. `wait` and `wait_for_nearest_pot` also emit `stay`.
Any failed inventory prerequisite or unreachable navigation goal emits `stay`:

```python
primitive_action = jnp.where(
    navigation_macro & can_execute & has_path,
    navigation_action,
    Actions.stay,
)
macro_reachable = ~navigation_macro | has_path
```

Evidence: [`_macro_to_primitive_action`](./overcooked.py#L455-L500).

### Simultaneous agent conflicts

Agents plan from the same pre-transition state. If they simultaneously enter
the same cell or swap cells, the base environment cancels those movements:

Evidence: base collision and swap resolution in
[`step_agents`](../overcooked_v3/overcooked.py#L888-L930).

The macro remains active unless another completion condition fires. On the next
tick it recomputes from the resulting positions. There is no path reservation,
right-of-way, or deadlock negotiation.

## Step 7: run one base Overcooked transition

The chosen primitives are passed to the parent environment:

```python
primitive_action_dict = {
    f"agent_{i}": primitive_actions[i] for i in range(self.num_agents)
}
obs, next_state, rewards, dones, info = super().step_env(
    key, state, primitive_action_dict
)
```

Evidence: [`_step_with_macro_replacements`](./overcooked.py#L231-L240).

The base transition applies agent movement, collisions, interactions, pot
timers, buttons, moving walls, pressure plates, conveyors, barrier timers,
orders, time, termination, observations, and rewards. Evidence:
[`OvercookedV3.step_env`](../overcooked_v3/overcooked.py#L741-L800).

## Step 8: determine whether the macro is done

Completion is checked after the base transition. The action-specific rules are:

| Macro | Done when |
| --- | --- |
| `wait` | After its one `stay` tick. |
| `get_ingredient_N` | The agent holds ingredient `N`, holds an incompatible object, or no such pile exists. |
| `get_plate` | The agent holds a plate, holds an incompatible object, or no plate pile exists. |
| `put_ingredient_in_nearest_pot` | The agent no longer holds an ingredient or no valid pot remains. |
| `get_soup_from_nearest_pot` | The agent holds a cooked dish, no longer holds a plate, or no ready pot remains. |
| `deliver` | The agent no longer holds a cooked dish or no goal exists. |
| `drop_on_nearest_counter` | Inventory becomes empty or no empty counter remains. |
| `pickup_from_nearest_counter` | Inventory becomes non-empty or no occupied counter remains. |
| `press_nearest_button` | The emitted primitive was `interact` or no button exists. |
| `stand_on_nearest_pressure_plate` | The agent occupies a plate or no plate exists. |
| `wait_for_nearest_pot` | A recipe pot is ready or no pot is still cooking. |

Every navigation macro also ends when its pre-transition flood field says no
valid target is reachable:

```python
return done | ~macro_reachable
```

Evidence: [`_macro_done_for_agent`](./overcooked.py#L517-L613).

Finally, every macro ends at `max_macro_steps` or when the environment ends:

```python
macro_done = (
    macro_done
    | (next_macro_step_count >= self.max_macro_steps)
    | dones["__all__"]
)
```

Evidence: [`_step_with_macro_replacements`](./overcooked.py#L242-L253).

## Step 9: store bookkeeping and return diagnostics

The active macro, done flag, and step count are stored in `next_state`:

```python
next_state = next_state.replace(
    current_macro_actions=current_macro_actions,
    macro_action_done=macro_done,
    macro_step_count=jnp.where(macro_done, 0, next_macro_step_count),
)
```

`info` reports:

- `current_macro_action`
- `macro_action_done`
- `macro_action_started`
- `primitive_action`

Evidence: state and info updates in
[`_step_with_macro_replacements`](./overcooked.py#L255-L275).

## Visualizing the flood fill

The scripted rollout can append a synchronized planner panel to its GIF:

```bash
python scripts/scripted_overcooked_v3_macro_cramped_room.py \
  --cooperative-barrier-demo \
  --flood-fill \
  --barrier-duration 30 \
  --output artifacts/overcooked_v3_macro_cooperative_barrier_flood_fill.gif
```

In this demo, agent 0 requests the plate behind a closed timed barrier and
initially receives an unreachable field. Agent 1 navigates to the linked button
and presses it. On the following tick the gate is open, agent 0's field becomes
finite, and agent 0 follows it through the barrier to the plate.

The left panel is the normal Overcooked render. The right panel uses:

- Numeric labels for the current distance-to-goal field.
- Purple `T` outlines for valid object targets.
- Green `G` cells for valid navigation destinations.
- A yellow `A0` outline for the planned agent.
- Red cells for currently closed barriers.
- `INF` for walkable cells outside every valid goal's reachable region.

The header reports the active macro, emitted primitive action, and reachability
flag. Implementation: [`scripted_overcooked_v3_macro_cramped_room.py`](../../../scripts/scripted_overcooked_v3_macro_cramped_room.py).
