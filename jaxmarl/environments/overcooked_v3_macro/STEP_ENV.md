# `OvercookedV3Macro.step_env` execution walkthrough

This document describes exactly what happens during one call to
`OvercookedV3Macro.step_env`.

The most important point is that **one call to `step_env` performs one primitive
Overcooked action per agent**. A macro such as `get_plate` is stored in the
state and advanced over multiple calls:

```text
requested macro
    -> keep the existing macro if it is still active
    -> choose one primitive action from the current state
    -> run one Overcooked V3 transition
    -> decide whether the macro is finished
    -> store the macro bookkeeping for the next call
```

The primary implementation is
[`overcooked_v3_macro/overcooked.py`](./overcooked.py). Primitive movement,
interaction, collision, reward, and world-update behavior is inherited from
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

Evidence: [`overcooked_v3/common.py`, `Actions`](../overcooked_v3/common.py#L210-L217).

The macro environment exposes these thirteen macro actions:

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
```

Evidence: [`overcooked_v3_macro/overcooked.py`, `MacroActions`](./overcooked.py#L45-L61).

## Step 1: read and sanitize the requested macros

The action dictionary contains one macro-action index per agent. `step_env`
collects these into an array and clips every index to the valid macro range:

```python
requested_macro_actions = jnp.array(
    [actions[f"agent_{i}"] for i in range(self.num_agents)],
    dtype=jnp.int32,
)
requested_macro_actions = jnp.clip(
    requested_macro_actions, 0, self.num_macro_actions - 1
)
```

Evidence: [`step_env`, requested actions](./overcooked.py#L197-L203).

`get_avail_actions` currently reports every macro as available. Validity is
therefore handled during execution rather than by policy-side action masking:

```python
return {
    f"agent_{i}": jnp.ones((self.num_macro_actions,), dtype=jnp.uint8)
    for i in range(self.num_agents)
}
```

Evidence: [`get_avail_actions`](./overcooked.py#L885-L890).

## Step 2: accept a new macro or continue the active one

An agent may start the newly requested macro only when its previous macro is
done. Otherwise, the new request is ignored and `current_macro_actions` keeps
the active macro:

```python
current_macro_actions = jnp.where(
    state.macro_action_done,
    requested_macro_actions,
    state.current_macro_actions,
)
macro_step_count = jnp.where(
    state.macro_action_done, 0, state.macro_step_count
)
```

Evidence: [`step_env`, macro latching](./overcooked.py#L205-L212).

For example, if `get_plate` is still active and the policy requests `deliver`,
the agent continues `get_plate`. The `deliver` request is not queued; the policy
must request it again after `macro_action_done` becomes true.

## Step 3: translate each active macro into one primitive action

Translation is vectorized across agents, but each agent is evaluated from the
same pre-transition state:

```python
return jax.vmap(
    lambda agent_idx, macro_action: self._macro_to_primitive_action(
        state, agent_idx, macro_action
    )
)(agent_idxs, macro_actions)
```

Evidence: [`_macro_to_primitive_actions`](./overcooked.py#L308-L316).

The default translation is `Actions.stay`. The matching macro branch replaces
it with a movement, facing, interaction, or waiting action:

```python
agent = self._agent_at(state, agent_idx)
primitive_action = jnp.array(Actions.stay, dtype=jnp.int32)

# ...one jnp.where branch per macro action...

return primitive_action.astype(jnp.int32)
```

Evidence: [`_macro_to_primitive_action`](./overcooked.py#L318-L388).

### How an interaction macro approaches its target

Pickup, placement, delivery, counter, and button macros share the same
interaction-navigation behavior:

1. Construct a mask of currently valid target cells.
2. Select the target with the nearest reachable adjacent walkable cell.
3. While not adjacent, emit one cardinal movement action.
4. When adjacent but facing the wrong direction, emit the cardinal action toward
   the target. A cardinal primitive both attempts movement and changes facing.
   For the usual non-walkable interaction objects (piles, pots, goals, walls,
   and buttons), movement is rejected and only the facing direction changes.
5. When adjacent and facing the target, emit `Actions.interact`.
6. If there is no target, emit `Actions.stay`.

The final choice between moving, facing, and interacting is:

```python
adjacent = (jnp.abs(dx) + jnp.abs(dy)) == 1
desired_dir = self._direction_from_delta(dx, dy)
face_action = self._dir_to_action[desired_dir]
interact_or_face = jnp.where(
    agent.dir == desired_dir, Actions.interact, face_action
)
move_action = self._action_to_adjacent_cell(state, agent, target_y, target_x)
action = jnp.where(adjacent, interact_or_face, move_action)
return jnp.where(has_target, action, Actions.stay).astype(jnp.int32)
```

Evidence: [`_action_to_interact_with_cell`](./overcooked.py#L625-L645).

### What each macro translates to

#### `wait`

`wait` has no matching replacement branch, so it retains the default
`Actions.stay`. It is marked complete after this transition.

```python
primitive_action = jnp.array(Actions.stay, dtype=jnp.int32)
# No `MacroActions.wait` replacement branch.
```

Evidence: default action in
[`_macro_to_primitive_action`](./overcooked.py#L321-L323), and its completion
condition in [`_macro_done_for_agent`](./overcooked.py#L475).

#### `get_ingredient_0`, `get_ingredient_1`, and `get_ingredient_2`

These macros require an empty inventory. They target the requested ingredient
pile, navigate beside it, face it, and interact. If the inventory is not empty,
the emitted primitive action is `stay`.

```python
inventory_empty = agent.inventory == DynamicObject.EMPTY
target_mask = state.grid[:, :, 0] == StaticObject.ingredient_pile(
    ingredient_idx
)
action = self._go_interact_with_static_object(state, agent, target_mask)
return jnp.where(inventory_empty, action, Actions.stay)
```

Evidence: [`_go_interact_with_ingredient`](./overcooked.py#L390-L398), called by
the three ingredient branches in
[`_macro_to_primitive_action`](./overcooked.py#L324-L337).

The base interaction code recognizes an ingredient pile and allows a pickup
only with an empty inventory:

```python
object_is_ingredient_pile = (
    fwd_pos_in_bounds & StaticObject.is_ingredient_pile(interact_item)
)
object_is_pile = object_is_plate_pile | object_is_ingredient_pile
# ...
successful_pickup = object_is_pile * inventory_is_empty + ...
```

Evidence: [`process_interact`, pile pickup](../overcooked_v3/overcooked.py#L1269-L1273)
and [`successful_pickup`](../overcooked_v3/overcooked.py#L1314-L1319).

#### `get_plate`

This macro targets a plate-pile static object and uses the shared
move/face/interact procedure:

```python
primitive_action = jnp.where(
    macro_action == MacroActions.get_plate,
    self._go_interact_with_static_object(
        state, agent, state.grid[:, :, 0] == StaticObject.PLATE_PILE
    ),
    primitive_action,
)
```

Evidence: [`get_plate` translation](./overcooked.py#L339-L345). The base pickup
rule is the same pile-plus-empty-inventory rule shown above.

#### `put_ingredient_in_nearest_pot`

This macro requires an ingredient in inventory. A pot is a valid target only if
it is not full, contains the same ingredient type or is empty, and is neither
cooked nor burned:

```python
inventory_is_ingredient = DynamicObject.is_ingredient(inventory)
pot_mask = self._valid_pot_placement_mask(state, inventory)
action = self._go_interact_with_static_object(state, agent, pot_mask)
return jnp.where(inventory_is_ingredient, action, Actions.stay)
```

```python
return (
    pot_mask
    & DynamicObject.is_ingredient(inventory)
    & (ingredient_counts < MAX_INGREDIENTS)
    & same_type
    & pot_not_finished
)
```

Evidence: [`_put_ingredient_in_nearest_pot`](./overcooked.py#L410-L417) and
[`_valid_pot_placement_mask`](./overcooked.py#L751-L776). The actual placement
is performed by the base interaction rule
[`successful_pot_placement`](../overcooked_v3/overcooked.py#L1334-L1336).

#### `get_soup_from_nearest_pot`

This macro requires a plate. It targets a pot containing the configured recipe
with `COOKED` set, then uses `interact` to transfer the soup to the plate:

```python
inventory_is_plate = agent.inventory == DynamicObject.PLATE
pot_mask = self._ready_recipe_pot_mask(state)
action = self._go_interact_with_static_object(state, agent, pot_mask)
return jnp.where(inventory_is_plate, action, Actions.stay)
```

```python
plated_recipe = state.recipe | DynamicObject.PLATE | DynamicObject.COOKED
return pot_mask & ((pot_contents | DynamicObject.PLATE) == plated_recipe)
```

Evidence: [`_get_soup_from_nearest_pot`](./overcooked.py#L419-L423) and
[`_ready_recipe_pot_mask`](./overcooked.py#L778-L782). The base environment's
successful dish-pickup condition is
[`successful_dish_pickup`](../overcooked_v3/overcooked.py#L1303-L1304).

#### `deliver`

This macro requires an inventory object with the `COOKED` bit. It approaches a
goal tile and interacts with it:

```python
inventory_is_dish = (agent.inventory & DynamicObject.COOKED) != 0
goal_mask = state.grid[:, :, 0] == StaticObject.GOAL
action = self._go_interact_with_static_object(state, agent, goal_mask)
return jnp.where(inventory_is_dish, action, Actions.stay)
```

Evidence: [`_deliver_to_goal`](./overcooked.py#L425-L429). The base environment
applies a successful delivery when the forward object is a goal and the
inventory is a dish:

```python
successful_delivery = object_is_goal * inventory_is_dish
```

Evidence: [`process_interact`, delivery](../overcooked_v3/overcooked.py#L1353).

#### `drop_on_nearest_counter`

This macro requires a non-empty inventory. It targets the nearest counter-like
cell whose dynamic-object layer is empty, then interacts to place the held item:

```python
can_drop = agent.inventory != DynamicObject.EMPTY
counter_mask = self._counter_like_static_mask(state.grid[:, :, 0])
target_mask = counter_mask & (state.grid[:, :, 1] == DynamicObject.EMPTY)
action = self._go_interact_with_static_object(state, agent, target_mask)
return jnp.where(can_drop, action, Actions.stay)
```

Evidence: [`_drop_on_nearest_counter`](./overcooked.py#L431-L436). Counter-like
objects include walls, moving walls, item conveyors, and player conveyors:
[`_counter_like_static_mask`](./overcooked.py#L740-L749). The base drop rule is
[`successful_drop`](../overcooked_v3/overcooked.py#L1348-L1351).

#### `pickup_from_nearest_counter`

This macro requires an empty inventory. It targets the nearest counter-like
cell whose dynamic-object layer is non-empty, then interacts to pick up the
item:

```python
can_pickup = agent.inventory == DynamicObject.EMPTY
counter_mask = self._counter_like_static_mask(state.grid[:, :, 0])
target_mask = counter_mask & (state.grid[:, :, 1] != DynamicObject.EMPTY)
action = self._go_interact_with_static_object(state, agent, target_mask)
return jnp.where(can_pickup, action, Actions.stay)
```

Evidence: [`_pickup_from_nearest_counter`](./overcooked.py#L438-L443), with the
base pickup behavior in
[`successful_pickup`](../overcooked_v3/overcooked.py#L1314-L1319).

#### `press_nearest_button`

This macro targets the nearest button and uses the shared interaction procedure:

```python
primitive_action = jnp.where(
    macro_action == MacroActions.press_nearest_button,
    self._go_interact_with_static_object(
        state, agent, state.grid[:, :, 0] == StaticObject.BUTTON
    ),
    primitive_action,
)
```

Evidence: [`press_nearest_button` translation](./overcooked.py#L371-L377). The
base environment detects `interact` while a button is in front of the agent and
then applies the configured button behavior:
[`step_agents`, button processing](../overcooked_v3/overcooked.py#L986-L1034).

#### `stand_on_nearest_pressure_plate`

A pressure plate differs from interaction targets because the agent must enter
the plate's walkable cell. The macro selects the nearest plate and moves onto it;
it does not emit `interact`:

```python
target_mask = state.grid[:, :, 0] == StaticObject.PRESSURE_PLATE
target_y, target_x, has_target = self._nearest_walkable_target(
    agent, target_mask
)
return self._action_to_walkable_cell(state, agent, target_y, target_x, has_target)
```

Evidence: [`_go_to_nearest_pressure_plate`](./overcooked.py#L445-L450). Once
already on the target, `_action_to_walkable_cell` emits `stay`:
[`_action_to_walkable_cell`](./overcooked.py#L674-L687).

#### `wait_for_nearest_pot`

Despite its name, this macro does not navigate to a pot. It emits `stay` on
every tick:

```python
primitive_action = jnp.where(
    macro_action == MacroActions.wait_for_nearest_pot,
    jnp.array(Actions.stay, dtype=jnp.int32),
    primitive_action,
)
```

Evidence: [`wait_for_nearest_pot` translation](./overcooked.py#L383-L388). It
continues until any recipe pot is ready or no pot has a positive cooking timer;
see Step 6 below.

## Step 4: choose the next movement while reacting to other agents

For a movement tick, the macro planner examines the four neighboring cells. It
filters out cells that are out of bounds, statically non-walkable, or currently
occupied by another agent. Among the remaining cells, it chooses the one with
the smallest precomputed static distance to the destination:

```python
candidate_unoccupied = self._cell_unoccupied_by_other_agents(
    state, agent, safe_y, safe_x
)
candidate_distances = self._distance_table[safe_y, safe_x, target_y, target_x]
scores = jnp.where(
    candidate_in_bounds & candidate_walkable & candidate_unoccupied,
    candidate_distances,
    INF_DISTANCE,
)
best_idx = jnp.argmin(scores)
has_step = scores[best_idx] < INF_DISTANCE
action = self._move_actions[best_idx]
return jnp.where(has_step, action, Actions.stay).astype(jnp.int32)
```

Evidence: [`_next_action_avoiding_agents`](./overcooked.py#L689-L717) and the
occupancy test in
[`_cell_unoccupied_by_other_agents`](./overcooked.py#L719-L730).

This is one-step reactive avoidance, not a reserved multi-agent path. Because
all agents choose from the same pre-transition state, two agents can still
simultaneously select the same destination or try to swap cells. The base
environment cancels same-cell collisions and swaps:

```python
# Same destination: collided agents retain their original positions.
new_agents = new_agents.replace(pos=_masked_positions(mask))

# Swaps: both agents retain their original positions.
swap_mask = _compute_swapped_agents(state.agents.pos, new_agents.pos)
new_agents = new_agents.replace(pos=_masked_positions(swap_mask))
```

Evidence: [`step_agents`, collision resolution](../overcooked_v3/overcooked.py#L888-L930).
On the next `step_env` call, the still-active macro recomputes another primitive
action from the resulting state.

## Step 5: run one base Overcooked V3 transition

The primitive-action array is converted back into the dictionary expected by
the base environment:

```python
primitive_action_dict = {
    f"agent_{i}": primitive_actions[i] for i in range(self.num_agents)
}

obs, next_state, rewards, dones, info = super().step_env(
    key, state, primitive_action_dict
)
```

Evidence: [`OvercookedV3Macro.step_env`](./overcooked.py#L217-L223).

The base transition then performs these world operations in order:

1. Apply primitive agent movement, collision resolution, interactions, pot
   timers, and button interactions through `step_agents`.
2. Process moving walls.
3. Process pressure plates.
4. Process item conveyors.
5. Process player conveyors.
6. Process timed barriers.
7. Process the order queue.
8. Increment environment time.
9. Check terminal state and construct observations, rewards, dones, and info.

The corresponding base code begins with:

```python
state, reward, shaped_rewards = self.step_agents(key, state, acts)

if self.enable_moving_walls:
    state = self._process_moving_walls(state)
if self.enable_pressure_plates:
    state = self._process_pressure_plates(state)
if self.enable_item_conveyors:
    state = self._process_item_conveyors(state)
if self.enable_player_conveyors:
    state = self._process_player_conveyors(state)

state = self._process_barrier_timers(state)
```

Evidence: [`OvercookedV3.step_env`](../overcooked_v3/overcooked.py#L741-L800).

## Step 6: decide whether each macro is finished

Completion is evaluated **after** the primitive transition, using `next_state`:

```python
macro_done = self._compute_macro_done(
    next_state,
    current_macro_actions,
    primitive_actions,
)
```

Evidence: [`OvercookedV3Macro.step_env`](./overcooked.py#L225-L230).

The action-specific completion conditions are:

| Macro | Marked done when |
| --- | --- |
| `wait` | Always, after its one `stay` tick. |
| `get_ingredient_N` | The agent holds ingredient `N`; the agent instead holds another object; or no pile of ingredient `N` exists. |
| `get_plate` | The agent holds a plate; the agent instead holds another object; or no plate pile exists. |
| `put_ingredient_in_nearest_pot` | The agent no longer holds an ingredient, normally because placement succeeded; or no pot remains valid for that ingredient. |
| `get_soup_from_nearest_pot` | The agent holds a cooked object; the agent no longer holds a plate; or no recipe-ready pot remains. |
| `deliver` | The agent no longer holds a cooked object, normally because delivery succeeded; or no goal exists. |
| `drop_on_nearest_counter` | The inventory is empty, normally because the drop succeeded; or no empty counter-like target remains. |
| `pickup_from_nearest_counter` | The inventory is non-empty, normally because pickup succeeded; or no occupied counter-like target remains. |
| `press_nearest_button` | The emitted primitive action was `interact`; or no button exists. |
| `stand_on_nearest_pressure_plate` | The agent is on a pressure plate; or no pressure plate exists. |
| `wait_for_nearest_pot` | Any recipe pot is ready; or no pot has a positive cooking timer. |

Evidence: all branches are in
[`_macro_done_for_agent`](./overcooked.py#L465-L559).

Because targets and completion masks are recomputed from current state on every
tick, an interaction macro can retarget if its previous target becomes invalid
but another valid target remains. If no valid target remains, its completion
condition normally ends the macro after the current transition.

There are also two unconditional termination guards:

```python
macro_done = (
    macro_done
    | (next_macro_step_count >= self.max_macro_steps)
    | dones["__all__"]
)
```

Evidence: [`step_env`, termination guards](./overcooked.py#L231-L235).

Thus every macro ends when its action-specific condition is met, when it reaches
`max_macro_steps` (80 by default), or when the whole environment terminates.
The step limit is especially important for persistent collisions or a target
that exists according to its mask but cannot actually be reached.

## Step 7: store macro state and expose debugging information

The environment stores the active macro and completion result. The counter is
reset to zero for completed macros and retained for continuing macros:

```python
next_state = next_state.replace(
    current_macro_actions=current_macro_actions,
    macro_action_done=macro_done,
    macro_step_count=jnp.where(macro_done, 0, next_macro_step_count),
)
```

Evidence: [`step_env`, state bookkeeping](./overcooked.py#L237-L241).

Finally, `info` exposes the macro, its completion flag, and the primitive action
actually selected for every agent:

```python
info["current_macro_action"] = {
    f"agent_{i}": current_macro_actions[i] for i in range(self.num_agents)
}
info["macro_action_done"] = {
    f"agent_{i}": macro_done[i] for i in range(self.num_agents)
}
info["primitive_action"] = {
    f"agent_{i}": primitive_actions[i] for i in range(self.num_agents)
}
```

Evidence: [`step_env`, info fields](./overcooked.py#L243-L254).

These fields are the most direct runtime evidence for tracing a rollout: they
show which macro was latched, which single primitive action it produced on this
tick, and whether the policy may select a new macro on the next tick.
