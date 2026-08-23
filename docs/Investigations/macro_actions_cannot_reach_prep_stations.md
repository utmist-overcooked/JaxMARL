# Macro actions cannot reach prep stations, sinks, or dirty-plate piles

Found while merging `feat_multi_satged_orders_and_actions` into the
macro-action + communication line.

## Summary

`overcooked_v3` gained three prep stations (cutting board, grill, blender) plus
the dish-washing sink and dirty-plate pile. The macro-action layer was never
extended to match, so a policy driving `overcooked_v3_macro[_interruptible]`
has no action that can interact with any of those tiles. On a prep-station or
dish-washing layout, macro agents can pick up raw ingredients and walk around,
but they can never process an ingredient, never fill a pot with the processed
form the recipe asks for, and therefore never deliver.

This is a gap in the macro layer, not a regression from the merge. It matters
now because the merge makes those layouts reachable from macro training for the
first time.

## Evidence

`MacroActions` (jaxmarl/environments/overcooked_v3_macro/overcooked.py) has 18
members. None of them name a prep station, a sink, or a dirty-plate pile:

    wait, get_ingredient_0..2, get_plate, put_ingredient_in_nearest_pot,
    get_soup_from_nearest_pot, deliver, drop_on_nearest_counter,
    pickup_from_nearest_counter, press_nearest_button,
    stand_on_pressure_plate_0..1, wait_for_nearest_pot, up, down, left, right

`drop_on_nearest_counter` / `pickup_from_nearest_counter` are the only generic
"put it on that tile" macros, and their target mask comes from
`_counter_like_static_mask`, which covers exactly four statics:

    WALL | MOVING_WALL | ITEM_CONVEYOR | PLAYER_CONVEYOR

`StaticObject.CUTTING_BOARD` (26), `GRILL` (27), `BLENDER` (28), `SINK` (29) and
`DIRTY_PLATE_PILE` (30) are all absent, so no macro ever plans a path to one or
emits an `interact` while facing one.

Corroborating run -- 3 x 400 steps of uniformly random macro actions on
`prep_kitchen_handoff` (two agents, orders cycling over the three prep dishes):

    event/prep_placement   0.0
    event/prep_action      0.0
    event/prep_pickup      0.0
    event/pot_placement    0.0
    event/delivery         0.0

Random exploration is weak evidence on its own, but combined with the mask above
it is consistent: the events are unreachable, not merely unlikely. The same
prep interactions do fire under primitive actions -- `tests/overcooked_v3/
test_prep_stations.py` drives them with a scripted primitive policy and passes.

## Consequence for training

Any macro-action run on `cutting_board_room`, `grill_room`, `blender_room`,
`prep_kitchen`, `*_handoff`, `prep_dish_kitchen`, or the `dish_washing_*`
layouts will train against an unsolvable task: the sparse delivery reward is
unreachable, and only the movement-shaping terms can ever fire. Multi-staged
*order queues* are fine -- those work on any layout with two or more orderable
dishes and need no new macros.

## What closing the gap would take

1. Add macros for the new tiles, e.g. `use_nearest_prep_station` (or one macro
   per station type so the policy can choose a chain), `wait_for_nearest_prep_
   station`, `wash_plate`, `get_dirty_plate`. Per-station-type macros match how
   `stand_on_pressure_plate_0/1` was already split for exactly this reason.
2. Give each new macro a target mask in the same `target_mask` chain that
   `drop_on_nearest_counter` uses, and a completion predicate alongside the
   existing ones.
3. Extend the valid-action mask so a macro is only offered when it can succeed
   (station empty and inventory holds the matching raw item; station holds the
   processed result and hands are empty; etc.).
4. Add macro-layer tests mirroring `tests/overcooked_v3/test_prep_stations.py`.
