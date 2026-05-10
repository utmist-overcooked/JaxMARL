# Reward-Event Histograms During Training

This document describes how to add reward-event histogram tracking to future
training scripts. The intended statistic is event count across training
updates/epochs, not average count per completed episode.

## Goal

Track what the policy is actually doing as training progresses. For each
training update, count how many times important environment events happened in
the rollout collected for that update, then log and save those counts.

For Overcooked-style environments, useful events include:

- `placement_in_pot`
- `plate_pickup`
- `soup_in_dish`
- `delivery`
- `pot_burned`

The output should answer questions like:

- Did deliveries start happening after a certain update?
- Is the agent picking up plates but never picking up soup?
- Is the agent placing ingredients in pots but failing downstream?
- Which part of the task chain is the bottleneck over training time?

## Do Not Infer From Episode Return

Do not infer event type from total episode return. Episode return is a sum, so
it is ambiguous. A total shaped reward can be produced by many different event
combinations.

Counting must happen at the step level while the environment still exposes the
raw per-step reward and/or event metadata.

## Prefer Explicit Event Flags

The robust pattern is for the environment to emit explicit event flags in
`info`, for example:

```python
info["reward_events"] = {
    "placement_in_pot": placement_in_pot.astype(jnp.float32),
    "plate_pickup": plate_pickup.astype(jnp.float32),
    "soup_in_dish": soup_in_dish.astype(jnp.float32),
    "delivery": delivery.astype(jnp.float32),
    "pot_burned": pot_burned.astype(jnp.float32),
}
```

Use reward-value inference only as a fallback when reward values are guaranteed
to be one-to-one with event types. In current Overcooked V3, this is not true:
`PLACEMENT_IN_POT` and `PLATE_PICKUP` both use shaped reward `0.1`, so reward
magnitude alone cannot distinguish them.

## Environment Implementation Pattern

Add event flags at the point where the environment already computes the event
conditions.

For Overcooked V3, this is inside `process_interact`, where the code already
computes booleans such as:

- `successful_pot_placement`
- `is_pot_placement_useful`
- `successful_plate_pickup`
- `is_plate_pickup_useful`
- `successful_dish_pickup`
- `is_dish_pickup_useful`
- `correct_delivery`

Return a small pytree of scalar flags from `process_interact`, scan those flags
across agents in `step_agents`, and include them in `step_env` info:

```python
reward_events = {
    "placement_in_pot": jnp.asarray(
        successful_pot_placement * is_pot_placement_useful,
        dtype=jnp.float32,
    ),
    "plate_pickup": jnp.asarray(
        successful_plate_pickup * is_plate_pickup_useful,
        dtype=jnp.float32,
    ),
    "soup_in_dish": jnp.asarray(
        successful_dish_pickup * is_dish_pickup_useful,
        dtype=jnp.float32,
    ),
    "delivery": jnp.asarray(correct_delivery, dtype=jnp.float32),
}
```

Pot burns are not interaction events. Track them where pot timers are updated.
For Overcooked V3, `_update_pot_timers` already computes `just_burned`; return
`jnp.asarray(is_active & just_burned, dtype=jnp.float32)` from the pot scan, sum
it across pots, and write the result to `reward_events["pot_burned"]`.

The exact event names can differ by environment, but they should be stable
string keys and should represent physical events, not reward broadcasts. If a
delivery reward is broadcast to all agents, count the delivery event once from
the actor that delivered, not once per rewarded agent.

## Training Script Aggregation Pattern

In on-policy training scripts such as IPPO/MAPPO, the rollout collection usually
returns a `traj_batch` with shape:

```text
NUM_STEPS x NUM_ACTORS x ...
```

After collecting the rollout and before averaging metrics, aggregate event
counts across the whole update:

```python
REWARD_EVENT_NAMES = (
    "placement_in_pot",
    "plate_pickup",
    "soup_in_dish",
    "delivery",
    "pot_burned",
)

reward_event_counts = {
    f"reward_events/{event}_count": traj_batch.info["reward_events"][event].sum()
    for event in REWARD_EVENT_NAMES
}

reward_event_rates = {
    f"reward_events/{event}_per_env_step": count / (config["NUM_STEPS"] * config["NUM_ENVS"])
    for event, count in zip(REWARD_EVENT_NAMES, reward_event_counts.values())
}
```

Then add those values to the metrics logged for the current update:

```python
metric.update(reward_event_counts)
metric.update(reward_event_rates)
```

Use the count metric as the primary histogram/time-series value. The normalized
rate is useful for comparing runs with different `NUM_STEPS` or `NUM_ENVS`.

## Output Files

For every training run, save a compact CSV with one row per update:

```text
update,placement_in_pot,plate_pickup,soup_in_dish,delivery,pot_burned
1,0,0,0,0,0
2,14,3,0,0,0
3,42,11,2,0,1
...
```

Also save a static plot with update on the x-axis and event count on the y-axis.
Use a non-interactive matplotlib backend:

```python
import matplotlib
matplotlib.use("Agg")
```

Never call `plt.show()` in training scripts, especially for SLURM jobs.

Recommended output directory:

```text
{WANDB_DIR}/histograms/
```

Recommended filenames:

```text
{algo}_{env}_{layout}_seed{seed}_reward_events_by_update.csv
{algo}_{env}_{layout}_seed{seed}_reward_events_by_update.png
```

## WandB / Monitor Logging

Log the same per-update counts to WandB when WandB is enabled:

```text
reward_events/placement_in_pot_count
reward_events/plate_pickup_count
reward_events/soup_in_dish_count
reward_events/delivery_count
reward_events/pot_burned_count
```

If a local progress monitor is used, show only one or two high-signal fields,
such as `delivery_count`, to avoid clutter.

## JAX Notes

Keep the event data JAX-friendly:

- Use dictionaries of arrays, not Python objects.
- Use `jnp.float32` or `jnp.int32` flags.
- Make the true and false branches of `jax.lax.cond` return the same pytree
  structure.
- Do aggregation with JAX operations inside the jitted train step.
- Convert to NumPy only after training returns to Python and files are saved.

## Checklist For Future Training Scripts

1. Identify the exact environment code that computes the event conditions.
2. Add explicit event flags to `info["reward_events"]`.
3. Preserve those flags through vectorized env stepping and wrappers.
4. In the training update, sum flags across `NUM_STEPS` and actors/envs.
5. Add per-update counts and normalized rates to metrics.
6. Save a CSV and static PNG after training.
7. Run a tiny smoke test with small `NUM_ENVS`, `NUM_STEPS`, and
   `TOTAL_TIMESTEPS`.
8. Inspect the CSV and verify the columns exist, even if all counts are zero in
   the smoke run.

## MAPPO Overcooked V3 Reference

The reference implementation in this branch applies the pattern to:

```text
jaxmarl/environments/overcooked_v3/overcooked.py
baselines/MAPPO/mappo_rnn_overcooked_v3.py
```

The environment emits `info["reward_events"]`; the MAPPO training script logs
`reward_events/*_count` per update and saves CSV/PNG files under:

```text
${WANDB_DIR}/histograms/
```
