# Overcooked V3 baselines

This package contains the Overcooked V3-specific model, trainer, evaluation,
and GIF logging code. Historical Overcooked and Overcooked V2 baselines are not
part of this interface.

Overcooked V3-specific models and trainers can live under `models/` and
`trainers/`. Existing algorithm-family entrypoints remain available as thin
compatibility wrappers while they move onto this interface.

## Add another model

1. Define the V3 architecture under `models/`, or import a genuinely shared
   architecture from its existing algorithm package.
2. Implement `RolloutPolicy.initial_state()` and `RolloutPolicy.act()` in a
   small adapter next to that model.
3. Construct `OvercookedV3Training` with a factory for that adapter.
4. Call `checkpoint_saved()` immediately after each checkpoint is written.

The shared runner controls environment reset/step/termination and records the
states required by `OvercookedV3Visualizer`. The adapter controls only
model-specific preprocessing and recurrent or communication state. If
`act()` returns actions, the runner uses them unchanged. If it returns a
distribution, the runner calls `mode()` to choose its highest-probability
action.

`NUM_CHECKPOINTS` is the number of checkpoint-save events in one run.
`ROLLOUT_GIF_COUNT` chooses how many of those checkpoints receive GIFs. The
checkpoint count must divide evenly by the GIF count: 20 checkpoints and 10
GIFs select checkpoints 2, 4, ..., 20, while 19 checkpoints and 10 GIFs raise
an error before training starts.

`ROLLOUT_GIF_ENV_SEED` stays fixed across checkpoints so progress GIFs are
directly comparable. `ROLLOUT_GIF_SEED_INDEX` chooses which vectorized training
seed supplies parameters when `NUM_SEEDS` is greater than one.
