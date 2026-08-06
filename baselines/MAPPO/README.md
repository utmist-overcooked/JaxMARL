# MAPPO Baseline

Pure JAX MAPPO implementation, based on the PureJaxRL PPO implementation.

## 🔎 Implementation Details
General features:
* Agents are controlled by a single network architecture (either FF or RNN).
* Parameters are shared between agents.
* Each script has a `WorldStateWrapper` which provides a global `"world_state"` observation.

## 🚀 Usage

If you have cloned JaxMARL and are in the repository root, you can run the algorithms as scripts, e.g.
```
python baselines/MAPPO/mappo_rnn_smax.py
```
Each file has a distinct config file which resides within [`config`](https://github.com/FLAIROx/JaxMARL/tree/main/baselines/MAPPO/config).
The config file contains the MAPPO hyperparameters, the environment's parameters and the `wandb` details (`wandb` is disabled by default).

## Overcooked V3 Macro Actions

The macro-action experiments use three explicit trainers so their rollout
semantics remain easy to inspect:

- `mappo_macro_boundary.py` selects a macro only after the previous macro ends.
  It accumulates discounted reward per agent and applies SMDP GAE at asynchronous
  decision events. `NUM_STEPS` must be a multiple of the environment's
  `max_steps` so pending actions never cross an on-policy update.
- `mappo_macro_every_step.py` samples a desired macro every primitive step.
  Selecting a different macro interrupts the active one; selecting the same
  macro continues it.
- `mappo_macro_replan.py` learns a `CONTINUE`/`REPLAN` gate and samples from the
  macro head only at idle or replanning decisions. The current macro is masked
  from replacement choices.

Run a variant from the repository root, for example:

```bash
python baselines/MAPPO/mappo_macro_boundary.py
python baselines/MAPPO/mappo_macro_every_step.py
python baselines/MAPPO/mappo_macro_replan.py
```

All three count `TOTAL_TIMESTEPS` in primitive environment steps for comparable
training budgets. Shared networks, losses, observation augmentation, and
logging utilities live in `mappo_macro_common.py`.

The macro trainers also:

- mask macros that are incompatible with the current inventory and world state;
- linearly anneal Overcooked shaped rewards over `REW_SHAPING_HORIZON` primitive
  steps;
- reject timestep budgets that would be silently truncated by rollout batching;
- log metrics to W&B during training without uploading policy files;
- run deterministic evaluation every `EVAL_INTERVAL_UPDATES` updates; and
- save resumable local checkpoints plus best and final actor/critic weights under
  `SAVE_PATH`.

Set `RESUME_FROM` to either a checkpoint `.npz` file or a `checkpoints` directory
containing `latest.json` to continue a stopped run. Checkpoints include optimizer,
environment, RNG, and policy state.
