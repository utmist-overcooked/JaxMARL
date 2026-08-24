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

## Protein Hyperparameter Sweep

`protein_sweep.py` runs [PufferLib's Protein](https://github.com/PufferAI/PufferLib/blob/4.0/pufferlib/sweep.py)
Bayesian-optimization sweep over any of the four macro trainers. Protein keeps
two Gaussian Processes — one over score, one over log-cost — and proposes points
around the Pareto frontier of past trials, trading predicted evaluation return
against predicted wall-clock cost (bounded by `--max-suggestion-cost` seconds).

The optimizer itself is vendored, single-file and dependency-free of PufferLib,
in `protein.py` (a near-verbatim copy of their `sweep.py`, MIT-licensed). Install
its extra dependencies with:

```bash
pip install -e '.[sweep]'   # torch, gpytorch, scikit-learn (scipy is already core)
```

All four trainers share the `run_experiment(config, make_train, name)` entry
point, so one driver targets any of them via `--target`:

```bash
cd baselines/MAPPO
python protein_sweep.py --target every_step --max-runs 30
python protein_sweep.py --target boundary  --max-runs 30
python protein_sweep.py --target replan    --max-runs 30
python protein_sweep.py --target every_step_comm --max-runs 30
```

Each trial merges Protein's suggestion into the trainer's base config, runs a
single-seed `run_experiment`, reads the best evaluation return from the run's
`best_eval.json`, measures wall-clock cost, and feeds both back with
`observe(...)`. Structurally invalid suggestions (e.g. a `NUM_MINIBATCHES` that
breaks `BATCH_SIZE` divisibility) are caught and reported as failed trials rather
than aborting the sweep.

### Search space (mirrors PufferLib's `default.ini`)

The search space and optimizer settings live in `config/sweep/protein.yaml`
(`config/sweep/protein_comm.yaml` for `every_step_comm`). The design follows
PufferLib's own sweep config:

- **`TOTAL_TIMESTEPS` is the searched cost lever** (`log_normal`, `scale=time`),
  wired up as Protein's `cost_param` so the cost GP is anchored to it. The driver
  snaps each suggestion to a whole number of `NUM_STEPS*NUM_ENVS` batches so the
  divisibility check always passes, and feeds the snapped value back to `observe`.
- **`NUM_ENVS` is fixed, not swept** (default 4096) — like PufferLib's
  `vec.total_agents`, it is a throughput/batch-size knob. A big fixed batch gives
  the most stable gradient updates; sweeping the budget then finds how many steps
  that regime needs. It is set via a `fixed:` block in the sweep YAML (a
  `protein_sweep.py` convention, not part of Protein) so the committed training
  configs are left untouched.
- **Algorithmic ranges are kept wide** (`LR` 1e-5–1e-2, `GAE_LAMBDA` 0.5–0.995,
  `HIDDEN_SIZE` 64–512, …) — Protein's GP narrows in, so a broad box is cheap.
- **Model size and Adam internals are searched too** — `NUM_LAYERS`
  (PufferLib's `policy.num_layers`) and `ADAM_B1` / `ADAM_B2` / `ADAM_EPS`
  (their `beta1` / `beta2` / `eps`). The GP's per-dimension (ARD) lengthscales
  suppress whichever of these don't matter for a given layout, which is what
  makes sweeping ~14 dimensions viable. The only PufferLib knobs left out are its
  V-trace / prioritized-replay parameters, which have no analogue in this
  synchronous on-policy MAPPO.
- **`NUM_STEPS`** (PufferLib's `horizon`) is provided commented-out; safe to
  enable for `every_step`/`replan`, but for `boundary` it is coupled to episode
  length, so leave it fixed there.

Two mechanical notes on the newly-swept knobs: `NUM_LAYERS` and `ADAM_*` are read
by the trainers with safe defaults (`NUM_LAYERS=2`, `ADAM_EPS=1e-5`, standard
betas), so configs without them are unchanged. Sweeping `NUM_LAYERS` changes the
network's parameter tree, so checkpoints are **not** interchangeable across depths
(same caveat as `USE_RNN`) — fine for a sweep since every trial starts fresh, but
it is deliberately **not** swept for `every_step_comm`, whose actor/critic are
frozen and must match the loaded checkpoint's depth.

Reserved keys (`method`, `metric`, `goal`, `metric_distribution`, `downsample`,
`early_stop_quantile`, `prune_pareto`, `max_runs`) configure Protein; `fixed:`
holds sweep-time base-config overrides; every other top-level key is a searchable
hyperparameter. Override base-config entries per sweep with `--override`:

```bash
python protein_sweep.py --target every_step --max-runs 40 \
    --max-suggestion-cost 1800 \
    --override WANDB_MODE=offline NUM_ENVS=2048
```

(`--override` pins a base-config value for the whole sweep and wins over the
`fixed:` block; it also overrides a searched key if you name one, effectively
removing it from the search.)

Per-trial checkpoints, a `sweep_results.jsonl` log, and the winning `best.json`
are written under `--save-path` (default `models/protein_sweep/<target>/`).

Notes:

- Observed cost is wall-clock **seconds**, while the *searched* cost dimension is
  `TOTAL_TIMESTEPS` — the two are monotonically related (more steps → more
  seconds), so Protein trades return against real runtime while still steering the
  training budget. Cost includes JAX compilation time; structural changes
  (`HIDDEN_SIZE`, `USE_RNN`) trigger recompilation and are billed accordingly.
- Protein's cost-aware `early_stop` is **not** wired into the scanned JAX
  training loop (that would require host callbacks mid-`lax.scan`); the sweep
  observes only completed-run score and cost.
- Set `method: random` or `method: pareto_genetic` in the sweep YAML to use a
  cheaper baseline optimizer (no torch/gpytorch GP dependency) instead of
  `method: protein`.
