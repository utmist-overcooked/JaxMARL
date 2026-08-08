# Overcooked V3 MARL Training — Run Setup & Handoff

Operational guide for training/diagnosing MARL algorithms on `overcooked_v3` maps in
this repo. Written as a handoff: read this before launching runs in a new session.

## Goal / context
Training and reward-tuning value-based (QMIX, VDN) and on-policy (IPPO) recurrent MARL
on `overcooked_v3` maps, primarily:
- `around_the_island` — single connected region, **solvable** (IPPO solves reliably).
- `coordinated_temporal_conveyor` (**CTC**) — two disconnected regions joined by a
  conveyor handoff; the hard map. IPPO solved it once (see below); QMIX has not.
- `maze_conveyor_hell` — exploration-hard; QMIX earns ~0 (bootstrapping fails).

## Environment / infrastructure (READ FIRST)
- Python: `/student/brownd58/dev/JaxMARL/.venv/bin/python`, `PYTHONPATH=/student/brownd58/dev/JaxMARL`.
- **ptxas PATH fix (required almost every run)** — without it JAX dies with
  `Unsupported .version 8.3; current version is '7.8'`:
  ```
  export PATH=/student/brownd58/dev/JaxMARL/.venv/lib/python3.10/site-packages/nvidia/cuda_nvcc/bin:$PATH
  ```
- `export XLA_PYTHON_CLIENT_MEM_FRACTION=0.99`
- **One ~16 GB GPU.** Run algorithms sequentially, not concurrently. To queue a second
  run, launch a watcher tmux: `while tmux has-session -t <first>; do sleep 30; done; bash <second.sh>`.
- Runs go in **detached tmux sessions**, stdout → `outputs/<name>_train.log`. Checkpoints →
  `checkpoints/<name>/`, GIFs → `outputs/<name>.gif`.
- wandb entity `zacharytang24-`; logged-in user is `dannyb3334`. Recent runs share project
  `ocv3_qlearning_coordinated_temporal_conveyor`.

## Launch patterns
Scripts live in `scripts/run_*.sh` and each create their own tmux session. Copy an
existing one and `sed` the tags (session name, WANDB_NAME, SAVE_PATH, SAVE_GIF_PATH, log path).

**IPPO** — `baselines/IPPO/ippo_rnn_overcooked_v3.py --config-name=ippo_rnn_overcooked_v3`,
top-level CLI keys: `WANDB_PROJECT=`, `WANDB_NAME=`, `ENV_KWARGS.layout=`, `SHAPED_REWARD_COEFF=`,
etc. New keys (not in the yaml) need a `+` prefix, e.g. `+LOAD_PATH=`.

**QMIX** — `baselines/QLearning/qmix_rnn.py +alg=ql_rnn_overcooked_v3`, alg-prefixed keys:
`alg.NUM_ENVS=`, `alg.ENV_KWARGS.layout=`, `alg.SHAPED_REWARD_COEFF=`. `PROJECT=` is
top-level; new alg keys need `+alg.` (e.g. `+alg.WANDB_NAME=`, `+alg.LOAD_PATH=`).

## Proven recipes
**IPPO around_the_island (the "superior" reference, run `bld6tz9b`):**
`max_steps=1000, SHAPED_REWARD_COEFF=30, ENT_COEF=0.01, REW_SHAPING_MIN_COEFF=0.1,
REW_SHAPING_HORIZON=15M, NUM_ENVS=128, NUM_STEPS=200, LR=5e-4, ANNEAL_LR=true,
GRU/FC=128, GAMMA=0.99, GAE=0.95, CLIP=0.2, VF=0.5, MAX_GRAD_NORM=0.5, alternating queue,
guard=1.` First deliveries ~13% in, saturates ~25%.

**IPPO CTC solved (run `zgm3h898`):** `coeff=20, min=0.1, ent=0.1, max_steps=400`, WITH the
dense+handoff shaping. Gave 70–82 deliveries/batch. **Dense shaping is required on CTC** —
event-only rewards give 0 deliveries (ablation `0fw9j526`); IPPO's 128 envs + entropy +
dense pull are what discover the cross-region serve chain.

**QMIX:** `+alg=ql_rnn_overcooked_v3`, NUM_ENVS=4 (memory), NUM_STEPS=400=max_steps
(full-episode BPTT by design), BUFFER_SIZE=512/BATCH=32, HIDDEN_SIZE=256, NUM_EPOCHS=8 for
value-based overcooked sweeps, Huber loss (in code) to avoid mixer divergence. **QMIX has
not solved CTC** — `event/dish_pickup` stays 0; 4 envs can't discover the serve chain.

## Reward shaping (`jaxmarl/environments/overcooked_v3/settings.py`)
`SHAPED_REWARDS` is read directly by the env and scaled by `SHAPED_REWARD_COEFF` then
annealed (`REW_SHAPING_MIN_COEFF + (1-min)*linear_decay(step, REW_SHAPING_HORIZON)`).
Env reads each key via `.get(name, 0.0)`, so a missing key just means that term is off.

Key effects / traps:
- `POT_START_COOKING` **farm trap**: too high (e.g. 5 × coeff) makes "cook → let burn →
  recook" net-positive (burn penalty is unscaled −5), so agents farm cook-starts and never
  serve. Keep effective value modest (~60 worked: 3×coeff20). Currently 2.0.
- Dense terms `TASK_PROGRESS` / `TASK_FACING` drive the empty→plate→pot→goal pull via a
  subtask target mask (`_task_target_mask`). Required for CTC serve discovery; ~0.05/0.01.
- `HANDOFF_DROP`/`HANDOFF_PICKUP` (~0.25) reinforce the conveyor handoff.
- `PLATE_PICKUP` guarded: only rewarded when soup is **cooked/ready** (not while cooking),
  and capped at `plate_pickup_guard × ready_pots` — stops plate pickup spam during cooking.
- `INGREDIENT_WASTE` (−0.004): fires when a **wrong-type** ingredient is dropped onto a
  conveyor (the "order switched → dump food" behavior); does NOT penalize correct handoffs.
- `IDLE_PENALTY` (−0.001): small penalty on the explicit `stay` no-op to stop greedy QMIX
  policies freezing in place (does not touch `interact`).

## Env kwargs that matter
- `pot_cook_time` / `pot_burn_time` default to `settings.POT_COOK_TIME` / `POT_BURN_TIME`
  (currently 60 / 90, CoGrid defaults). **If a script passes them on the CLI it overrides
  settings** — omit or match settings to honor a settings change.
- `order_queue_mode=alternating` **alternates the required ingredient type (0↔1) per order**
  (needs ≥2 ingredient piles). The front order flips type after each delivery; with
  `order_expiration_time=0` both types are always pending. This is why CTC has two piles and
  why agents holding the now-wrong type may dump it.
- `enable_item_conveyors=true` for conveyor maps (auto-enabled from layout if unset).
  `enable_player_conveyors=false`. `plate_pickup_guard=1`. `max_orders`, `order_generation_rate`.

## Warm-start (added this session)
Both trainers support `LOAD_PATH` to initialize from a previous checkpoint (online + target
nets for QMIX; replay buffer still starts empty):
- IPPO: `+LOAD_PATH=checkpoints/<run>/` (loads `model.msgpack`).
- QMIX: `+alg.LOAD_PATH=checkpoints/<run>/<env_name>/..._vmap0.safetensors` (full file path).
Confirm with the `[warm-start] ... initialized from ...` log line.

## GIF generation
Training auto-saves a GIF to `SAVE_GIF_PATH` at the end (full episode). QMIX does NOT upload
it to the wandb media panel (saved to disk only) — IPPO does. To re-render any checkpoint at
any seed:
- IPPO: `scripts/generate_ippo_v3_gif.py --checkpoint <model.msgpack> --layout <l> --output <g> --seed <n>`
- QMIX: `scripts/generate_qmix_v3_gif.py --checkpoint <...vmap0.safetensors> --output <g> --seed <n>`
Both rebuild the CTC env with the correct kwargs (order queue, conveyors). GIFs are
full-episode; the visualizer adds per-frame markers so encoders don't merge static frames.

## Monitoring (wandb API)
```python
import wandb; r = wandb.Api().run("zacharytang24-/<project>/<run_id>")
h = r.history(keys=["env_step","event/delivery","event/dish_pickup","event/pot_start_cooking",
                    "event/pot_burn","event/pot_placement"], pandas=False)
```
Success signal on CTC: **`event/dish_pickup` > 0** (the serve step that's the hard part),
then `event/delivery`. `pot_burn ≈ pot_start_cooking` with `dish_pickup=0` = cook-burn farm.

## Common pitfalls
- `import jaxmarl` triggers GPU init → ptxas error if PATH not set (see above).
- OOM scales with `BUFFER_SIZE × NUM_STEPS × obs_size`; CTC needed BUFFER_SIZE≤512. Long
  episodes (NUM_STEPS=1000) blow up QMIX memory.
- Editing `settings.py` can truncate the moving-walls/buttons/barriers constants
  (`MAX_MOVING_WALLS`, `MAX_BUTTONS`, `MAX_BARRIERS`, `MAX_BUTTON_TARGETS`,
  `DEFAULT_BARRIER_DURATION`) that the env imports — keep them.
- After changing `SHAPED_REWARDS`, smoke-test: build the env + one `step` to catch missing-key
  / dtype errors before a long run.
