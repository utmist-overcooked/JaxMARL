# Handoff: run MAPPO + communication (CommNet & FSQ) on Overcooked V3

Paste the "Prompt" section below into a Claude Code session on the new machine.
Everything after it is reference detail.

---

## Prompt (paste this)

> I'm on a new machine with a working NVIDIA driver. In `/student/brownd58/dev/JaxMARL`
> (branch `test_daniel_mappo`) there is a MAPPO-RNN with pluggable inter-agent
> communication at `baselines/MAPPO/mappo_rnn_overcooked_v3_fsq.py`, supporting
> `COMM_TYPE=commnet|fsq|none`. It is ported and unit-verified but has never run on a GPU.
>
> Please:
> 1. Verify the GPU works for JAX before anything else (see "GPU checks" in
>    `docs/HANDOFF_mappo_comm_runs.md` — this repo has two separate, recurring
>    NVIDIA failure modes; check both).
> 2. Launch the CommNet run on `around_the_island`, then the FSQ run after it
>    finishes. One GPU: never run them concurrently.
>    - `bash scripts/run_mappo_commnet_comm_around_the_island.sh`
>    - `bash scripts/run_mappo_fsq_comm_around_the_island.sh`
>    Both wait for a working GPU and self-launch; set `FSQ_WAIT_GPU=0` to skip waiting.
>    Use detached tmux, logs to `outputs/<name>_train.log`.
> 3. Report per-run: `env_step`, `returned_episode_returns`, entropy, and for FSQ also
>    `fsq_unique_codes` / `fsq_code_entropy` (these are 0 by design under CommNet).
>
> First do a 2-update smoke test of each on GPU before committing to the full 10M run.
> Read the "Known traps" section first — there are three environment landmines that
> will silently waste a run.

---

## What this code is

MAPPO-RNN, centralized critic over concatenated partial observations, actors partially
observed (`agent_view_size=2`) and talking to each other each step.

`COMM_TYPE` selects the channel:

| value | mechanism |
|---|---|
| `commnet` | Continuous (Sukhbaatar et al. 2016). Each agent reads the **mean of the other agents'** hidden states, folded in over `COMMNET_ROUNDS` rounds of `h <- tanh(H h + C c)`. Scales past 2 agents. |
| `fsq` | Discrete. Each agent quantizes its hidden state into one code of a `prod(FSQ_LEVELS)`-word codebook (default `[5,5,5]` = 125 codes) and reads its partner's code. 2 agents only (partner = flip of the agent axis). |
| `none` | Ablation: no channel, identical network shape. |

`DISABLE_FSQ_COMM: True` is a legacy switch that overrides `COMM_TYPE` to `none`.

Ported from `origin/codex/overcooked-fsq-distill`'s
`mappo_rnn_overcooked_v3_fsq_distill.py` **with the full-observation teacher
distillation removed** — no teacher checkpoint is needed; it trains from environment
reward alone.

### Already verified (CPU)
- Trains end-to-end and saves actor/critic checkpoints, under both `commnet` and `fsq`.
- FSQ: codes land exactly on the quantization grid; partner routing correct both ways.
- CommNet: messages are continuous (not on a lattice); mean-of-others routing correct.
- **Gradients reach the message head in both modes** — the FSQ straight-through
  estimator works. This is the failure that would otherwise be silent: a dead channel
  trains happily and just never learns to communicate.

### Never done
Any GPU run. Any run long enough to show learning. No comm-vs-no-comm comparison yet.

---

## Known traps

**1. `scripts/` is gitignored** (`.gitignore` line 3), as are `outputs/` and
`checkpoints/`. So `git pull` will **not** bring the launch scripts. Home is on NFS
(`localhost:/student`), so if the new box mounts the same `/student` everything is
already in place and nothing needs copying. If it does not, copy `scripts/` manually.

**2. Uncommitted work.** Committed already: prep stations, dish washing, order rotation
(`036b1c5`, `7294095`). Still uncommitted:
- `M jaxmarl/environments/overcooked_v3/layouts.py` — the `prep_kitchen_handoff_orders_alt_dishes` layout
- `M tests/overcooked_v3/test_dish_washing.py` — its tests
- `?? baselines/MAPPO/{fsq.py, fsq_viewer.py, mappo_rnn_overcooked_v3_fsq.py}` and
  `?? baselines/MAPPO/config/mappo_rnn_overcooked_v3_{fsq,commnet}.yaml` — **the entire
  comm port**

Commit these before relying on git to move anything.

**3. A stale `jaxmarl` is installed in the venv.** `.venv/.../site-packages/jaxmarl` is a
real directory (not an editable install) and is **months out of date** — no prep
stations, no dish washing, no new layouts. Any script that does not put the repo root on
`sys.path` silently imports it and runs old code. The comm script and the IPPO baselines
insert the repo root themselves; ad-hoc scripts need `PYTHONPATH=/student/brownd58/dev/JaxMARL`.
Sanity check:
```bash
python -c "import jaxmarl, jaxmarl.environments.overcooked_v3.common as c; \
print(jaxmarl.__file__, hasattr(c.DynamicObject,'DIRTY'))"   # must print the repo path and True
```

**4. flax `nn.scan` is broken in this venv** (flax 0.10.4 + jax 0.4.38 → `jax.api_util`
has no `debug_info`). **Every stock MAPPO baseline here fails because of it**, e.g.
`mappo_rnn_mpe.py`. The comm script sidesteps it with a manual GRU + `jax.lax.scan`
(same approach as the working IPPO V3 baseline). If you port another MAPPO variant, expect
this and reuse `ScannedRNN` from `mappo_rnn_overcooked_v3_fsq.py`.

---

## GPU checks — two distinct NVIDIA failure modes

Check **both** before launching; they look different and have different fixes.

### A. ptxas version skew (the long-standing one)
```
XlaRuntimeError: INTERNAL: ptxas exited with non-zero error code
ptxas ... Unsupported .version 8.3; current version is '7.8'
```
The system `ptxas` is older than the PTX that JAX emits. Fix, needed on essentially
every run (already inside all the launch scripts):
```bash
export PATH=/student/brownd58/dev/JaxMARL/.venv/lib/python3.10/site-packages/nvidia/cuda_nvcc/bin:$PATH
```
Forgetting it makes `import jaxmarl` itself blow up, because importing triggers GPU init.

### B. Driver/library version mismatch (what killed the old machine)
```
nvidia-smi -> Failed to initialize NVML: Driver/library version mismatch
JAX        -> kernel version 535.309.1 does not match DSO version 580.173.2
              No visible GPU devices -> silently falls back to CPU
```
Cause: the driver was upgraded 535 → 580, which swapped the userspace libraries, but the
running kernel still had the **old 535 module** loaded. Linux cannot replace an in-use
module, so it persists until reloaded or rebooted. Diagnose:
```bash
cat /proc/driver/nvidia/version                              # loaded kernel module
readlink /usr/lib/x86_64-linux-gnu/libcuda.so.1              # userspace library
```
These two **must** match. If they do not, there is **no user-space workaround** — the old
matching libraries get deleted on upgrade, the cached `.deb` is a ~12 KB transitional stub,
and the archive only offers that stub. `libcuda.so` is driver-supplied and cannot come from
pip. It requires root: `scripts/fix_nvidia_driver.sh` (stops the display manager, reloads
`nvidia_drm`/`nvidia_modeset`/`nvidia_uvm`/`nvidia`, verifies, rolls back on failure) — a
reboot also works. On the old box nothing held the GPU except the `gdm3`/Xorg session.

**The dangerous part: JAX falls back to CPU silently.** A run will appear to start
normally and be ~100x too slow. Always confirm before a long run:
```bash
python -c "import jax; print(jax.devices())"    # must list a CudaDevice, not CpuDevice
```

---

## The runs

Both scripts poll for a working GPU and launch themselves; `FSQ_WAIT_GPU=0` runs
immediately instead. The FSQ script additionally waits for the CommNet tmux session to
disappear, so they serialize on a single GPU.

```bash
cd /student/brownd58/dev/JaxMARL
tmux new-session -d -s commnet "bash scripts/run_mappo_commnet_comm_around_the_island.sh"
tmux new-session -d -s fsq     "bash scripts/run_mappo_fsq_comm_around_the_island.sh"
```

Both: `around_the_island`, `agent_view_size=2`, `max_steps=400`, 10M steps,
256 envs x 256 steps, LR 2.5e-4 annealed, `ENT_COEF=0.04`, GRU 128 / FC 64, seed 0.
CommNet uses `COMMNET_ROUNDS=2`; FSQ uses `FSQ_LEVELS=[5,5,5]`.
W&B projects `ocv3_mappo_commnet_comm` and `ocv3_mappo_fsq_comm`, entity `zacharytang24-`.

Smoke test first (~1 min on GPU):
```bash
export PATH=/student/brownd58/dev/JaxMARL/.venv/lib/python3.10/site-packages/nvidia/cuda_nvcc/bin:$PATH
python baselines/MAPPO/mappo_rnn_overcooked_v3_fsq.py COMM_TYPE=commnet COMMNET_ROUNDS=2 \
  ENV_KWARGS.layout=around_the_island TOTAL_TIMESTEPS=131072 NUM_ENVS=64 NUM_STEPS=64 \
  NUM_MINIBATCHES=4 WANDB_MODE=disabled USE_RICH_MONITOR=False DISABLE_CHECKPOINTS=True \
  CHECKPOINT_GIF=False CHECKPOINT_FSQ_VIEWER=False
```

### Worth running afterwards
The `none` ablation, so the comm results mean something:
```bash
python baselines/MAPPO/mappo_rnn_overcooked_v3_fsq.py COMM_TYPE=none ...
```
Three runs (`commnet` / `fsq` / `none`), identical otherwise, is the actual experiment.

### Reading the results
- `around_the_island` is known-solvable — IPPO solves it reliably, so a flat return curve
  means something is wrong, not that the task is hard.
- FSQ only: `fsq_unique_codes` rising above 1 is the signal that the codebook is being
  used. Stuck at 1 = collapsed channel (all agents emit the centre code, which is also the
  correct value at init). Under CommNet these metrics are hard-zeroed on purpose — there is
  no codebook — so do not read them as a collapse.

---

## Also in flight on the old machine (Overcooked V3 env work)

Independent of the comm runs, but the same repo:
- Prep stations (cutting board / grill / blender) with chop, grill-and-burn, blend chains.
- Dish washing: finite plate stack, delivery dirties a plate, sink washes it; plates are
  conserved. Toggle `enable_dish_washing`; off by default and byte-identical to the old
  observation schema.
- Orders now rotate through every dish a layout can make (`order_queue_mode=alternating`).
- `prep_kitchen_handoff` restructured: food right, machines left, counter handoff.

**Open issue there:** `prep_kitchen_handoff` trained 30M steps for **0 deliveries**. The
map is provably solvable (scripted rollout completes a delivery), but the food-supplying
agent gets no shaped reward for passing items across the counter, so the chain is never
discovered. Naive handoff drop/pickup shaping would be farmable on a static counter —
drop/pick-up is a free reversible cycle. See
`.claude/.../memory/overcooked-v3-shaping-farm-traps.md` for the same class of bug that
had already silently destroyed a grill run.
