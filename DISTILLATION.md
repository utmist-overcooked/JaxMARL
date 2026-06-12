# FSQ Communication Distillation Handoff

## Context

This worktree implements a partially observable Overcooked V3 MAPPO student
with a discrete FSQ communication bottleneck, trained with policy distillation
from a privileged full-observation MAPPO teacher.

The intended experimental setup is:

- Teacher: privileged full-observation MAPPO actor/critic, no communication.
- Student actor: partial observation only, with current-step quantized communication.
- Student critic: centralized over concatenated partial observations only.
- Distillation: actor policy only, via `KL(teacher || student)`.
- Rollout actions: always sampled from the student to keep PPO on-policy.
- Environment scope: two-agent Overcooked V3 only.

Communication protocol:

```text
partial_obs_t -> CNN -> GRU -> FSQ message_t -> exchange messages -> action distribution
```

The student action head receives:

```text
own GRU output + partner quantized message
```

It does not receive its own message as an extra input. The intent is to keep
the channel honest: own recurrent state already contains local information,
while the explicit message input represents the partner channel.

## Design Decisions

FSQ settings:

- `FSQ_LEVELS = [5, 5, 5]`
- codebook size is `125`
- no message regularization yet
- code usage is logged during training and can be analyzed post-hoc with the
  FSQ viewer script

Distillation settings:

- `DISTILL_COEF = 1.0`
- `DISTILL_TEMPERATURE = 1.0`
- `DISTILL_DECAY_FRACTION = 0.30`
- distillation weight follows cosine decay from `1.0` to `0.0` over the first
  30% of total training timesteps

Loss:

```text
actor_loss =
  PPO clipped policy loss
  - ENT_COEF * entropy
  + distill_weight(t) * DISTILL_TEMPERATURE^2 * KL(teacher || student)
```

Important PPO correctness point:

Teacher actions are not used for rollout. The student always samples rollout
actions, because the baseline PPO update assumes `traj_batch.log_prob` comes
from the same policy being updated.

## Implemented Files

### `baselines/MAPPO/fsq.py`

Import-safe FSQ helper adapted from the Google Research FSQ implementation.
The local version is a frozen dataclass with input validation and a
`quantize_and_index()` convenience method.

### `baselines/MAPPO/mappo_rnn_overcooked_v3_fsq_distill.py`

New trainer based on the Overcooked V3 MAPPO RNN trainer.

Major changes:

- Adds `CommActorRNN` with FSQ communication.
- Adds `TeacherActorRNN` matching the full-observation actor parameter structure.
- Loads teacher actor params from `TEACHER_ACTOR_PATH`.
- Accepts teacher checkpoints saved either as a bare params tree or as a
  Flax variables dict containing a top-level `params` collection.
- Builds full-observation teacher observations from the same underlying env
  state using a separate `OvercookedV3(..., agent_view_size=None)` object.
- Maintains teacher recurrent hidden state in rollout state.
- Stores `teacher_logits`, `distill_weight`, `comm_code`, and `comm_index` in
  each transition.
- Adds KL distillation loss in the actor update.
- Keeps student critic unchanged: concat partial observations only.
- Uses env-level shuffling so both agents from each env remain paired during
  PPO reruns.
- Logs FSQ usage metrics.
- Saves periodic safetensors checkpoints and can render checkpoint GIFs after
  the training JIT returns.

Logged communication/distillation metrics include:

- `teacher_kl`
- `distill_loss`
- `distill_weight`
- `fsq_unique_codes`
- `fsq_code_entropy`
- `fsq_usage_max_frac`
- per-code count metrics under `fsq/code_*_count`
- per-dimension level counts under `fsq/dim_*_level_*_count`

The code currently supports only two agents and explicitly raises if
`env.num_agents != 2`.

### `baselines/MAPPO/config/mappo_rnn_overcooked_v3_fsq_distill.yaml`

Hydra config for FSQ distillation. Key fields:

```yaml
"FSQ_LEVELS": [5, 5, 5]
"DISTILL_COEF": 1.0
"DISTILL_TEMPERATURE": 1.0
"DISTILL_DECAY_FRACTION": 0.30
"TEACHER_ACTOR_PATH": "/path/to/full_obs_actor.safetensors"
"DISABLE_CHECKPOINTS": False
"CHECKPOINT_GIF": True
"CHECKPOINT_GIF_COUNT": 10
```

The current local default points at an `asymm_advantages_recipes_right` teacher
checkpoint. Override `TEACHER_ACTOR_PATH` for other layouts or machines.
`CHECKPOINT_GIF_COUNT` controls how many evenly spaced checkpoint actors are
saved and rendered into GIFs per run. GIF rendering is intentionally performed
after the training JIT returns, not inside the checkpoint callback, to avoid
nesting extra JAX render work inside `jax.debug.callback`.

### `slurm_scripts/mappo_rnn_overcooked_v3_fsq_distill.sbatch`

Yonsei SLURM runner for the FSQ distillation trainer. It uses the 3090 base
partition and mirrors the bad-node exclusions from the local cluster notes.

Important runtime details:

- activates this worktree's `venv`
- unsets `LD_LIBRARY_PATH` around the Python process
- defaults to `asymm_advantages_recipes_right`
- uses a concrete teacher checkpoint path unless overridden

Example:

```bash
sbatch slurm_scripts/mappo_rnn_overcooked_v3_fsq_distill.sbatch
```

Common overrides:

```bash
LAYOUT=cramped_room \
TEACHER_ACTOR_PATH=/path/to/full_obs_cramped_room_actor.safetensors \
sbatch slurm_scripts/mappo_rnn_overcooked_v3_fsq_distill.sbatch
```

### `slurm_scripts/mappo_rnn_overcooked_v3_fsq_distill_ctc_10run.sbatch`

Yonsei SLURM array runner for a 10-run `coordinated_temporal_conveyor_harder`
FSQ distillation sweep against the CTC full-observation teacher checkpoint.
It sets the CTC worktree first on `PYTHONPATH`, applies the conveyor/handoff
environment overrides, logs to W&B under `zacharytang24-/overcookedv3-mappo-full-obs`,
and varies distillation strength, distillation duration, teacher temperature,
FSQ channel size, partial-observation radius, and one no-distillation control.

Example:

```bash
sbatch slurm_scripts/mappo_rnn_overcooked_v3_fsq_distill_ctc_10run.sbatch
```

### `slurm_scripts/mappo_rnn_overcooked_v3_fsq_distill_local.sh`

Interactive/local runner with `smoke` and `full` modes:

```bash
./slurm_scripts/mappo_rnn_overcooked_v3_fsq_distill_local.sh smoke
./slurm_scripts/mappo_rnn_overcooked_v3_fsq_distill_local.sh full asymm_advantages_recipes_right
```

### `play_scripts/analyze_fsq_overcooked_v3.py`

Offline analyzer for trained FSQ student checkpoints. It collects rollout
states by FSQ code and writes an HTML viewer with example frames and code
usage counts.

Example:

```bash
python play_scripts/analyze_fsq_overcooked_v3.py \
  --config baselines/MAPPO/config/mappo_rnn_overcooked_v3_fsq_distill.yaml \
  --actor-path /path/to/student_actor.safetensors \
  --out-dir outputs/fsq_code_viewer
```

## Verification Run In This Worktree

Syntax checks:

```bash
source venv/bin/activate
python -m py_compile \
  baselines/MAPPO/fsq.py \
  baselines/MAPPO/mappo_rnn_overcooked_v3_fsq_distill.py \
  play_scripts/analyze_fsq_overcooked_v3.py

bash -n \
  slurm_scripts/mappo_rnn_overcooked_v3_fsq_distill.sbatch \
  slurm_scripts/mappo_rnn_overcooked_v3_fsq_distill_local.sh
```

Import check:

```bash
source venv/bin/activate
env -u LD_LIBRARY_PATH JAX_PLATFORMS=cpu python - <<'PY'
from baselines.MAPPO.mappo_rnn_overcooked_v3_fsq_distill import CommActorRNN, load_actor_params
from baselines.MAPPO.fsq import FSQ
fsq = FSQ(levels=(5, 5, 5))
print(CommActorRNN.__name__)
print(load_actor_params.__name__)
print(fsq.codebook_size)
PY
```

Observed output:

```text
CommActorRNN
load_actor_params
125
```

## Recommended Next Verification

1. Real teacher checkpoint CPU smoke:

```bash
cd /home/tangzach/JaxMARL/.worktrees/codex/overcooked-fsq-distill
source venv/bin/activate

env -u LD_LIBRARY_PATH JAX_PLATFORMS=cpu python baselines/MAPPO/mappo_rnn_overcooked_v3_fsq_distill.py \
  ENV_KWARGS.layout=cramped_room \
  TOTAL_TIMESTEPS=2 \
  NUM_ENVS=2 \
  NUM_STEPS=1 \
  NUM_MINIBATCHES=1 \
  UPDATE_EPOCHS=1 \
  WANDB_MODE=disabled \
  USE_RICH_MONITOR=False \
  DISABLE_CHECKPOINTS=True \
  TEACHER_ACTOR_PATH=/path/to/full_obs_cramped_room_actor.safetensors
```

2. Slurm smoke:

```bash
LAYOUT=cramped_room \
TEACHER_ACTOR_PATH=/path/to/full_obs_cramped_room_actor.safetensors \
MODE=smoke \
sbatch slurm_scripts/mappo_rnn_overcooked_v3_fsq_distill.sbatch
```

3. Short offline run before full training:

- run around `1e6` timesteps
- inspect `teacher_kl`, `distill_weight`, `fsq_unique_codes`,
  `fsq_usage_max_frac`, `entropy`, `approx_kl`, `clip_frac`, and returns

4. Full run:

- only after real-teacher CPU smoke and Slurm smoke pass

## Known Risks / Follow-Up Work

1. Teacher actor path is manual.

The SLURM script assumes one layout per run and uses one explicit teacher actor
path. If later running multiple layouts in one job, add layout-aware teacher
path resolution.

2. No message regularization.

Code collapse is allowed for now. Watch `fsq_unique_codes` and
`fsq_usage_max_frac`. If collapse is undesirable after initial runs, add usage
regularization later.

3. Agent order assumption.

The communication exchange currently relies on agent-major ordering:

```text
[agent_0 env_0..N, agent_1 env_0..N]
```

This matches the current batching logic in the MAPPO Overcooked V3 trainer. If
batching changes, re-check message pairing.

4. Two-agent only.

The implementation explicitly targets two-agent Overcooked V3. Generalizing to
more agents requires replacing `jnp.flip(..., axis=1)` with a real message
aggregation scheme.

## Expected Files To Commit

```text
DISTILLATION.md
baselines/MAPPO/config/mappo_rnn_overcooked_v3_fsq_distill.yaml
baselines/MAPPO/fsq.py
baselines/MAPPO/mappo_rnn_overcooked_v3_fsq_distill.py
play_scripts/analyze_fsq_overcooked_v3.py
slurm_scripts/mappo_rnn_overcooked_v3_fsq_distill.sbatch
slurm_scripts/mappo_rnn_overcooked_v3_fsq_distill_ctc_10run.sbatch
slurm_scripts/mappo_rnn_overcooked_v3_fsq_distill_local.sh
```
