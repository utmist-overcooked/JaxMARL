# FSQ Communication Distillation Handoff

## Context

This branch implements a new partially observable Overcooked V3 MAPPO student with a discrete FSQ communication bottleneck, trained with policy distillation from a full-observation MAPPO teacher.

The intended experimental setup is:

- Teacher: privileged full-observation MAPPO actor/critic, no communication.
- Student actor: partial observation only, with current-step quantized communication.
- Student critic: centralized over concatenated partial observations only.
- Distillation: actor policy only, via `KL(teacher || student)`.
- Rollout actions: always sampled from the student to keep PPO on-policy.
- Environment scope: two-agent Overcooked V3 only.

This worktree was forked from local `feature/maddpg` at `797f24a`, on branch `zac/overcooked-fsq-distill`.

## Design Decisions

Communication protocol:

```text
partial_obs_t -> CNN -> GRU -> FSQ message_t -> exchange messages -> action distribution
```

The student action head receives:

```text
own GRU output + partner quantized message
```

It does not receive its own message as an extra input. The intent is to keep the channel honest: own recurrent state already contains local information, while the explicit message input represents the partner channel.

FSQ settings:

- `FSQ_LEVELS = [5, 5, 5]`
- codebook size is `125`
- no message regularization yet
- raw code usage is logged first; deeper situation/code analysis should be a post-hoc rollout/debug script

Distillation settings:

- `DISTILL_COEF = 1.0`
- `DISTILL_TEMPERATURE = 1.0`
- `DISTILL_DECAY_FRACTION = 0.30`
- distillation weight follows cosine decay from `1.0` to `0.0` over the first 30% of total training timesteps

Loss:

```text
actor_loss =
  PPO clipped policy loss
  - ENT_COEF * entropy
  + distill_weight(t) * DISTILL_TEMPERATURE^2 * KL(teacher || student)
```

Important PPO correctness point:

Teacher actions are not used for rollout. The student always samples rollout actions, because the baseline PPO update assumes `traj_batch.log_prob` comes from the same policy being updated.

## Implemented Files

### `FSQ.py`

Copied from the main checkout and made import-safe. The example code is now guarded by:

```python
if __name__ == "__main__":
```

This prevents training imports from printing example output or running assertions.

### `baselines/MAPPO/mappo_rnn_overcooked_v3_fsq_distill.py`

New trainer based on `mappo_rnn_overcooked_v3.py`.

Major changes:

- Adds `ActorRNN` with FSQ communication.
- Adds `TeacherActorRNN` matching the full-observation actor parameter structure.
- Loads teacher actor params from `TEACHER_ACTOR_PATH`.
- Builds full-observation teacher observations from the same underlying env state using a separate `OvercookedV3(..., agent_view_size=None)` object.
- Maintains teacher recurrent hidden state in rollout state.
- Stores `teacher_logits` in each transition.
- Adds KL distillation loss in the actor update.
- Keeps student critic unchanged: concat partial observations only.
- Replaces flattened-agent minibatch shuffling with env-level shuffling so both agents from each env remain paired during PPO reruns.
- Logs FSQ usage metrics.

Logged communication/distillation metrics include:

- `teacher_student_kl`
- `distill_loss`
- `distill_weight`
- `comm_code_unique`
- `comm_code_top1_frac`
- `comm_code_hist`
- `comm_dim0_hist`
- `comm_dim1_hist`
- `comm_dim2_hist`

The code currently supports only two agents and explicitly raises if `env.num_agents != 2`.

### `baselines/MAPPO/config/mappo_rnn_overcooked_v3_fsq_distill.yaml`

New Hydra config. Key fields:

```yaml
"FSQ_LEVELS": [5, 5, 5]
"DISTILL_COEF": 1.0
"DISTILL_TEMPERATURE": 1.0
"DISTILL_DECAY_FRACTION": 0.30
"TEACHER_ACTOR_PATH": ""
"DISABLE_CHECKPOINTS": False
```

`TEACHER_ACTOR_PATH` must be supplied for real runs.

### `slurm_scripts/mappo_rnn_overcooked_v3_fsq_distill.sh`

Single-layout Slurm script. Usage:

```bash
sbatch slurm_scripts/mappo_rnn_overcooked_v3_fsq_distill.sh smoke cramped_room /scratch/.../teacher_actor.safetensors
sbatch slurm_scripts/mappo_rnn_overcooked_v3_fsq_distill.sh full  cramped_room /scratch/.../teacher_actor.safetensors
```

The script fails fast if the teacher actor path is missing or does not exist.

Before submitting, create scratch directories on the login node:

```bash
mkdir -p $SCRATCH/jaxmarl/logs \
  $SCRATCH/jaxmarl/overcookedv3-mappo-fsq-distill-smoke/models \
  $SCRATCH/jaxmarl/overcookedv3-mappo-fsq-distill/models \
  $SCRATCH/jaxmarl/wandb-cache \
  $SCRATCH/jaxmarl/wandb-config
```

Cluster note: repo docs say to prefer `def-cglee` unless the run is explicitly part of `rrg-cglee`. The script currently has `#SBATCH --account=rrg-cglee`; override at submission if needed:

```bash
sbatch --account=def-cglee slurm_scripts/mappo_rnn_overcooked_v3_fsq_distill.sh ...
```

## Verification Already Run

Syntax checks:

```bash
python -m py_compile FSQ.py baselines/MAPPO/mappo_rnn_overcooked_v3_fsq_distill.py
bash -n slurm_scripts/mappo_rnn_overcooked_v3_fsq_distill.sh
```

Import check:

- confirmed trainer imports successfully
- confirmed `REPO_ROOT` resolves to the worktree, not the main checkout

Teacher checkpoint compatibility check:

- initialized `TeacherActorRNN` from the new distill trainer
- initialized `ActorRNN` from `mappo_rnn_overcooked_v3_full_obs.py`
- compared flattened parameter key sets
- result: key sets are equal

One-update CPU JIT smoke:

- used tiny config: `NUM_ENVS=2`, `NUM_STEPS=1`, `NUM_MINIBATCHES=1`, `UPDATE_EPOCHS=1`
- used randomly initialized teacher params
- verified full train step compiles and runs
- output:

```text
smoke ok
comm_code_hist_shape (1, 125)
comm_dim_hists_shape (1, 3, 5)
distill_weight [1.]
```

This smoke exercises:

- FSQ message generation
- current-step message exchange
- teacher hidden state and teacher logits
- `KL(teacher || student)`
- env-level paired minibatching
- FSQ histogram metrics

## Recommended Next Verification

1. Real teacher checkpoint CPU smoke:

```bash
cd /project/rrg-cglee/zachtang/JaxMARL/.worktrees/zac/overcooked-fsq-distill
source venv/bin/activate

export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export XLA_FLAGS='--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1'

python baselines/MAPPO/mappo_rnn_overcooked_v3_fsq_distill.py \
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

This is the most important next check because it validates real saved teacher params, not randomly initialized dummy params.

2. Slurm smoke:

```bash
sbatch --account=def-cglee \
  slurm_scripts/mappo_rnn_overcooked_v3_fsq_distill.sh \
  smoke cramped_room /path/to/full_obs_cramped_room_actor.safetensors
```

3. Short offline run before full training:

- run around `1e6` timesteps
- confirm no immediate divergence
- inspect `teacher_student_kl`, `distill_weight`, `comm_code_unique`, `comm_code_top1_frac`, `entropy`, `approx_kl`, `clip_frac`, and returns

4. Full run:

- only after real-teacher CPU smoke and Slurm smoke pass

## Known Risks / Follow-Up Work

1. Teacher actor path is manual.

The Slurm script assumes one layout per run and takes one explicit teacher actor path. If later running multiple layouts in one job, add layout-aware teacher path resolution.

2. Raw communication logging only.

The trainer logs code usage but does not yet explain what each code means. Recommended follow-up is a separate rollout analysis script that writes rows like:

```text
step,agent,layout,code,dim0,dim1,dim2,inventory,partner_inventory,near_pot,near_goal,partner_visible
```

3. No message regularization.

Code collapse is allowed for now. Watch `comm_code_unique` and `comm_code_top1_frac`. If collapse is undesirable after initial runs, add usage regularization later.

4. Agent order assumption.

The communication exchange currently relies on agent-major ordering:

```text
[agent_0 env_0..N, agent_1 env_0..N]
```

This matches the current batching logic in the MAPPO Overcooked V3 trainer. If batching changes, re-check message pairing.

5. Two-agent only.

The implementation explicitly targets two-agent Overcooked V3. Generalizing to more agents requires replacing `jnp.flip(..., axis=1)` with a real message aggregation scheme.

## Current Git State Before Handoff Commit

Expected files to commit on this branch:

```text
DISTILLATION.md
FSQ.py
baselines/MAPPO/config/mappo_rnn_overcooked_v3_fsq_distill.yaml
baselines/MAPPO/mappo_rnn_overcooked_v3_fsq_distill.py
slurm_scripts/mappo_rnn_overcooked_v3_fsq_distill.sh
```
