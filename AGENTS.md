# Compute Cluster
See CLUSTER.md on notes on how to use this compute cluster. 

NOTE that the cluster / login node that you are running on RIGHT NOW has no GPUs on it. You will have to look into CLUSTER.md to see how to srun into a GPU node. 

Preferred if you go into `srun3090_base` if you ever want to debug or run ANY jobs.

When writing or running future `sbatch` scripts, mirror the bad-node exclusions from `~/.bashrc`:

- 3090 base (`base_suma_rtx3090` / `base_qos`): `#SBATCH --exclude=node19,node13,node16,node08,node10,node21,node14,node04,node05`
- 3090 big (`big_suma_rtx3090` / `big_qos`): `#SBATCH --exclude=node19,node13,node16,node08,node10,node21,node14,node04,node18`
- 4090 (`suma_rtx4090`): `#SBATCH --exclude=node19,node13,node16,node08,node10,node21,node34,node31`
- A5000 (`asus_a5000`): `#SBATCH --exclude=node19,node13,node16,node08,node10,node21,node34`

# Debugging / running 
Always run `source venv/bin/activate` before running any Python scripts.

## New worktrees

When asked to make a new worktree, fork it from `main` unless the user specifies a different base. Create it under `.worktrees/<branch_name>` and do all work from that worktree. Always copy `AGENTS.md` and `CLUSTER.md` into the new worktree after creating it so the local instructions are available there too.

If Python scripts need to be run from a new worktree and `venv/` is absent, copy the base repo's `venv/` into the new worktree first, then activate it with `source venv/bin/activate`. If the copied environment is broken because of path-specific virtualenv metadata, recreate it locally in the worktree.

## FSQ distillation work

When the user asks about FSQ, finite scalar quantization, FSQ communication, FSQ distillation, FSQ diagnostics, or no-FSQ ablations, work from:

```bash
cd /home/tangzach/JaxMARL/.worktrees/codex/overcooked-fsq-distill
```

That worktree is the important FSQ branch:

```text
codex/overcooked-fsq-distill
```

Use it by default for FSQ-related edits, Slurm scripts, diagnostics viewers, W&B run management, and checkpoint/GIF work unless the user explicitly requests another branch.

# JaxMARL GPU setup

Working stack:

```text
Python 3.11
jax==0.4.38
jaxlib==0.4.38
jax-cuda12-plugin==0.4.38
jax-cuda12-pjrt==0.4.38
flax==0.10.2
```

Important fixes:

- Install GPU JAX with `venv/bin/python -m pip install --upgrade 'jax[cuda12]==0.4.38'`.
- Keep `flax==0.10.2`; newer Flax broke with `jax.api_util.debug_info` missing from JAX 0.4.38.
- Unset `LD_LIBRARY_PATH` before running JAX. The CUDA module's `/opt/ohpc/pub/apps/cuda/12.8/lib64` path caused JAX CUDA segfaults.

Quick GPU check from an allocated GPU node:

```bash
cd /home/tangzach/JaxMARL
env -u LD_LIBRARY_PATH JAX_PLATFORMS=cuda,cpu venv/bin/python - <<'PY'
import jax
print(jax.default_backend())
print(jax.devices())
PY
```

Smoke test:

```bash
./slurm_scripts/mappo_rnn_overcooked_v3_full_obs_100m_local.sh smoke
```

Expected result: `All layouts complete.`

Node note: `node05` had a driver/UVM failure (`cuInit(0) == 999`, `/dev/nvidia-uvm` returned `EIO`). Use another node; `node03` worked.
