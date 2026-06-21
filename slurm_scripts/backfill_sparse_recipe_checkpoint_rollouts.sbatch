#!/bin/bash
#SBATCH -J sparse_recipe_gifs
#SBATCH -o /home/tangzach/JaxMARL/.worktrees/codex/overcooked-fsq-distill/outputs/logs/%x_%A_%a.out
#SBATCH -e /home/tangzach/JaxMARL/.worktrees/codex/overcooked-fsq-distill/outputs/logs/%x_%A_%a.err
#SBATCH -p base_suma_rtx3090
#SBATCH -q base_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=8:00:00
#SBATCH --array=0-32
#SBATCH --exclude=node19,node13,node16,node08,node10,node21,node14,node04,node05

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/tangzach/JaxMARL/.worktrees/codex/overcooked-fsq-distill}"
CTC_PROJECT_DIR="${CTC_PROJECT_DIR:-/home/tangzach/JaxMARL/.worktrees/codex/ctc-mappo-fullobs}"
SPARSE_ROOT="${SPARSE_ROOT:-/scratch/tangzach/jaxmarl/ctc_sparse_11run}"
RUN_GLOB="${RUN_GLOB:-sparse_*_5m_*}"

# Keep disabled for local viewer regeneration only. Set BACKFILL_WANDB_MODE=online
# if you explicitly want the regenerated rollout media logged back to W&B.
BACKFILL_WANDB_MODE="${BACKFILL_WANDB_MODE:-disabled}"
WANDB_ENTITY="${WANDB_ENTITY:-zacharytang24-}"
WANDB_PROJECT="${WANDB_PROJECT:-overcookedv3-zac-distillation}"

CHECKPOINT_GIF_NAMESPACE="${CHECKPOINT_GIF_NAMESPACE:-checkpoint_rollouts_by_recipe}"
CHECKPOINT_GIF_MAX_STEPS="${CHECKPOINT_GIF_MAX_STEPS:-150}"
CHECKPOINT_GIF_EPSILON="${CHECKPOINT_GIF_EPSILON:-0.0}"
CHECKPOINT_GIF_SEED="${CHECKPOINT_GIF_SEED:-0}"
CHECKPOINT_GIF_FPS="${CHECKPOINT_GIF_FPS:-8}"
CHECKPOINT_GIF_TILE_SIZE="${CHECKPOINT_GIF_TILE_SIZE:-32}"
DRY_RUN="${DRY_RUN:-0}"

cd "$PROJECT_DIR"

# shellcheck disable=SC1091
source "${PROJECT_DIR}/venv/bin/activate"

if [[ -n "${PYTHON:-}" ]]; then
    PYTHON_BIN="$PYTHON"
elif [[ -x "${PROJECT_DIR}/venv/bin/python" ]]; then
    PYTHON_BIN="${PROJECT_DIR}/venv/bin/python"
else
    PYTHON_BIN="python"
fi

export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda,cpu}"
export PYTHONPATH="${CTC_PROJECT_DIR}:${PROJECT_DIR}/baselines/MAPPO:${PROJECT_DIR}${PYTHONPATH:+:$PYTHONPATH}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-${SPARSE_ROOT}/wandb-cache}"
export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-${SPARSE_ROOT}/wandb-config}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${SPARSE_ROOT}/matplotlib}"
export PROJECT_DIR
export SPARSE_ROOT
export RUN_GLOB
export BACKFILL_WANDB_MODE
export WANDB_ENTITY
export WANDB_PROJECT
export CHECKPOINT_GIF_NAMESPACE
export CHECKPOINT_GIF_MAX_STEPS
export CHECKPOINT_GIF_EPSILON
export CHECKPOINT_GIF_SEED
export CHECKPOINT_GIF_FPS
export CHECKPOINT_GIF_TILE_SIZE
export DRY_RUN

mkdir -p \
    "${PROJECT_DIR}/outputs/logs" \
    "$WANDB_CACHE_DIR" \
    "$WANDB_CONFIG_DIR" \
    "$MPLCONFIGDIR"

echo "START: $(date)"
echo "HOST: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "project_dir=$PROJECT_DIR"
echo "ctc_project_dir=$CTC_PROJECT_DIR"
echo "sparse_root=$SPARSE_ROOT"
echo "run_glob=$RUN_GLOB"
echo "array_task_id=${SLURM_ARRAY_TASK_ID:-0}"
echo "backfill_wandb_mode=$BACKFILL_WANDB_MODE"
echo "checkpoint_gif_namespace=$CHECKPOINT_GIF_NAMESPACE"
echo "checkpoint_gif_max_steps=$CHECKPOINT_GIF_MAX_STEPS"
echo "dry_run=$DRY_RUN"

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv,noheader || true
else
    echo "nvidia-smi not found"
fi

env -u LD_LIBRARY_PATH "$PYTHON_BIN" -u - <<'PY'
import os
import re
from pathlib import Path

import wandb
from omegaconf import OmegaConf

from jaxmarl.wrappers.baselines import load_params
from mappo_rnn_overcooked_v3_fsq_distill import render_and_log_checkpoint_gif


def update_from_actor(path: Path) -> int:
    return int(path.name.split("_", 1)[0])


def checkpoint_actor_paths(run_dir: Path) -> list[tuple[int, Path]]:
    by_update: dict[int, Path] = {}
    for path in run_dir.glob("models/*/*_actor.safetensors"):
        if not path.name.split("_", 1)[0].isdigit():
            continue
        update = update_from_actor(path)
        existing = by_update.get(update)
        if existing is None or path.stat().st_mtime > existing.stat().st_mtime:
            by_update[update] = path
    return sorted(by_update.items())


def run_name_from_dir(run_dir: Path) -> str:
    match = re.match(r"(.+)_\d+$", run_dir.name)
    return match.group(1) if match else run_dir.name


def wandb_run_id(run_dir: Path) -> str | None:
    runs = sorted((run_dir / "wandb").glob("run-*-*"))
    if not runs:
        return None
    return runs[-1].name.rsplit("-", 1)[-1]


def load_config(run_dir: Path) -> tuple[dict, Path]:
    config_paths = sorted(run_dir.glob("models/*_config.yaml"))
    if len(config_paths) != 1:
        raise RuntimeError(
            f"Expected one config under {run_dir}/models, got {config_paths}"
        )
    config = OmegaConf.to_container(OmegaConf.load(config_paths[0]), resolve=False)
    return config, config_paths[0]


project_dir = Path(os.environ["PROJECT_DIR"])
sparse_root = Path(os.environ["SPARSE_ROOT"])
run_glob = os.environ["RUN_GLOB"]
task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
mode = os.environ["BACKFILL_WANDB_MODE"]
namespace = os.environ["CHECKPOINT_GIF_NAMESPACE"]

run_dirs = sorted(
    path
    for path in sparse_root.glob(run_glob)
    if path.is_dir() and (path / "models").is_dir()
)
print(f"Discovered {len(run_dirs)} sparse run directories.")
for index, run_dir in enumerate(run_dirs):
    print(f"  [{index:02d}] {run_dir.name}")

if task_id >= len(run_dirs):
    print(f"Task {task_id} has no run directory; exiting cleanly.")
    raise SystemExit(0)

run_dir = run_dirs[task_id]
run_name = run_name_from_dir(run_dir)
config, config_path = load_config(run_dir)
checkpoint_paths = checkpoint_actor_paths(run_dir)
if not checkpoint_paths:
    raise RuntimeError(f"No checkpoint actor files found under {run_dir}/models")

config["CHECKPOINT_GIF"] = True
config["CHECKPOINT_GIF_NAMESPACE"] = namespace
config["CHECKPOINT_GIF_MEDIA_KEY"] = f"{namespace}/rollout"
config["CHECKPOINT_GIF_OUTPUT_DIR"] = str(run_dir / "checkpoint_rollouts")
config["CHECKPOINT_GIF_MAX_STEPS"] = int(os.environ["CHECKPOINT_GIF_MAX_STEPS"])
config["CHECKPOINT_GIF_EPSILON"] = float(os.environ["CHECKPOINT_GIF_EPSILON"])
config["CHECKPOINT_GIF_SEED"] = int(os.environ["CHECKPOINT_GIF_SEED"])
config["CHECKPOINT_GIF_FPS"] = int(os.environ["CHECKPOINT_GIF_FPS"])
config["CHECKPOINT_GIF_TILE_SIZE"] = int(os.environ["CHECKPOINT_GIF_TILE_SIZE"])
config["CHECKPOINT_FSQ_VIEWER"] = True
config["WANDB_MODE"] = mode
config["WANDB_DIR"] = str(run_dir)

print(f"Selected task {task_id}: {run_name}")
print(f"run_dir={run_dir}")
print(f"config_path={config_path}")
print(f"checkpoints={len(checkpoint_paths)}")
print(f"disable_fsq={bool(config.get('DISABLE_FSQ_COMM', False))}")
print(f"output_dir={config['CHECKPOINT_GIF_OUTPUT_DIR']}")

if os.environ["DRY_RUN"] == "1":
    print("DRY_RUN=1: stopping before W&B init, parameter loading, or rendering.")
    raise SystemExit(0)

run = None
if mode != "disabled":
    run_id = wandb_run_id(run_dir)
    if run_id is None:
        raise RuntimeError(f"Cannot find W&B run id under {run_dir}/wandb")
    run = wandb.init(
        entity=os.environ["WANDB_ENTITY"],
        project=os.environ["WANDB_PROJECT"],
        id=run_id,
        name=run_name,
        resume="allow",
        dir=str(run_dir),
        mode=mode,
    )

try:
    for rollout_index, (update, actor_path) in enumerate(checkpoint_paths):
        print(f"{run_name}: rendering update={update} actor={actor_path}")
        actor_params = load_params(actor_path)
        render_and_log_checkpoint_gif(
            actor_params=actor_params,
            config=config,
            update=update,
            run_name=run_name,
            checkpoint_interval=max(update, 1),
            rollout_index=rollout_index,
            actor_path=actor_path,
            config_path=config_path,
        )
finally:
    if run is not None:
        run.finish()

print(f"Finished {run_name}.")
PY

echo "END: $(date)"
