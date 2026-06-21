#!/bin/bash
#SBATCH -J sparse_combo_gif
#SBATCH -o /home/tangzach/JaxMARL/.worktrees/codex/overcooked-fsq-distill/outputs/logs/%x_%j.out
#SBATCH -e /home/tangzach/JaxMARL/.worktrees/codex/overcooked-fsq-distill/outputs/logs/%x_%j.err
#SBATCH -p base_suma_rtx3090
#SBATCH -q base_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=1:00:00
#SBATCH --exclude=node19,node13,node16,node08,node10,node21,node14,node04,node05

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/tangzach/JaxMARL/.worktrees/codex/overcooked-fsq-distill}"
CTC_PROJECT_DIR="${CTC_PROJECT_DIR:-/home/tangzach/JaxMARL/.worktrees/codex/ctc-mappo-fullobs}"
RUN_DIR="${RUN_DIR:-/scratch/tangzach/jaxmarl/ctc_sparse_11run/sparse_fsq_ctc_harder_0_baseline_seed0_5m_1688651}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${RUN_DIR}/models/sparse_fsq_ctc_harder_0_baseline_seed0_5m_20260617}"
ACTOR_PATH="${ACTOR_PATH:-${CHECKPOINT_DIR}/76_actor.safetensors}"
CONFIG_PATH="${CONFIG_PATH:-${RUN_DIR}/models/mappo_rnn_overcooked_v3_fsq_distill_coordinated_temporal_conveyor_harder_seed0_config.yaml}"
RUN_NAME="${RUN_NAME:-combined_gif_test}"
WANDB_RUN_ID="${WANDB_RUN_ID:-0m5ynavz}"
WANDB_MODE="${WANDB_MODE:-online}"
NAMESPACE="${NAMESPACE:-checkpoint_rollouts_by_recipe}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/scratch/tangzach/jaxmarl/combined_gif_test}"

# shellcheck disable=SC1091
source "${PROJECT_DIR}/venv/bin/activate"

export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda,cpu}"
export PYTHONPATH="${PROJECT_DIR}/baselines/MAPPO:${CTC_PROJECT_DIR}:${PROJECT_DIR}${PYTHONPATH:+:$PYTHONPATH}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-${OUTPUT_ROOT}/wandb-cache}"
export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-${OUTPUT_ROOT}/wandb-config}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${OUTPUT_ROOT}/matplotlib}"
export RUN_DIR
export ACTOR_PATH
export CONFIG_PATH
export RUN_NAME
export WANDB_RUN_ID
export WANDB_MODE
export NAMESPACE

mkdir -p "${PROJECT_DIR}/outputs/logs" "$OUTPUT_ROOT" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR" "$MPLCONFIGDIR"
cd "$OUTPUT_ROOT"

echo "START: $(date)"
echo "HOST: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "PROJECT_DIR=$PROJECT_DIR"
echo "CTC_PROJECT_DIR=$CTC_PROJECT_DIR"
echo "RUN_DIR=$RUN_DIR"
echo "ACTOR_PATH=$ACTOR_PATH"
echo "CONFIG_PATH=$CONFIG_PATH"
echo "WANDB_RUN_ID=$WANDB_RUN_ID"
echo "WANDB_MODE=$WANDB_MODE"
echo "NAMESPACE=$NAMESPACE"

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv,noheader || true
fi

env -u LD_LIBRARY_PATH "${PROJECT_DIR}/venv/bin/python" -u - <<'PY'
import os
from pathlib import Path

import wandb
from omegaconf import OmegaConf

from jaxmarl.wrappers.baselines import load_params
from mappo_rnn_overcooked_v3_fsq_distill import (
    _checkpoint_gif_namespace,
    render_and_log_checkpoint_gif,
)

run_dir = Path(os.environ["RUN_DIR"])
actor_path = Path(os.environ["ACTOR_PATH"])
config_path = Path(os.environ["CONFIG_PATH"])
run_name = os.environ["RUN_NAME"]
namespace = os.environ["NAMESPACE"]
wandb_mode = os.environ["WANDB_MODE"]

config = OmegaConf.to_container(OmegaConf.load(config_path), resolve=False)
config["CHECKPOINT_GIF"] = True
config["CHECKPOINT_GIF_NAMESPACE"] = namespace
config["CHECKPOINT_GIF_MEDIA_KEY"] = f"{namespace}/combined_gif_test"
config["CHECKPOINT_GIF_OUTPUT_DIR"] = str(run_dir / namespace)
config["CHECKPOINT_GIF_MAX_STEPS"] = int(config.get("CHECKPOINT_GIF_MAX_STEPS", 150))
config["CHECKPOINT_GIF_EPSILON"] = 0.0
config["CHECKPOINT_GIF_SEED"] = 0
config["CHECKPOINT_GIF_FPS"] = int(config.get("CHECKPOINT_GIF_FPS", 8))
config["CHECKPOINT_GIF_TILE_SIZE"] = int(config.get("CHECKPOINT_GIF_TILE_SIZE", 32))
config["CHECKPOINT_FSQ_VIEWER"] = False
config["WANDB_MODE"] = wandb_mode
config["WANDB_DIR"] = str(run_dir)

if wandb_mode != "disabled":
    run = wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        id=os.environ["WANDB_RUN_ID"],
        name=config["WANDB_RUN_NAME"],
        resume="allow",
        mode=wandb_mode,
        dir=str(run_dir),
    )
    resolved_namespace = _checkpoint_gif_namespace(config)
    wandb.define_metric(f"{resolved_namespace}/rollout_index")
    wandb.define_metric(
        f"{resolved_namespace}/*",
        step_metric=f"{resolved_namespace}/rollout_index",
    )
else:
    run = None

try:
    print(f"Rendering combined checkpoint GIF for actor={actor_path}")
    render_and_log_checkpoint_gif(
        actor_params=load_params(str(actor_path)),
        config=config,
        update=76,
        run_name=run_name,
        checkpoint_interval=76,
        rollout_index=0,
        actor_path=actor_path,
        config_path=config_path,
    )
finally:
    if run is not None:
        wandb.finish()
PY

echo "Generated GIF files:"
find "${RUN_DIR}/${NAMESPACE}/${RUN_NAME}" -maxdepth 1 -type f -name '*.gif' -printf '%s %p\n' | sort -n
echo "END: $(date)"
