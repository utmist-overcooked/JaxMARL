#!/bin/bash
#SBATCH -J recipe_gifs
#SBATCH -o /home/tangzach/JaxMARL/.worktrees/codex/overcooked-fsq-distill/outputs/logs/%x_%j.out
#SBATCH -e /home/tangzach/JaxMARL/.worktrees/codex/overcooked-fsq-distill/outputs/logs/%x_%j.err
#SBATCH -p base_suma_rtx3090
#SBATCH -q base_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=8:00:00
#SBATCH --exclude=node19,node13,node16,node08,node10,node21,node14,node04,node05

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/tangzach/JaxMARL/.worktrees/codex/overcooked-fsq-distill}"
CTC_PROJECT_DIR="${CTC_PROJECT_DIR:-/home/tangzach/JaxMARL/.worktrees/codex/ctc-mappo-fullobs}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/scratch/tangzach/jaxmarl/recipe_checkpoint_rollouts_backfill}"

cd "$PROJECT_DIR"

# shellcheck disable=SC1091
source "${PROJECT_DIR}/venv/bin/activate"

export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda,cpu}"
export PYTHONPATH="${CTC_PROJECT_DIR}:${PROJECT_DIR}/baselines/MAPPO:${PROJECT_DIR}${PYTHONPATH:+:$PYTHONPATH}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-${OUTPUT_ROOT}/wandb-cache}"
export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-${OUTPUT_ROOT}/wandb-config}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${OUTPUT_ROOT}/matplotlib}"

mkdir -p \
    "${PROJECT_DIR}/outputs/logs" \
    "$OUTPUT_ROOT" \
    "$WANDB_CACHE_DIR" \
    "$WANDB_CONFIG_DIR" \
    "$MPLCONFIGDIR"

echo "START: $(date)"
echo "HOST: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "project_dir=$PROJECT_DIR"
echo "ctc_project_dir=$CTC_PROJECT_DIR"

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv,noheader || true
fi

env -u LD_LIBRARY_PATH "${PROJECT_DIR}/venv/bin/python" -u \
    "${PROJECT_DIR}/baselines/MAPPO/backfill_recipe_checkpoint_rollouts.py" \
    --entity zacharytang24- \
    --project overcookedv3-zac-distillation \
    --namespace checkpoint_rollouts_by_recipe \
    --max-steps 150 \
    --fps 8 \
    --tile-size 32 \
    --mode online

echo "END: $(date)"
