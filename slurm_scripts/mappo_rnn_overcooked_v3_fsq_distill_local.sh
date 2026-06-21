#!/bin/bash
set -euo pipefail

# Local/interactive runner for MAPPO RNN FSQ distillation.
#
# Examples:
#   ./slurm_scripts/mappo_rnn_overcooked_v3_fsq_distill_local.sh smoke
#   ./slurm_scripts/mappo_rnn_overcooked_v3_fsq_distill_local.sh full

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/outputs}"

if [[ -f "${PROJECT_DIR}/venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "${PROJECT_DIR}/venv/bin/activate"
elif [[ -z "${PYTHON:-}" ]]; then
    echo "Missing ${PROJECT_DIR}/venv. Set PYTHON=/path/to/python or copy/recreate venv."
    exit 1
fi

PYTHON_BIN="${PYTHON:-python}"

export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda,cpu}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-${OUTPUT_ROOT}/wandb-cache}"
export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-${OUTPUT_ROOT}/wandb-config}"

MODE="${1:-smoke}"
LAYOUT="${2:-asymm_advantages_recipes_right}"

if [[ "$MODE" == "full" ]]; then
    NUM_ENVS="${NUM_ENVS:-2048}"
    NUM_STEPS="${NUM_STEPS:-256}"
    NUM_MINIBATCHES="${NUM_MINIBATCHES:-64}"
    UPDATE_EPOCHS="${UPDATE_EPOCHS:-4}"
    TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-100000000}"
    REW_SHAPING_HORIZON="${REW_SHAPING_HORIZON:-50000000}"
    WANDB_MODE="${WANDB_MODE:-online}"
    WANDB_PROJECT="${WANDB_PROJECT:-overcookedv3-mappo-fsq-distill}"
    DISABLE_CHECKPOINTS="${DISABLE_CHECKPOINTS:-False}"
    CHECKPOINT_GIF="${CHECKPOINT_GIF:-True}"
elif [[ "$MODE" == "smoke" ]]; then
    NUM_ENVS="${NUM_ENVS:-1}"
    NUM_STEPS="${NUM_STEPS:-2}"
    NUM_MINIBATCHES="${NUM_MINIBATCHES:-1}"
    UPDATE_EPOCHS="${UPDATE_EPOCHS:-1}"
    TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-$((NUM_ENVS * NUM_STEPS))}"
    REW_SHAPING_HORIZON="${REW_SHAPING_HORIZON:-$((TOTAL_TIMESTEPS / 2))}"
    WANDB_MODE="${WANDB_MODE:-disabled}"
    WANDB_PROJECT="${WANDB_PROJECT:-overcookedv3-mappo-fsq-distill-smoke}"
    DISABLE_CHECKPOINTS="${DISABLE_CHECKPOINTS:-True}"
    CHECKPOINT_GIF="${CHECKPOINT_GIF:-True}"
else
    echo "Unknown mode: $MODE. Use 'smoke' or 'full'."
    exit 1
fi

TEACHER_ACTOR_PATH="${TEACHER_ACTOR_PATH:-/home/tangzach/JaxMARL/outputs/overcookedv3-mappo-full-obs/models/mappo_v3_full_obs_full_asymm_advantages_recipes_right_20260527/150_actor.safetensors}"
WANDB_DIR="${WANDB_DIR:-${OUTPUT_ROOT}/${WANDB_PROJECT}}"
CHECKPOINT_GIF_OUTPUT_DIR="${CHECKPOINT_GIF_OUTPUT_DIR:-${WANDB_DIR}/checkpoint_rollouts}"
CHECKPOINT_GIF_MAX_STEPS="${CHECKPOINT_GIF_MAX_STEPS:-150}"
CHECKPOINT_GIF_EPSILON="${CHECKPOINT_GIF_EPSILON:-0}"
CHECKPOINT_GIF_SEED="${CHECKPOINT_GIF_SEED:-0}"
CHECKPOINT_GIF_FPS="${CHECKPOINT_GIF_FPS:-8}"
CHECKPOINT_GIF_TILE_SIZE="${CHECKPOINT_GIF_TILE_SIZE:-32}"
RUN_DIR="${WANDB_DIR}/runs/${MODE}_${LAYOUT}"

mkdir -p "${OUTPUT_ROOT}/logs" "${WANDB_DIR}/models" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR" "$CHECKPOINT_GIF_OUTPUT_DIR"

echo "mode=$MODE"
echo "layout=$LAYOUT"
echo "teacher_actor_path=$TEACHER_ACTOR_PATH"
echo "project_dir=$PROJECT_DIR"
echo "python=$PYTHON_BIN"
echo "checkpoint_gif=$CHECKPOINT_GIF"
echo "checkpoint_gif_output_dir=$CHECKPOINT_GIF_OUTPUT_DIR"
echo "checkpoint_gif_max_steps=$CHECKPOINT_GIF_MAX_STEPS"
echo "checkpoint_gif_epsilon=$CHECKPOINT_GIF_EPSILON"

env -u LD_LIBRARY_PATH "$PYTHON_BIN" "${PROJECT_DIR}/baselines/MAPPO/mappo_rnn_overcooked_v3_fsq_distill.py" \
    hydra.run.dir="$RUN_DIR" \
    hydra.output_subdir=null \
    ENV_KWARGS.layout="$LAYOUT" \
    TEACHER_ACTOR_PATH="$TEACHER_ACTOR_PATH" \
    TOTAL_TIMESTEPS="$TOTAL_TIMESTEPS" \
    REW_SHAPING_HORIZON="$REW_SHAPING_HORIZON" \
    NUM_ENVS="$NUM_ENVS" \
    NUM_STEPS="$NUM_STEPS" \
    NUM_MINIBATCHES="$NUM_MINIBATCHES" \
    UPDATE_EPOCHS="$UPDATE_EPOCHS" \
    WANDB_MODE="$WANDB_MODE" \
    ENTITY=null \
    PROJECT="$WANDB_PROJECT" \
    WANDB_DIR="$WANDB_DIR" \
    "++WANDB_RUN_NAME=mappo_v3_fsq_distill_${MODE}_${LAYOUT}" \
    USE_RICH_MONITOR=False \
    "++DISABLE_CHECKPOINTS=$DISABLE_CHECKPOINTS" \
    CHECKPOINT_GIF="$CHECKPOINT_GIF" \
    CHECKPOINT_GIF_OUTPUT_DIR="$CHECKPOINT_GIF_OUTPUT_DIR" \
    CHECKPOINT_GIF_MAX_STEPS="$CHECKPOINT_GIF_MAX_STEPS" \
    CHECKPOINT_GIF_EPSILON="$CHECKPOINT_GIF_EPSILON" \
    CHECKPOINT_GIF_SEED="$CHECKPOINT_GIF_SEED" \
    CHECKPOINT_GIF_FPS="$CHECKPOINT_GIF_FPS" \
    CHECKPOINT_GIF_TILE_SIZE="$CHECKPOINT_GIF_TILE_SIZE"
