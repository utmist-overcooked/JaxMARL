#!/bin/bash
set -euo pipefail

# Local RTX GPU runner for MAPPO RNN full-observation Overcooked V3.
#
# Examples:
#   ./slurm_scripts/mappo_rnn_overcooked_v3_full_obs_100m_local.sh
#   ./slurm_scripts/mappo_rnn_overcooked_v3_full_obs_100m_local.sh smoke forced_coord
#   ./slurm_scripts/mappo_rnn_overcooked_v3_full_obs_100m_local.sh full
#
# Optional overrides:
#   PYTHON=/path/to/python NUM_ENVS=1024 NUM_STEPS=256 ./... full
#   SMOKE_NUM_ENVS=1 SMOKE_NUM_STEPS=8 WANDB_MODE=online ./... smoke cramped_room

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/outputs}"

if [[ -n "${PYTHON:-}" ]]; then
    PYTHON_BIN="$PYTHON"
elif [[ -x "${PROJECT_DIR}/venv/bin/python" ]]; then
    PYTHON_BIN="${PROJECT_DIR}/venv/bin/python"
elif [[ -x "${PROJECT_DIR}/.venv/bin/python" ]]; then
    PYTHON_BIN="${PROJECT_DIR}/.venv/bin/python"
else
    PYTHON_BIN="python3"
fi

export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda,cpu}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-${OUTPUT_ROOT}/wandb-cache}"
export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-${OUTPUT_ROOT}/wandb-config}"

MODE="${1:-smoke}"
if [[ $# -gt 0 ]]; then
    shift
fi

DEFAULT_NUM_ENVS=2048
DEFAULT_NUM_STEPS=256
DEFAULT_NUM_MINIBATCHES=64
DEFAULT_LR=0.002
DEFAULT_UPDATE_EPOCHS=4
DEFAULT_FC_DIM_SIZE=64
DEFAULT_GRU_HIDDEN_DIM=128

ALL_LAYOUTS=(
    cramped_room
    two_rooms
    follow_the_leader
    around_the_island_nerfed
    asymm_advantages_recipes_right
    coordinated_temporal_conveyor
    forced_coord
    maze_conveyor_hell
    middle_conveyor
    race_against_the_clock
)

if [[ "$MODE" == "full" ]]; then
    NUM_ENVS="${NUM_ENVS:-$DEFAULT_NUM_ENVS}"
    NUM_STEPS="${NUM_STEPS:-$DEFAULT_NUM_STEPS}"
    NUM_MINIBATCHES="${NUM_MINIBATCHES:-$DEFAULT_NUM_MINIBATCHES}"
    LR="${LR:-$DEFAULT_LR}"
    UPDATE_EPOCHS="${UPDATE_EPOCHS:-$DEFAULT_UPDATE_EPOCHS}"
    FC_DIM_SIZE="${FC_DIM_SIZE:-$DEFAULT_FC_DIM_SIZE}"
    GRU_HIDDEN_DIM="${GRU_HIDDEN_DIM:-$DEFAULT_GRU_HIDDEN_DIM}"
    TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-100000000}"
    REW_SHAPING_HORIZON="${REW_SHAPING_HORIZON:-50000000}"
    WANDB_MODE="${WANDB_MODE:-online}"
    WANDB_PROJECT="${WANDB_PROJECT:-overcookedv3-mappo-full-obs-100m}"
    DISABLE_CHECKPOINTS="${DISABLE_CHECKPOINTS:-False}"
    if [[ $# -gt 0 ]]; then
        LAYOUTS=("$@")
    else
        LAYOUTS=("${ALL_LAYOUTS[@]}")
    fi
elif [[ "$MODE" == "smoke" ]]; then
    NUM_ENVS="${NUM_ENVS:-${SMOKE_NUM_ENVS:-1}}"
    NUM_STEPS="${NUM_STEPS:-${SMOKE_NUM_STEPS:-2}}"
    NUM_MINIBATCHES="${NUM_MINIBATCHES:-${SMOKE_NUM_MINIBATCHES:-1}}"
    LR="${LR:-$DEFAULT_LR}"
    UPDATE_EPOCHS="${UPDATE_EPOCHS:-${SMOKE_UPDATE_EPOCHS:-1}}"
    FC_DIM_SIZE="${FC_DIM_SIZE:-${SMOKE_FC_DIM_SIZE:-8}}"
    GRU_HIDDEN_DIM="${GRU_HIDDEN_DIM:-${SMOKE_GRU_HIDDEN_DIM:-16}}"
    TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-$((NUM_ENVS * NUM_STEPS))}"
    REW_SHAPING_HORIZON="${REW_SHAPING_HORIZON:-$((TOTAL_TIMESTEPS / 2))}"
    WANDB_MODE="${WANDB_MODE:-disabled}"
    WANDB_PROJECT="${WANDB_PROJECT:-overcookedv3-mappo-full-obs-100m-smoke}"
    DISABLE_CHECKPOINTS="${DISABLE_CHECKPOINTS:-True}"
    if [[ $# -gt 0 ]]; then
        LAYOUTS=("$@")
    else
        LAYOUTS=(cramped_room)
    fi
else
    echo "Unknown mode: $MODE. Use 'smoke' or 'full'."
    exit 1
fi

WANDB_DIR="${WANDB_DIR:-${OUTPUT_ROOT}/${WANDB_PROJECT}}"
mkdir -p "${OUTPUT_ROOT}/logs" "${WANDB_DIR}/models" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR"

echo "mode=$MODE"
echo "project_dir=$PROJECT_DIR"
echo "output_root=$OUTPUT_ROOT"
echo "wandb_dir=$WANDB_DIR"
echo "layouts=${LAYOUTS[*]}"
echo "total_timesteps=$TOTAL_TIMESTEPS"
echo "num_envs=$NUM_ENVS"
echo "num_steps=$NUM_STEPS"
echo "num_minibatches=$NUM_MINIBATCHES"
echo "update_epochs=$UPDATE_EPOCHS"
echo "fc_dim_size=$FC_DIM_SIZE"
echo "gru_hidden_dim=$GRU_HIDDEN_DIM"
echo "lr=$LR"
echo "wandb_mode=$WANDB_MODE"
echo "hostname=$(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "python=$PYTHON_BIN"

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv,noheader
else
    echo "nvidia-smi not found"
fi

"$PYTHON_BIN" - <<'PY'
import os
import jax

print("CUDA_VISIBLE_DEVICES repr:", repr(os.environ.get("CUDA_VISIBLE_DEVICES")))
print("JAX devices:", jax.devices())
PY

for LAYOUT in "${LAYOUTS[@]}"; do
    RUN_DIR="${WANDB_DIR}/runs/${MODE}_${LAYOUT}"

    echo "============================================================"
    echo "Starting full-obs MAPPO on layout: $LAYOUT"
    echo "run_dir=$RUN_DIR"
    echo "============================================================"

    "$PYTHON_BIN" "${PROJECT_DIR}/baselines/MAPPO/mappo_rnn_overcooked_v3_full_obs.py" \
        hydra.run.dir="$RUN_DIR" \
        hydra.output_subdir=null \
        ENV_KWARGS.layout="$LAYOUT" \
        ++ENV_KWARGS.enable_item_conveyors=True \
        ++ENV_KWARGS.enable_player_conveyors=True \
        TOTAL_TIMESTEPS="$TOTAL_TIMESTEPS" \
        REW_SHAPING_HORIZON="$REW_SHAPING_HORIZON" \
        NUM_ENVS="$NUM_ENVS" \
        NUM_STEPS="$NUM_STEPS" \
        NUM_MINIBATCHES="$NUM_MINIBATCHES" \
        UPDATE_EPOCHS="$UPDATE_EPOCHS" \
        FC_DIM_SIZE="$FC_DIM_SIZE" \
        GRU_HIDDEN_DIM="$GRU_HIDDEN_DIM" \
        LR="$LR" \
        WANDB_MODE="$WANDB_MODE" \
        ENTITY=null \
        PROJECT="$WANDB_PROJECT" \
        WANDB_DIR="$WANDB_DIR" \
        "++WANDB_RUN_NAME=mappo_v3_full_obs_${MODE}_${LAYOUT}" \
        USE_RICH_MONITOR=False \
        "++DISABLE_CHECKPOINTS=$DISABLE_CHECKPOINTS"

    echo "Finished layout: $LAYOUT"
    echo ""
done

echo "All layouts complete."
