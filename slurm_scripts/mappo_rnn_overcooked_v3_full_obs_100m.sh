#!/bin/bash
#SBATCH --job-name=mappo_island_5m
#SBATCH --partition=base_suma_rtx3090
#SBATCH --qos=base_qos
#SBATCH --exclude=node19,node13,node16,node08,node10,node21,node14,node04,node05
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/zachtang/jaxmarl/logs/%x_%j.out
#SBATCH --error=/scratch/zachtang/jaxmarl/logs/%x_%j.err

set -euo pipefail

# Default is a one-update smoke test. Submit with `full` for the 5M-step
# around_the_island_nerfed run:
#   sbatch slurm_scripts/mappo_rnn_overcooked_v3_full_obs_100m.sh full
#
# Pre-create dirs on the login node before submitting:
#   mkdir -p /scratch/zachtang/jaxmarl/logs \
#     /scratch/zachtang/jaxmarl/overcookedv3-mappo-island-5m/models \
#     /scratch/zachtang/jaxmarl/overcookedv3-mappo-island-5m-smoke/models \
#     /scratch/zachtang/jaxmarl/wandb-cache \
#     /scratch/zachtang/jaxmarl/wandb-config

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR=${PROJECT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}
SCRATCH_ROOT=${SCRATCH:-/scratch/zachtang}

source ${PROJECT_DIR}/venv/bin/activate

cd "$PROJECT_DIR"

export PYTHONUNBUFFERED=1
export JAX_PLATFORMS=cuda,cpu
export SCRATCH="$SCRATCH_ROOT"
unset LD_LIBRARY_PATH
export WANDB_CACHE_DIR="$SCRATCH_ROOT/jaxmarl/wandb-cache"
export WANDB_CONFIG_DIR="$SCRATCH_ROOT/jaxmarl/wandb-config"

MODE="${1:-smoke}"
LAYOUT=around_the_island_nerfed
NUM_ENVS=128
NUM_STEPS=256
NUM_MINIBATCHES=32
UPDATE_EPOCHS=4
LR=0.0005
ENT_COEF=0.06
SHAPED_REWARD_SCALE=5.0
DELIVERY_REWARD=40.0
DENSE_TASK_SHAPING=True
USE_ACTION_MASK=True
PRINT_METRIC_INTERVAL=1

if [[ "$MODE" == "full" ]]; then
    TOTAL_TIMESTEPS=5000000
    REW_SHAPING_HORIZON=5000000
    WANDB_MODE=offline
    WANDB_PROJECT=overcookedv3-mappo-island-250
    DISABLE_CHECKPOINTS=False
elif [[ "$MODE" == "smoke" ]]; then
    TOTAL_TIMESTEPS=$((NUM_ENVS * NUM_STEPS))
    REW_SHAPING_HORIZON=$((TOTAL_TIMESTEPS / 2))
    WANDB_MODE=disabled
    WANDB_PROJECT=overcookedv3-mappo-island-5m-smoke
    DISABLE_CHECKPOINTS=True
else
    echo "Unknown mode: $MODE. Use 'smoke' or 'full'."
    exit 1
fi

WANDB_DIR="$SCRATCH_ROOT/jaxmarl/${WANDB_PROJECT}"

if [[ ! -d "$SCRATCH_ROOT/jaxmarl/logs" || ! -d "$WANDB_DIR" || ! -d "$WANDB_DIR/models" ]]; then
    echo "Missing scratch output directories."
    echo "Run this on the login node before sbatch:"
    echo "  mkdir -p $SCRATCH_ROOT/jaxmarl/logs $WANDB_DIR/models $WANDB_CACHE_DIR $WANDB_CONFIG_DIR"
    exit 1
fi

echo "mode=$MODE"
echo "project_dir=$PROJECT_DIR"
echo "scratch_root=$SCRATCH_ROOT"
echo "layout=$LAYOUT"
echo "total_timesteps=$TOTAL_TIMESTEPS"
echo "rew_shaping_horizon=$REW_SHAPING_HORIZON"
echo "num_envs=$NUM_ENVS"
echo "num_steps=$NUM_STEPS"
echo "num_minibatches=$NUM_MINIBATCHES"
echo "update_epochs=$UPDATE_EPOCHS"
echo "lr=$LR"
echo "ent_coef=$ENT_COEF"
echo "shaped_reward_scale=$SHAPED_REWARD_SCALE"
echo "delivery_reward=$DELIVERY_REWARD"
echo "dense_task_shaping=$DENSE_TASK_SHAPING"
echo "use_action_mask=$USE_ACTION_MASK"
echo "wandb_project=$WANDB_PROJECT"
echo "hostname=$(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unset}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "python=$(which python)"
nvidia-smi -L
python - <<'PY'
import os
import jax

print("CUDA_VISIBLE_DEVICES repr:", repr(os.environ.get("CUDA_VISIBLE_DEVICES")))
print("JAX devices:", jax.devices())
PY

echo "============================================================"
echo "Starting full-obs MAPPO on layout: $LAYOUT"
echo "============================================================"

srun --ntasks=1 --gres=gpu:1 python -u ${PROJECT_DIR}/baselines/MAPPO/mappo_rnn_overcooked_v3_full_obs.py \
    ENV_KWARGS.layout="$LAYOUT" \
    ENV_KWARGS.random_agent_positions=False \
    ++ENV_KWARGS.delivery_reward="$DELIVERY_REWARD" \
    "++ENV_KWARGS.dense_task_shaping=$DENSE_TASK_SHAPING" \
    ++ENV_KWARGS.enable_item_conveyors=True \
    ++ENV_KWARGS.enable_player_conveyors=True \
    TOTAL_TIMESTEPS="$TOTAL_TIMESTEPS" \
    REW_SHAPING_HORIZON="$REW_SHAPING_HORIZON" \
    NUM_ENVS="$NUM_ENVS" \
    NUM_STEPS="$NUM_STEPS" \
    NUM_MINIBATCHES="$NUM_MINIBATCHES" \
    UPDATE_EPOCHS="$UPDATE_EPOCHS" \
    LR="$LR" \
    ENT_COEF="$ENT_COEF" \
    SHAPED_REWARD_SCALE="$SHAPED_REWARD_SCALE" \
    "++USE_ACTION_MASK=$USE_ACTION_MASK" \
    WANDB_MODE="$WANDB_MODE" \
    ENTITY=null \
    PROJECT="$WANDB_PROJECT" \
    WANDB_DIR="$WANDB_DIR" \
    "++WANDB_RUN_NAME=mappo_v3_full_obs_${MODE}_${LAYOUT}_handoffevents_decay250" \
    USE_RICH_MONITOR=False \
    "++DISABLE_CHECKPOINTS=$DISABLE_CHECKPOINTS" \
    "++PRINT_METRIC_INTERVAL=$PRINT_METRIC_INTERVAL"

echo "Finished layout: $LAYOUT"
echo ""

echo "All layouts complete."
