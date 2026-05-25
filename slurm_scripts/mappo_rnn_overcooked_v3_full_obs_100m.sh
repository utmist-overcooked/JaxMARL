#!/bin/bash
#SBATCH --job-name=mappo_full_obs_100m
#SBATCH --account=rrg-cglee
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=23:59:00
#SBATCH --output=/scratch/zachtang/jaxmarl/logs/%x_%j.out
#SBATCH --error=/scratch/zachtang/jaxmarl/logs/%x_%j.err

set -euo pipefail

# Default is a one-update smoke test. Submit with `full` for 100M steps:
#   sbatch slurm_scripts/mappo_rnn_overcooked_v3_full_obs_100m.sh full
#
# Pre-create dirs on the login node before submitting:
#   mkdir -p $SCRATCH/jaxmarl/logs \
#     $SCRATCH/jaxmarl/overcookedv3-mappo-full-obs-100m-smoke/models \
#     $SCRATCH/jaxmarl/overcookedv3-mappo-full-obs-100m/models \
#     $SCRATCH/jaxmarl/wandb-cache \
#     $SCRATCH/jaxmarl/wandb-config

PROJECT_DIR=$HOME/links/projects/rrg-cglee/zachtang/JaxMARL

source ${PROJECT_DIR}/venv/bin/activate

cd $SCRATCH

export PYTHONUNBUFFERED=1
export JAX_PLATFORMS=cuda,cpu
export WANDB_CACHE_DIR="$SCRATCH/jaxmarl/wandb-cache"
export WANDB_CONFIG_DIR="$SCRATCH/jaxmarl/wandb-config"

MODE="${1:-smoke}"
NUM_ENVS=2048
NUM_STEPS=256
NUM_MINIBATCHES=64
LR=0.002

if [[ "$MODE" == "full" ]]; then
    TOTAL_TIMESTEPS=100000000
    REW_SHAPING_HORIZON=50000000
    WANDB_MODE=offline
    WANDB_PROJECT=overcookedv3-mappo-full-obs-100m
    DISABLE_CHECKPOINTS=False
elif [[ "$MODE" == "smoke" ]]; then
    TOTAL_TIMESTEPS=$((NUM_ENVS * NUM_STEPS))
    REW_SHAPING_HORIZON=$((TOTAL_TIMESTEPS / 2))
    WANDB_MODE=disabled
    WANDB_PROJECT=overcookedv3-mappo-full-obs-100m-smoke
    DISABLE_CHECKPOINTS=True
else
    echo "Unknown mode: $MODE. Use 'smoke' or 'full'."
    exit 1
fi

WANDB_DIR="$SCRATCH/jaxmarl/${WANDB_PROJECT}"

if [[ ! -d "$SCRATCH/jaxmarl/logs" || ! -d "$WANDB_DIR" || ! -d "$WANDB_DIR/models" ]]; then
    echo "Missing scratch output directories."
    echo "Run this on the login node before sbatch:"
    echo "  mkdir -p $SCRATCH/jaxmarl/logs $WANDB_DIR/models $WANDB_CACHE_DIR $WANDB_CONFIG_DIR"
    exit 1
fi

LAYOUTS=(
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

echo "mode=$MODE"
echo "total_timesteps=$TOTAL_TIMESTEPS"
echo "num_envs=$NUM_ENVS"
echo "num_steps=$NUM_STEPS"
echo "lr=$LR"
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

for LAYOUT in "${LAYOUTS[@]}"; do
    echo "============================================================"
    echo "Starting full-obs MAPPO on layout: $LAYOUT"
    echo "============================================================"

    srun --ntasks=1 --gpus-per-task=1 python ${PROJECT_DIR}/baselines/MAPPO/mappo_rnn_overcooked_v3_full_obs.py \
        ENV_KWARGS.layout="$LAYOUT" \
        ++ENV_KWARGS.enable_item_conveyors=True \
        ++ENV_KWARGS.enable_player_conveyors=True \
        TOTAL_TIMESTEPS="$TOTAL_TIMESTEPS" \
        REW_SHAPING_HORIZON="$REW_SHAPING_HORIZON" \
        NUM_ENVS="$NUM_ENVS" \
        NUM_STEPS="$NUM_STEPS" \
        NUM_MINIBATCHES="$NUM_MINIBATCHES" \
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
