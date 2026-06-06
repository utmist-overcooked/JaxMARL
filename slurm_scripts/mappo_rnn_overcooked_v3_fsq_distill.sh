#!/bin/bash
#SBATCH --job-name=mappo_fsq_distill
#SBATCH --account=rrg-cglee
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=23:59:00
#SBATCH --output=/scratch/zachtang/jaxmarl/logs/%x_%j.out
#SBATCH --error=/scratch/zachtang/jaxmarl/logs/%x_%j.err

set -euo pipefail

# Usage:
#   sbatch slurm_scripts/mappo_rnn_overcooked_v3_fsq_distill.sh smoke cramped_room /scratch/.../teacher_actor.safetensors
#   sbatch slurm_scripts/mappo_rnn_overcooked_v3_fsq_distill.sh full  cramped_room /scratch/.../teacher_actor.safetensors
#
# Pre-create dirs on the login node before submitting:
#   mkdir -p $SCRATCH/jaxmarl/logs \
#     $SCRATCH/jaxmarl/overcookedv3-mappo-fsq-distill-smoke/models \
#     $SCRATCH/jaxmarl/overcookedv3-mappo-fsq-distill/models \
#     $SCRATCH/jaxmarl/wandb-cache \
#     $SCRATCH/jaxmarl/wandb-config

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=${PROJECT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}

source ${PROJECT_DIR}/venv/bin/activate

cd $SCRATCH

export PYTHONUNBUFFERED=1
export JAX_PLATFORMS=cuda,cpu
export WANDB_CACHE_DIR="$SCRATCH/jaxmarl/wandb-cache"
export WANDB_CONFIG_DIR="$SCRATCH/jaxmarl/wandb-config"

MODE="${1:-smoke}"
LAYOUT="${2:-cramped_room}"
TEACHER_ACTOR_PATH="${3:-}"

NUM_ENVS=256
NUM_STEPS=256
NUM_MINIBATCHES=64
LR=0.00025

if [[ -z "$TEACHER_ACTOR_PATH" ]]; then
    echo "Missing TEACHER_ACTOR_PATH argument."
    exit 1
fi

if [[ ! -f "$TEACHER_ACTOR_PATH" ]]; then
    echo "Teacher actor checkpoint not found: $TEACHER_ACTOR_PATH"
    exit 1
fi

if [[ "$MODE" == "full" ]]; then
    TOTAL_TIMESTEPS=30000000
    REW_SHAPING_HORIZON=15000000
    WANDB_MODE=offline
    WANDB_PROJECT=overcookedv3-mappo-fsq-distill
    DISABLE_CHECKPOINTS=False
elif [[ "$MODE" == "smoke" ]]; then
    TOTAL_TIMESTEPS=$((NUM_ENVS * NUM_STEPS))
    REW_SHAPING_HORIZON=$((TOTAL_TIMESTEPS / 2))
    WANDB_MODE=disabled
    WANDB_PROJECT=overcookedv3-mappo-fsq-distill-smoke
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

echo "mode=$MODE"
echo "layout=$LAYOUT"
echo "teacher_actor_path=$TEACHER_ACTOR_PATH"
echo "total_timesteps=$TOTAL_TIMESTEPS"
echo "num_envs=$NUM_ENVS"
echo "num_steps=$NUM_STEPS"
echo "num_minibatches=$NUM_MINIBATCHES"
echo "lr=$LR"
echo "wandb_project=$WANDB_PROJECT"
echo "hostname=$(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unset}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "python=$(which python)"
nvidia-smi -L

srun --ntasks=1 --gpus-per-task=1 python ${PROJECT_DIR}/baselines/MAPPO/mappo_rnn_overcooked_v3_fsq_distill.py \
    ENV_KWARGS.layout="$LAYOUT" \
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
    TEACHER_ACTOR_PATH="$TEACHER_ACTOR_PATH" \
    "++WANDB_RUN_NAME=mappo_v3_fsq_distill_${MODE}_${LAYOUT}" \
    USE_RICH_MONITOR=False \
    "++DISABLE_CHECKPOINTS=$DISABLE_CHECKPOINTS"
