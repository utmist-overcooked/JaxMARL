#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

CURRENT_TIME=$(date +"%Y%m%d_%H%M%S")
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_PROJECT="${WANDB_PROJECT:-ippo_v1_cramped_room_current_params_${CURRENT_TIME}}"
WANDB_NAME="${WANDB_NAME:-ippo_cnn_overcooked_v1_cramped_room_current_params}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-15000000}"
REW_SHAPING_HORIZON="${REW_SHAPING_HORIZON:-$TOTAL_TIMESTEPS}"
MAX_STEPS="${MAX_STEPS:-1000}"
NUM_ENVS="${NUM_ENVS:-128}"
NUM_STEPS="${NUM_STEPS:-200}"
SAVE_GIF_PATH="${SAVE_GIF_PATH:-$ROOT_DIR/outputs/ippo_v1_cramped_room_current_params_${CURRENT_TIME}.gif}"

mkdir -p "$(dirname "$SAVE_GIF_PATH")"

echo "Starting IPPO v1 training for cramped_room"
echo "W&B project: $WANDB_PROJECT"
echo "Total timesteps: $TOTAL_TIMESTEPS"
echo "Episode max steps: $MAX_STEPS"
echo "Reward shaping horizon: $REW_SHAPING_HORIZON"

"$PYTHON_BIN" "$ROOT_DIR/baselines/IPPO/ippo_cnn_overcooked.py" \
  WANDB_MODE="$WANDB_MODE" \
  PROJECT="$WANDB_PROJECT" \
  +WANDB_NAME="$WANDB_NAME" \
  ENTITY="dannyb3334-university-of-toronto" \
  ENV_KWARGS.layout=cramped_room \
  +ENV_KWARGS.max_steps="$MAX_STEPS" \
  TOTAL_TIMESTEPS="$TOTAL_TIMESTEPS" \
  NUM_ENVS="$NUM_ENVS" \
  NUM_STEPS="$NUM_STEPS" \
  UPDATE_EPOCHS=4 \
  NUM_MINIBATCHES=8 \
  LR=0.0005 \
  GAMMA=0.99 \
  GAE_LAMBDA=0.95 \
  CLIP_EPS=0.2 \
  ENT_COEF=0.01 \
  VF_COEF=0.5 \
  MAX_GRAD_NORM=0.5 \
  REW_SHAPING_HORIZON="$REW_SHAPING_HORIZON" \
  ACTIVATION=relu \
  SEED=42 \
  NUM_SEEDS=1 \
  TUNE=False \
  SAVE_GIF_PATH="$SAVE_GIF_PATH"

echo "Training finished."
