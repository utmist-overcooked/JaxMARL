#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

CURRENT_TIME=$(date +"%Y%m%d_%H%M%S")
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_PROJECT="${WANDB_PROJECT:-ippo_v1_cramped_room_${CURRENT_TIME}}"

echo "Starting IPPO v1 training for cramped_room"
echo "W&B project: $WANDB_PROJECT"

"$PYTHON_BIN" "$ROOT_DIR/baselines/IPPO/ippo_cnn_overcooked.py" \
  WANDB_MODE="$WANDB_MODE" \
  PROJECT="$WANDB_PROJECT" \
  ENTITY="dannyb3334-university-of-toronto" \
  ENV_KWARGS.layout=cramped_room \
  TOTAL_TIMESTEPS=5000000 \
  NUM_ENVS=64 \
  NUM_STEPS=256 \
  UPDATE_EPOCHS=4 \
  NUM_MINIBATCHES=16 \
  LR=0.0005 \
  GAMMA=0.99 \
  GAE_LAMBDA=0.95 \
  CLIP_EPS=0.2 \
  ENT_COEF=0.01 \
  VF_COEF=0.5 \
  MAX_GRAD_NORM=0.5 \
  REW_SHAPING_HORIZON=2500000 \
  ACTIVATION=relu \
  SEED=42 \
  NUM_SEEDS=1 \
  TUNE=False \
  SAVE_GIF_PATH=/workspace/JaxMARL/outputs/ippo_v1_cramped_room_inference.gif

echo "Training finished."
