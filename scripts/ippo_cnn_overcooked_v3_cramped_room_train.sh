#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

CURRENT_TIME=$(date +"%Y%m%d_%H%M%S")
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_PROJECT="${WANDB_PROJECT:-ippo_v3_cnn_cramped_${CURRENT_TIME}}"
GIF_PATH="/workspace/JaxMARL/outputs/ippo_v3_cnn_cramped_room_inference.gif"

echo "Starting IPPO CNN v3 training for cramped_room"
echo "W&B project: $WANDB_PROJECT"
echo "GIF will be saved to: $GIF_PATH"

"$PYTHON_BIN" "$ROOT_DIR/baselines/IPPO/ippo_cnn_overcooked_v3.py" \
  --config-name=ippo_cnn_overcooked_v3 \
  WANDB_MODE="$WANDB_MODE" \
  PROJECT="$WANDB_PROJECT" \
  ENTITY="dannyb3334-university-of-toronto" \
  ENV_KWARGS.layout=cramped_room \
  ENV_KWARGS.max_steps=400 \
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
  ACTIVATION=relu \
  ANNEAL_LR=true \
  SHAPED_REWARD_COEFF=30.0 \
  REW_SHAPING_HORIZON=2500000 \
  SEED=42 \
  NUM_SEEDS=1 \
  SAVE_GIF_PATH="$GIF_PATH"

echo "Training finished. GIF: $GIF_PATH"
