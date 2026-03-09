#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

CURRENT_TIME=$(date +"%Y%m%d_%H%M%S")
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_PROJECT="${WANDB_PROJECT:-overcookedv3_ippo_optimal_${CURRENT_TIME}}"
SAVE_PATH="${SAVE_PATH:-$ROOT_DIR/checkpoints/ippo_v3_optimal_around_the_island_${CURRENT_TIME}}"

mkdir -p "$SAVE_PATH"

echo "Starting optimal IPPO training for around_the_island"
echo "W&B project: $WANDB_PROJECT"
echo "Save path: $SAVE_PATH"

"$PYTHON_BIN" "$ROOT_DIR/baselines/IPPO/ippo_rnn_overcooked_v3.py" \
  --config-name=ippo_rnn_overcooked_v3 \
  WANDB_MODE="$WANDB_MODE" \
  WANDB_PROJECT="$WANDB_PROJECT" \
  WANDB_NAME="ippo_v3_optimal_around_the_island" \
  ENV_KWARGS.layout=around_the_island \
  ENV_KWARGS.max_steps=400 \
  TOTAL_TIMESTEPS=120000000 \
  NUM_ENVS=128 \
  NUM_STEPS=200 \
  UPDATE_EPOCHS=4 \
  NUM_MINIBATCHES=8 \
  LR=0.0005 \
  GAMMA=0.99 \
  GAE_LAMBDA=0.95 \
  CLIP_EPS=0.2 \
  VF_COEF=0.5 \
  ENT_COEF=0.01 \
  ENT_COEF_MIN=0.0 \
  ENTROPY_FLOOR=0.0 \
  ENTROPY_FLOOR_COEF=0.0 \
  MAX_GRAD_NORM=0.5 \
  GRU_HIDDEN_DIM=128 \
  FC_DIM_SIZE=128 \
  REW_SHAPING_HORIZON=60000000 \
  SHAPED_REWARD_COEFF=30.0 \
  SAVE_PATH="$SAVE_PATH"

echo "Training finished. Model should be in: $SAVE_PATH/model.msgpack"