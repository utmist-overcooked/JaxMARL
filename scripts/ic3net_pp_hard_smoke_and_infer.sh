#!/usr/bin/env bash

set -euo pipefail

RUN_NAME="${1:-}"

if [[ -z "$RUN_NAME" ]]; then
  echo "Usage: $0 <run_name>"
  echo "Example: $0 smoke_$(date +%Y%m%d_%H%M%S)"
  exit 1
fi

ROOT_DIR="/workspace/JaxMARL"
PYTHON_BIN="/usr/bin/python3"
RUN_DIR="$ROOT_DIR/checkpoints/$RUN_NAME"
MODEL_PATH="$RUN_DIR/model.msgpack"
GIF_PATH="$RUN_DIR/inference.gif"

mkdir -p "$RUN_DIR"

echo "[1/2] Smoke training -> $RUN_DIR"
JAX_PLATFORMS=cuda WANDB_MODE=disabled "$PYTHON_BIN" "$ROOT_DIR/baselines/IC3Net/ic3net_train.py" \
  --config-name=ic3net_pp_hard \
  WANDB_MODE=disabled \
  SAVE_PATH="$RUN_DIR" \
  +CHECKPOINT_EVERY=1 \
  LOG_EVERY=1 \
  NUM_ENVS=1 \
  NUM_STEPS=8 \
  TOTAL_TIMESTEPS=8 \
  ENV_KWARGS.max_steps=8

if [[ ! -f "$MODEL_PATH" ]]; then
  echo "ERROR: Expected model not found at $MODEL_PATH"
  exit 2
fi

echo "[2/2] Inference -> $GIF_PATH"
JAX_PLATFORMS=cuda "$PYTHON_BIN" "$ROOT_DIR/baselines/IC3Net/ic3net_infer.py" \
  --config-name=ic3net_pp_hard_infer \
  MODEL_PATH="$MODEL_PATH" \
  SAVE_GIF="$GIF_PATH" \
  NUM_EPISODES=2 \
  MAX_STEPS=8

echo "Done."
echo "Model: $MODEL_PATH"
echo "GIF:   $GIF_PATH"