#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv-jaxcuda/bin/python}"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"

CURRENT_TIME="$(date +%Y%m%d_%H%M%S)"
LAYOUT="${LAYOUT:-cramped_room}"
DEMO_GLOB="${DEMO_GLOB:-$ROOT_DIR/demos/overcooked_v3/*.npz}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT_DIR/checkpoints/ippo_overcooked_v3_bc_warmstart_${CURRENT_TIME}}"
BC_SAVE_PATH="${BC_SAVE_PATH:-$OUTPUT_ROOT/bc}"
RL_SAVE_PATH="${RL_SAVE_PATH:-$OUTPUT_ROOT/rl}"

echo "Using python: $PYTHON_BIN"
echo "Demo glob: $DEMO_GLOB"
echo "Behavior cloning output: $BC_SAVE_PATH"
echo "RL output: $RL_SAVE_PATH"

"$PYTHON_BIN" "$ROOT_DIR/baselines/IPPO/behavior_clone_overcooked_v3.py" \
  --config-name=behavior_clone_overcooked_v3 \
  "ENV_KWARGS.layout=$LAYOUT" \
  "DEMO_GLOB=$DEMO_GLOB" \
  "SAVE_PATH=$BC_SAVE_PATH"

"$PYTHON_BIN" "$ROOT_DIR/baselines/IPPO/ippo_rnn_overcooked_v3.py" \
  --config-name=ippo_rnn_overcooked_v3 \
  "ENV_KWARGS.layout=$LAYOUT" \
  "PRETRAINED_CHECKPOINT=$BC_SAVE_PATH/best/model.msgpack" \
  "SAVE_PATH=$RL_SAVE_PATH"