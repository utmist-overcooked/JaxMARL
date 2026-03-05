#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"

# JAX memory: disable pre-allocation so we can scale NUM_ENVS higher
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

# Generate timestamp-based project name
CURRENT_TIME=$(date +"%Y%m%d_%H%M%S")
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_PROJECT="${WANDB_PROJECT:-overcookedv3_ic3net_${CURRENT_TIME}}"
WANDB_SWEEP_COUNT="${WANDB_SWEEP_COUNT:-20}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT_DIR/checkpoints/ic3net_overcooked_v3_all_maps_sweeps_${CURRENT_TIME}}"

INFER_NUM_EPISODES="${INFER_NUM_EPISODES:-10}"
INFER_MAX_STEPS="${INFER_MAX_STEPS:-200}"
INFER_DETERMINISTIC="${INFER_DETERMINISTIC:-false}"

mkdir -p "$OUTPUT_ROOT"

readarray -t MAPS < <(
  export ROOT_DIR
  "$PYTHON_BIN" - <<'PY'
import re
import os
from pathlib import Path

root_dir = Path(os.environ["ROOT_DIR"])
layouts_path = root_dir / "jaxmarl" / "environments" / "overcooked_v3" / "layouts.py"
text = layouts_path.read_text(encoding="utf-8")

if "overcooked_v3_layouts" not in text:
  raise SystemExit("ERROR: Could not find overcooked_v3_layouts in layouts.py")

matches = re.findall(r'^\s*"([a-zA-Z0-9_]+)"\s*:\s*Layout\.from_string\(', text, flags=re.MULTILINE)
for map_name in sorted(set(matches)):
  print(map_name)
PY
)

if [[ ${#MAPS[@]} -eq 0 ]]; then
  echo "ERROR: No overcooked_v3 maps found in overcooked_v3_layouts"
  exit 1
fi

echo "Found ${#MAPS[@]} overcooked_v3 maps:"
printf '  - %s\n' "${MAPS[@]}"

for MAP_NAME in "${MAPS[@]}"; do
  MAP_DIR="$OUTPUT_ROOT/$MAP_NAME"
  TRAIN_SAVE_PATH="$MAP_DIR/train_sweeps"
  mkdir -p "$TRAIN_SAVE_PATH"

  echo
  echo "============================================================"
  echo "[TRAIN SWEEP] map=$MAP_NAME"
  echo "save_path=$TRAIN_SAVE_PATH"
  echo "============================================================"

  if ! WANDB_SWEEP=TRUE "$PYTHON_BIN" "$ROOT_DIR/baselines/IC3Net/ic3net_train.py" \
    --config-name=ic3net_overcooked_v3_cramped_room \
    ENV_KWARGS.layout="$MAP_NAME" \
    SAVE_PATH="$TRAIN_SAVE_PATH" \
    NUM_ENVS=128 \
    USE_SHAPED_REWARD=true \
    SHAPED_REWARD_COEFF=1.0 \
    WANDB_MODE="$WANDB_MODE" \
    WANDB_PROJECT="$WANDB_PROJECT" \
    WANDB_NAME="ic3net_overcooked_v3_${MAP_NAME}_train_sweep" \
    +WANDB_SWEEP=TRUE \
    +WANDB_SWEEP_COUNT="$WANDB_SWEEP_COUNT"; then
    echo "WARNING: train sweep failed for map=$MAP_NAME; continuing to next map"
    continue
  fi

  readarray -t MODELS < <(find "$TRAIN_SAVE_PATH" -type f -name "model.msgpack" | sort)

  if [[ ${#MODELS[@]} -eq 0 ]]; then
    echo "WARNING: No model.msgpack found under $TRAIN_SAVE_PATH; skipping infer sweep for $MAP_NAME"
    continue
  fi

  echo
  echo "[INFER SWEEP] map=$MAP_NAME (models=${#MODELS[@]})"
  for MODEL_PATH in "${MODELS[@]}"; do
    echo "  -> infer model: $MODEL_PATH"

    if ! WANDB_SWEEP=TRUE "$PYTHON_BIN" "$ROOT_DIR/baselines/IC3Net/ic3net_infer.py" \
      --config-name=ic3net_overcooked_v3_cramped_room \
      ENV_KWARGS.layout="$MAP_NAME" \
      MODEL_PATH="$MODEL_PATH" \
      NUM_EPISODES="$INFER_NUM_EPISODES" \
      MAX_STEPS="$INFER_MAX_STEPS" \
      DETERMINISTIC="$INFER_DETERMINISTIC" \
      +WANDB_SWEEP=TRUE \
      +WANDB_NAME="ic3net_overcooked_v3_${MAP_NAME}_infer_sweep"; then
      echo "WARNING: infer failed for model=$MODEL_PATH; continuing"
    fi
  done
done

echo
echo "All train+infer sweeps completed."
echo "Output root: $OUTPUT_ROOT"
