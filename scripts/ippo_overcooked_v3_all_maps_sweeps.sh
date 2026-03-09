#!/usr/bin/env bash
# IPPO sweep over all overcooked_v3 maps (train only, no infer step).
# Mirrors the IC3Net sweep script structure.

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
WANDB_PROJECT="${WANDB_PROJECT:-overcookedv3_ippo_${CURRENT_TIME}}"
WANDB_SWEEP_COUNT="${WANDB_SWEEP_COUNT:-20}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT_DIR/checkpoints/ippo_overcooked_v3_all_maps_sweeps_${CURRENT_TIME}}"

mkdir -p "$OUTPUT_ROOT"

# Discover all overcooked_v3 maps
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
echo "W&B project: $WANDB_PROJECT"
echo "Output root: $OUTPUT_ROOT"

for MAP_NAME in "${MAPS[@]}"; do
  MAP_DIR="$OUTPUT_ROOT/$MAP_NAME"
  TRAIN_SAVE_PATH="$MAP_DIR/train_sweeps"
  mkdir -p "$TRAIN_SAVE_PATH"

  echo
  echo "============================================================"
  echo "[IPPO TRAIN SWEEP] map=$MAP_NAME"
  echo "save_path=$TRAIN_SAVE_PATH"
  echo "============================================================"

  if ! "$PYTHON_BIN" "$ROOT_DIR/baselines/IPPO/ippo_rnn_overcooked_v3.py" \
    --config-name=ippo_rnn_overcooked_v3 \
    ENV_KWARGS.layout="$MAP_NAME" \
    SAVE_PATH="$TRAIN_SAVE_PATH" \
    NUM_ENVS=128 \
    REW_SHAPING_HORIZON=80000000 \
    SHAPED_REWARD_COEFF=40.0 \
    ENT_COEF=0.05 \
    WANDB_MODE="$WANDB_MODE" \
    WANDB_PROJECT="$WANDB_PROJECT" \
    WANDB_NAME="ippo_overcooked_v3_${MAP_NAME}_train_sweep" \
    +WANDB_SWEEP=true \
    +WANDB_SWEEP_COUNT="$WANDB_SWEEP_COUNT"; then
    echo "WARNING: train sweep failed for map=$MAP_NAME; continuing to next map"
    continue
  fi
done

echo
echo "All IPPO train sweeps completed."
echo "Output root: $OUTPUT_ROOT"
