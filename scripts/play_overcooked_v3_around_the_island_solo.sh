#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv-jaxcuda/bin/python}"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LAYOUT="${LAYOUT:-around_the_island}"
HUMAN_AGENT="${HUMAN_AGENT:-0}"
PARTNER_MODE="${PARTNER_MODE:-scripted}"
MAX_STEPS="${MAX_STEPS:-400}"
FPS="${FPS:-10}"
TILE_SIZE="${TILE_SIZE:-48}"
POT_COOK_TIME="${POT_COOK_TIME:-20}"
POT_BURN_TIME="${POT_BURN_TIME:-10}"
PRINT_ALL_REWARDS="${PRINT_ALL_REWARDS:-0}"

DEMO_ROOT="${DEMO_ROOT:-$ROOT_DIR/demos/overcooked_v3/around_the_island_solo}"
RECORD_DIR="${RECORD_DIR:-$DEMO_ROOT/$TIMESTAMP}"

echo "Using python: $PYTHON_BIN"
echo "JAX_PLATFORMS: $JAX_PLATFORMS"
echo "Layout: $LAYOUT"
echo "Human agent: $HUMAN_AGENT"
echo "Partner mode: $PARTNER_MODE"
echo "Record dir: $RECORD_DIR"
echo
echo "Controls: WASD to move, SPACE to interact, R to reset, Q to quit"
echo "Rewards: sparse reward and per-agent shaped rewards are shown in the HUD and console"
echo

ARGS=(
  "$ROOT_DIR/play_scripts/play_overcooked_v3.py"
  "--layout" "$LAYOUT"
  "--max-steps" "$MAX_STEPS"
  "--fps" "$FPS"
  "--tile-size" "$TILE_SIZE"
  "--pot-cook-time" "$POT_COOK_TIME"
  "--pot-burn-time" "$POT_BURN_TIME"
  "--partner-mode" "$PARTNER_MODE"
  "--human-agent" "$HUMAN_AGENT"
  "--record-dir" "$RECORD_DIR"
)

if [[ "$PRINT_ALL_REWARDS" == "1" ]]; then
  ARGS+=("--print-all-rewards")
fi

exec "$PYTHON_BIN" "${ARGS[@]}"