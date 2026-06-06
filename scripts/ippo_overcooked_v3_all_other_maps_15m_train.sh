#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
SWEEP_ID="${SWEEP_ID:-$(date +"%Y%m%d_%H%M%S")}"
SWEEP_TAG="${SWEEP_TAG:-order_visible_plateguard_burn40_15m}"
SWEEP_PROJECT="${SWEEP_PROJECT:-overcookedv3_ippo_cnn_all_other_maps_${SWEEP_TAG}_${SWEEP_ID}}"
SWEEP_DIR="${SWEEP_DIR:-$ROOT_DIR/outputs/v3_map_sweep_${SWEEP_TAG}_${SWEEP_ID}}"

TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-15000000}"
REW_SHAPING_HORIZON="${REW_SHAPING_HORIZON:-$TOTAL_TIMESTEPS}"
REW_SHAPING_MIN_COEFF="${REW_SHAPING_MIN_COEFF:-0.10}"
MAX_STEPS="${MAX_STEPS:-1000}"
NUM_ENVS="${NUM_ENVS:-128}"
NUM_STEPS="${NUM_STEPS:-200}"
POT_COOK_TIME="${POT_COOK_TIME:-20}"
POT_BURN_TIME="${POT_BURN_TIME:-$((POT_COOK_TIME * 2))}"
ENABLE_ORDER_QUEUE="${ENABLE_ORDER_QUEUE:-true}"
MAX_ORDERS="${MAX_ORDERS:-5}"
ORDER_GENERATION_RATE="${ORDER_GENERATION_RATE:-1.0}"
ORDER_EXPIRATION_TIME="${ORDER_EXPIRATION_TIME:-0}"
ORDER_QUEUE_MODE="${ORDER_QUEUE_MODE:-alternating}"

LAYOUTS=(
  cramped_room
  asymm_advantages
  coord_ring
  forced_coord
  counter_circuit
  cramped_room_v2
  conveyor_demo
  player_conveyor_demo
  player_conveyor_loop
  middle_conveyor
  follow_the_leader
  single_file
)

mkdir -p "$SWEEP_DIR" "$ROOT_DIR/checkpoints"

echo "Starting Overcooked V3 all-other-layouts IPPO CNN sweep"
echo "Sweep id: $SWEEP_ID"
echo "W&B project: $SWEEP_PROJECT"
echo "Sweep dir: $SWEEP_DIR"
echo "Total timesteps per layout: $TOTAL_TIMESTEPS"
echo "Episode max steps: $MAX_STEPS"
echo "Pot timing: cook=$POT_COOK_TIME; burn_expiry=$POT_BURN_TIME"
echo "Order queue: enabled=$ENABLE_ORDER_QUEUE; mode=$ORDER_QUEUE_MODE; rate=$ORDER_GENERATION_RATE; expiry=$ORDER_EXPIRATION_TIME; max=$MAX_ORDERS"
echo "Reward shaping horizon: $REW_SHAPING_HORIZON; floor: $REW_SHAPING_MIN_COEFF"
echo "L2/delivery-distance reward and PLATE_PICKUP_DURING_COOKING must be disabled in settings.py."

for layout in "${LAYOUTS[@]}"; do
  run_tag="${layout}_${SWEEP_TAG}_${SWEEP_ID}"
  log_path="$SWEEP_DIR/${layout}.log"
  save_path="$ROOT_DIR/checkpoints/ippo_cnn_v3_${run_tag}"
  save_gif_path="$ROOT_DIR/outputs/ippo_cnn_v3_${run_tag}.gif"
  wandb_name="ippo_cnn_overcooked_v3_${run_tag}"
  enable_item_conveyors=false
  enable_player_conveyors=false

  case "$layout" in
    conveyor_demo|middle_conveyor)
      enable_item_conveyors=true
      ;;
    player_conveyor_demo|player_conveyor_loop)
      enable_player_conveyors=true
      ;;
  esac

  mkdir -p "$save_path"
  echo "[$(date +"%Y-%m-%d %H:%M:%S")] START $layout" | tee -a "$SWEEP_DIR/status.log"

  if env \
    PYTHON_BIN="$PYTHON_BIN" \
    LAYOUT="$layout" \
    ENABLE_ITEM_CONVEYORS="$enable_item_conveyors" \
    ENABLE_PLAYER_CONVEYORS="$enable_player_conveyors" \
    TOTAL_TIMESTEPS="$TOTAL_TIMESTEPS" \
    REW_SHAPING_HORIZON="$REW_SHAPING_HORIZON" \
    REW_SHAPING_MIN_COEFF="$REW_SHAPING_MIN_COEFF" \
    MAX_STEPS="$MAX_STEPS" \
    NUM_ENVS="$NUM_ENVS" \
    NUM_STEPS="$NUM_STEPS" \
    POT_COOK_TIME="$POT_COOK_TIME" \
    POT_BURN_TIME="$POT_BURN_TIME" \
    ENABLE_ORDER_QUEUE="$ENABLE_ORDER_QUEUE" \
    MAX_ORDERS="$MAX_ORDERS" \
    ORDER_GENERATION_RATE="$ORDER_GENERATION_RATE" \
    ORDER_EXPIRATION_TIME="$ORDER_EXPIRATION_TIME" \
    ORDER_QUEUE_MODE="$ORDER_QUEUE_MODE" \
    WANDB_PROJECT="$SWEEP_PROJECT" \
    WANDB_NAME="$wandb_name" \
    SAVE_PATH="$save_path" \
    SAVE_GIF_PATH="$save_gif_path" \
    "$ROOT_DIR/scripts/ippo_overcooked_v3_around_the_island_optimal_train.sh" \
    > "$log_path" 2>&1; then
    echo "[$(date +"%Y-%m-%d %H:%M:%S")] DONE  $layout gif=$save_gif_path" | tee -a "$SWEEP_DIR/status.log"
  else
    exit_code=$?
    echo "[$(date +"%Y-%m-%d %H:%M:%S")] FAIL  $layout exit_code=$exit_code log=$log_path" | tee -a "$SWEEP_DIR/status.log"
  fi
done

echo "Sweep finished. Status: $SWEEP_DIR/status.log"
