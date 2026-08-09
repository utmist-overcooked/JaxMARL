#!/usr/bin/env bash
# Launch the 15M-step MAPPO RNN run on the `prep_kitchen` layout, logged to W&B.
#
# The centralised-critic counterpart of run_ippo_prep_kitchen_15m.sh: same
# layout, same primitive actions, same order queue and shaping, but the critic
# is conditioned on the joint world state (both agents' grid observations
# stacked on the channel axis) instead of the acting agent's own view.
#
# The run is started inside a detached tmux session so it survives SSH drops.
#
# Usage:
#   scripts/run_mappo_prep_kitchen_15m.sh                 # launch in tmux
#   DRY_RUN=1 scripts/run_mappo_prep_kitchen_15m.sh       # print the command only
#   FOREGROUND=1 scripts/run_mappo_prep_kitchen_15m.sh    # run here, no tmux
#
#   # CommNet communication between agents, full observability
#   ALGO=commnet scripts/run_mappo_prep_kitchen_15m.sh
#
#   # CommNet with a 5x5 egocentric view instead of the whole grid
#   ALGO=commnet AGENT_VIEW_SIZE=2 scripts/run_mappo_prep_kitchen_15m.sh
#
# Environment overrides: SEED, TOTAL_TIMESTEPS, NUM_ENVS, NUM_STEPS, MAX_STEPS,
# ENT_COEF, LAYOUT, POT_COOK_TIME, POT_BURN_TIME, ENTITY, WANDB_PROJECT,
# RUN_NAME, TMUX_SESSION, EXTRA_ARGS, ALGO, AGENT_VIEW_SIZE, COMM_PASSES.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "error: no interpreter at $PYTHON_BIN (create it with 'uv venv .venv' + 'uv pip install -e .[algs,dev]')" >&2
  exit 1
fi

SEED="${SEED:-42}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-15000000}"
NUM_ENVS="${NUM_ENVS:-128}"
NUM_STEPS="${NUM_STEPS:-5}"
MAX_STEPS="${MAX_STEPS:-800}"
LAYOUT="${LAYOUT:-prep_kitchen}"
# Burning disabled: it was the only base-reward signal the agent ever saw, so
# filling a pot was net-punished. pot_burn_time=0 keeps cooked soup ready.
POT_COOK_TIME="${POT_COOK_TIME:-20}"
POT_BURN_TIME="${POT_BURN_TIME:-0}"
# ALGO=mappo (no communication) or commnet (CommNet block between agents).
ALGO="${ALGO:-mappo}"
# null = full observability; an integer gives each agent that egocentric radius.
AGENT_VIEW_SIZE="${AGENT_VIEW_SIZE:-null}"
COMM_PASSES="${COMM_PASSES:-2}"
# Dish washing needs a layout with a sink (S) and a dirty plate pile (D).
# INITIAL_DIRTY_PLATES seeds the dirty pile so the wash loop is reachable
# before the first delivery; 0 reproduces the original behaviour.
ENABLE_DISH_WASHING="${ENABLE_DISH_WASHING:-false}"
NUM_PLATES="${NUM_PLATES:-3}"
INITIAL_DIRTY_PLATES="${INITIAL_DIRTY_PLATES:-0}"
# Item conveyor belts only actually move items when this is on; with it off
# the belt tiles behave as ordinary counters.
ENABLE_ITEM_CONVEYORS="${ENABLE_ITEM_CONVEYORS:-false}"
ENT_COEF="${ENT_COEF:-0.03}"
ENTROPY_FLOOR="${ENTROPY_FLOOR:-0.8}"
ENTROPY_FLOOR_COEF="${ENTROPY_FLOOR_COEF:-0.01}"
REW_SHAPING_HORIZON="${REW_SHAPING_HORIZON:-$TOTAL_TIMESTEPS}"
REW_SHAPING_MIN_COEFF="${REW_SHAPING_MIN_COEFF:-0.10}"
SHAPED_REWARD_COEFF="${SHAPED_REWARD_COEFF:-30.0}"

ENTITY="${ENTITY:-dannyb3334-university-of-toronto}"
WANDB_PROJECT="${WANDB_PROJECT:-overcookedv3_mappo_rnn_prep_kitchen_15m}"
case "$ALGO" in
  mappo)   TRAINER=baselines/MAPPO/mappo_rnn_overcooked_v3.py ;;
  commnet) TRAINER=baselines/MAPPO/mappo_rnn_overcooked_v3_commnet.py ;;
  *) echo "error: ALGO must be 'mappo' or 'commnet', got '$ALGO'" >&2; exit 1 ;;
esac

RUN_NAME="${RUN_NAME:-${ALGO}_rnn_v3_prep_kitchen_15m_seed${SEED}}"

SAVE_PATH="${SAVE_PATH:-$REPO_ROOT/checkpoints/$RUN_NAME}"
SAVE_GIF_PATH="${SAVE_GIF_PATH:-$REPO_ROOT/outputs/$RUN_NAME.gif}"
mkdir -p "$(dirname "$SAVE_GIF_PATH")"

TMUX_SESSION="${TMUX_SESSION:-mappo_prep_kitchen_15m_seed${SEED}}"

# jaxlib's CUDA build wants ptxas from the pip-installed nvcc package.
NVCC_BIN="$REPO_ROOT/.venv/lib/python3.10/site-packages/nvidia/cuda_nvcc/bin"

# COMM_PASSES only exists in the commnet config; passing it to the plain MAPPO
# trainer would be rejected by Hydra's struct mode.
COMM_ARGS=""
if [[ "$ALGO" == "commnet" ]]; then
  COMM_ARGS="COMM_PASSES=$COMM_PASSES"
fi

CMD=(
  env
  "PATH=$NVCC_BIN:$PATH"
  "PYTHONPATH=$REPO_ROOT"
  "XLA_PYTHON_CLIENT_PREALLOCATE=false"
  "$PYTHON_BIN" "$TRAINER"
  "ENV_KWARGS.layout=$LAYOUT"
  "ENV_KWARGS.agent_view_size=$AGENT_VIEW_SIZE"
  "ENV_KWARGS.enable_dish_washing=$ENABLE_DISH_WASHING"
  "ENV_KWARGS.num_plates=$NUM_PLATES"
  "ENV_KWARGS.initial_dirty_plates=$INITIAL_DIRTY_PLATES"
  "ENV_KWARGS.max_steps=$MAX_STEPS"
  "ENV_KWARGS.pot_cook_time=$POT_COOK_TIME"
  "ENV_KWARGS.pot_burn_time=$POT_BURN_TIME"
  ENV_KWARGS.enable_order_queue=true
  ENV_KWARGS.max_orders=5
  ENV_KWARGS.order_generation_rate=1.0
  ENV_KWARGS.order_expiration_time=0
  ENV_KWARGS.order_queue_mode=alternating
  "ENV_KWARGS.enable_item_conveyors=$ENABLE_ITEM_CONVEYORS"
  ENV_KWARGS.enable_player_conveyors=false
  "TOTAL_TIMESTEPS=$TOTAL_TIMESTEPS"
  "NUM_ENVS=$NUM_ENVS"
  "NUM_STEPS=$NUM_STEPS"
  UPDATE_EPOCHS=4
  NUM_MINIBATCHES=8
  LR=0.0005
  GAMMA=0.99
  GAE_LAMBDA=0.95
  CLIP_EPS=0.2
  VF_COEF=0.5
  "ENT_COEF=$ENT_COEF"
  ENT_COEF_MIN=0.0
  "ENTROPY_FLOOR=$ENTROPY_FLOOR"
  "ENTROPY_FLOOR_COEF=$ENTROPY_FLOOR_COEF"
  MAX_GRAD_NORM=0.5
  ${COMM_ARGS:-}
  GRU_HIDDEN_DIM=256
  FC_DIM_SIZE=256
  ACTIVATION=relu
  ANNEAL_LR=true
  "REW_SHAPING_HORIZON=$REW_SHAPING_HORIZON"
  "REW_SHAPING_MIN_COEFF=$REW_SHAPING_MIN_COEFF"
  "SHAPED_REWARD_COEFF=$SHAPED_REWARD_COEFF"
  WANDB_MODE=online
  "ENTITY=$ENTITY"
  "WANDB_PROJECT=$WANDB_PROJECT"
  "WANDB_NAME=$RUN_NAME"
  "SAVE_PATH=$SAVE_PATH"
  "SAVE_GIF_PATH=$SAVE_GIF_PATH"
  LOG_EVERY=100
  CHECKPOINT_EVERY=5000
  "SEED=$SEED"
  ${EXTRA_ARGS:-}
)

printf 'launch command:\n'
printf '  %q' "${CMD[@]}"
printf '\n\n'

if [[ -n "${DRY_RUN:-}" ]]; then
  exit 0
fi

if [[ -n "${FOREGROUND:-}" ]]; then
  exec "${CMD[@]}"
fi

if ! command -v tmux >/dev/null 2>&1; then
  echo "error: tmux not found; re-run with FOREGROUND=1 to train in this shell" >&2
  exit 1
fi

if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
  echo "error: tmux session '$TMUX_SESSION' already exists; kill it or set TMUX_SESSION" >&2
  exit 1
fi

# Keep the pane alive after the run so the tail of the log stays readable.
tmux new-session -d -s "$TMUX_SESSION" \
  "$(printf '%q ' "${CMD[@]}"); echo; echo '=== run finished (exit '\$?') ==='; exec bash"

cat <<EOF
started tmux session: $TMUX_SESSION
  algorithm   : $ALGO RNN (centralised critic over joint world state)
  view        : agent_view_size=$AGENT_VIEW_SIZE (null = full observability)
  dish washing: $ENABLE_DISH_WASHING (plates=$NUM_PLATES, seeded dirty=$INITIAL_DIRTY_PLATES)
  layout      : $LAYOUT (multi-stage orders, 3 prep chains)
  episode     : $MAX_STEPS steps, pot_burn_time=$POT_BURN_TIME
  budget      : $TOTAL_TIMESTEPS env steps
  wandb       : $ENTITY/$WANDB_PROJECT/$RUN_NAME
  checkpoint  : $SAVE_PATH
  gif         : $SAVE_GIF_PATH

attach : tmux attach -t $TMUX_SESSION
detach : Ctrl-b d
kill   : tmux kill-session -t $TMUX_SESSION
EOF
