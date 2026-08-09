#!/usr/bin/env bash
# Launch the full 15M-step IPPO RNN run on the `prep_kitchen` layout, logged to W&B.
#
# `prep_kitchen` is the large multi-stage-order kitchen: all three prep chains
# (lettuce -> cutting board, meat -> grill, carrot -> blender) plus pot, plates
# and delivery in one open room, with a recipe indicator driving an alternating
# three-dish order queue. Prep stations are auto-detected from the layout, so no
# extra env flag is needed to turn them on.
#
# The run is started inside a detached tmux session so it survives SSH drops.
#
# Usage:
#   scripts/run_ippo_prep_kitchen_15m.sh                 # launch in tmux
#   SEED=7 scripts/run_ippo_prep_kitchen_15m.sh          # different seed
#   DRY_RUN=1 scripts/run_ippo_prep_kitchen_15m.sh       # print the command only
#   FOREGROUND=1 scripts/run_ippo_prep_kitchen_15m.sh    # run here, no tmux
#
# Environment overrides: SEED, TOTAL_TIMESTEPS, NUM_ENVS, NUM_STEPS, ENT_COEF,
# ENTITY, WANDB_PROJECT, RUN_NAME, TMUX_SESSION, EXTRA_ARGS.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "error: no interpreter at $PYTHON_BIN (create it with 'uv venv .venv' + 'uv pip install -e .[algs,dev]')" >&2
  exit 1
fi

# Training budget and rollout shape. NUM_STEPS=5 keeps PPO updates frequent,
# matching the 15M reference run in docs/ippo_rnn_overcooked_v3_15m_typical_run.md.
SEED="${SEED:-42}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-15000000}"
NUM_ENVS="${NUM_ENVS:-128}"
NUM_STEPS="${NUM_STEPS:-5}"
# prep_kitchen dishes need chop/grill/blend before the pot, so episodes are long.
MAX_STEPS="${MAX_STEPS:-1000}"
# Three prep chains means a wider action funnel than a plain onion kitchen, so
# exploration is held up a bit longer than the single-chain default.
ENT_COEF="${ENT_COEF:-0.03}"
ENTROPY_FLOOR="${ENTROPY_FLOOR:-0.8}"
ENTROPY_FLOOR_COEF="${ENTROPY_FLOOR_COEF:-0.01}"
# Shaping anneals across the whole run but never fully to zero: the prep events
# (prep_placement / prep_action / prep_pickup) are the only signal early on.
REW_SHAPING_HORIZON="${REW_SHAPING_HORIZON:-$TOTAL_TIMESTEPS}"
REW_SHAPING_MIN_COEFF="${REW_SHAPING_MIN_COEFF:-0.10}"
SHAPED_REWARD_COEFF="${SHAPED_REWARD_COEFF:-30.0}"

ENTITY="${ENTITY:-dannyb3334-university-of-toronto}"
WANDB_PROJECT="${WANDB_PROJECT:-overcookedv3_ippo_rnn_prep_kitchen_15m}"
RUN_NAME="${RUN_NAME:-ippo_rnn_v3_prep_kitchen_15m_seed${SEED}}"

SAVE_PATH="${SAVE_PATH:-$REPO_ROOT/checkpoints/$RUN_NAME}"
SAVE_GIF_PATH="${SAVE_GIF_PATH:-$REPO_ROOT/outputs/$RUN_NAME.gif}"
mkdir -p "$(dirname "$SAVE_GIF_PATH")"

TMUX_SESSION="${TMUX_SESSION:-ippo_prep_kitchen_15m_seed${SEED}}"

# jaxlib's CUDA build wants ptxas from the pip-installed nvcc package.
NVCC_BIN="$REPO_ROOT/.venv/lib/python3.10/site-packages/nvidia/cuda_nvcc/bin"

CMD=(
  env
  "PATH=$NVCC_BIN:$PATH"
  "PYTHONPATH=$REPO_ROOT"
  "XLA_PYTHON_CLIENT_PREALLOCATE=false"
  "$PYTHON_BIN" baselines/IPPO/ippo_rnn_overcooked_v3.py
  ENV_KWARGS.layout=prep_kitchen
  "ENV_KWARGS.max_steps=$MAX_STEPS"
  ENV_KWARGS.pot_cook_time=20
  ENV_KWARGS.pot_burn_time=40
  # Alternating queue over the layout's three recipes: chopped lettuce soup,
  # grilled meat soup, carrot puree soup. order_expiration_time=0 disables
  # expiry so agents are never penalised for a slow multi-stage dish.
  ENV_KWARGS.enable_order_queue=true
  ENV_KWARGS.max_orders=5
  ENV_KWARGS.order_generation_rate=1.0
  ENV_KWARGS.order_expiration_time=0
  ENV_KWARGS.order_queue_mode=alternating
  ENV_KWARGS.enable_item_conveyors=false
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
  GRU_HIDDEN_DIM=128
  FC_DIM_SIZE=128
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
  layout      : prep_kitchen (multi-stage orders, 3 prep chains)
  budget      : $TOTAL_TIMESTEPS env steps
  wandb       : $ENTITY/$WANDB_PROJECT/$RUN_NAME
  checkpoint  : $SAVE_PATH
  gif         : $SAVE_GIF_PATH

attach : tmux attach -t $TMUX_SESSION
detach : Ctrl-b d
kill   : tmux kill-session -t $TMUX_SESSION
EOF
