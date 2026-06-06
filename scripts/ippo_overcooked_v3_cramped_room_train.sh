#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CURRENT_TIME=$(date +"%Y%m%d_%H%M%S")

LAYOUT="${LAYOUT:-cramped_room}"
POT_COOK_TIME="${POT_COOK_TIME:-20}"
POT_BURN_TIME="${POT_BURN_TIME:-$((POT_COOK_TIME * 2))}"
WANDB_PROJECT="${WANDB_PROJECT:-overcookedv3_ippo_rnn_${LAYOUT}_order_visible_plateguard_burn${POT_BURN_TIME}_${CURRENT_TIME}}"
WANDB_NAME="${WANDB_NAME:-ippo_rnn_overcooked_v3_${LAYOUT}_order_visible_plateguard_burn${POT_BURN_TIME}}"
SAVE_PATH="${SAVE_PATH:-$ROOT_DIR/checkpoints/ippo_rnn_v3_${LAYOUT}_order_visible_plateguard_burn${POT_BURN_TIME}_${CURRENT_TIME}}"
SAVE_GIF_PATH="${SAVE_GIF_PATH:-$ROOT_DIR/outputs/ippo_rnn_v3_${LAYOUT}_order_visible_plateguard_burn${POT_BURN_TIME}_${CURRENT_TIME}.gif}"

exec env \
  LAYOUT="$LAYOUT" \
  POT_COOK_TIME="$POT_COOK_TIME" \
  POT_BURN_TIME="$POT_BURN_TIME" \
  WANDB_PROJECT="$WANDB_PROJECT" \
  WANDB_NAME="$WANDB_NAME" \
  SAVE_PATH="$SAVE_PATH" \
  SAVE_GIF_PATH="$SAVE_GIF_PATH" \
  "$ROOT_DIR/scripts/ippo_rnn_overcooked_v3_layout_train.sh"
