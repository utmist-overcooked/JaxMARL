#!/usr/bin/env bash
# Rerun grill_room at 30M after the PREP_PLACEMENT farm fix.
#
# The original grill_room run (ippo_grill_room_30m_20260730) trained against the
# buggy grill, which handed raw meat back on early pickup. place -> pick up ->
# place then paid PREP_PLACEMENT every two steps forever, so the policy farmed
# pickups (~12.5k/batch) and never delivered. Stations are now commitments, so
# placement pays once per unit. Everything else is identical to the sweep.
#
# Waits for the main sweep's tmux session to end so the two never share the GPU.
set -uo pipefail

REPO=/student/brownd58/dev/JaxMARL
NAME=ippo_grill_room_30m_fixed_20260730
QUEUE_LOG=$REPO/outputs/ippo_prep_all_30m_20260730_queue.log

run_it() {
  cd "$REPO"
  export PATH=$REPO/.venv/lib/python3.10/site-packages/nvidia/cuda_nvcc/bin:$PATH
  export XLA_PYTHON_CLIENT_MEM_FRACTION=0.99
  export PYTHONUNBUFFERED=1

  while tmux has-session -t jaxmarl_ippo_prep_all_30m_20260730 2>/dev/null; do sleep 60; done

  echo "[$(date +%F_%T)] starting grill_room RERUN (post-fix) -> $NAME" >> "$QUEUE_LOG"
  "$REPO/.venv/bin/python" "$REPO/baselines/IPPO/ippo_rnn_overcooked_v3.py" \
    --config-name=ippo_rnn_overcooked_v3 \
    WANDB_MODE=online WANDB_PROJECT=ocv3_prep_stations WANDB_NAME="$NAME" \
    ENTITY=zacharytang24- \
    ENV_KWARGS.layout=grill_room ENV_KWARGS.max_steps=400 \
    ENV_KWARGS.pot_cook_time=20 ENV_KWARGS.pot_burn_time=40 \
    ENV_KWARGS.enable_order_queue=false \
    ENV_KWARGS.enable_item_conveyors=false ENV_KWARGS.enable_player_conveyors=false \
    TOTAL_TIMESTEPS=30000000 NUM_ENVS=128 NUM_STEPS=200 UPDATE_EPOCHS=4 NUM_MINIBATCHES=8 \
    LR=0.0005 GAMMA=0.99 GAE_LAMBDA=0.95 CLIP_EPS=0.2 VF_COEF=0.5 \
    ENT_COEF=0.1 ENT_COEF_MIN=0.01 ENTROPY_FLOOR=0.0 ENTROPY_FLOOR_COEF=0.0 \
    MAX_GRAD_NORM=0.5 GRU_HIDDEN_DIM=128 FC_DIM_SIZE=128 ACTIVATION=relu ANNEAL_LR=true \
    REW_SHAPING_HORIZON=30000000 REW_SHAPING_MIN_COEFF=0.1 SHAPED_REWARD_COEFF=20.0 \
    SEED=42 \
    SAVE_PATH="$REPO/checkpoints/$NAME" SAVE_GIF_PATH="$REPO/outputs/${NAME}.gif" \
    LOG_EVERY=1 CHECKPOINT_EVERY=5000 \
    > "$REPO/outputs/${NAME}_train.log" 2>&1
  echo "[$(date +%F_%T)] grill_room RERUN exit=$? " >> "$QUEUE_LOG"
}

if [ "${INSIDE_RERUN_TMUX:-0}" = "1" ]; then
  run_it
else
  tmux new-session -d -s jaxmarl_ippo_grill_rerun_20260730 \
    "INSIDE_RERUN_TMUX=1 bash $REPO/scripts/run_ippo_rnn_v3_grill_room_30m_rerun_20260730.sh"
  echo "queued grill_room rerun (waits for main sweep to finish)"
fi
