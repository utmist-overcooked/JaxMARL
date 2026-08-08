#!/usr/bin/env bash
# Sequential IPPO-RNN sweep over every multi-stage prep-station map, 30M steps each.
#
# Runs one after another in a single tmux session (one ~16GB GPU, so never concurrent).
# A failure in one layout is logged and the queue continues to the next.
#
#   env: no order queue, no conveyors, max_steps=400, cook=20, burn=40, default prep
#        timings (chop_stages=3, grill 15+30, blend 10), full obs.
#        The *_room maps have a single fixed recipe; prep_kitchen(_handoff) sample
#        among [[5,5,5],[6,6,6],[7,7,7]] via their R indicator.
#   hyperparams: same recipe that solved prep_kitchen_handoff at 15M (run freysm5y,
#        28-58 deliveries/batch) - SHAPED_REWARD_COEFF=20, MIN_COEFF=0.1,
#        ENT_COEF 0.1 -> 0.01, shaping horizon scaled to the full 30M.
#
# Order: the three single-station rooms first (cheapest signal on whether each prep
# mechanic is learnable at all), then the counter-handoff variants.
# prep_kitchen_handoff runs last - it already has a completed 15M run for comparison.
set -uo pipefail

REPO=/student/brownd58/dev/JaxMARL
PY=$REPO/.venv/bin/python
TAG=30m_20260730
QUEUE_LOG=$REPO/outputs/ippo_prep_all_${TAG}_queue.log

LAYOUTS=(
  cutting_board_room
  grill_room
  blender_room
  prep_kitchen
  cutting_board_handoff
  grill_handoff
  blender_handoff
  prep_kitchen_handoff
)

run_queue() {
  cd "$REPO"
  export PATH=$REPO/.venv/lib/python3.10/site-packages/nvidia/cuda_nvcc/bin:$PATH
  export XLA_PYTHON_CLIENT_MEM_FRACTION=0.99
  export PYTHONUNBUFFERED=1

  echo "[$(date +%F_%T)] queue start: ${#LAYOUTS[@]} layouts x 30M steps" >> "$QUEUE_LOG"

  for i in "${!LAYOUTS[@]}"; do
    LAYOUT=${LAYOUTS[$i]}
    NAME=ippo_${LAYOUT}_${TAG}
    LOG=$REPO/outputs/${NAME}_train.log

    echo "[$(date +%F_%T)] ($((i+1))/${#LAYOUTS[@]}) starting $LAYOUT -> $LOG" >> "$QUEUE_LOG"

    "$PY" "$REPO/baselines/IPPO/ippo_rnn_overcooked_v3.py" \
      --config-name=ippo_rnn_overcooked_v3 \
      WANDB_MODE=online \
      WANDB_PROJECT=ocv3_prep_stations \
      WANDB_NAME="$NAME" \
      ENTITY=zacharytang24- \
      ENV_KWARGS.layout="$LAYOUT" \
      ENV_KWARGS.max_steps=400 \
      ENV_KWARGS.pot_cook_time=20 \
      ENV_KWARGS.pot_burn_time=40 \
      ENV_KWARGS.enable_order_queue=false \
      ENV_KWARGS.enable_item_conveyors=false \
      ENV_KWARGS.enable_player_conveyors=false \
      TOTAL_TIMESTEPS=30000000 \
      NUM_ENVS=128 NUM_STEPS=200 UPDATE_EPOCHS=4 NUM_MINIBATCHES=8 \
      LR=0.0005 GAMMA=0.99 GAE_LAMBDA=0.95 CLIP_EPS=0.2 VF_COEF=0.5 \
      ENT_COEF=0.1 ENT_COEF_MIN=0.01 ENTROPY_FLOOR=0.0 ENTROPY_FLOOR_COEF=0.0 \
      MAX_GRAD_NORM=0.5 GRU_HIDDEN_DIM=128 FC_DIM_SIZE=128 ACTIVATION=relu \
      ANNEAL_LR=true \
      REW_SHAPING_HORIZON=30000000 REW_SHAPING_MIN_COEFF=0.1 SHAPED_REWARD_COEFF=20.0 \
      SEED=42 \
      SAVE_PATH="$REPO/checkpoints/$NAME" \
      SAVE_GIF_PATH="$REPO/outputs/${NAME}.gif" \
      LOG_EVERY=1 CHECKPOINT_EVERY=5000 \
      > "$LOG" 2>&1

    STATUS=$?
    if [ $STATUS -eq 0 ]; then
      echo "[$(date +%F_%T)] ($((i+1))/${#LAYOUTS[@]}) DONE $LAYOUT" >> "$QUEUE_LOG"
    else
      echo "[$(date +%F_%T)] ($((i+1))/${#LAYOUTS[@]}) FAILED $LAYOUT (exit $STATUS) - see $LOG" >> "$QUEUE_LOG"
    fi
  done

  echo "[$(date +%F_%T)] queue finished" >> "$QUEUE_LOG"
}

# Re-exec inside tmux unless already there (allows `bash script.sh` to detach itself).
if [ "${INSIDE_QUEUE_TMUX:-0}" = "1" ]; then
  run_queue
else
  tmux new-session -d -s jaxmarl_ippo_prep_all_${TAG} \
    "INSIDE_QUEUE_TMUX=1 bash $REPO/scripts/run_ippo_rnn_v3_prep_all_30m_20260730.sh"
  echo "launched tmux session jaxmarl_ippo_prep_all_${TAG}"
  echo "queue log: $QUEUE_LOG"
fi
