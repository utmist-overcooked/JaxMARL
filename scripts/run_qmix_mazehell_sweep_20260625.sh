#!/usr/bin/env bash
# QMIX-on-mazehell sweep (12 runs, sequential on one GPU).
#   Swept : SHAPED_REWARD_COEFF {1.0,2.5,5.0} x TARGET_UPDATE_INTERVAL {10,200} x NUM_EPOCHS {4,8}
#   Fixed : LR=5e-5, EPS_DECAY=0.3, HIDDEN_SIZE=256, MIXER_EMBEDDING_DIM=64,
#           NUM_STEPS=500, max_steps=1000, TOTAL=15M, NUM_ENVS=4, BUFFER 256/16
#   Early stopping: stop a run once greedy eval reaches >=3 deliveries/episode.
#   GIF: rendered at end of each run as usual (SAVE_GIF_PATH).
# Usage:  bash scripts/run_qmix_mazehell_sweep_20260625.sh        # launches detached tmux
#         bash scripts/run_qmix_mazehell_sweep_20260625.sh inner  # runs the loop (used by tmux)
set -euo pipefail

ROOT=/student/brownd58/dev/JaxMARL
SESSION=jaxmarl_qmix_mazehell_sweep_20260625
PROJECT=ocv3_qlearning_maze_conveyor_hell

if [ "${1:-}" != "inner" ]; then
    tmux new-session -d -s "$SESSION" "bash $ROOT/scripts/run_qmix_mazehell_sweep_20260625.sh inner"
    echo "launched sweep in tmux session: $SESSION"
    exit 0
fi

cd "$ROOT"
export PYTHONPATH=$ROOT:${PYTHONPATH:-}
export PATH=$ROOT/.venv/lib/python3.10/site-packages/nvidia/cuda_nvcc/bin:$PATH
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.99
export PYTHONUNBUFFERED=1
mkdir -p "$ROOT/outputs" "$ROOT/checkpoints"

for COEFF in 1.0 2.5 5.0; do
  for TUI in 10 200; do
    for EPOCHS in 8; do  # epochs=4 dropped: too few grad steps; use 8 only
      TAG="coeff${COEFF//./p}_tui${TUI}_ep${EPOCHS}_20260625"
      echo "=== [$(date)] starting $TAG ==="
      "$ROOT/.venv/bin/python" "$ROOT/baselines/QLearning/qmix_rnn.py" \
        +alg=ql_rnn_overcooked_v3 WANDB_MODE=online PROJECT="$PROJECT" \
        +alg.WANDB_NAME="qmix_mazehell_${TAG}" ENTITY=zacharytang24- SEED=42 \
        SAVE_PATH="$ROOT/checkpoints/qmix_mazehell_sweep_${TAG}" \
        +alg.SAVE_GIF_PATH="$ROOT/outputs/qmix_mazehell_sweep_${TAG}.gif" +alg.GIF_NUM_STEPS=1000 \
        +alg.EARLY_STOP_DELIVERIES=3 \
        alg.ENV_KWARGS.layout=maze_conveyor_hell alg.ENV_KWARGS.agent_view_size=null \
        alg.ENV_KWARGS.max_steps=1000 alg.ENV_KWARGS.pot_cook_time=20 alg.ENV_KWARGS.pot_burn_time=1000000 \
        alg.ENV_KWARGS.enable_order_queue=true alg.ENV_KWARGS.max_orders=5 alg.ENV_KWARGS.order_generation_rate=1.0 \
        alg.ENV_KWARGS.order_expiration_time=0 alg.ENV_KWARGS.recipe_mode=fixed \
        +alg.ENV_KWARGS.plate_pickup_guard=2 alg.ENV_KWARGS.enable_item_conveyors=true alg.ENV_KWARGS.enable_player_conveyors=false \
        +alg.ENV_KWARGS.dish_to_target_agent=1 +alg.ENV_KWARGS.dish_to_target_col=9 \
        +alg.ENV_KWARGS.dish_to_target_row=6 +alg.ENV_KWARGS.dish_to_target_progress_reward=0.1 \
        +alg.ENV_KWARGS.ingredient_to_pot_progress_reward=0.1 +alg.ENV_KWARGS.plate_to_pot_progress_reward=0.1 \
        alg.TOTAL_TIMESTEPS=15000000 alg.NUM_ENVS=4 alg.NUM_STEPS=500 alg.BUFFER_SIZE=128 alg.BUFFER_BATCH_SIZE=8 \
        alg.HIDDEN_SIZE=256 alg.MIXER_EMBEDDING_DIM=64 alg.MIXER_HYPERNET_HIDDEN_DIM=256 alg.MIXER_INIT_SCALE=0.001 \
        alg.EPS_START=1.0 alg.EPS_FINISH=0.05 alg.EPS_DECAY=0.3 alg.MAX_GRAD_NORM=10 \
        alg.TARGET_UPDATE_INTERVAL="${TUI}" alg.TAU=1.0 alg.NUM_EPOCHS="${EPOCHS}" alg.LR=0.00005 \
        alg.LEARNING_STARTS=10000 alg.LR_LINEAR_DECAY=false alg.GAMMA=0.99 alg.REW_SCALE=1.0 \
        alg.SHAPED_REWARD_COEFF="${COEFF}" alg.REW_SHAPING_HORIZON=15000000 alg.REW_SHAPING_MIN_COEFF=0.0 \
        alg.TEST_DURING_TRAINING=true alg.TEST_NUM_ENVS=8 alg.TEST_NUM_STEPS=1000 alg.TEST_INTERVAL=0.05 \
        > "$ROOT/outputs/qmix_mazehell_sweep_${TAG}.log" 2>&1 || echo "!!! run $TAG failed (continuing)"
      echo "=== [$(date)] finished $TAG ==="
    done
  done
done
echo "=== sweep complete ==="
