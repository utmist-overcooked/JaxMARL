#!/usr/bin/env bash
# IPPO-RNN on prep_kitchen_handoff for 30M steps.
#
# Supersedes the 15M run (wandb freysm5y, 28-58 deliveries/batch). Two things
# differ beyond the doubled budget:
#   - double the steps, so the shaping horizon is stretched to 30M to match
#   - the grill PREP_PLACEMENT farm is fixed. This map contains a grill, and the
#     15M run trained against the version that handed raw meat back, which made
#     place -> pick up -> place pay out forever. Stations are now commitments.
# Hyperparameters are otherwise the proven handoff recipe (coeff=20, min=0.1,
# entropy 0.1 -> 0.01, seed 42).
set -uo pipefail

REPO=/student/brownd58/dev/JaxMARL
NAME=ippo_prep_kitchen_handoff_30m_20260730

tmux new-session -d -s jaxmarl_ippo_pkh_30m_20260730 \
  "cd $REPO && export PATH=$REPO/.venv/lib/python3.10/site-packages/nvidia/cuda_nvcc/bin:\$PATH && export XLA_PYTHON_CLIENT_MEM_FRACTION=0.99 && PYTHONUNBUFFERED=1 $REPO/.venv/bin/python $REPO/baselines/IPPO/ippo_rnn_overcooked_v3.py --config-name=ippo_rnn_overcooked_v3 WANDB_MODE=online WANDB_PROJECT=ocv3_prep_stations WANDB_NAME=$NAME ENTITY=zacharytang24- ENV_KWARGS.layout=prep_kitchen_handoff ENV_KWARGS.max_steps=400 ENV_KWARGS.pot_cook_time=20 ENV_KWARGS.pot_burn_time=40 ENV_KWARGS.enable_order_queue=false ENV_KWARGS.enable_item_conveyors=false ENV_KWARGS.enable_player_conveyors=false TOTAL_TIMESTEPS=30000000 NUM_ENVS=128 NUM_STEPS=200 UPDATE_EPOCHS=4 NUM_MINIBATCHES=8 LR=0.0005 GAMMA=0.99 GAE_LAMBDA=0.95 CLIP_EPS=0.2 VF_COEF=0.5 ENT_COEF=0.1 ENT_COEF_MIN=0.01 ENTROPY_FLOOR=0.0 ENTROPY_FLOOR_COEF=0.0 MAX_GRAD_NORM=0.5 GRU_HIDDEN_DIM=128 FC_DIM_SIZE=128 ACTIVATION=relu ANNEAL_LR=true REW_SHAPING_HORIZON=30000000 REW_SHAPING_MIN_COEFF=0.1 SHAPED_REWARD_COEFF=20.0 SEED=42 SAVE_PATH=$REPO/checkpoints/$NAME SAVE_GIF_PATH=$REPO/outputs/${NAME}.gif LOG_EVERY=1 CHECKPOINT_EVERY=5000 > $REPO/outputs/${NAME}_train.log 2>&1"

echo "launched tmux jaxmarl_ippo_pkh_30m_20260730"
echo "log: $REPO/outputs/${NAME}_train.log"
