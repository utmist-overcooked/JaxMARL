#!/bin/bash
# Example: running a CARBS hyperparameter sweep locally.
#
# CARBS sweeps hyperparameters defined in the CARBS_SWEEP section of
# baselines/IPPO/config/ippo_rnn_overcooked_v3.yaml. To add/remove params
# from the sweep, comment/uncomment them in that YAML file.
#
# The search centers come from the YAML defaults (e.g. LR: 0.00025),
# and any Hydra overrides below take precedence.
#
# WHEN RUNNING THIS MAKE SURE TO CHECK WHERE RESULTS ARE SAVED TO.

cd "$(dirname "$0")/.."
source venv/bin/activate

export PYTHONUNBUFFERED=1

python baselines/IPPO/ippo_rnn_overcooked_v3.py \
    --config-name ippo_rnn_overcooked_v3 \
    TUNE=True \
    CARBS_NUM_TRIALS=20 \
    NUM_ENVS=512 \
    TOTAL_TIMESTEPS=1e7 \
    REW_SHAPING_HORIZON=5e6 \
    ENV_KWARGS.layout=cramped_room \
    ENV_KWARGS.agent_view_size=null \
    WANDB_MODE=disabled \
    USE_RICH_MONITOR=False
