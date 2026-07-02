#!/bin/bash
#SBATCH --job-name=mappo_full_obs_ctc_harder
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#
# Train an "optimal" full-observation MAPPO parent (teacher) policy on
# coordinated_temporal_conveyor_harder. This is the teacher that the FSQ
# distillation student later distills from.
#
# The goal behaviour to look for in the trained policy:
#   1. the top ("blue") agent first sends ingredients down the conveyor so the
#      bottom ("green") agent can start cooking, then
#   2. while green is busy cooking, blue cooks its own dishes.
#
# Runs one W&B run per seed sequentially so the best seed can be picked as the
# teacher. (Once the SEEDS/WANDB_GROUP change from the multi-seed PR is merged
# into this branch you can replace the loop below with a single call passing
# SEEDS=[0,1,2] and WANDB_GROUP.)
#
# IMPORTANT: OvercookedV3 disables conveyor movement by default, so the
# ++ENV_KWARGS.enable_item_conveyors/enable_player_conveyors overrides below
# are mandatory — without them the belt is static decoration. delivery_reward
# and random_agent_positions=False match the existing CTC run convention.
# (enforce_handoff_roles / wrong_delivery_reward from the cluster scripts are
# NOT passed: those kwargs only exist in the cluster worktree env.)
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$PROJECT_DIR"

if [[ -f "${PROJECT_DIR}/venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "${PROJECT_DIR}/venv/bin/activate"
fi

export PYTHONUNBUFFERED=1

LAYOUT="${LAYOUT:-coordinated_temporal_conveyor_harder}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-5e6}"
REW_SHAPING_HORIZON="${REW_SHAPING_HORIZON:-3e6}"
SHAPED_REWARD_SCALE="${SHAPED_REWARD_SCALE:-1.0}"
MAX_STEPS="${MAX_STEPS:-400}"
DELIVERY_REWARD="${DELIVERY_REWARD:-100.0}"
WANDB_MODE="${WANDB_MODE:-offline}"
WANDB_PROJECT="${WANDB_PROJECT:-overcookedv3-mappo-full-obs}"
WANDB_ENTITY="${WANDB_ENTITY:-null}"
WANDB_GROUP="${WANDB_GROUP:-mappo_v3_full_obs_${LAYOUT}}"
WANDB_DIR="${WANDB_DIR:-${SCRATCH:-.}/jaxmarl/${WANDB_PROJECT}}"
# space separated list of seeds, e.g. "0 1 2"
SEEDS="${SEEDS:-0 1 2}"

mkdir -p "$WANDB_DIR"

for SEED in $SEEDS; do
    echo "=== full-obs teacher | layout=$LAYOUT seed=$SEED ==="
    python "${PROJECT_DIR}/baselines/MAPPO/mappo_rnn_overcooked_v3_full_obs.py" \
        ENV_KWARGS.layout="$LAYOUT" \
        ENV_KWARGS.agent_view_size=null \
        ENV_KWARGS.shaped_rewards=True \
        ENV_KWARGS.max_steps="$MAX_STEPS" \
        ENV_KWARGS.random_agent_positions=False \
        "++ENV_KWARGS.enable_item_conveyors=True" \
        "++ENV_KWARGS.enable_player_conveyors=True" \
        "++ENV_KWARGS.delivery_reward=${DELIVERY_REWARD}" \
        TOTAL_TIMESTEPS="$TOTAL_TIMESTEPS" \
        REW_SHAPING_HORIZON="$REW_SHAPING_HORIZON" \
        SHAPED_REWARD_SCALE="$SHAPED_REWARD_SCALE" \
        SEED="$SEED" \
        NUM_SEEDS=1 \
        WANDB_MODE="$WANDB_MODE" \
        ENTITY="$WANDB_ENTITY" \
        PROJECT="$WANDB_PROJECT" \
        WANDB_DIR="$WANDB_DIR" \
        "++WANDB_RUN_NAME=mappo_v3_full_obs_${LAYOUT}_seed${SEED}" \
        USE_RICH_MONITOR=False
done
