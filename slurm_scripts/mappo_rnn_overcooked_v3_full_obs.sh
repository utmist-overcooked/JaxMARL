#!/bin/bash
#SBATCH --job-name=mappo_rnn_overcooked_v3_full_obs
#SBATCH --account=rrg-cglee
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/zachtang/jaxmarl/logs/%x_%j.out
#SBATCH --error=/scratch/zachtang/jaxmarl/logs/%x_%j.err

# Trillium notes:
#   - $HOME and $PROJECT are read-only on compute nodes
#   - All output must go to $SCRATCH
#   - --mem is ignored (1 GPU = ~188 GiB)
#   - Must request exactly 1 or 4 GPUs (--gpus-per-node, not --gres)
#   - Pre-create scratch dirs on the login node before submitting:
#       mkdir -p $SCRATCH/jaxmarl/logs $SCRATCH/jaxmarl/overcookedv3-mappo-full-obs-sweep

PROJECT_DIR=$HOME/links/projects/rrg-cglee/zachtang/JaxMARL

source ${PROJECT_DIR}/venv/bin/activate

cd $SCRATCH

export PYTHONUNBUFFERED=1

WANDB_PROJECT=overcookedv3-mappo-full-obs-sweep
WANDB_DIR="$SCRATCH/jaxmarl/${WANDB_PROJECT}"

if [[ ! -d "$SCRATCH/jaxmarl/logs" || ! -d "$WANDB_DIR" ]]; then
    echo "Missing scratch output directories."
    echo "Run this on the login node before sbatch:"
    echo "  mkdir -p $SCRATCH/jaxmarl/logs $WANDB_DIR"
    exit 1
fi

LAYOUTS=("cramped_room_v2" "asymm_advantages_recipes_center" "asymm_advantages_recipes_right" "asymm_advantages_recipes_left" "two_rooms")

for LAYOUT in "${LAYOUTS[@]}"; do
    python ${PROJECT_DIR}/baselines/MAPPO/mappo_rnn_overcooked_v3_full_obs.py \
        ENV_KWARGS.layout="$LAYOUT" \
        TOTAL_TIMESTEPS=1e7 \
        REW_SHAPING_HORIZON=5e6 \
        WANDB_MODE=offline \
        ENTITY=null \
        PROJECT="$WANDB_PROJECT" \
        WANDB_DIR="$WANDB_DIR" \
        "++WANDB_RUN_NAME=mappo_v3_full_obs_${LAYOUT}" \
        USE_RICH_MONITOR=False
done
