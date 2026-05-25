#!/bin/bash
#SBATCH --job-name=mappo_rnn_overcooked_v3_full_obs_verify
#SBATCH --account=rrg-cglee
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/zachtang/jaxmarl/logs/%x_%j.out
#SBATCH --error=/scratch/zachtang/jaxmarl/logs/%x_%j.err

# Trillium notes:
#   - $HOME and $PROJECT are read-only on compute nodes
#   - All output must go to $SCRATCH
#   - Pre-create scratch dirs on the login node before submitting:
#       mkdir -p $SCRATCH/jaxmarl/logs $SCRATCH/jaxmarl/overcookedv3-mappo-full-obs-verify

PROJECT_DIR=$HOME/links/projects/rrg-cglee/zachtang/JaxMARL

source ${PROJECT_DIR}/venv/bin/activate

cd $SCRATCH

export PYTHONUNBUFFERED=1

WANDB_PROJECT=overcookedv3-mappo-full-obs-verify
WANDB_DIR="$SCRATCH/jaxmarl/${WANDB_PROJECT}"

if [[ ! -d "$SCRATCH/jaxmarl/logs" || ! -d "$WANDB_DIR" ]]; then
    echo "Missing scratch output directories."
    echo "Run this on the login node before sbatch:"
    echo "  mkdir -p $SCRATCH/jaxmarl/logs $WANDB_DIR"
    exit 1
fi

LAYOUTS=("cramped_room" "asymm_advantages_recipes_right")

for LAYOUT in "${LAYOUTS[@]}"; do
    python ${PROJECT_DIR}/baselines/MAPPO/mappo_rnn_overcooked_v3_full_obs.py \
        ENV_KWARGS.layout="$LAYOUT" \
        TOTAL_TIMESTEPS=5000000 \
        NUM_ENVS=2048 \
        LR=0.002 \
        WANDB_MODE=offline \
        ENTITY=null \
        PROJECT="$WANDB_PROJECT" \
        WANDB_DIR="$WANDB_DIR" \
        "++WANDB_RUN_NAME=mappo_v3_full_obs_verify_${LAYOUT}" \
        USE_RICH_MONITOR=False
done
