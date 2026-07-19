#!/bin/bash
#SBATCH --job-name=mappo_rnn_overcooked_v3
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

# Project root (read-only on compute, but venv/python still loadable)
PROJECT_DIR=$HOME/links/projects/rrg-cglee/zachtang/JaxMARL

# Activate venv
source ${PROJECT_DIR}/venv/bin/activate

# Change to scratch so SLURM output files and any stray writes land here
cd $SCRATCH

export PYTHONUNBUFFERED=1

WANDB_PROJECT=overcookedv3-mappo-sweep
WANDB_DIR="$SCRATCH/jaxmarl/${WANDB_PROJECT}"
mkdir -p "$WANDB_DIR"

LAYOUTS=("cramped_room_v2" "asymm_advantages_recipes_center" "asymm_advantages_recipes_right" "asymm_advantages_recipes_left" "two_rooms")

for LAYOUT in "${LAYOUTS[@]}"; do
    python ${PROJECT_DIR}/baselines/MAPPO/mappo_rnn_overcooked_v3.py \
        ENV_KWARGS.layout="$LAYOUT" \
        TOTAL_TIMESTEPS=1e7 \
        REW_SHAPING_HORIZON=5e6 \
        WANDB_MODE=offline \
        ENTITY=null \
        PROJECT="$WANDB_PROJECT" \
        WANDB_DIR="$WANDB_DIR" \
        "++WANDB_RUN_NAME=mappo_v3_${LAYOUT}" \
        USE_RICH_MONITOR=False
done
