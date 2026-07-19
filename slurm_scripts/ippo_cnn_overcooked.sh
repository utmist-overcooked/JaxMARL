#!/bin/bash
#SBATCH --job-name=ippo_cnn_overcooked
#SBATCH --account=rrg-cglee
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

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

# Run training -- pass Hydra overrides as args to this script
# Examples:
#   sbatch slurm_scripts/ippo_cnn_overcooked.sh
#   sbatch slurm_scripts/ippo_cnn_overcooked.sh TUNE=False
#   sbatch slurm_scripts/ippo_cnn_overcooked.sh CARBS_NUM_TRIALS=100 TOTAL_TIMESTEPS=2e7
python ${PROJECT_DIR}/baselines/IPPO/ippo_cnn_overcooked.py "$@"
