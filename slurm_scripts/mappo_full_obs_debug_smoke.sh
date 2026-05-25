#!/bin/bash
set -euo pipefail

PROJECT_DIR=$HOME/links/projects/rrg-cglee/zachtang/JaxMARL
WANDB_DIR=$SCRATCH/jaxmarl/full_obs_debug_smoke

if [[ ! -d "$WANDB_DIR" || ! -d "$WANDB_DIR/models" ]]; then
    echo "Missing pre-created output directories under $WANDB_DIR"
    exit 1
fi

# The repo's venv activation script calls `module load ...`; in debugjob this
# can disturb the CUDA environment. Keep the venv PATH change but skip modules.
module() { return 0; }
source ${PROJECT_DIR}/venv/bin/activate
cd "$SCRATCH"

export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_PLATFORMS=cuda,cpu

echo "python: $(which python)"
python - <<'PY'
import sys
print(sys.executable)
PY
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv,noheader

LAYOUTS=("cramped_room" "asymm_advantages_recipes_right")

for LAYOUT in "${LAYOUTS[@]}"; do
    echo
    echo "=== full-obs smoke layout=${LAYOUT} ==="
    python ${PROJECT_DIR}/baselines/MAPPO/mappo_rnn_overcooked_v3_full_obs.py \
        hydra.run.dir="$WANDB_DIR" \
        hydra.output_subdir=null \
        ENV_KWARGS.layout="$LAYOUT" \
        TOTAL_TIMESTEPS=524288 \
        NUM_ENVS=2048 \
        NUM_STEPS=256 \
        NUM_MINIBATCHES=64 \
        LR=0.002 \
        WANDB_MODE=disabled \
        ENTITY=null \
        PROJECT=null \
        WANDB_DIR="$WANDB_DIR" \
        "++WANDB_RUN_NAME=debug_full_obs_smoke_${LAYOUT}" \
        USE_RICH_MONITOR=False \
        "++DISABLE_CHECKPOINTS=True"
done
