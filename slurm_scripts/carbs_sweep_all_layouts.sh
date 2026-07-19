#!/bin/bash
#SBATCH --job-name=carbs-all-layouts
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=12:00:00
#SBATCH --account=rrg-cglee
#SBATCH --output=/scratch/zachtang/jaxmarl/logs/carbs_sweep_all_layouts_%j.out

cd /project/rrg-cglee/zachtang/JaxMARL
source venv/bin/activate

export PYTHONUNBUFFERED=1

LAYOUTS=(
    cramped_room
    asymm_advantages
    coord_ring
    forced_coord
    counter_circuit
    cramped_room_v2
    conveyor_demo
    player_conveyor_demo
    player_conveyor_loop
    middle_conveyor
    follow_the_leader
    around_the_island
    around_the_island_nerfed
    single_file
)

for LAYOUT in "${LAYOUTS[@]}"; do
    echo "============================================================"
    echo "Starting CARBS sweep on layout: $LAYOUT"
    echo "============================================================"

    python baselines/IPPO/ippo_rnn_overcooked_v3.py \
        --config-name ippo_rnn_overcooked_v3 \
        TUNE=True \
        CARBS_NUM_TRIALS=20 \
        NUM_ENVS=512 \
        TOTAL_TIMESTEPS=1e7 \
        REW_SHAPING_HORIZON=5e6 \
        ENV_KWARGS.layout="$LAYOUT" \
        ENV_KWARGS.agent_view_size=null \
        WANDB_MODE=disabled \
        USE_RICH_MONITOR=False

    echo "Finished layout: $LAYOUT"
    echo ""
done

echo "All layouts complete."
