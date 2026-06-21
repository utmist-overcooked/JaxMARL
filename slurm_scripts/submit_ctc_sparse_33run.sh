#!/bin/bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/tangzach/JaxMARL/.worktrees/codex/overcooked-fsq-distill}"
SBATCH_SCRIPT="${PROJECT_DIR}/slurm_scripts/mappo_rnn_overcooked_v3_ctc_sparse_single.sbatch"
OUTPUT_ROOT="${OUTPUT_ROOT:-/scratch/tangzach/jaxmarl/ctc_sparse_11run}"
SEEDS="${SEEDS:-0 1 2}"

cd "$PROJECT_DIR"
mkdir -p outputs/logs "$OUTPUT_ROOT"

submit_variant() {
    local variant_idx="$1"
    local variant_name="$2"
    local disable_fsq="$3"
    local agent_view_size="$4"
    local fsq_levels="$5"
    local distill_coef="$6"
    local decay_fraction="$7"
    local temperature="$8"

    local run_name_prefix
    if [[ "$disable_fsq" == "True" ]]; then
        run_name_prefix="sparse_nofsq_ctc_harder_${variant_name}"
    else
        run_name_prefix="sparse_fsq_ctc_harder_${variant_idx}_${variant_name}"
    fi

    echo "Submitting ${run_name_prefix} seeds: ${SEEDS}"
    OUTPUT_ROOT="$OUTPUT_ROOT" \
    RUN_NAME_PREFIX="$run_name_prefix" \
    SEEDS="$SEEDS" \
    DISABLE_FSQ_COMM="$disable_fsq" \
    AGENT_VIEW_SIZE="$agent_view_size" \
    FSQ_LEVELS="$fsq_levels" \
    DISTILL_COEF="$distill_coef" \
    DISTILL_DECAY_FRACTION="$decay_fraction" \
    DISTILL_TEMPERATURE="$temperature" \
    SHAPED_REWARD_SCALE=0.0 \
    sbatch --parsable "$SBATCH_SCRIPT"
}

submit_variant 0 baseline False 2 "[5,5,5]" 1.0 0.30 1.0
submit_variant 2 stronger_distill False 2 "[5,5,5]" 2.0 0.30 1.0
submit_variant 3 longer_distill False 2 "[5,5,5]" 1.0 0.60 1.0
submit_variant 4 strong_long_distill False 2 "[5,5,5]" 2.0 0.60 1.0
submit_variant 5 soft_teacher False 2 "[5,5,5]" 1.0 0.60 2.0
submit_variant 6 tiny_channel False 2 "[3,3,3]" 1.0 0.60 1.0
submit_variant 7 big_channel False 2 "[7,7,7]" 1.0 0.60 1.0
submit_variant 8 larger_partial_view False 3 "[5,5,5]" 1.0 0.60 1.0
submit_variant 9 no_distill_control False 2 "[5,5,5]" 0.0 0.30 1.0
submit_variant nofsq strong_long True 2 "[5,5,5]" 2.0 0.60 1.0
submit_variant nofsq distill05 True 2 "[5,5,5]" 0.5 0.60 1.0
