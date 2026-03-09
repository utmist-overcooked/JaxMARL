#!/bin/bash
cd /workspace/JaxMARL
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
python baselines/IPPO/ippo_rnn_overcooked_v3.py \
    TOTAL_TIMESTEPS=2_000_000 \
    NUM_ENVS=64 \
    REW_SHAPING_HORIZON=1_000_000 \
    > /tmp/ippo_v3_smoke.log 2>&1
echo "SMOKE_TEST_DONE exit=$?" >> /tmp/ippo_v3_smoke.log
