# Overcooked V3 IPPO CNN/RNN 15M Sweep Runbook

This runbook documents how to run and read the current Overcooked V3 IPPO CNN
and IPPO RNN experiments. It is written for the May 24, 2026 configuration where
the L2 delivery-distance reward is off and `PLATE_PICKUP_DURING_COOKING` is off.

## Current Experiment Setup

Core training setup:

- Repository: `/student/brownd58/dev/JaxMARL`
- Environment: `overcooked_v3`
- CNN trainer: `baselines/IPPO/ippo_cnn_overcooked_v3.py`
- RNN trainer: `baselines/IPPO/ippo_rnn_overcooked_v3.py`
- CNN single-layout launcher: `scripts/ippo_overcooked_v3_around_the_island_optimal_train.sh`
- CNN sweep launcher: `scripts/ippo_overcooked_v3_all_other_maps_15m_train.sh`
- RNN single-layout launcher: `scripts/ippo_rnn_overcooked_v3_layout_train.sh`
- RNN sweep launcher: `scripts/ippo_rnn_overcooked_v3_all_other_maps_15m_train.sh`
- Python: `/student/brownd58/dev/JaxMARL/.venv/bin/python`
- JAX platform: `cuda,cpu`
- CUDA `ptxas`: `.venv/lib/python3.10/site-packages/nvidia/cuda_nvcc/bin/ptxas`

Training budget:

- `TOTAL_TIMESTEPS=15000000`
- `REW_SHAPING_HORIZON=15000000`
- `REW_SHAPING_MIN_COEFF=0.10`
- `MAX_STEPS=1000`
- `NUM_ENVS=128`
- `NUM_STEPS=200`
- Expected PPO updates: `585`
- Actual env steps per run: `14,976,000`

Environment mechanics:

- Pot cook time: `20`
- Pot burn time: `0`
- Burn penalty: `0.0`
- Order queue: disabled
- Order expiration: disabled
- Around-the-island plates were moved beside the pot in the layout.

Current reward structure:

| Reward | Raw value | Active in learning | Notes |
| --- | ---: | --- | --- |
| `DELIVERY_REWARD` | `20.0` | yes | Sparse correct-delivery reward. |
| `INGREDIENT_PICKUP` | `0.1` | yes | Guarded to avoid ingredient pickup/drop farming. |
| `PLACEMENT_IN_POT` | `0.2` | yes | Paid for useful recipe ingredient placement. |
| `SOUP_IN_DISH` | `0.6` | yes | Paid when a plate picks up finished soup from pot. |
| `PLATE_PICKUP` | `0.1` | yes | Guarded to only pay when a plate is useful. |
| `PLATE_PICKUP_DURING_COOKING` | `0.0` | no | Disabled ablation. |
| `DISH_TO_GOAL_PROGRESS` | `0.0` | no | L2/delivery-distance progress is logged only. |
| `POT_START_COOKING` | `0.2` | no | Present in settings, but not added as reward by env. |

Learning reward formula:

```text
learning_reward =
  sparse_delivery_reward
  + SHAPED_REWARD_COEFF * shaped_reward * anneal_factor

anneal_factor =
  REW_SHAPING_MIN_COEFF
  + (1 - REW_SHAPING_MIN_COEFF)
  * linear_decay(env_step, REW_SHAPING_HORIZON)
```

For these runs, `SHAPED_REWARD_COEFF=30.0`, so an active raw shaped reward of
`0.6` starts with an effective value of `18.0` and decays to a floor value of
`1.8`.

## W&B Reward Structure Logging

Future IPPO CNN and RNN V3 runs automatically log reward structure to W&B:

- `reward_structure` in W&B config
- `reward_structure` in W&B summary
- `reward_structure/table` at run start
- scalar values like `reward_structure/raw/soup_in_dish`
- active flags like `reward_structure/active/dish_to_goal_progress`

Use this to confirm whether a reward was actually active in learning. For the
current ablation, W&B should show:

- `reward_structure/active/dish_to_goal_progress = 0`
- `reward_structure/active/plate_pickup_during_cooking = 0`
- `reward_structure/pot_burn_time = 0`
- `reward_structure/burn_enabled = 0`

## Run One Layout

Use the single-layout shell scripts for a specific layout. Both scripts accept
`LAYOUT`, conveyor flags, and output overrides.

CNN example for `cramped_room`:

```bash
cd /student/brownd58/dev/JaxMARL

PYTHON_BIN=/student/brownd58/dev/JaxMARL/.venv/bin/python \
LAYOUT=cramped_room \
TOTAL_TIMESTEPS=15000000 \
REW_SHAPING_HORIZON=15000000 \
REW_SHAPING_MIN_COEFF=0.10 \
MAX_STEPS=1000 \
WANDB_PROJECT=overcookedv3_ippo_cnn_cramped_room_manual_$(date +"%Y%m%d_%H%M%S") \
WANDB_NAME=ippo_cnn_overcooked_v3_cramped_room_manual \
scripts/ippo_overcooked_v3_around_the_island_optimal_train.sh
```

RNN example for `around_the_island`:

```bash
cd /student/brownd58/dev/JaxMARL

PYTHON_BIN=/student/brownd58/dev/JaxMARL/.venv/bin/python \
LAYOUT=around_the_island \
TOTAL_TIMESTEPS=15000000 \
REW_SHAPING_HORIZON=15000000 \
REW_SHAPING_MIN_COEFF=0.10 \
MAX_STEPS=1000 \
WANDB_PROJECT=overcookedv3_ippo_rnn_around_the_island_manual_$(date +"%Y%m%d_%H%M%S") \
WANDB_NAME=ippo_rnn_overcooked_v3_around_the_island_manual \
scripts/ippo_rnn_overcooked_v3_layout_train.sh
```

For conveyor maps, turn on the matching conveyor mechanic:

```bash
ENABLE_ITEM_CONVEYORS=true LAYOUT=middle_conveyor scripts/ippo_overcooked_v3_around_the_island_optimal_train.sh
ENABLE_PLAYER_CONVEYORS=true LAYOUT=player_conveyor_demo scripts/ippo_overcooked_v3_around_the_island_optimal_train.sh
ENABLE_ITEM_CONVEYORS=true LAYOUT=middle_conveyor scripts/ippo_rnn_overcooked_v3_layout_train.sh
ENABLE_PLAYER_CONVEYORS=true LAYOUT=player_conveyor_demo scripts/ippo_rnn_overcooked_v3_layout_train.sh
```

## Run The All-Other-Maps Sweep

The sweep excludes `around_the_island` because we have been testing that map
separately. It runs these layouts sequentially:

```text
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
single_file
```

Launch the CNN sweep directly:

```bash
cd /student/brownd58/dev/JaxMARL

PYTHON_BIN=/student/brownd58/dev/JaxMARL/.venv/bin/python \
scripts/ippo_overcooked_v3_all_other_maps_15m_train.sh
```

Launch the RNN sweep directly:

```bash
cd /student/brownd58/dev/JaxMARL

PYTHON_BIN=/student/brownd58/dev/JaxMARL/.venv/bin/python \
scripts/ippo_rnn_overcooked_v3_all_other_maps_15m_train.sh
```

Launch the CNN sweep in tmux:

```bash
cd /student/brownd58/dev/JaxMARL

SWEEP_ID=$(date +"%Y%m%d_%H%M%S")
SWEEP_TAG=farmguard_nodist_no_cookplate_floor10_15m
SESSION_NAME=jaxmarl_v3_maps_15m_${SWEEP_ID}
SWEEP_DIR=/student/brownd58/dev/JaxMARL/outputs/v3_map_sweep_${SWEEP_TAG}_${SWEEP_ID}
SWEEP_PROJECT=overcookedv3_ippo_cnn_all_other_maps_${SWEEP_TAG}_${SWEEP_ID}

mkdir -p "$SWEEP_DIR"

tmux new-session -d -s "$SESSION_NAME" \
  "cd /student/brownd58/dev/JaxMARL && \
   PYTHON_BIN=/student/brownd58/dev/JaxMARL/.venv/bin/python \
   SWEEP_ID='$SWEEP_ID' \
   SWEEP_TAG='$SWEEP_TAG' \
   SWEEP_PROJECT='$SWEEP_PROJECT' \
   SWEEP_DIR='$SWEEP_DIR' \
   scripts/ippo_overcooked_v3_all_other_maps_15m_train.sh \
   > '$SWEEP_DIR/sweep.log' 2>&1"
```

Launch the RNN sweep in tmux by changing the tag/project/launcher:

```bash
SWEEP_ID=$(date +"%Y%m%d_%H%M%S")
SWEEP_TAG=rnn_farmguard_nodist_no_cookplate_floor10_15m
SESSION_NAME=jaxmarl_v3_rnn_maps_15m_${SWEEP_ID}
SWEEP_DIR=/student/brownd58/dev/JaxMARL/outputs/v3_rnn_map_sweep_${SWEEP_TAG}_${SWEEP_ID}
SWEEP_PROJECT=overcookedv3_ippo_rnn_all_other_maps_${SWEEP_TAG}_${SWEEP_ID}

tmux new-session -d -s "$SESSION_NAME" \
  "cd /student/brownd58/dev/JaxMARL && \
   PYTHON_BIN=/student/brownd58/dev/JaxMARL/.venv/bin/python \
   SWEEP_ID='$SWEEP_ID' \
   SWEEP_TAG='$SWEEP_TAG' \
   SWEEP_PROJECT='$SWEEP_PROJECT' \
   SWEEP_DIR='$SWEEP_DIR' \
   scripts/ippo_rnn_overcooked_v3_all_other_maps_15m_train.sh \
   > '$SWEEP_DIR/sweep.log' 2>&1"
```

## Completed Sweeps

CNN all-other-maps sweep:

- W&B project: `overcookedv3_ippo_cnn_all_other_maps_farmguard_nodist_no_cookplate_floor10_15m_20260524_014811`
- W&B URL: `https://wandb.ai/zacharytang24-/overcookedv3_ippo_cnn_all_other_maps_farmguard_nodist_no_cookplate_floor10_15m_20260524_014811`
- Sweep directory: `/student/brownd58/dev/JaxMARL/outputs/v3_map_sweep_farmguard_nodist_no_cookplate_floor10_15m_20260524_014811`

RNN all-other-maps sweep:

- W&B project: `overcookedv3_ippo_rnn_all_other_maps_rnn_farmguard_nodist_no_cookplate_floor10_15m_20260524_024519`
- W&B URL: `https://wandb.ai/zacharytang24-/overcookedv3_ippo_rnn_all_other_maps_rnn_farmguard_nodist_no_cookplate_floor10_15m_20260524_024519`
- Sweep directory: `/student/brownd58/dev/JaxMARL/outputs/v3_rnn_map_sweep_rnn_farmguard_nodist_no_cookplate_floor10_15m_20260524_024519`

RNN around-the-island 15M run:

- W&B project: `overcookedv3_ippo_rnn_around_the_island_rnn_farmguard_nodist_no_cookplate_floor10_15m_20260524_081512`
- W&B URL: `https://wandb.ai/zacharytang24-/overcookedv3_ippo_rnn_around_the_island_rnn_farmguard_nodist_no_cookplate_floor10_15m_20260524_081512/runs/w1vseg8t`
- Log: `/student/brownd58/dev/JaxMARL/outputs/ippo_rnn_v3_around_the_island_rnn_farmguard_nodist_no_cookplate_floor10_15m_20260524_081512_train.log`
- Checkpoint: `/student/brownd58/dev/JaxMARL/checkpoints/ippo_rnn_v3_around_the_island_rnn_farmguard_nodist_no_cookplate_floor10_15m_20260524_081512/model.msgpack`
- GIF: `/student/brownd58/dev/JaxMARL/outputs/ippo_rnn_v3_around_the_island_rnn_farmguard_nodist_no_cookplate_floor10_15m_20260524_081512.gif`

First attempted sweep to ignore:

- Sweep id: `20260524_014712`
- Issue: Hydra rejected conveyor kwargs because the script used `ENV_KWARGS.enable_item_conveyors` instead of `+ENV_KWARGS.enable_item_conveyors`.
- Fix: the main launcher now uses `+ENV_KWARGS.enable_item_conveyors` and `+ENV_KWARGS.enable_player_conveyors`.

## Monitor A Sweep

Check tmux:

```bash
tmux ls
tmux attach -t <session_name>
```

Detach from tmux without stopping training:

```text
Ctrl-b d
```

Check sweep status:

```bash
tail -n 50 /student/brownd58/dev/JaxMARL/outputs/v3_map_sweep_farmguard_nodist_no_cookplate_floor10_15m_20260524_014811/status.log
tail -n 50 /student/brownd58/dev/JaxMARL/outputs/v3_rnn_map_sweep_rnn_farmguard_nodist_no_cookplate_floor10_15m_20260524_024519/status.log
```

Check a layout log:

```bash
tail -n 80 /student/brownd58/dev/JaxMARL/outputs/v3_map_sweep_farmguard_nodist_no_cookplate_floor10_15m_20260524_014811/cramped_room.log
tail -n 80 /student/brownd58/dev/JaxMARL/outputs/v3_rnn_map_sweep_rnn_farmguard_nodist_no_cookplate_floor10_15m_20260524_024519/cramped_room.log
```

Watch logs live:

```bash
tail -f /student/brownd58/dev/JaxMARL/outputs/v3_map_sweep_farmguard_nodist_no_cookplate_floor10_15m_20260524_014811/status.log
tail -f /student/brownd58/dev/JaxMARL/outputs/v3_rnn_map_sweep_rnn_farmguard_nodist_no_cookplate_floor10_15m_20260524_024519/status.log
```

## Artifact Locations

For sweep id `20260524_014811`, each layout writes:

```text
Log:
/student/brownd58/dev/JaxMARL/outputs/v3_map_sweep_farmguard_nodist_no_cookplate_floor10_15m_20260524_014811/<layout>.log

Checkpoint:
/student/brownd58/dev/JaxMARL/checkpoints/ippo_cnn_v3_<layout>_farmguard_nodist_no_cookplate_floor10_15m_20260524_014811/model.msgpack

GIF:
/student/brownd58/dev/JaxMARL/outputs/ippo_cnn_v3_<layout>_farmguard_nodist_no_cookplate_floor10_15m_20260524_014811.gif
```

Examples:

```text
/student/brownd58/dev/JaxMARL/outputs/ippo_cnn_v3_cramped_room_farmguard_nodist_no_cookplate_floor10_15m_20260524_014811.gif
/student/brownd58/dev/JaxMARL/checkpoints/ippo_cnn_v3_cramped_room_farmguard_nodist_no_cookplate_floor10_15m_20260524_014811/model.msgpack
```

For the RNN sweep id `20260524_024519`, each layout writes:

```text
Log:
/student/brownd58/dev/JaxMARL/outputs/v3_rnn_map_sweep_rnn_farmguard_nodist_no_cookplate_floor10_15m_20260524_024519/<layout>.log

Checkpoint:
/student/brownd58/dev/JaxMARL/checkpoints/ippo_rnn_v3_<layout>_rnn_farmguard_nodist_no_cookplate_floor10_15m_20260524_024519/model.msgpack

GIF:
/student/brownd58/dev/JaxMARL/outputs/ippo_rnn_v3_<layout>_rnn_farmguard_nodist_no_cookplate_floor10_15m_20260524_024519.gif
```

## How To Read The Results

Primary success metrics:

- `delivery`: count of successful deliveries in the PPO update.
- `event/delivery`: same event count logged for charts.
- `delivery_count.agent_0` and `delivery_count.agent_1`: which agent completed deliveries.
- `base_reward_per_step`: sparse reward only. This is the best compact signal for delivery performance.
- `combined_reward_per_step`: sparse plus shaped reward after coefficient and anneal.
- `event/dish_pickup`: cooked soup picked up from pot with plate.

Supporting behavior metrics:

- `event/pot_placement`: ingredients placed into pots.
- `event/pot_start_cooking`: pot filled and cooking started. Logged only, not rewarded.
- `event/pickup` and `event/drop`: useful for spotting object farming or thrashing.
- `event/dish_to_goal_progress`: signed distance progress is still logged, but reward weight is `0.0`.
- `loss/entropy`: policy randomness. Very low entropy can mean the policy has collapsed into a fixed routine.
- `anneal_factor`: shaped reward multiplier. In the current 15M setup it decays from `1.0` to about `0.103`.

What to look for:

- Good run: `delivery` becomes nonzero early and remains nonzero near update `585`.
- Strong run: late `dish_pickup` is close to late `delivery`, meaning soup pickup is translating into delivery.
- Suspicious run: high `dish_pickup` but `delivery=0`, meaning the agent gets soup but does not finish.
- Suspicious run: high `pickup/drop` with low `pot_placement`, meaning object handling may be noisy or farming.
- Failed run: no `step=...` metrics, or Hydra/JAX error before training.

## Quick Result Extraction

Summarize final metric lines for a sweep:

```bash
cd /student/brownd58/dev/JaxMARL

/student/brownd58/dev/JaxMARL/.venv/bin/python - <<'PY'
from pathlib import Path
import re

sweep = Path('/student/brownd58/dev/JaxMARL/outputs/v3_map_sweep_farmguard_nodist_no_cookplate_floor10_15m_20260524_014811')
# For RNN, use:
# sweep = Path('/student/brownd58/dev/JaxMARL/outputs/v3_rnn_map_sweep_rnn_farmguard_nodist_no_cookplate_floor10_15m_20260524_024519')
line_re = re.compile(
    r'step=(\d+)/(\d+).*?env_step=(\d+).*?'
    r'base_rew/step=([0-9.\-]+).*?combined_rew/step=([0-9.\-]+).*?'
    r'delivery=([0-9.\-]+).*?dish_pickup=([0-9.\-]+).*?'
    r'entropy=([0-9.\-]+).*?anneal=([0-9.\-]+)'
)

print('layout,state,step,env_step,delivery,dish_pickup,base_rew_per_step,combined_rew_per_step,entropy,anneal')
for log in sorted(sweep.glob('*.log')):
    if log.name in {'sweep.log', 'status.log'}:
        continue
    last = None
    for line in log.read_text(errors='ignore').splitlines():
        m = line_re.search(line)
        if m:
            last = m.groups()
    if last:
        step, total, env_step, base, combined, delivery, dish, entropy, anneal = last
        state = 'done' if step == total else 'running'
        print(f'{log.stem},{state},{step}/{total},{env_step},{delivery},{dish},{base},{combined},{entropy},{anneal}')
    else:
        print(f'{log.stem},no_metrics,,,,,,,,')
PY
```

Find final W&B summaries in a log:

```bash
rg -n "Run summary|delivery|base_reward_per_step|combined_reward_per_step|GIF saved" \
  /student/brownd58/dev/JaxMARL/outputs/v3_map_sweep_farmguard_nodist_no_cookplate_floor10_15m_20260524_014811/*.log

rg -n "Run summary|delivery|base_reward_per_step|combined_reward_per_step|GIF saved|Saved GIF" \
  /student/brownd58/dev/JaxMARL/outputs/v3_rnn_map_sweep_rnn_farmguard_nodist_no_cookplate_floor10_15m_20260524_024519/*.log
```

List GIFs from a sweep:

```bash
ls -lh /student/brownd58/dev/JaxMARL/outputs/ippo_cnn_v3_*_farmguard_nodist_no_cookplate_floor10_15m_20260524_014811.gif
ls -lh /student/brownd58/dev/JaxMARL/outputs/ippo_rnn_v3_*_rnn_farmguard_nodist_no_cookplate_floor10_15m_20260524_024519.gif
```

## Snapshot Results

Around-the-island 15M ablations:

| Run | Model | L2 reward | `PLATE_PICKUP_DURING_COOKING` | Final delivery | Notes |
| --- | --- | --- | ---: | ---: | --- |
| `farmguard_nodist_floor10_15m_20260524_011835` | CNN | off | `0.2` | `384` | Learned deliveries by 15M; agent split was mostly `agent_0`. |
| `farmguard_nodist_no_cookplate_floor10_15m_20260524_012655` | CNN | off | `0.0` | `384` | Also learned deliveries by 15M; agent split was mostly `agent_1`. |
| `around_the_island_rnn_farmguard_nodist_no_cookplate_floor10_15m_20260524_081512` | RNN | off | `0.0` | `735` | Strongest around-the-island 15M run so far; split was `agent_0=384`, `agent_1=351`. |

CNN all-other-maps sweep snapshot from `20260524_014811`:

| Layout | State | Step | Delivery | Dish pickup | Base reward/step | Combined reward/step | Entropy |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `asymm_advantages` | done | `585/585` | `896` | `896` | `0.7000` | `0.7811` | `0.1904` |
| `conveyor_demo` | done | `585/585` | `768` | `767` | `0.6000` | `0.6656` | `0.2904` |
| `coord_ring` | done | `585/585` | `768` | `768` | `0.6000` | `0.6664` | `0.1799` |
| `counter_circuit` | done | `585/585` | `640` | `640` | `0.5000` | `0.5571` | `0.0467` |
| `cramped_room` | done | `585/585` | `896` | `896` | `0.7000` | `0.7810` | `0.0763` |
| `cramped_room_v2` | done | `585/585` | `443` | `730` | `0.3461` | `0.3947` | `0.5848` |
| `follow_the_leader` | done | `585/585` | `639` | `512` | `0.4992` | `0.5564` | `0.3004` |
| `forced_coord` | done | `585/585` | `640` | `767` | `0.5000` | `0.5663` | `0.2019` |
| `middle_conveyor` | done | `585/585` | `513` | `513` | `0.4008` | `0.4510` | `0.0691` |
| `player_conveyor_demo` | done | `585/585` | `384` | `384` | `0.3000` | `0.3772` | `0.0230` |
| `player_conveyor_loop` | done | `585/585` | `0` | `0` | `0.0000` | `0.0000` | `0.7045` |
| `single_file` | done | `585/585` | `640` | `768` | `0.5000` | `0.5640` | `0.0928` |

RNN all-other-maps sweep snapshot from `20260524_024519`:

| Layout | State | Step | Delivery | Dish pickup | Base reward/step | Combined reward/step | Entropy |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `asymm_advantages` | done | `585/585` | `1730` | `1696` | `1.3516` | `1.5053` | `0.1375` |
| `conveyor_demo` | done | `585/585` | `757` | `749` | `0.5914` | `0.6596` | `0.4816` |
| `coord_ring` | done | `585/585` | `931` | `910` | `0.7273` | `0.8067` | `0.0881` |
| `counter_circuit` | done | `585/585` | `642` | `641` | `0.5016` | `0.5635` | `0.0220` |
| `cramped_room` | done | `585/585` | `890` | `893` | `0.6953` | `0.7759` | `0.7518` |
| `cramped_room_v2` | done | `585/585` | `428` | `787` | `0.3344` | `0.3862` | `0.5864` |
| `follow_the_leader` | done | `585/585` | `630` | `640` | `0.4922` | `0.5481` | `0.4332` |
| `forced_coord` | done | `585/585` | `638` | `639` | `0.4984` | `0.5939` | `0.2049` |
| `middle_conveyor` | done | `585/585` | `635` | `637` | `0.4961` | `0.5499` | `0.1471` |
| `player_conveyor_demo` | done | `585/585` | `384` | `512` | `0.3000` | `0.3834` | `0.1489` |
| `player_conveyor_loop` | done | `585/585` | `0` | `0` | `0.0000` | `0.0000` | `0.6326` |
| `single_file` | done | `585/585` | `647` | `748` | `0.5055` | `0.5656` | `0.4328` |

## Notes And Caveats

- The first all-other-maps sweep id `20260524_014712` should not be used for results because it hit the Hydra append issue before training.
- The RNN all-other-maps sweep id `20260524_024519` and the RNN around-the-island run id `20260524_081512` completed successfully.
- W&B media uploads should include one final GIF per completed run. If a GIF exists locally but not in W&B, upload it manually by resuming the W&B run and logging `wandb.Video(..., format='gif')`.
- The `dish_to_goal_progress` metric can be negative or positive, but it currently has zero reward weight.
- `POT_START_COOKING` appears in reward settings for compatibility/history, but the environment currently only logs the event.
- `player_conveyor_loop` had zero deliveries for both CNN and RNN; inspect the GIF and map mechanics before drawing a final conclusion.
