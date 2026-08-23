# IPPO RNN Overcooked V3 15M Typical Run

This is the compact reference for the default 15M Overcooked V3 run we have
been using most often in experiments.

## What This Run Is

- Algorithm: `IPPO RNN`
- Environment: `overcooked_v3`
- Layout: `around_the_island`
- Observation: partial, with `agent_view_size=3`
- Training budget: `15,000,000` env steps
- Goal: learn cooperative delivery behavior while keeping the setup small enough
  to iterate on quickly

In this codebase, `agent_view_size=3` means a `7x7` local crop around each
agent, not a `3x3` crop.

## Typical Launch

```bash
cd /student/brownd58/dev/JaxMARL

PATH=/student/brownd58/dev/JaxMARL/.venv/lib/python3.10/site-packages/nvidia/cuda_nvcc/bin:$PATH \
.venv/bin/python baselines/IPPO/ippo_rnn_overcooked_v3.py \
  ENV_KWARGS.layout=around_the_island \
  +ENV_KWARGS.agent_view_size=3 \
  ENV_KWARGS.max_steps=1000 \
  ENV_KWARGS.pot_cook_time=20 \
  ENV_KWARGS.pot_burn_time=40 \
  ENV_KWARGS.enable_order_queue=true \
  ENV_KWARGS.max_orders=5 \
  ENV_KWARGS.order_generation_rate=1.0 \
  ENV_KWARGS.order_expiration_time=0 \
  ENV_KWARGS.order_queue_mode=alternating \
  ENV_KWARGS.enable_item_conveyors=false \
  ENV_KWARGS.enable_player_conveyors=false \
  TOTAL_TIMESTEPS=15000000 \
  NUM_ENVS=128 \
  NUM_STEPS=5 \
  UPDATE_EPOCHS=4 \
  NUM_MINIBATCHES=8 \
  LR=0.0005 \
  GAMMA=0.99 \
  GAE_LAMBDA=0.95 \
  CLIP_EPS=0.2 \
  VF_COEF=0.5 \
  ENT_COEF=0.03 \
  ENT_COEF_MIN=0.0 \
  ENTROPY_FLOOR=0.8 \
  ENTROPY_FLOOR_COEF=0.01 \
  MAX_GRAD_NORM=0.5 \
  GRU_HIDDEN_DIM=128 \
  FC_DIM_SIZE=128 \
  ACTIVATION=relu \
  ANNEAL_LR=true \
  REW_SHAPING_HORIZON=15000000 \
  REW_SHAPING_MIN_COEFF=0.10 \
  SHAPED_REWARD_COEFF=30.0 \
  WANDB_MODE=online \
  WANDB_LOG_HISTORY_TABLE=false \
  ENTITY=zacharytang24- \
  WANDB_PROJECT=overcookedv3_ippo_rnn_around_the_island_15m_typical \
  WANDB_NAME=ippo_rnn_overcooked_v3_around_the_island_partialobs3_T5 \
  SAVE_PATH=/student/brownd58/dev/JaxMARL/checkpoints/ippo_rnn_v3_around_the_island_partialobs3_T5 \
  SAVE_GIF_PATH=/student/brownd58/dev/JaxMARL/outputs/ippo_rnn_v3_around_the_island_partialobs3_T5.gif \
  SEED=42
```

## Parameter Guide

### Environment

| Param | Meaning |
| --- | --- |
| `ENV_KWARGS.layout` | Which map to train on. `around_the_island` is the main cooperative baseline. |
| `ENV_KWARGS.agent_view_size` | Local observation radius. `3` means each agent sees a `7x7` crop. |
| `ENV_KWARGS.max_steps` | Maximum steps per episode. `1000` gives agents time to finish long routes. |
| `ENV_KWARGS.pot_cook_time` | Steps needed for soup to finish cooking. |
| `ENV_KWARGS.pot_burn_time` | Steps after cooking before soup burns. |
| `ENV_KWARGS.enable_order_queue` | Enables the queue of active delivery orders. |
| `ENV_KWARGS.max_orders` | Maximum number of orders kept in the queue. |
| `ENV_KWARGS.order_generation_rate` | How aggressively new orders are added. `1.0` means fully active. |
| `ENV_KWARGS.order_expiration_time` | When `0`, orders do not expire from age. |
| `ENV_KWARGS.order_queue_mode` | How orders are sampled. `alternating` forces a predictable alternation. |
| `ENV_KWARGS.enable_item_conveyors` | Enables conveyor tiles that move items. |
| `ENV_KWARGS.enable_player_conveyors` | Enables conveyor tiles that move players. |

### Training Budget

| Param | Meaning |
| --- | --- |
| `TOTAL_TIMESTEPS` | Total environment steps to train for. `15,000,000` is the standard short run. |
| `NUM_ENVS` | Number of parallel environment instances. More envs means more throughput. |
| `NUM_STEPS` | Rollout length per update. `5` keeps updates frequent for quick iteration. |
| `UPDATE_EPOCHS` | PPO passes over each rollout batch. |
| `NUM_MINIBATCHES` | How many minibatches each rollout is split into. |

### Optimization

| Param | Meaning |
| --- | --- |
| `LR` | Adam learning rate. |
| `GAMMA` | Discount factor for future rewards. |
| `GAE_LAMBDA` | GAE smoothing factor for advantage estimation. |
| `CLIP_EPS` | PPO clipping range. |
| `VF_COEF` | Weight of the value-function loss. |
| `MAX_GRAD_NORM` | Gradient clipping threshold. |
| `ANNEAL_LR` | If true, linearly reduces the learning rate during training. |

### Entropy Control

| Param | Meaning |
| --- | --- |
| `ENT_COEF` | Main entropy bonus. Encourages exploration and prevents early collapse. |
| `ENT_COEF_MIN` | Lower bound for the entropy coefficient if annealing is used. |
| `ENTROPY_FLOOR` | Target minimum policy entropy before the floor penalty starts. |
| `ENTROPY_FLOOR_COEF` | Extra penalty applied when entropy drops below the floor. |

### Reward Shaping

| Param | Meaning |
| --- | --- |
| `REW_SHAPING_HORIZON` | Number of steps over which shaped rewards anneal down. |
| `REW_SHAPING_MIN_COEFF` | Final shaping multiplier at the end of annealing. |
| `SHAPED_REWARD_COEFF` | Multiplies all shaped rewards before they are added to the learning signal. |

### Model

| Param | Meaning |
| --- | --- |
| `GRU_HIDDEN_DIM` | Hidden size of the recurrent state. |
| `FC_DIM_SIZE` | Width of the fully connected layer after the encoder. |
| `ACTIVATION` | Nonlinearity used by the network. |

### Logging and Outputs

| Param | Meaning |
| --- | --- |
| `WANDB_MODE` | `online` syncs to W&B. |
| `WANDB_LOG_HISTORY_TABLE` | Keeps the table fallback off by default so W&B charts stay scalar-based. |
| `ENTITY` | W&B account or team. |
| `WANDB_PROJECT` | W&B project name. |
| `WANDB_NAME` | Display name for the run. |
| `SAVE_PATH` | Directory where the checkpoint is written. |
| `SAVE_GIF_PATH` | Path where the inference GIF is saved after training. |
| `SEED` | Random seed for reproducibility. |

## What To Watch In W&B

The main charts to watch are:

- `delivery`
- `event/dish_pickup`
- `event/pot_start_cooking`
- `event/pot_burn`
- `loss/value`
- `loss/entropy`
- `combined_reward_per_step`

If the run is healthy, you usually want to see:

- `event/delivery` rising over time
- `loss/entropy` not collapsing too quickly
- `delivery` eventually becoming nonzero instead of only shaping events

## Notes

- `agent_view_size=0` is now a true zero-radius view and produces a `1x1`
  observation.
- `DISH_TO_GOAL_PROGRESS` is logged as an event in this setup, but its reward
  weight is `0.0` unless explicitly re-enabled.
- The table-backed W&B history upload is disabled by default to keep the normal
  scalar charts clean.
