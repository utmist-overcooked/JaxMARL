# CTC FSQ Teacher-Distillation Comparison (IPPO vs QMIX vs MAPPO)

Distilling three privileged **full-observation** teachers (IPPO, QMIX, MAPPO) into
partially-observed **FSQ-communication** student policies on the
`coordinated_temporal_conveyor` (CTC) Overcooked-V3 map, and comparing the students
on a single **common environment**.

- **wandb project:** `zacharytang24-/ocv3_ctc_comparison`
- **Date:** 2026-07-05/06
- **Common env (all teachers + students):** `layout=coordinated_temporal_conveyor`,
  `max_steps=400`, `pot_cook_time=60`, `pot_burn_time=90`, `enable_order_queue=true`,
  `max_orders=5`, `order_generation_rate=1.0`, `order_expiration_time=0`,
  `order_queue_mode=alternating`, `plate_pickup_guard=1`, `enable_item_conveyors=true`,
  `enable_player_conveyors=false`, `random_agent_positions=false`.
  Teachers see the **full grid** (`agent_view_size=null`); students see a partial
  view (`agent_view_size=2`) + an FSQ comm channel.

---

## Result (headline)

Deliveries per episode, 256 episodes each, common env (400 steps, pots 60/90):

| Eval mode | IPPO-teacher | QMIX-teacher | MAPPO-teacher |
|---|---|---|---|
| Greedy, fixed start  | 4.00 ± 0.00 | 4.00 ± 0.00 | 4.00 ± 0.00 |
| Greedy, random start | 3.97 ± 0.35 | 3.97 ± 0.35 | 3.98 ± 0.25 |
| Sampled, fixed start | 3.63 ± 0.49 | 3.99 ± 0.12 | 4.00 ± 0.06 |
| Sampled, random start| 2.90 ± 0.36 | 3.97 ± 0.29 | 3.98 ± 0.12 |

(max observed = 4 in every cell)

**Findings**

1. **All three students solve the task** — every arm reaches ~4 deliveries/episode
   greedy from ~100% of starts. The FSQ partial-obs+comm student recovered each
   teacher's serving behaviour through the communication bottleneck.
2. **Greedy is a tie (~4) because the env caps deliveries.** With slow 60/90 pots +
   order pacing, ~4 is the achievable ceiling per 400-step episode; the limiter is the
   *environment*, not the policy. Greedy delivery count therefore cannot discriminate
   teacher quality here.
3. **The only real difference is policy sharpness under sampling.** The IPPO-teacher
   student is "looser" (sampled drops to 2.9–3.6); the QMIX/MAPPO students stay pinned
   at ~4 even when sampling. This traces to teacher entropy: the MAPPO teacher was
   razor-sharp (entropy 0.07), the IPPO teacher trained with high entropy
   (`ENT_COEF` 0.1→0.01), so its student inherited softer logits.

**To actually expose teacher-quality differences through students**, the env needs
more delivery headroom (faster pots, longer horizon, or more orders) so the ceiling is
above what the weak-teacher (QMIX) student can reach — requires retraining teachers +
students in the higher-ceiling env.

### Teacher quality (for reference, on the common env)

| Teacher | Metric | Notes |
|---|---|---|
| IPPO  | ~126–167 deliveries/batch at convergence, `dish_pickup≈delivery`, `pot_burn≈0` | strong, clean serving |
| MAPPO | `returned_episode_returns` ~60 (≈2–3 deliv/ep), teacher logits entropy 0.07 | strong, very peaked |
| QMIX  | greedy delivers ~4/ep; Q-values low-spread (~0.009 across actions) | learned but weak/soft |

---

## Artifacts

### Teachers (full-obs, common env)
| Teacher | Checkpoint | wandb |
|---|---|---|
| IPPO  | `checkpoints/ippo_ctc_15m_qmixenv_20260705/model.msgpack` | `ocv3_ctc_comparison/cfnb8n5q` |
| QMIX  | `checkpoints/qmix_ctc_15m_handoff_20260702/overcooked_v3_coordinated_temporal_conveyor/qmix_rnn_overcooked_v3_coordinated_temporal_conveyor_seed42_vmap0.safetensors` | `ocv3_ctc_comparison/urm30dyu` |
| MAPPO | `outputs/mappo_ctc_15m_qmixenv_20260705/models/mappo_rnn_overcooked_v3_full_obs_coordinated_temporal_conveyor_seed42_vmap0_actor.safetensors` | `ocv3_ctc_comparison/4oh2trk0` |

### Distilled students (partial-obs + FSQ comm)
| Student | Actor checkpoint (stem `+ _vmap0_actor.safetensors`) | wandb | GIF |
|---|---|---|---|
| IPPO-teacher  | `outputs/mappo_fsq_ippo_distill_ctc_qmixenv_20260705/models/mappo_rnn_overcooked_v3_fsq_ippo_distill_coordinated_temporal_conveyor_seed0` | `qoewotbw` | `outputs/mappo_fsq_ippo_distill_ctc_qmixenv_20260705.gif` |
| QMIX-teacher  | `outputs/mappo_fsq_qmix_distill_ctc_20260705/models/mappo_rnn_overcooked_v3_fsq_qmix_distill_coordinated_temporal_conveyor_seed0` | `3iyo9qev` | `outputs/mappo_fsq_qmix_distill_ctc_20260705.gif` |
| MAPPO-teacher | `outputs/mappo_fsq_mappo_distill_ctc_20260705/models/mappo_rnn_overcooked_v3_fsq_mappo_distill_coordinated_temporal_conveyor_seed0` | `hjidn461` | `outputs/mappo_fsq_mappo_distill_ctc_20260705.gif` |

---

## Steps to reproduce

Environment prerequisites (every run):
```bash
cd /student/brownd58/dev/JaxMARL
export PATH=/student/brownd58/dev/JaxMARL/.venv/lib/python3.10/site-packages/nvidia/cuda_nvcc/bin:$PATH  # ptxas fix
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.99
```
One ~16 GB GPU — runs are **sequential** (each `run_*.sh` opens a detached tmux session;
launch the next only after the previous finishes).

### 1. Train the three teachers (full obs, common env)
```bash
bash scripts/run_ippo_rnn_v3_ctc_15m_qmixenv_20260705.sh   # IPPO  -> checkpoints/ippo_ctc_15m_qmixenv_20260705/
bash scripts/run_mappo_rnn_v3_ctc_15m_qmixenv_20260705.sh  # MAPPO -> outputs/mappo_ctc_15m_qmixenv_20260705/models/
# QMIX teacher was trained earlier: scripts/run_qmix_rnn_v3_ctc_15m_handoff_20260702.sh
```

### 2. Distill each teacher into an FSQ student (partial obs + comm)
```bash
bash scripts/run_mappo_fsq_ippo_distill_ctc_qmixenv_20260705.sh  # IPPO-teacher  -> run qoewotbw
bash scripts/run_mappo_fsq_qmix_distill_ctc_20260705.sh          # QMIX-teacher  -> run 3iyo9qev
bash scripts/run_mappo_fsq_mappo_distill_ctc_20260705.sh         # MAPPO-teacher -> run hjidn461
```

### 3. Render a GIF for any student
```bash
python scripts/generate_fsq_ippo_distill_gif.py \
  --checkpoint <...>_vmap0_actor.safetensors \
  --config     <...>_config.yaml \
  --output     outputs/<name>.gif --seed 0   # add --deterministic for argmax
```

### 4. Batched delivery eval (the comparison numbers above)
```bash
python scripts/eval_fsq_students_deliveries.py --n-envs 256 --seed 0                  # greedy, fixed start
python scripts/eval_fsq_students_deliveries.py --n-envs 256 --seed 0 --sampled        # sampled
python scripts/eval_fsq_students_deliveries.py --n-envs 256 --seed 0 --random-start   # + episode diversity
python scripts/eval_fsq_students_deliveries.py --n-envs 256 --seed 0 --sampled --random-start
```

---

## Source files

### Training / distillation
- `baselines/IPPO/ippo_rnn_overcooked_v3.py` — IPPO teacher trainer (manual-GRU `ActorCriticRNN`).
- `baselines/MAPPO/mappo_rnn_overcooked_v3_full_obs.py` — full-obs MAPPO teacher trainer.
- `baselines/QLearning/qmix_rnn.py` — QMIX teacher trainer (`RNNQNetwork` + mixer).
- `baselines/MAPPO/mappo_rnn_overcooked_v3_fsq_ippo_distill.py` (+ `config/…fsq_ippo_distill.yaml`)
- `baselines/MAPPO/mappo_rnn_overcooked_v3_fsq_qmix_distill.py` (+ `config/…fsq_qmix_distill.yaml`)
- `baselines/MAPPO/mappo_rnn_overcooked_v3_fsq_mappo_distill.py` (+ `config/…fsq_mappo_distill.yaml`)

### Launch scripts (`scripts/`)
- Teachers: `run_ippo_rnn_v3_ctc_15m_qmixenv_20260705.sh`,
  `run_mappo_rnn_v3_ctc_15m_qmixenv_20260705.sh`, `run_qmix_rnn_v3_ctc_15m_handoff_20260702.sh`
- Distill: `run_mappo_fsq_ippo_distill_ctc_qmixenv_20260705.sh`,
  `run_mappo_fsq_qmix_distill_ctc_20260705.sh`, `run_mappo_fsq_mappo_distill_ctc_20260705.sh`
- Tools: `generate_fsq_ippo_distill_gif.py` (renders any FSQ student),
  `eval_fsq_students_deliveries.py` (batched delivery eval)

### Env / settings
- `jaxmarl/environments/overcooked_v3/overcooked.py`, `.../settings.py` (shaped rewards)
- `jaxmarl/wrappers/baselines.py` — `CTRolloutManager` (QMIX obs preprocessing), `save_params`/`load_params`

---

## Key technical notes / gotchas

- **`nn.scan` is broken** on this stack (jax 0.4.38 / flax 0.10.4): flax `axes_scan`
  calls the nonexistent `jax.api_util.debug_info`. Every recurrent net here uses a
  **manual-GRU `ScannedRNN`** drop-in (input projections outside `jax.lax.scan`,
  recurrent weights as raw params). This was patched into the FSQ student, the distill
  teacher nets, and `mappo_rnn_overcooked_v3_full_obs.py`.
- **Teacher observation formats differ:**
  - IPPO / MAPPO teachers: full **grid** obs (`agent_view_size=null`), CNN encoder.
  - QMIX teacher: `flatten(full_obs)=2400` **+ 2-dim agent one-hot = 2402**, MLP+GRU
    (matches `CTRolloutManager._preprocess_obs`). `HIDDEN_SIZE=256`.
- **QMIX is value-based** — its per-agent Q-values have a tiny across-action spread
  (~0.009), so `softmax(Q/temp=1)` is nearly uniform. The QMIX distill **standardizes
  Q per action-vector** (`TEACHER_Q_STANDARDIZE=true`) and uses `DISTILL_TEMPERATURE=0.5`
  so the KL target reflects the greedy ranking. IPPO/MAPPO teachers are policies → distill
  on logits directly at temp 1.
- **Metric aggregation differs between trainers:** IPPO logs `event/delivery` as a **sum**
  (hundreds); QMIX logs it as a per-step **mean** (~0.004). Not directly comparable —
  use the batched eval script for apples-to-apples numbers.
- **Eval determinism:** with `random_agent_positions=false` and deterministic Overcooked
  dynamics, greedy replays one identical episode across all envs (std=0). Use `--sampled`
  and/or `--random-start` for a distribution.
- **ptxas PATH fix** is required or JAX dies with `Unsupported .version 8.3`.
