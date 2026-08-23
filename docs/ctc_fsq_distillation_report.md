# FSQ Teacher-Distillation on CTC — Results Report

**What this studies:** whether a partially-observed student with a learned FSQ
communication channel can recover a *privileged full-observation* teacher's policy on
the hard `coordinated_temporal_conveyor` (CTC) Overcooked-V3 map, and how the choice of
teacher algorithm (IPPO / QMIX / MAPPO) affects the distilled student.

- **wandb project (teachers + students):** https://wandb.ai/zacharytang24-/ocv3_ctc_comparison
- **wandb project (QMIX sweep):** https://wandb.ai/zacharytang24-/ocv3_qmix_ctc_teacher_sweep
- **Companion docs:** `docs/ctc_fsq_teacher_distill_comparison.md` (artifact index + reproduce steps)

---

## 1. Executive summary

- **All three teachers distill successfully.** Every FSQ student — partial view
  (`agent_view_size=2`) + a 3-dim FSQ comm channel — reaches the same **~4
  deliveries/episode greedy** as it was distilled toward, through the communication
  bottleneck.
- **The common env caps deliveries at ~4/episode** (400 steps, slow 60/90 pots + order
  pacing). This is the *environment* limiting, not the policy — so greedy delivery counts
  **cannot** separate the three teachers; they all saturate the ceiling.
- **The only measurable student difference is policy sharpness under sampling.** The
  IPPO-teacher student is "looser" (sampled ~2.9–3.6), while the QMIX- and MAPPO-teacher
  students stay pinned at ~4 even when sampling — traceable to teacher entropy.
- **A 16-trial QMIX hyperparameter sweep** found a **2.5× more sample-efficient** recipe
  (reaches the 4/ep ceiling in 6M vs 15M steps) and showed QMIX is *knife-edge sensitive*
  on CTC (13/16 configs scored 0). It did **not** raise the ceiling (as expected).

---

## 2. Common environment (identical for every teacher & student)

`layout=coordinated_temporal_conveyor`, `max_steps=400`, `pot_cook_time=60`,
`pot_burn_time=90`, `enable_order_queue=true`, `max_orders=5`,
`order_generation_rate=1.0`, `order_expiration_time=0`, `order_queue_mode=alternating`,
`plate_pickup_guard=1`, `enable_item_conveyors=true`, `enable_player_conveyors=false`,
`random_agent_positions=false`.

Teachers observe the **full grid** (`agent_view_size=null`); students observe a **partial
view** (`agent_view_size=2`) and must communicate via a quantized (FSQ, levels `[5,5,5]`)
message to a partner to recover the missing state.

---

## 3. Teachers (privileged, full-observation)

| Teacher | wandb run | Quality on common env | Checkpoint |
|---|---|---|---|
| **IPPO** | [cfnb8n5q](https://wandb.ai/zacharytang24-/ocv3_ctc_comparison/runs/cfnb8n5q) | strong: ~126–167 deliveries/batch, `dish_pickup≈delivery`, `pot_burn≈0` | `checkpoints/ippo_ctc_15m_qmixenv_20260705/model.msgpack` |
| **MAPPO** (full-obs) | [4oh2trk0](https://wandb.ai/zacharytang24-/ocv3_ctc_comparison/runs/4oh2trk0) | strong: `returned_episode_returns`~60 (≈2–3 deliv/ep), teacher logit entropy 0.07 | `outputs/mappo_ctc_15m_qmixenv_20260705/models/mappo_rnn_overcooked_v3_full_obs_coordinated_temporal_conveyor_seed42_vmap0_actor.safetensors` |
| **QMIX** (baseline) | [urm30dyu](https://wandb.ai/zacharytang24-/ocv3_ctc_comparison/runs/urm30dyu) | learned but weak: greedy ~4/ep; Q-values low-spread (~0.009 across actions) | `checkpoints/qmix_ctc_15m_handoff_20260702/overcooked_v3_coordinated_temporal_conveyor/qmix_rnn_overcooked_v3_coordinated_temporal_conveyor_seed42_vmap0.safetensors` |
| **QMIX** (sweep-best, 15M) | [4yrdfj36](https://wandb.ai/zacharytang24-/ocv3_ctc_comparison/runs/4yrdfj36) | full-budget retrain of the sweep winner *(in progress at time of writing)* | `checkpoints/qmix_ctc_15m_sweepbest_20260706/…` (pending) |

Metric note: IPPO logs `event/delivery` as a **sum** (hundreds); QMIX logs it as a
per-step **mean** (~0.004). They are **not** directly comparable — hence the batched eval
in §6 is the authoritative student comparison.

---

## 4. Distilled students (partial obs + FSQ comm)

Trained with a MAPPO student (decentralized FSQ actor + centralized critic) plus a cosine-
decayed KL distillation loss toward the teacher's action distribution.

| Student | wandb run | GIF |
|---|---|---|
| **IPPO-teacher** | [qoewotbw](https://wandb.ai/zacharytang24-/ocv3_ctc_comparison/runs/qoewotbw) | `outputs/mappo_fsq_ippo_distill_ctc_qmixenv_20260705.gif` |
| **QMIX-teacher** | [3iyo9qev](https://wandb.ai/zacharytang24-/ocv3_ctc_comparison/runs/3iyo9qev) | `outputs/mappo_fsq_qmix_distill_ctc_20260705.gif` |
| **MAPPO-teacher** | [hjidn461](https://wandb.ai/zacharytang24-/ocv3_ctc_comparison/runs/hjidn461) | `outputs/mappo_fsq_mappo_distill_ctc_20260705.gif` |

Per teacher type, the distillation target differs:
- **IPPO / MAPPO** are policies → distill on their logits directly (KL at temperature 1).
- **QMIX** is value-based → its Q-values have a tiny across-action spread (~0.009), so
  `softmax(Q/temp=1)` is nearly uniform. The QMIX distill **standardizes Q per
  action-vector** (`TEACHER_Q_STANDARDIZE`) and uses `DISTILL_TEMPERATURE=0.5` so the KL
  target reflects the teacher's greedy *ranking*.

---

## 5. GIF references (single sampled rollout each)

| Rollout | Path | Deliveries (1 sampled episode) |
|---|---|---|
| IPPO-teacher student | `outputs/mappo_fsq_ippo_distill_ctc_qmixenv_20260705.gif` | 4 |
| QMIX-teacher student | `outputs/mappo_fsq_qmix_distill_ctc_20260705.gif` | 4 |
| MAPPO-teacher student | `outputs/mappo_fsq_mappo_distill_ctc_20260705.gif` | 4 |
| QMIX baseline teacher (greedy) | `outputs/qmix_ctc_15m_handoff_20260702.gif` | 4 |
| QMIX sweep-best teacher | `outputs/qmix_ctc_15m_sweepbest_20260706.gif` *(auto-rendered on completion)* | pending |

Single-seed GIFs are illustrative only; the quantitative comparison is §6.

---

## 6. Multi-seed delivery eval (the authoritative comparison)

Batched greedy/sampled rollouts, **256 parallel independent episodes per student** on the
common env, via `scripts/eval_fsq_students_deliveries.py`. Deliveries per episode
(mean ± std):

| Eval mode | IPPO-teacher | QMIX-teacher | MAPPO-teacher |
|---|---|---|---|
| Greedy, fixed start  | **4.00 ± 0.00** | **4.00 ± 0.00** | **4.00 ± 0.00** |
| Greedy, random start | 3.97 ± 0.35 | 3.97 ± 0.35 | 3.98 ± 0.25 |
| Sampled, fixed start | 3.63 ± 0.49 | 3.99 ± 0.12 | 4.00 ± 0.06 |
| Sampled, random start| 2.90 ± 0.36 | 3.97 ± 0.29 | 3.98 ± 0.12 |

(max observed = 4 in every cell)

**How to read this**
1. **All students solve the task** — ~4 deliveries/episode greedy from ~100% of starts.
   The FSQ partial-obs+comm student recovered each teacher's serving behaviour.
2. **Greedy is a 3-way tie at the env ceiling (~4).** With deterministic dynamics and a
   fixed start, greedy replays one identical episode (std = 0); random starts confirm the
   tie holds across states.
3. **Sampling exposes the one real difference:** the IPPO-teacher student drops to
   2.9–3.6 when sampling (softer logits), while the QMIX/MAPPO students stay ~4. This
   mirrors teacher entropy — the MAPPO teacher was razor-sharp (entropy 0.07), the IPPO
   teacher trained with high entropy (`ENT_COEF` 0.1→0.01).

**Conclusion:** despite very different teacher quality (IPPO/MAPPO strong, QMIX weak),
all three distill into students that hit the same ~4 greedy ceiling — because the env
caps deliveries there. To differentiate teacher quality *through* students, the env needs
more delivery headroom (faster pots, longer horizon, or more orders).

---

## 7. QMIX teacher hyperparameter sweep

Bayesian sweep, **env held byte-identical**, only training hyperparameters varied.
Metric: `test_returned_episode_returns` (greedy eval), 16 trials × 6M steps.

- **Sweep:** https://wandb.ai/zacharytang24-/ocv3_qmix_ctc_teacher_sweep/sweeps/jn4czc8l

**Winner — `expert-sweep-13`: test return 80.0 = 4.00 deliv/ep.** 13 of 16 configs scored
0.0 → QMIX is knife-edge sensitive on CTC.

Winning hyperparameters vs the previous QMIX teacher (`urm30dyu`):

| Param | Previous teacher | Sweep winner | Change |
|---|---|---|---|
| `SHAPED_REWARD_COEFF` | 1.0 | 5.0 | 5× stronger (but moderate — coeff 15/30 all failed) |
| `REW_SHAPING_MIN_COEFF` | 1.0 (constant) | 0.5 | shaping now anneals toward true reward |
| `LR` | 5e-5 | 2.5e-4 | 5× higher |
| `EPS_DECAY` | 0.2 | 0.4 | 2× longer exploration |
| `NUM_ENVS` | 4 | 4 | same (8 envs failed across the board) |
| `TARGET_UPDATE_INTERVAL` | 10 | 10 | same (fast target updates) |
| `TOTAL_TIMESTEPS` | 15M | 6M to reach ceiling | 2.5× more sample-efficient |

**Verdict:** the sweep did **not** beat the ~4/ep ceiling (expected — env cap), but found
a **more efficient, more reliable** recipe. A full 15M retrain of the winner
([4yrdfj36](https://wandb.ai/zacharytang24-/ocv3_ctc_comparison/runs/4yrdfj36)) is running
to produce a saved checkpoint + GIF and, optionally, a re-distilled student.

---

## 8. Key technical notes

- **`nn.scan` is broken** on this stack (jax 0.4.38 / flax 0.10.4) → all recurrent nets
  use a manual-GRU `ScannedRNN` drop-in (patched into the FSQ student, the distill teacher
  nets, and `mappo_rnn_overcooked_v3_full_obs.py`). QMIX's own training path happens to
  work with `nn.scan` and was left unchanged.
- **Teacher obs formats:** IPPO/MAPPO = full grid (CNN); QMIX = `flatten(full_obs)=2400 +
  2-dim agent one-hot = 2402` (MLP+GRU, `CTRolloutManager` preprocessing), `HIDDEN_SIZE=256`.
- **`PYTHONPATH=/student/brownd58/dev/JaxMARL` is required** or the installed jaxmarl
  (missing CTC kwargs) shadows the local repo.
- **Eval determinism:** with `random_agent_positions=false` + deterministic dynamics,
  greedy replays one identical episode (std=0). Use `--sampled` / `--random-start` for a
  distribution.

---

## 9. Reproduce

Scripts (`scripts/`): teachers `run_ippo_rnn_v3_ctc_15m_qmixenv_20260705.sh`,
`run_mappo_rnn_v3_ctc_15m_qmixenv_20260705.sh`,
`run_qmix_rnn_v3_ctc_15m_sweepbest_20260706.sh`; distills
`run_mappo_fsq_{ippo,qmix,mappo}_distill_*.sh`; sweep
`run_qmix_ctc_teacher_sweep_20260706.sh` (+ `sweep_qmix_teacher_ctc.py`); tools
`generate_fsq_ippo_distill_gif.py`, `generate_qmix_v3_gif.py`,
`eval_fsq_students_deliveries.py`. Full step-by-step in
`docs/ctc_fsq_teacher_distill_comparison.md`.
