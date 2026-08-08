# CTC run comparison: metric normalization & eval-mode documentation

Companion to `docs/ctc_fsq_distillation_report.md`. Covers (1) how to put every
run in wandb project `ocv3_ctc_comparison` on one scale, and (2) what "greedy"
and "sampled" mean in the delivery evals. Runs after 2026-07-06 excluded.

## 1. Why the raw wandb charts don't line up

Each trainer logs `event/*` with a different aggregation over a different batch
shape, so identical policies produce wildly different chart magnitudes:

| Trainer | Aggregation | Window shape | Code |
|---|---|---|---|
| IPPO (`ippo_rnn_overcooked_v3.py`) | **sum** over window | 200 steps × 2 agents × 128 envs | `metric[event_key] = event_values.sum()` (l. 637) |
| QMIX (`qmix_rnn.py`) | **mean** per step/agent | 400 steps × 4 envs × 2 agents | `jax.tree.map(lambda x: x.mean(), infos)` (l. 624) |
| MAPPO (`mappo_rnn_overcooked_v3_full_obs.py`) | **mean** per step/agent | 256 steps × 256 envs × 2 agents | l. 839 |
| FSQ students (`mappo_rnn_overcooked_v3_fsq_*_distill.py`) | **mean** per step/agent | 256 steps × 256 envs × 2 agents | l. 1049 |

The env emits each `event/*` as a per-agent vector (one slot per agent; slots
sum to the env total for that step) — see `overcooked.py` step info
construction — so the per-agent mean must be scaled back up by both the agent
count and the episode length.

**Conversion to events per episode (per env):**

```
mean-logged runs (QMIX, MAPPO, students):
    events_per_episode = wandb_value × num_agents (2) × max_steps
sum-logged runs (IPPO):
    events_per_episode = wandb_value × max_steps / (NUM_STEPS × NUM_ENVS)
common throughput unit (compares 400- and 800-step envs):
    events_per_400 = events_per_episode × 400 / max_steps
```

Caveats:
- Training curves include exploration (QMIX ε-greedy floor 0.05; PPO-family
  sampled actions), so they sit below greedy eval numbers.
- `returned_episode_returns` is **not** comparable across runs (different
  shaping coefficients / conventions); compare only within a family.
- Alt-env runs (800-step episodes, pots 20/40 instead of 60/90) are an easier
  condition; per-400-step normalization makes throughput comparable but not the
  task difficulty.

## 2. Normalized end-of-training numbers (mean of last 10 logged points)

| Run | Env | Steps | Deliv/ep | Deliv/400 | Dish/400 | Cook-starts/400 | Burns/400 |
|---|---|---|---|---|---|---|---|
| ippo_ctc_15m_20260705 | 400st, 60/90 | 15M | 2.30 | 2.30 | 2.35 | 2.87 | 0.01 |
| mappo_ctc_15m_20260705 | 400st, 60/90 | 15M | 2.14 | 2.14 | 2.17 | 2.84 | 0.01 |
| qmix_ctc_15m_20260703 | 400st, 60/90 | 15M | 3.20 | 3.20 | 3.27 | 3.77 | 0.00 |
| ippo_ctc_15m_alt_env_20260703 | 800st, 20/40 | 15M | 10.50 | 5.25 | 5.27 | 5.46 | 0.00 |
| qmix_ctc_15m_sweepbest_20260706 | 400st, 60/90 | 15M | **0.03 (failed)** | 0.03 | 1.02 | 1.18 | 0.10 |
| mappo_fsq_ippo_distill_ctc_qmixenv_20260705 | 400st, 60/90 | 30M | 3.64 | 3.64 | 3.76 | 4.04 | 0.00 |
| mappo_fsq_mappo_distill_ctc_20260705 | 400st, 60/90 | 30M | 4.04 | 4.04 | 4.05 | 4.68 | 0.00 |
| mappo_fsq_qmix_distill_ctc_20260705 | 400st, 60/90 | 30M | 4.05 | 4.05 | 4.05 | 4.71 | 0.00 |
| mappo_fsq_ippo_distill_ctc_20260705 (alt env) | 800st, 20/40 | 30M | 15.94 | 7.97 | 7.97 | 8.48 | 0.00 |

Notes:
- `qmix_ctc_15m_sweepbest_20260706` **never learned to serve** — its greedy
  `test_deliveries_per_episode` is 0 through all 15M steps. The working QMIX
  teacher is the 20260703 run.
- Cross-check: the students' training-end sampled throughput (3.64 / 4.05 /
  4.04) reproduces the sampled fixed-start eval row (3.63 / 3.99 / 4.00) almost
  exactly — the normalization is consistent with the held-out eval.
- Burns ≈ 0 everywhere: no cook-burn farming in any of these runs.

Interactive version (charts over training, tooltips, both metrics):
Claude artifact `80a233e4` — "CTC teachers vs FSQ students — normalized".

## 3. How greedy vs sampled eval actions are generated

Source: `scripts/eval_fsq_students_deliveries.py` (the 256-episode delivery
benchmark). The three arms load the **FSQ student** checkpoints (each named for
the teacher it was distilled from), all on the common env (400 steps, pots
60/90) taken from each run's saved config.

Both modes use the same network output. The student `ActorRNN` maps
(observation, GRU hidden) → action logits and wraps them in
`pi = distrax.Categorical(logits=actor_mean)` — a softmax distribution over the
discrete action set. Per agent, per step:

- **Greedy** (`deterministic=True`, the default):
  `action = jnp.argmax(pi.logits, axis=-1)` — the mode of the policy, i.e. the
  single highest-logit action every step. No randomness enters action
  selection, so with a fixed start the whole episode is a deterministic replay:
  all 256 "episodes" are the same trajectory. That is why greedy/fixed-start is
  exactly 4.00 ± 0.00 for every student.
- **Sampled** (`--sampled`):
  `action = pi.sample(seed=ak)` — an action drawn from the full softmax
  distribution, with a fresh PRNG key split each step. This is exactly the
  behavior policy used during (MA/I)PPO training. Higher-entropy policies lose
  more here: the IPPO-distilled student (entropy ≈ 1.30) drops to 3.63
  fixed-start / 2.90 random-start, while the MAPPO/QMIX-distilled students
  (entropy ≈ 1.23) barely move.
- **Fixed vs random start**: fixed start is the training reset. `--random-start`
  sets `random_agent_positions=True`, randomizing agent spawn positions — the
  only source of episode diversity under greedy actions, and slightly
  out-of-distribution since students trained with fixed starts.

There is no temperature or top-k anywhere: greedy = argmax of the logits,
sampled = one draw from the untempered softmax.
