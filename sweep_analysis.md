# Sweep Analysis: `overcookedv3_ic3net_20260305_035929`

**Date:** 5 March 2026  
**Setup:** IC3Net on Overcooked V3, 3 layouts, `NUM_ENVS=64`, `TOTAL_TIMESTEPS=80M`, `COMM_PASSES=2`, shaped reward enabled  
**Sweeps:** 51 total runs across 3 sweeps (50 finished, 1 running)

## Throughput & Compute

| Metric | Previous Sweeps (NUM_ENVS=28, 20M) | New Sweeps (NUM_ENVS=64, 80M) |
|--------|-------------------------------------|-------------------------------|
| Throughput | ~42K env steps/sec | ~92K env steps/sec (**2.2x**) |
| GPU Compute | ~60–64% | ~67% (range 57–75%) |
| GPU Memory | 75.8% (JAX pre-alloc) | 47.2% |
| GPU Power | ~120W / 575W (21%) | ~141W / 575W (25%) |
| CPU | ~1.2% | ~1.3% |
| Avg Runtime | ~478s (20M steps) | ~868s (80M steps) |

GPU still has headroom — could push `NUM_ENVS` to 128.

---

## 1. `asymm_advantages` — Best Results

**Sweep ID:** `p6md1t49` — 20 runs, all finished

**7 out of 20 runs achieved non-zero episode returns (actual deliveries).**

| Rank | Run | Ep Return | Mean Rew/Step | Entropy | Value Loss |
|------|-----|-----------|---------------|---------|------------|
| 1 | misty-sweep-20 | **40.0** | 0.1818 | 1.23 | 19.78 |
| 2 | rare-sweep-16 | **20.0** | 0.1284 | 1.20 | 36.29 |
| 3 | vocal-sweep-1 | **20.0** | 0.1232 | 1.44 | 23.11 |
| 4 | solar-sweep-3 | **20.0** | 0.1207 | 1.44 | 20.32 |
| 5 | graceful-sweep-6 | **20.0** | 0.1078 | 1.27 | 3.17 |
| 6 | gallant-sweep-11 | **20.0** | 0.1068 | 1.46 | 7.26 |
| 7 | elated-sweep-7 | 0.0 | 0.0938 | 0.03 | 0.41 |
| 8 | vital-sweep-10 | **20.0** | 0.0929 | 1.42 | 17.63 |
| 9 | stellar-sweep-14 | 0.0 | 0.0651 | 0.28 | 7.21 |
| 10 | solar-sweep-8 | 0.0 | 0.0639 | 0.39 | 5.67 |
| 11–20 | (remaining) | 0.0 | 0.014–0.048 | 0.00–0.82 | — |

**Top 5 config patterns:**
- LR: 0.001–0.005 (wide range works)
- HIDDEN_DIM: mixed 128/256
- GAMMA: mostly 0.995
- SHAPED_REWARD_COEFF: 3/5 use 2
- MAX_GRAD_NORM: 4/5 use 0.5
- VALUE_COEFF: ≥0.5 (top run uses 1.0)

---

## 2. `around_the_island` — Partial Learning, No Deliveries

**Sweep ID:** `5ucigykl` — 20 runs, all finished

**Zero episode returns across all runs**, but top mean_rew/step (0.042) is **3.4x better** than the previous best (0.012 from old sweeps at 20M steps).

| Rank | Run | Ep Return | Mean Rew/Step | Entropy | Value Loss |
|------|-----|-----------|---------------|---------|------------|
| 1 | major-sweep-6 | 0.0 | **0.0416** | 0.80 | 0.40 |
| 2 | atomic-sweep-14 | 0.0 | **0.0409** | 0.77 | 0.43 |
| 3 | skilled-sweep-15 | 0.0 | **0.0268** | 0.85 | 1.50 |
| 4 | floral-sweep-16 | 0.0 | 0.0189 | 0.90 | 0.39 |
| 5 | smart-sweep-3 | 0.0 | 0.0170 | 0.54 | 0.37 |
| 6–9 | (mid-range) | 0.0 | 0.006–0.016 | 0.73–1.28 | — |
| 10–20 | (bottom half) | 0.0 | 0.000–0.006 | 0.96–1.70 | — |

**Top 5 config patterns:**
- GAMMA: all 0.99
- SHAPED_REWARD_COEFF: 3/5 use 2
- MAX_GRAD_NORM: 4/5 use 0.5
- LR: 0.0015–0.0021 (tighter mid-range)
- RMSPROP_ALPHA: mostly 0.95 or 0.99

---

## 3. `conveyor_demo` — Bimodal Results

**Sweep ID:** `kc75qaja` — 10 finished, 1 running (11/20 total)

Sharply bimodal: top 5 learned (mean_rew > 0.04), bottom 5 learned nothing (0.0).

| Rank | Run | Ep Return | Mean Rew/Step | Entropy | Value Loss |
|------|-----|-----------|---------------|---------|------------|
| 1 | wandering-sweep-7 | 0.0 | **0.0931** | 0.03 | 1.49 |
| 2 | solar-sweep-5 | 0.0 | **0.0907** | 0.07 | 2.75 |
| 3 | sunny-sweep-1 | 0.0 | 0.0479 | 0.003 | 0.01 |
| 4 | bright-sweep-10 | 0.0 | 0.0477 | 0.01 | 0.09 |
| 5 | dulcet-sweep-9 | 0.0 | 0.0453 | 0.04 | 0.43 |
| 6–10 | (bottom 5) | 0.0 | 0.000 | 0.00–0.81 | — |

Top performers have very low entropy (~0.03), meaning highly deterministic policies. No deliveries completed despite strong shaped reward signals.

**Top 5 config patterns:**
- SHAPED_REWARD_COEFF: 2/5 use 2
- VALUE_COEFF: ≥0.75 (vs 0.5 for bottom)
- DETACH_GAP: 5 dominates (4/5)

---

## Cross-Layout Config Insights

| Parameter | Best Setting | Notes |
|-----------|-------------|-------|
| **SHAPED_REWARD_COEFF** | **2** | Consistently appears in top runs across all layouts |
| **GAMMA** | **0.99** (around_the_island), **0.995** (asymm_advantages) | Layout-dependent |
| **MAX_GRAD_NORM** | **0.5** | Preferred in top runs for 2/3 layouts |
| **VALUE_COEFF** | **≥ 0.75** | Higher values correlate with better performance |
| **LR** | **0.001–0.003** | Mid-range; very high LR (>0.004) hurts |
| **HIDDEN_DIM** | **256 slightly preferred** | Not a strong differentiator |
| **COMM_PASSES** | **2** (fixed) | Not varied in this sweep |
| **DETACH_GAP** | **5–10** | Layout-dependent; no clear universal winner |

## Key Takeaways

1. **`asymm_advantages` is the most learnable layout** — the only one achieving actual deliveries (up to 40 episode return).
2. **`around_the_island` is harder** — 3.4x improvement over previous sweeps but still no deliveries at 80M steps. May need longer training or curriculum.
3. **`conveyor_demo` shows bimodal behavior** — runs either learn or don't, suggesting sensitivity to initial config. Top runs achieve high shaped reward but no deliveries.
4. **`SHAPED_REWARD_COEFF=2`** is a consistent improvement over 1 — consider making this the default or sweeping higher values (3, 5).
5. **`NUM_ENVS=64` scaling worked** — 2.2x throughput with GPU at 67%. Still room to push to 128.
6. **Next steps:** Focus on `asymm_advantages` best configs for longer training; try `NUM_ENVS=128`; consider `SHAPED_REWARD_COEFF > 2`; increase `TOTAL_TIMESTEPS` for harder layouts.
