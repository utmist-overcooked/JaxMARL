# Adding Reward Shaping (Developer Guide)

This guide explains how to add reward-shaping features to Overcooked V3 for use with the MAPPO baselines. It covers two common patterns:

- Interaction-specific shaped rewards (e.g., +x when interacting with an object)
- Potential-Based Reward Shaping (PBRS) — a distance-to-completion heuristic implemented as a potential function φ(s)

The instructions below assume JAX-compatible code (use `jax.numpy` and `jax.lax` only in JITed paths) and the repository layout used by this project.

**Files of interest**
- Environment: `jaxmarl/environments/overcooked_v3/overcooked.py`
- Layouts: `jaxmarl/environments/overcooked_v3/layouts.py`
- Settings/constants: `jaxmarl/environments/overcooked_v3/settings.py`
- MAPPO training: `baselines/MAPPO/mappo_rnn_overcooked_v3_full_obs.py`
- MAPPO config: `baselines/MAPPO/config/mappo_rnn_overcooked_v3_full_obs.yaml`

Principles and constraints
- Keep shaping logic in the environment layer. MAPPO consumes `info['shaped_reward']` and mixes it into the final reward; the algorithm need not change.
- JIT compatibility: code that will be traced/jitted must use `jax.numpy` (`jnp`) and `jax.lax` control flow. Avoid Python-side branching or mutations in those functions.
- Shapes: shaped rewards must be either a scalar per-agent (shape `(num_agents,)`) or a mapping that MAPPO already expects. Prefer the `(num_agents,)` convention and expose the final shaped values in `info['shaped_reward']` as a dict mapping agent names to floats.
- Toggle: Provide an env-level toggle (e.g., `use_potential_shaping`) so experiments can enable/disable new shaping without code changes.

1) Interaction-specific shaped rewards (simple)

Where to add:
- In `process_interact(...)` or wherever interaction logic lives inside `overcooked.py`.

Pattern:
- Compute a small shaped term during interaction (e.g., picking up a required item, stepping on a button) and accumulate it into a per-agent `shaped_reward` value.
- Respect an env toggle: only add these terms when `self.shaped_rewards_enabled` (or a dedicated `self.use_legacy_shaping`) is True.

Example (pseudocode, JAX style):

```
# inside process_interact(...)
shaped_reward = jnp.array(0.0)
# successful pickup detection using jnp ops
successful_pickup = ...  # boolean/0-1 jnp value
useful_pickup = ...
shaped_reward = shaped_reward + (successful_pickup * useful_pickup * 0.1)
```

Notes:
- Keep the per-action shaped amounts small relative to sparse episode reward. Use training config `SHAPED_REWARD_SCALE` to globally scale shaped terms.
- If you add many small shaped terms, consider centralizing them or moving to PBRS (next section) to avoid exploit loops.

2) Potential-Based Reward Shaping (PBRS)

Rationale:
- PBRS guarantees policy invariance under certain conditions (Ng et al., 1999) and reduces reward-hacking by rewarding progress toward milestones instead of repeatable micro-actions.
- Implement PBRS by computing φ(s) and emitting shaped reward r_s = γ φ(s') − φ(s).

Where to add:
- Add a JAX-friendly `get_state_potential(state) -> jnp.ndarray` method in `overcooked.py` (or a small helper module under `jaxmarl/environments/overcooked_v3/`).
- In the environment transition path (e.g., `step_env(...)`) compute `phi_s = get_state_potential(state)` before applying actions, and `phi_s_prime = get_state_potential(next_state)` after state updates, then compute `shaped = gamma * phi_s_prime - phi_s`.

Designing φ(s):
- φ(s) must be deterministic and increasing with progress toward task completion.
- Prefer discrete milestones mapped to numeric potentials (e.g., start=0, button_pressed=10, entered_zone=20, holding_item=30, delivered=40). Use per-agent potentials where appropriate.
- Implement only with `jnp.where`, `jnp.any`, `jnp.sum`, `jax.lax.cond` / `jax.lax.select` and vectorized ops; avoid Python loops over agents.

Example template (conceptual):

```
def get_state_potential(self, state: State) -> jnp.ndarray:
    # returns shape (num_agents,)
    # milestone A: any button toggled -> 10
    # milestone B: agent on gated region -> +10
    # milestone C: agent holding task item -> +10
    # milestone D: delivery -> +40
    # Use jnp ops only. Return float32 jnp array per agent.
```

Integration in `step_env(...)`:
- At the start of `step_env`, call `phi_s = self.get_state_potential(state)`.
- After all environment updates (movement, interactions, conveyors, timers) compute `phi_s_prime = self.get_state_potential(state)` (the new state after update).
- Compute `gamma` for PBRS: make it configurable (`self.potential_gamma`) but default to the main `GAMMA` (or 0.99) if unspecified.
- Compute final shaped reward: `pbrs_shaped = (gamma * phi_s_prime) - phi_s` and, if desired, scale by `self.potential_scale`.
- Put final values into `info['shaped_reward']` as a dict mapping `agent_i` -> float. MAPPO training uses this key.

Important: If PBRS is enabled, disable legacy instantaneous shaped terms (e.g., `SHAPED_REWARDS` entry additions) to avoid overlapping signals. You can do this by branching on a `use_potential_shaping` flag in `process_interact` and elsewhere.

3) Config and knobs

Add environment-level config keys in `ENV_KWARGS` in MAPPO config files, for example:

```
ENV_KWARGS:
  layout: button_gated_zones
  shaped_rewards: True
  use_potential_shaping: True      # enable PBRS
  potential_scale: 0.1            # scale applied to computed PBRS
  potential_gamma: null           # optional override for gamma; otherwise agent GAMMA or 0.99
```

Training-side mixing (MAPPO):
- MAPPO already expects `info['shaped_reward']` and mixes it with the environment reward using an anneal schedule and `SHAPED_REWARD_SCALE`.
- No algorithmic changes are required there — just ensure shape and key names match.
- The mixing typically looks like:

```
reward = original_reward + anneal * SHAPED_REWARD_SCALE * info['shaped_reward']
```

4) Common pitfalls and anti-patterns

- Reward loops: granting immediate positive shaped reward for easily repeatable actions (e.g., standing on a button that toggles each step) will be exploited. PBRS mitigates this by rewarding progress (difference of φ).
- JIT pitfalls: referencing Python lists, dicts with dynamic keys, or calling non-jittable functions inside `get_state_potential` will break jitted training. Keep the function pure JAX.
- Shape mismatch: ensure `info['shaped_reward']` is a mapping with per-agent floats or a structure compatible with MAPPO's expected shape.

5) Testing and debug tips

- Add debug traces to `info` under a debug mode only (e.g., `info['debug_potential'] = {'phi': phi_s, 'phi_prime': phi_s_prime}`) so you can inspect shaping values without affecting production runs.
- Unit test `get_state_potential` using synthetic `State` objects representing each milestone. Assert strict monotonicity across milestone-advancing states.
- Run a short deterministic rollout (NUM_ENVS=1, NUM_STEPS small) and print `info['shaped_reward']` to validate PBRS output.

6) Example workflow to add a new shaped feature

1. Pick a milestone or interaction (e.g., "agent places special object on counter").
2. Add detection code in `process_interact(...)` or the appropriate state update function using `jnp` ops to produce a boolean per-agent signal.
3. If using PBRS, incorporate the milestone into `get_state_potential` as an incremental potential value. If not using PBRS, add a small instantaneous shaped term but keep it gated behind `self.shaped_rewards_enabled`.
4. Update config docs and set conservative default scale values: `potential_scale` small (0.05–0.2) and anneal shaping in training config using `REW_SHAPING_HORIZON`.
5. Run quick local rollout and inspect `info['shaped_reward']` and `info['debug_potential']` if enabled.

7) Contributing checklist

- [ ] Follow JAX best practices (use `jnp` and `jax.lax` only in jitted code)
- [ ] Add a config toggle for the feature and a conservative default
- [ ] Avoid introducing repeatable positive rewards unless encapsulated by PBRS
- [ ] Add unit tests for `get_state_potential` and any new per-step shaping logic
- [ ] Add a short README note listing the new shaping keys and recommended scales

If you need, create a small template patch: add a `get_state_potential` stub and a `use_potential_shaping` flag in `overcooked.py`, and update `ENV_KWARGS` in the MAPPO config. This repository already contains an example implementation you can follow.

---

This guide is meant to be a compact how-to for developers adding reward shaping to the Overcooked V3 env used by MAPPO. Keep shaping minimal and focused on guiding exploration rather than replacing sparse task rewards.