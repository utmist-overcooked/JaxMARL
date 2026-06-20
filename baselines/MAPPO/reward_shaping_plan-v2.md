# Action Plan: Implement Potential-Based Reward Shaping (PBRS)

## Objective
Implement a potential-based reward shaping mechanism for the `button_gated_zones` environment to resolve reward hacking (infinite point farming). The goal is to reward sequence progression (distance to completion) without rewriting the core JAX MAPPO architecture or network layers. 

## Constraints
- **Strictly additive/modifying:** Do not rewrite the core JAX environment loop, MAPPO architecture, or rollout buffers.
- **JIT Compatibility:** All new reward logic must use `jax.numpy` and `jax.lax` control flow (e.g., `jnp.where`, `jax.lax.cond`) to maintain `@jax.jit` compatibility.

---

## Phase 1: Environment State Audit
1. **Target File:** Locate the environment script defining the `button_gated_zones` layout and the `step` function (e.g., `overcooked_env.py` or layout-specific logic).
2. **Identify State Variables:** Find the exact `env_state` variables that track:
   - Button interaction status (pressed/unpressed).
   - Zone gating status (walls open/closed).
   - Subtask item possession (agent holding object).
   - Subtask completion status.

## Phase 2: Define the Potential Function $\Phi(s)$
Create a new JAX-compatible helper function `get_state_potential(state)` that assigns a strictly increasing scalar value based on milestone completion. 

*Proposed Milestone Hierarchy:*
- `0`: Base state (Start)
- `10`: Milestone 1 (Button successfully pressed / walls toggled)
- `20`: Milestone 2 (Agent successfully entered the gated zone)
- `30`: Milestone 3 (Agent interacted with the required subtask object)
- `40`: Milestone 4 (Subtask successfully completed)

## Phase 3: Implement Shaped Reward Calculation
1. Modify the `step` function to calculate the potential of the state *before* the action (`phi_s`) and *after* the action (`phi_s_prime`).
2. Calculate the potential difference using the discount factor ($\gamma pprox 0.99$):
   ```python
   # Pseudocode for JAX implementation
   phi_s = get_state_potential(state)
   # ... environment transition logic ...
   phi_s_prime = get_state_potential(next_state)
   
   shaped_reward = (gamma * phi_s_prime) - phi_s
   ```
3. Add `shaped_reward` to the agent's reward dictionary/array.

## Phase 4: Purge Exploitable Dense Rewards
1. Identify and **remove** any legacy shaped rewards in the `step` function that reward single, repeatable actions (e.g., `+1` for simply picking up an onion, `+1` for standing on a button).
2. Ensure that only the new `shaped_reward` from the potential function dictates the dense reward stream.

## Phase 5: Configuration Sync
1. Verify the layout configuration in the YAML file.
2. Ensure `SHAPED_REWARD_SCALE` is lowered (e.g., to `0.1` or `0.05`) so the potential-based shaping guides exploration without overriding the sparse global reward signal.
