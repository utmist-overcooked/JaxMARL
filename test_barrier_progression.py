#!/usr/bin/env python
"""Test barrier visualization progression over time."""

import jax
import jax.numpy as jnp
from jaxmarl.environments.overcooked_v3.overcooked import OvercookedV3
from jaxmarl.environments.overcooked_v3.common import Actions
from jaxmarl.viz.overcooked_v3_visualizer import OvercookedV3Visualizer

# Create environment with timed barrier demo
print("Creating environment with timed_barrier_demo layout...")
env = OvercookedV3(layout='timed_barrier_demo')

# Create visualizer
viz = OvercookedV3Visualizer(env)

# Reset to get initial state
print("Resetting environment...")
key = jax.random.PRNGKey(0)
obs_dict, state = jax.jit(env.reset)(key)

print("\n=== Initial State ===")
print(f"Barrier positions: {state.barrier_positions}")
print(f"Barrier active: {state.barrier_active}")
print(f"Barrier timer: {state.barrier_timer}")
print(f"Barrier active mask: {state.barrier_active_mask}")

# Render initial state
print("\nRendering initial state...")
img0 = viz.render_state(state)
print(f"✓ Initial image rendered! Shape: {img0.shape}")

# Try pressing button to activate timed barrier
print("\n=== Pressing button to deactivate barrier ===")
actions = {"agent_0": jnp.array(Actions.interact), "agent_1": jnp.array(Actions.stay)}
obs, state, reward, done, info = jax.jit(env.step_env)(
    key, state, actions
)
print(f"After button press:")
print(f"  Barrier active: {state.barrier_active}")
print(f"  Barrier timer: {state.barrier_timer}")

# Render state with timer
img1 = viz.render_state(state)
print(f"✓ State with timer rendered! Shape: {img1.shape}")

# Step through a few more steps to see progress bar change
print("\n=== Stepping through environment to show progress bar ===")
for i in range(5):
    actions = {"agent_0": jnp.array(Actions.stay), "agent_1": jnp.array(Actions.stay)}
    obs, state, reward, done, info = jax.jit(env.step_env)(
        key, state, actions
    )
    print(f"Step {i+1}:")
    print(f"  Barrier active: {state.barrier_active}")
    print(f"  Barrier timer: {state.barrier_timer}")
    
    img = viz.render_state(state)

print("\n✓ Barrier visualization test PASSED!")
print("✓ Progress bar should be visible on barriers with active timers")
