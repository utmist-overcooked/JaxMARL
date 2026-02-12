#!/usr/bin/env python
"""Test script for barrier visualization."""

import jax
from jaxmarl.environments.overcooked_v3.overcooked import OvercookedV3
from jaxmarl.viz.overcooked_v3_visualizer import OvercookedV3Visualizer

# Create environment with timed barrier demo
print("Creating environment with timed_barrier_demo layout...")
env = OvercookedV3(layout='timed_barrier_demo')

# Reset to get initial state
print("Resetting environment...")
key = jax.random.PRNGKey(0)
obs_dict, state = jax.jit(env.reset)(key)

# Create visualizer
print("Creating visualizer...")
viz = OvercookedV3Visualizer(env)

# Render initial state
print("Rendering initial state...")
try:
    img = viz.render_state(state)
    print(f"✓ Image rendered successfully! Shape: {img.shape}")
    print("✓ Barrier visualization test PASSED!")
except Exception as e:
    print(f"✗ Error rendering: {e}")
    import traceback
    traceback.print_exc()
