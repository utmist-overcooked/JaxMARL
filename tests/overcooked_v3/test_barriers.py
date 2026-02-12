"""Test script for the barrier system in Overcooked V3."""

import jax
import jax.numpy as jnp
from jaxmarl import make

def test_barriers():
    """Test that barriers block movement when active and allow it when inactive."""
    
    # Create environment with barrier demo layout
    env = make("overcooked_v3", layout="barrier_demo")
    
    # Initialize RNG key
    key = jax.random.PRNGKey(0)
    
    # Reset environment
    key, reset_key = jax.random.split(key)
    obs, state = env.reset(reset_key)
    
    print("=== Barrier Test ===")
    print(f"Initial agent positions: {state.agents.pos.x}, {state.agents.pos.y}")
    print(f"Barrier positions: {state.barrier_positions}")
    print(f"Barrier active states: {state.barrier_active}")
    print(f"Barrier active mask: {state.barrier_active_mask}")
    
    # Test 1: Agent trying to move through inactive barrier (should succeed)
    print("\n--- Test 1: Move through INACTIVE barrier ---")
    actions = {"agent_0": 2, "agent_1": 4}  # agent_0 moves left (toward first barrier)
    key, step_key = jax.random.split(key)
    obs, state, rewards, dones, info = env.step_env(step_key, state, actions)
    print(f"Agent 0 position after left move: ({state.agents.pos.x[0]}, {state.agents.pos.y[0]})")
    print(f"Expected: Should have moved onto or past the inactive barrier at (3, 1)")
    
    # Test 2: Manually activate barriers and try to move through
    print("\n--- Test 2: Move through ACTIVE barrier (should be blocked) ---")
    
    # First move agent right once to get adjacent to barrier at (3, 1)
    actions = {"agent_0": 0, "agent_1": 4}  # agent_0 moves right
    print(f"Agent starting at: ({state.agents.pos.x[0]}, {state.agents.pos.y[0]})")
    
    # Move right once: (1,1) -> (2,1), adjacent to barrier at (3,1)
    key, step_key = jax.random.split(key)
    obs, state, rewards, dones, info = env.step_env(step_key, state, actions)
    print(f"After move 1: Agent at ({state.agents.pos.x[0]}, {state.agents.pos.y[0]}) - adjacent to barrier at (3,1)")
    
    # Now activate all barriers
    print(f"\nActivating all barriers...")
    new_barrier_active = jnp.ones_like(state.barrier_active, dtype=jnp.bool_)
    state = state.replace(barrier_active=new_barrier_active)
    print(f"Barriers now active")
    
    # Try to move right again - should be blocked by active barrier at (3, 1)
    agent_0_pos_before = (state.agents.pos.x[0], state.agents.pos.y[0])
    key, step_key = jax.random.split(key)
    obs, state, rewards, dones, info = env.step_env(step_key, state, actions)
    agent_0_pos_after = (state.agents.pos.x[0], state.agents.pos.y[0])
    
    print(f"Agent position before: {agent_0_pos_before}")
    print(f"Agent position after trying to move right: {agent_0_pos_after}")
    
    if agent_0_pos_before == agent_0_pos_after:
        print("✓ Movement blocked by active barrier (correct!)")
    else:
        print("✗ Agent moved through active barrier (incorrect!)")
    
    # Test 3: Deactivate barrier and try again
    print("\n--- Test 3: Move through INACTIVE barrier (should succeed) ---")
    new_barrier_active = jnp.zeros_like(state.barrier_active, dtype=jnp.bool_)
    state = state.replace(barrier_active=new_barrier_active)
    print(f"All barriers now inactive: {state.barrier_active}")
    
    agent_0_pos_before = (state.agents.pos.x[0], state.agents.pos.y[0])
    actions = {"agent_0": 0, "agent_1": 4}  # agent_0 moves right
    key, step_key = jax.random.split(key)
    obs, state, rewards, dones, info = env.step_env(step_key, state, actions)
    agent_0_pos_after = (state.agents.pos.x[0], state.agents.pos.y[0])
    
    print(f"Agent 0 position before: {agent_0_pos_before}")
    print(f"Agent 0 position after right move: {agent_0_pos_after}")
    
    if agent_0_pos_before != agent_0_pos_after:
        print("✓ Movement allowed through inactive barrier (correct!)")
    else:
        print("✗ Agent blocked by inactive barrier (incorrect!)")
    
    print("\n=== Test Complete ===")


def test_timed_barriers():
    """Test timed barrier functionality with button interaction."""
    
    # Create environment with timed barrier demo layout (barrier_duration=5)
    env = make("overcooked_v3", layout="timed_barrier_demo", barrier_duration=5)
    
    # Initialize RNG key
    key = jax.random.PRNGKey(0)
    
    # Reset environment
    key, reset_key = jax.random.split(key)
    obs, state = env.reset(reset_key)
    
    print("\n\n=== Timed Barrier Test ===")
    print(f"Initial agent positions: Agent 0 at ({state.agents.pos.x[0]}, {state.agents.pos.y[0]})")
    print(f"Barrier position: {state.barrier_positions[0]}")
    print(f"Barrier active: {state.barrier_active[0]}")
    print(f"Barrier timer: {state.barrier_timer[0]}")
    print(f"Barrier duration: {state.barrier_duration[0]}")
    print(f"Button position: {state.button_positions[0]}")
    
    # Test 1: Try to move through active barrier (should be blocked)
    print("\n--- Test 1: Try to move through active barrier ---")
    actions = {"agent_0": 0, "agent_1": 4}  # agent_0 moves right toward barrier
    for i in range(2):
        key, step_key = jax.random.split(key)
        obs, state, rewards, dones, info = env.step_env(step_key, state, actions)
    print(f"Agent 0 position: ({state.agents.pos.x[0]}, {state.agents.pos.y[0]})")
    print(f"Barrier active: {state.barrier_active[0]}")
    print("Should be blocked by active barrier")
    
    # Test 2: Press button to deactivate barrier
    print("\n--- Test 2: Press button to deactivate barrier ---")
    # Move agent down to button
    actions = {"agent_0": 1, "agent_1": 4}  # agent_0 moves down
    key, step_key = jax.random.split(key)
    obs, state, rewards, dones, info = env.step_env(step_key, state, actions)
    print(f"Agent 0 position after moving down: ({state.agents.pos.x[0]}, {state.agents.pos.y[0]})")
    
    # Press button (interact action)
    actions = {"agent_0": 5, "agent_1": 4}  # agent_0 interacts
    key, step_key = jax.random.split(key)
    obs, state, rewards, dones, info = env.step_env(step_key, state, actions)
    print(f"After button press:")
    print(f"  Barrier active: {state.barrier_active[0]}")
    print(f"  Barrier timer: {state.barrier_timer[0]}")
    print(f"  Expected timer: 29 (barrier_duration - 1, since timer decrements on same step)")
    
    if not state.barrier_active[0] and state.barrier_timer[0] == 29:
        print("✓ Barrier deactivated and timer set (correct!)")
    else:
        print("✗ Barrier state incorrect after button press")
    
    # Test 3: Move through deactivated barrier
    print("\n--- Test 3: Move through deactivated barrier ---")
    actions = {"agent_0": 0, "agent_1": 4}  # agent_0 moves right
    key, step_key = jax.random.split(key)
    obs, state, rewards, dones, info = env.step_env(step_key, state, actions)
    print(f"Agent 0 position: ({state.agents.pos.x[0]}, {state.agents.pos.y[0]})")
    print(f"Barrier timer: {state.barrier_timer[0]}")
    print("Should have moved through inactive barrier")
    
    # Test 4: Wait for timer to expire and barrier to reactivate
    print("\n--- Test 4: Wait for barrier timer to expire ---")
    print(f"Simulating {state.barrier_timer[0]} steps...")
    
    steps_to_simulate = int(state.barrier_timer[0])
    for step in range(steps_to_simulate):
        actions = {"agent_0": 4, "agent_1": 4}  # both agents stay
        key, step_key = jax.random.split(key)
        obs, state, rewards, dones, info = env.step_env(step_key, state, actions)
        
        if step % 10 == 0 or step == steps_to_simulate - 1:
            print(f"  Step {step + 1}: Timer = {state.barrier_timer[0]}, Active = {state.barrier_active[0]}")
    
    print(f"\nAfter timer expired:")
    print(f"  Barrier active: {state.barrier_active[0]}")
    print(f"  Barrier timer: {state.barrier_timer[0]}")
    
    if state.barrier_active[0] and state.barrier_timer[0] == 0:
        print("✓ Barrier reactivated after timer expired (correct!)")
    else:
        print("✗ Barrier did not reactivate properly")
    
    # Test 5: Verify barrier blocks when trying to move INTO it
    print("\n--- Test 5: Verify barrier blocks after reactivation ---")
    # First, move agent away from barrier (left from current position)
    actions = {"agent_0": 2, "agent_1": 4}  # agent_0 moves left
    key, step_key = jax.random.split(key)
    obs, state, rewards, dones, info = env.step_env(step_key, state, actions)
    print(f"Moved agent away from barrier to: ({state.agents.pos.x[0]}, {state.agents.pos.y[0]})")
    
    # Now try to move right back INTO the reactivated barrier - should be blocked
    agent_pos_before = (state.agents.pos.x[0], state.agents.pos.y[0])
    actions = {"agent_0": 0, "agent_1": 4}  # agent_0 moves right (toward barrier at x=3)
    key, step_key = jax.random.split(key)
    obs, state, rewards, dones, info = env.step_env(step_key, state, actions)
    agent_pos_after = (state.agents.pos.x[0], state.agents.pos.y[0])
    
    print(f"Agent position before attempting to enter barrier: {agent_pos_before}")
    print(f"Agent position after: {agent_pos_after}")
    
    if agent_pos_before == agent_pos_after:
        print("✓ Movement blocked by reactivated barrier (correct!)")
    else:
        print("✗ Agent moved through reactivated barrier (incorrect!)")
    
    print("\n=== Timed Barrier Test Complete ===")


if __name__ == "__main__":
    test_barriers()
    test_timed_barriers()
