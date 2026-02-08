import jax
import jax.numpy as jnp
from jaxmarl.environments.traffic_junction.traffic_junction import TrafficJunction  
from jaxmarl.environments.traffic_junction.tj_visualizer import TrafficVisualizer


MAX_AGENTS = 10


def run_test():
    # 1. Setup the Environment
    
    env = TrafficJunction(
        max_agents=MAX_AGENTS, 
        spawn_prob=0.05, 
        max_steps=100,
        view_size=3,
        collision_penalty=-10.0,
        time_penalty=-0.01,
        grid_size=14
    )
    
    key = jax.random.PRNGKey(42)
    key, key_reset = jax.random.split(key)
    
    # 2. Reset the environment
    obs, state = env.reset(key_reset)
    
    state_seq = [state]
    cumulative_reward = 0.0
    total_collisions = 0
    
    print("Starting Simulation...")
    print(f"{'Step':<8} | {'Active':<8} | {'Step Reward':<12} | {'Total Collisions':<16}")
    print("-" * 60)

    for s in range(env.max_steps):
        key, key_step = jax.random.split(key)
        
        # 3. Generate Random Actions (0: Brake, 1: Gas)
        actions_list = jax.random.randint(key, (MAX_AGENTS,), 0, 2)
        actions = {agent: actions_list[i] for i, agent in enumerate(env.agents)}
        
        # 4. Step the Environment
        obs, state, rewards, dones, info = env.step(key_step, state, actions)
        
        state_seq.append(state)

        # --- REWARD & COLLISION TRACKING ---
        # Sum rewards across all agents for this step
        step_reward = sum(rewards.values())
        cumulative_reward += step_reward
        
        # In our JAX step, we return the collision_mask in the 'info' dict or similar.
        # If your step returns collisions directly in rewards (penalty), 
        # we can infer it, but let's assume we sum the collision logic:
        # Note: Adjust 'info.get' based on your exact 'step' return structure
        step_collisions = jnp.sum(jnp.array(list(info.values()))) if info else 0
        total_collisions += int(step_collisions)
        
        # Print status every 10 steps
        if s % 10 == 0:
            active_count = jnp.sum(state.active)
            print(f"{s:<8} | {active_count:<8} | {step_reward:<12.2f} | {total_collisions:<16}")

        if dones["__all__"]:
            print(f"Simulation ended early at step {s}")
            break

    # 5. Final Report
    print("-" * 60)
    print(f"Final Cumulative Reward: {cumulative_reward:.2f}")
    print(f"Total Collisions over {len(state_seq)} steps: {total_collisions}")

    # 6. Run Visualization
    print("\nLaunching Visualizer...")
    viz = TrafficVisualizer(env, state_seq)
    viz.animate(save_fname=None)

if __name__ == "__main__":
    run_test()