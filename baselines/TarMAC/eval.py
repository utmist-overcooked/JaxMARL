import os
import argparse
import pickle
import jax
import jax.numpy as jnp
import numpy as np
import imageio
from jaxmarl import make as jaxmarl_make
from jaxmarl.viz.overcooked_v3_visualizer import OvercookedV3Visualizer

# Import your underlying modules cleanly from your directory code
from tarmac import TarMACCell, TarMACConfig 

def load_checkpoint(checkpoint_path):
    """Loads and deserializes the JAX parameter dictionary from a pickle file."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"No checkpoint found at path: {checkpoint_path}")
    
    print(f"📦 Loading trained weights from {checkpoint_path}...")
    with open(checkpoint_path, "rb") as f:
        params = pickle.load(f)
    return params

def run_evaluation(checkpoint_path, output_gif_path, layout_name, max_steps, num_envs):
    """Runs a parallelized greedy rollout directly using TarMACCell to preserve array dimensions."""
    
    # 1. Environment and Visualizer Setup
    env_kwargs = {
        "layout": layout_name,
        "max_steps": max_steps
    }
    eval_env = jaxmarl_make("overcooked_v3", **env_kwargs)
    viz = OvercookedV3Visualizer(eval_env)
    
    # Vectorize reset and step environments to enforce static training batch dimensions [32, ...]
    vmap_reset = jax.vmap(eval_env.reset, in_axes=(0,))
    vmap_step = jax.vmap(eval_env.step, in_axes=(0, 0, 0))
    
    # 2. Extract and Align Parameter Scope Keys
    raw_checkpoint = load_checkpoint(checkpoint_path)
    if "actor" in raw_checkpoint:
        actor_params = raw_checkpoint["actor"]
    elif "params" in raw_checkpoint:
        actor_params = raw_checkpoint["params"]
    else:
        actor_params = raw_checkpoint
        
    # Unnest the Scan module key layer if your checkpoint was compiled via the sequence wrapper
    if "params" in actor_params and "ScanTarMACCell_0" in actor_params["params"]:
        cell_params = {"params": actor_params["params"]["ScanTarMACCell_0"]}
    elif "ScanTarMACCell_0" in actor_params:
        cell_params = {"params": actor_params["ScanTarMACCell_0"]}
    else:
        cell_params = {"params": actor_params}

    # 3. Instantiate the Core Cell Unit Module Directly
    config = TarMACConfig(hidden_dim=128, msg_dim=32, key_dim=16, num_rounds=1)
    num_agents = len(eval_env.agents)
    action_dim = 6
    
    cell_module = TarMACCell(action_dim=action_dim, config=config)
    
    # 4. Initialize Tracking State Tensors
    rng = jax.random.PRNGKey(42)
    rng, rng_reset = jax.random.split(rng)
    obs, env_state = vmap_reset(jax.random.split(rng_reset, num_envs))
    
    # Initialize the structural rnn_carry states using the cell's static initialization method
    rnn_carry = TarMACCell.initialize_carry(
        hidden_dim=config.hidden_dim, 
        msg_dim=config.msg_dim, 
        batch_size=num_envs, 
        num_agents=num_agents
    )
    dones = {a: jnp.zeros((num_envs,), dtype=bool) for a in eval_env.agents}
    
    frames = []
    total_raw_rewards = 0.0
    steps_executed = 0
    
    print(f"🎬 Starting cell-level evaluation rollout loop (Batch Dimension: {num_envs})...")
    
    # 5. Sequential Step Execution
    for step in range(max_steps):
        # Unbatch environment index 0 to render the visual RGB matrix frame
        single_state = jax.tree_util.tree_map(lambda x: x[0], env_state)
        frame = viz.render_state(single_state)
        frames.append(frame)
        
        # Format tensors matching exact 3-dimensional cell layout expectations [Batch=32, Agents, ...]
        obs_tensor = jnp.stack([obs[a] for a in eval_env.agents], axis=1)        # [32, N, H, W, C]
        dones_tensor = jnp.stack([dones[a] for a in eval_env.agents], axis=1)  # [32, N]
        
        # Package inputs cleanly into a 3-rank tuple structure mapping directly to __call__(carry, inputs)
        packaged_inputs = (obs_tensor, dones_tensor)
        
        # Execute the forward trace directly using cell_module
        rnn_carry, (logits, _, _) = cell_module.apply(
            cell_params, rnn_carry, packaged_inputs
        )
        
        # PURE GREEDY ACTION EVALUATION: Select index with highest logit value (Argmax)
        actions = jnp.argmax(logits, axis=-1)  # Shape: [32, N]
        actions_dict = {a: actions[:, i] for i, a in enumerate(eval_env.agents)}
        
        # Transition environment physics simultaneously across the batch
        rng, rng_step = jax.random.split(rng)
        obs, env_state, rewards, next_dones, _ = vmap_step(
            jax.random.split(rng_step, num_envs), env_state, actions_dict
        )
        
        # Track raw environment 0 performance metrics
        total_raw_rewards += sum([float(rewards[a][0]) for a in eval_env.agents])
        steps_executed += 1
        
        dones = {a: next_dones[a] for a in eval_env.agents}
        
        if next_dones["__all__"][0]:
            print(f"🏁 Environment 0 reached terminal state naturally at step {step + 1}.")
            break

    print(f"📉 Evaluation Completed. Steps: {steps_executed} | Env 0 Accumulated Raw Score: {total_raw_rewards}")
    
    # 6. Compress and Write out Animated GIF
    print(f"💾 Saving visual animation track to {output_gif_path}...")
    imageio.mimsave(output_gif_path, frames, fps=8)
    print("✨ GIF compiled successfully! Open it up to view your trained coordination loop.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained TarMAC checkpoint on OvercookedV3.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to your saved .pkl parameter checkpoint file.")
    parser.add_argument("--output", type=str, default="my_kitchen_dance.gif", help="Output file path name for your compiled animated GIF.")
    parser.add_argument("--layout", type=str, default="cramped_room", help="Name key string of the map layout.")
    parser.add_argument("--max_steps", type=int, default=400, help="Maximum frame sequence step length budget per episode run.")
    parser.add_argument("--num_envs", type=int, default=32, help="Parallel batch dimension to align Flax Dense matrices.")
    
    args = parser.parse_args()
    
    run_evaluation(
        checkpoint_path=args.checkpoint,
        output_gif_path=args.output,
        layout_name=args.layout,
        max_steps=args.max_steps,
        num_envs=args.num_envs
    )