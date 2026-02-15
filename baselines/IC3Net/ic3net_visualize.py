"""IC3Net inference with GUI visualization for JaxMARL.

Interactive visualization of trained IC3Net policies using matplotlib.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization
import jaxmarl
from jaxmarl.environments.mpe import MPEVisualizer
import matplotlib.pyplot as plt
import argparse

from baselines.IC3Net.models import IndependentMLP, CommNetDiscrete


def run_episode_and_visualize(
    env,
    network,
    params,
    rng,
    max_steps=100,
    has_talk=False,
    comm_action_one=False,
):
    """Run a single episode and collect states for visualization."""
    
    # Reset environment
    rng, _rng = jax.random.split(rng)
    obs, env_state = env.reset(_rng)
    
    # Initialize comm_action for IC3Net
    num_agents = len(env.agents)
    if has_talk:
        comm_action = jnp.zeros(num_agents, dtype=jnp.int32)
    else:
        comm_action = None
    
    # Collect states
    state_seq = [env_state]
    episode_return = 0.0
    
    print(f"Running episode for {max_steps} steps...")
    
    for step in range(max_steps):
        # Stack observations
        obs_batch = jnp.stack([obs[a] for a in env.agents], axis=0)
        obs_batch = jnp.expand_dims(obs_batch, 0)  # Add batch dim: (1, N, obs_dim)
        
        # Forward pass
        if has_talk:
            logits, value, talk_logits = network.apply(
                params,
                obs_batch,
                comm_action=comm_action,
            )
            
            # Deterministic action selection (argmax)
            action = jnp.argmax(logits[0], axis=-1)
            action_talk = jnp.argmax(talk_logits[0], axis=-1)
            
            # Update comm_action for next step
            if comm_action_one:
                comm_action = jnp.ones(num_agents, dtype=jnp.int32)
            else:
                comm_action = action_talk
            
            # Print communication status
            talk_status = ["TALK" if int(action_talk[i]) == 1 else "SILENT" 
                          for i in range(num_agents)]
            if step % 10 == 0:
                print(f"  Step {step}: Communication status: {talk_status}")
        else:
            logits, value = network.apply(params, obs_batch)
            action = jnp.argmax(logits[0], axis=-1)
        
        # Convert to dict
        action_dict = {agent_id: action[i] for i, agent_id in enumerate(env.agents)}
        
        # Step environment
        rng, _rng = jax.random.split(rng)
        obs, env_state, reward, done, info = env.step(_rng, env_state, action_dict)
        
        # Collect state
        state_seq.append(env_state)
        
        # Sum rewards
        total_reward = sum(reward[a] for a in env.agents)
        episode_return += float(total_reward)
        
        if done["__all__"]:
            print(f"  Episode ended at step {step + 1}")
            break
    
    print(f"Episode return: {episode_return:.2f}")
    return state_seq, episode_return


def main():
    parser = argparse.ArgumentParser(description="IC3Net Inference with GUI")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model checkpoint")
    parser.add_argument("--env_name", type=str, default="MPE_simple_spread_v3", help="Environment name")
    parser.add_argument("--baseline", type=str, default="ic3net", choices=["ic", "iric", "commnet", "ic3net"], 
                       help="Baseline type")
    parser.add_argument("--hidden_dim", type=int, default=64, help="Hidden dimension")
    parser.add_argument("--comm_passes", type=int, default=1, help="Communication passes")
    parser.add_argument("--comm_mode", type=str, default="avg", choices=["avg", "sum"], help="Communication mode")
    parser.add_argument("--share_weights", action="store_true", help="Share communication weights")
    parser.add_argument("--comm_action_one", action="store_true", help="All agents always talk")
    parser.add_argument("--max_steps", type=int, default=100, help="Maximum steps per episode")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--save_gif", type=str, default=None, help="Path to save animation as GIF")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("IC3Net Inference with GUI")
    print("=" * 60)
    
    # Check model exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model not found at {args.model_path}")
        print("\nTo train a model first, run:")
        print("  .venv/bin/python baselines/IC3Net/ic3net_train.py")
        return
    
    # Load model
    print(f"\nLoading model from {args.model_path}...")
    with open(args.model_path, "rb") as f:
        params = serialization.from_bytes(None, f.read())
    
    # Setup environment
    print(f"Setting up environment: {args.env_name}")
    env = jaxmarl.make(args.env_name)
    num_agents = env.num_agents
    
    # Get observation and action dimensions
    obs_shape = env.observation_space(env.agents[0]).shape
    obs_dim = int(np.prod(obs_shape))
    action_dim = env.action_space(env.agents[0]).n
    
    print(f"Environment: {num_agents} agents, obs_dim={obs_dim}, action_dim={action_dim}")
    
    # Build network
    if args.baseline in ("ic", "iric"):
        network = IndependentMLP(
            action_dim=action_dim,
            hidden_dim=args.hidden_dim,
        )
        has_talk = False
    else:
        hard_attn = (args.baseline == "ic3net")
        network = CommNetDiscrete(
            num_agents=num_agents,
            action_dim=action_dim,
            hidden_dim=args.hidden_dim,
            comm_passes=args.comm_passes,
            comm_mode=args.comm_mode,
            hard_attn=hard_attn,
            share_weights=args.share_weights,
        )
        has_talk = hard_attn
    
    print(f"Model: {args.baseline.upper()}, has_talk={has_talk}")
    
    # Run episode
    rng = jax.random.PRNGKey(args.seed)
    state_seq, episode_return = run_episode_and_visualize(
        env=env,
        network=network,
        params=params,
        rng=rng,
        max_steps=args.max_steps,
        has_talk=has_talk,
        comm_action_one=args.comm_action_one,
    )
    
    # Visualize
    print(f"\nCreating visualization with {len(state_seq)} frames...")
    print("Close the matplotlib window to exit.")
    
    try:
        viz = MPEVisualizer(env, state_seq)
        viz.animate(save_fname=args.save_gif, view=True)
    except Exception as e:
        print(f"Visualization error: {e}")
        print("Note: Make sure you're running on a system with display support.")
    
    print("\n" + "=" * 60)
    print("Inference complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
