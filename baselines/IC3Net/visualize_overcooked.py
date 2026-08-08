"""Visualize IC3Net on Overcooked environment."""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization
import jaxmarl
from jaxmarl.viz.overcooked_visualizer import OvercookedVisualizer
import distrax

from baselines.IC3Net.models import CommNetDiscrete, CommNetLSTM
from jaxmarl.environments.overcooked.layouts import overcooked_layouts
from flax.core.frozen_dict import FrozenDict


def visualize_overcooked():
    """Visualize IC3Net policy on Overcooked."""
    
    print("=" * 60)
    print("IC3Net Visualization on Overcooked")
    print("=" * 60)
    
    # Configuration
    model_path = "checkpoints/ic3net_overcooked_medium/model.msgpack"
    config = {
        "ENV_NAME": "overcooked",
        "ENV_KWARGS": {"layout": FrozenDict(overcooked_layouts["coord_ring"]), "max_steps": 200},
        "HIDDEN_DIM": 128,
        "COMM_PASSES": 1,
        "COMM_MODE": "avg",
        "MAX_STEPS": 200,
        "SEED": 123,
        "RECURRENT": True,
    }
    
    # Check model exists
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Train a model first by running:")
        print("  .venv/bin/python baselines/IC3Net/train_overcooked.py")
        return
    
    # Load model
    print(f"\nLoading model from {model_path}...")
    with open(model_path, "rb") as f:
        params = serialization.from_bytes(None, f.read())
    
    # Setup environment
    print(f"Setting up environment: {config['ENV_NAME']}")
    env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    num_agents = env.num_agents
    
    # Get dimensions
    obs_shape = env.observation_space(env.agents[0]).shape
    obs_dim = int(np.prod(obs_shape))
    action_dim = env.action_space(env.agents[0]).n
    
    print(f"Environment: {num_agents} agents, obs_shape: {obs_shape}, action_dim: {action_dim}")
    
    # Build network
    recurrent = config.get("RECURRENT", False)
    if recurrent:
        network = CommNetLSTM(
            num_agents=num_agents,
            action_dim=action_dim,
            hidden_dim=config["HIDDEN_DIM"],
            comm_passes=config["COMM_PASSES"],
            comm_mode=config["COMM_MODE"],
            hard_attn=True,  # IC3Net
        )
        print("Model: IC3Net LSTM with communication")
    else:
        network = CommNetDiscrete(
            num_agents=num_agents,
            action_dim=action_dim,
            hidden_dim=config["HIDDEN_DIM"],
            comm_passes=config["COMM_PASSES"],
            comm_mode=config["COMM_MODE"],
            hard_attn=True,  # IC3Net
        )
        print("Model: IC3Net with communication")
    
    # Run episode
    rng = jax.random.PRNGKey(config["SEED"])
    rng, _rng = jax.random.split(rng)
    obs, env_state = env.reset(_rng)
    
    state_seq = [env_state]
    comm_action = jnp.zeros((1, num_agents), dtype=jnp.int32)
    max_steps = config["MAX_STEPS"]
    episode_return = 0.0
    
    # Initialize hidden state for recurrent models
    if recurrent:
        hidden_dim = config["HIDDEN_DIM"]
        hstate = jnp.zeros((1, num_agents, hidden_dim))
        cstate = jnp.zeros((1, num_agents, hidden_dim))
    
    print(f"\nRunning episode for up to {max_steps} steps...")
    
    for step in range(max_steps):
        # Stack observations
        obs_batch = jnp.stack([obs[a] for a in env.agents], axis=0)
        
        # Handle spatial observations (Overcooked has shape (H, W, C))
        if len(obs_batch.shape) > 2:  # (N, H, W, C) -> flatten
            obs_batch = obs_batch.reshape(num_agents, -1)
        
        obs_batch = jnp.expand_dims(obs_batch, 0)  # Add batch dim
        
        # Forward pass
        if recurrent:
            logits, value, talk_logits, (hstate, cstate) = network.apply(
                params,
                obs_batch,
                carry=(hstate, cstate),
                comm_action=comm_action,
            )
        else:
            logits, value, talk_logits = network.apply(
                params,
                obs_batch,
                comm_action=comm_action,
            )
        
        # Stochastic action selection (non-deterministic)
        rng, _rng = jax.random.split(rng)
        action_dist = distrax.Categorical(logits=logits[0])
        action = action_dist.sample(seed=_rng)
        
        rng, _rng = jax.random.split(rng)
        talk_dist = distrax.Categorical(logits=talk_logits[0])
        action_talk = talk_dist.sample(seed=_rng)
        
        # Update comm
        comm_action = jnp.expand_dims(action_talk, 0)
        
        # Print communication periodically
        if step % 50 == 0:
            talk_status = ["TALK" if int(action_talk[i]) == 1 else "SILENT" 
                          for i in range(num_agents)]
            print(f"  Step {step}: Communication: {talk_status}, Actions: {[int(action[i]) for i in range(num_agents)]}")
        
        # Step environment
        action_dict = {a: action[i] for i, a in enumerate(env.agents)}
        rng, _rng = jax.random.split(rng)
        obs, env_state, reward, done, info = env.step(_rng, env_state, action_dict)
        
        state_seq.append(env_state)
        
        # Sum rewards
        total_reward = sum(reward[a] for a in env.agents)
        episode_return += float(total_reward)
        
        if done["__all__"]:
            print(f"\nEpisode ended at step {step + 1}")
            break
    
    print(f"Episode return: {episode_return:.2f}")
    print(f"Episode length: {len(state_seq)} steps")
    
    # Visualize
    output_file = "/tmp/ic3net_overcooked.gif"
    print(f"\nCreating visualization with {len(state_seq)} frames...")
    print(f"Saving animation to {output_file}...")
    
    try:
        viz = OvercookedVisualizer()
        # animate(self, state_seq, agent_view_size, filename="animation.gif")
        viz.animate(state_seq, env.agent_view_size, filename=output_file)
        print(f"\n✓ Animation saved to: {output_file}")
        print("\nTo view the animation:")
        print(f"  xdg-open {output_file}")
        print(f"  Or copy to your local machine")
    except Exception as e:
        print(f"\n✗ Visualization error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("Visualization complete!")
    print("=" * 60)


if __name__ == "__main__":
    visualize_overcooked()
