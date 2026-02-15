"""IC3Net-family inference script for JaxMARL.

Evaluate trained IC3Net/CommNet/IC/IRIC policies.
"""
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization
from typing import Dict
import jaxmarl
import hydra
from omegaconf import OmegaConf
import time
import os

from baselines.IC3Net.models import IndependentMLP, CommNetDiscrete, IndependentLSTM, CommNetLSTM


def make_infer(config):
    """Create inference function for IC3Net-family models."""
    
    # Setup environment
    env_kwargs = config.get("ENV_KWARGS", {}).copy()
    
    # Handle layout string for Overcooked (convert to actual layout dict)
    if config["ENV_NAME"] == "overcooked" and "layout" in env_kwargs:
        layout_name = env_kwargs["layout"]
        if isinstance(layout_name, str):
            from jaxmarl.environments.overcooked.layouts import overcooked_layouts
            from flax.core.frozen_dict import FrozenDict
            env_kwargs["layout"] = FrozenDict(overcooked_layouts[layout_name])
    
    env = jaxmarl.make(config["ENV_NAME"], **env_kwargs)
    num_agents = env.num_agents
    
    # Get observation and action dimensions
    obs_shape = env.observation_space(env.agents[0]).shape
    obs_dim = int(np.prod(obs_shape))
    action_dim = env.action_space(env.agents[0]).n
    
    def infer(rng, params):
        """Run inference episodes."""
        
        # Build network
        baseline = config.get("BASELINE", "ic3net")
        recurrent = config.get("RECURRENT", False)
        
        if baseline in ("ic", "iric"):
            if recurrent:
                network = IndependentLSTM(
                    action_dim=action_dim,
                    hidden_dim=config.get("HIDDEN_DIM", 64),
                )
            else:
                network = IndependentMLP(
                    action_dim=action_dim,
                    hidden_dim=config.get("HIDDEN_DIM", 64),
                )
            has_talk = False
        else:
            hard_attn = (baseline == "ic3net")
            if recurrent:
                network = CommNetLSTM(
                    num_agents=num_agents,
                    action_dim=action_dim,
                    hidden_dim=config.get("HIDDEN_DIM", 64),
                    comm_passes=config.get("COMM_PASSES", 1),
                    comm_mode=config.get("COMM_MODE", "avg"),
                    hard_attn=hard_attn,
                    share_weights=config.get("SHARE_WEIGHTS", False),
                )
            else:
                network = CommNetDiscrete(
                    num_agents=num_agents,
                    action_dim=action_dim,
                    hidden_dim=config.get("HIDDEN_DIM", 64),
                    comm_passes=config.get("COMM_PASSES", 1),
                    comm_mode=config.get("COMM_MODE", "avg"),
                    hard_attn=hard_attn,
                    share_weights=config.get("SHARE_WEIGHTS", False),
                )
            has_talk = hard_attn
        
        # Run episodes
        def run_episode(rng):
            """Run a single episode."""
            # Reset environment
            rng, _rng = jax.random.split(rng)
            obs, env_state = env.reset(_rng)
            
            # Initialize comm_action for IC3Net
            if has_talk:
                comm_action = jnp.zeros(num_agents, dtype=jnp.int32)
            else:
                comm_action = None
            
            # Initialize hidden state for recurrent models
            if recurrent:
                hidden_dim = config.get("HIDDEN_DIM", 64)
                init_hstate = jnp.zeros((1, num_agents, hidden_dim))
                init_cstate = jnp.zeros((1, num_agents, hidden_dim))
                hidden_state = (init_hstate, init_cstate)
            else:
                hidden_state = None
            
            episode_return = 0.0
            
            def step_fn(carry, _):
                """Single step function."""
                obs, env_state, comm_action, hidden_state, ep_return, rng = carry
                
                # Stack observations
                obs_batch = jnp.stack([obs[a] for a in env.agents], axis=0)
                
                # Handle spatial observations (Overcooked has shape (H, W, C))
                if len(obs_batch.shape) > 2:  # (N, H, W, C) -> flatten
                    obs_batch = obs_batch.reshape(num_agents, -1)
                
                obs_batch = jnp.expand_dims(obs_batch, 0)  # Add batch dim: (1, N, obs_dim)
                
                # Forward pass
                if recurrent:
                    if has_talk:
                        logits, value, talk_logits, hidden_state = network.apply(
                            params,
                            obs_batch,
                            carry=hidden_state,
                            comm_action=comm_action,
                        )
                        
                        # Deterministic action selection
                        action = jnp.argmax(logits[0], axis=-1)
                        action_talk = jnp.argmax(talk_logits[0], axis=-1)
                        
                        # Update comm_action for next step
                        if config.get("COMM_ACTION_ONE", False):
                            comm_action = jnp.ones(num_agents, dtype=jnp.int32)
                        else:
                            comm_action = action_talk
                    else:
                        logits, value, _, hidden_state = network.apply(
                            params,
                            obs_batch,
                            carry=hidden_state,
                        )
                        action = jnp.argmax(logits[0], axis=-1)
                else:
                    if has_talk:
                        logits, value, talk_logits = network.apply(
                            params,
                            obs_batch,
                            comm_action=comm_action,
                        )
                        
                        # Deterministic action selection
                        action = jnp.argmax(logits[0], axis=-1)
                        action_talk = jnp.argmax(talk_logits[0], axis=-1)
                        
                        # Update comm_action for next step
                        if config.get("COMM_ACTION_ONE", False):
                            comm_action = jnp.ones(num_agents, dtype=jnp.int32)
                        else:
                            comm_action = action_talk
                    else:
                        logits, value = network.apply(params, obs_batch)
                        action = jnp.argmax(logits[0], axis=-1)
                
                # Convert to dict
                action_dict = {agent_id: action[i] for i, agent_id in enumerate(env.agents)}
                
                # Step environment
                rng, _rng = jax.random.split(rng)
                obs, env_state, reward, done, info = env.step(_rng, env_state, action_dict)
                
                # Sum rewards
                total_reward = sum(reward[a] for a in env.agents)
                ep_return = ep_return + total_reward
                
                return (obs, env_state, comm_action, hidden_state, ep_return, rng), done["__all__"]
            
            # Run episode steps
            max_steps = config.get("MAX_STEPS", 50)
            init_carry = (obs, env_state, comm_action, hidden_state, episode_return, rng)
            
            final_carry, dones = jax.lax.scan(
                step_fn,
                init_carry,
                None,
                length=max_steps,
            )
            
            final_return = final_carry[4]  # Updated index from 3 to 4
            return final_return
        
        # Run multiple episodes
        num_episodes = config.get("NUM_EPISODES", 10)
        rngs = jax.random.split(rng, num_episodes)
        
        returns = jax.vmap(run_episode)(rngs)
        
        return {
            "returns": returns,
            "mean_return": jnp.mean(returns),
            "std_return": jnp.std(returns),
        }
    
    return infer


@hydra.main(version_base=None, config_path="config", config_name="ic3net_mpe_infer")
def main(config):
    """Main inference entry point."""
    config = OmegaConf.to_container(config)
    
    # Load model
    if not config.get("MODEL_PATH"):
        raise ValueError("MODEL_PATH must be specified for inference")
    
    model_path = config["MODEL_PATH"]
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    print(f"Loading model from {model_path}")
    with open(model_path, "rb") as f:
        params = serialization.from_bytes(None, f.read())
    
    # Run inference
    rng = jax.random.PRNGKey(config.get("SEED", 0))
    infer_fn = make_infer(config)
    
    # JIT compile for speed
    infer_jit = jax.jit(lambda rng: infer_fn(rng, params))
    
    print(f"Running {config.get('NUM_EPISODES', 10)} episodes...")
    results = infer_jit(rng)
    
    print(f"\nInference complete!")
    print(f"Mean return: {results['mean_return']:.2f} ± {results['std_return']:.2f}")
    print(f"All returns: {results['returns']}")


if __name__ == "__main__":
    main()
