"""Quick training test for IC3Net on Overcooked.

Tests the IC3Net implementation on the Overcooked cooperative cooking environment.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState
import distrax
import jaxmarl
from jaxmarl.wrappers.baselines import LogWrapper

from baselines.IC3Net.models import CommNetDiscrete


def train_overcooked():
    """Train IC3Net on Overcooked environment."""
    
    print("=" * 60)
    print("IC3Net Training on Overcooked")
    print("=" * 60)
    
    # Configuration
    config = {
        "ENV_NAME": "overcooked",
        "ENV_KWARGS": {},  # layout will be default
        "BASELINE": "ic3net",
        "HIDDEN_DIM": 128,
        "COMM_PASSES": 2,
        "COMM_MODE": "avg",
        "NUM_ENVS": 4,
        "NUM_STEPS": 128,
        "NUM_UPDATES": 200,  # Quick training for demo
        "LR": 3e-4,
        "MAX_GRAD_NORM": 0.5,
        "GAMMA": 0.99,
        "VALUE_COEFF": 0.5,
        "ENTROPY_COEFF": 0.01,
        "SEED": 42,
    }
    
    print(f"\nEnvironment: {config['ENV_NAME']}")
    print(f"Training for {config['NUM_UPDATES']} updates...")
    
    # Setup environment
    env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    num_agents = env.num_agents
    
    # Get observation and action dimensions
    obs_shape = env.observation_space(env.agents[0]).shape
    obs_dim = int(np.prod(obs_shape))  # Flatten the observation
    action_dim = env.action_space(env.agents[0]).n
    
    print(f"Agents: {num_agents}, obs_shape: {obs_shape}, obs_dim: {obs_dim}, action_dim: {action_dim}")
    
    env = LogWrapper(env)
    
    # Initialize network (IC3Net with communication)
    network = CommNetDiscrete(
        num_agents=num_agents,
        action_dim=action_dim,
        hidden_dim=config["HIDDEN_DIM"],
        comm_passes=config["COMM_PASSES"],
        comm_mode=config["COMM_MODE"],
        hard_attn=True,  # IC3Net
    )
    
    print(f"Model: IC3Net with {config['COMM_PASSES']} communication passes")
    
    # Initialize
    rng = jax.random.PRNGKey(config["SEED"])
    rng, _rng = jax.random.split(rng)
    
    init_obs = jnp.zeros((1, num_agents, obs_dim))
    init_comm = jnp.zeros((1, num_agents), dtype=jnp.int32)
    network_params = network.init(_rng, init_obs, comm_action=init_comm)
    
    # Setup optimizer
    tx = optax.chain(
        optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
        optax.rmsprop(learning_rate=config["LR"], decay=0.99, eps=1e-5),
    )
    
    train_state = TrainState.create(
        apply_fn=network.apply,
        params=network_params,
        tx=tx,
    )
    
    print("Model initialized successfully")
    
    # Initialize environment
    rng, _rng = jax.random.split(rng)
    reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
    obsv, env_state = jax.vmap(env.reset)(reset_rng)
    
    print(f"Environment initialized with {config['NUM_ENVS']} parallel envs")
    
    # Training function
    def update_step(carry, unused):
        """Single update step."""
        train_state, env_state, obs, rng = carry
        
        # Collect one rollout
        def env_step(carry, unused):
            train_state, env_state, obs, comm_action, rng = carry
            
            # Stack and flatten observations
            # obs is a dict with shape {agent: (B, height, width, channels)}
            # We need (B, N, obs_dim)
            obs_list = []
            for a in env.agents:
                # obs[a] has shape (B, height, width, channels)
                # Flatten spatial dimensions: (B, height*width*channels)
                flat_obs = obs[a].reshape(config["NUM_ENVS"], -1)
                obs_list.append(flat_obs)
            # Stack agents: (B, N, obs_dim)
            obs_batch = jnp.stack(obs_list, axis=1)
            
            # Forward pass with communication
            logits, value, talk_logits = network.apply(
                train_state.params, obs_batch, comm_action=comm_action
            )
            
            # Sample actions
            rng, _rng = jax.random.split(rng)
            action_dist = distrax.Categorical(logits=logits)
            action = action_dist.sample(seed=_rng)
            log_prob = action_dist.log_prob(action)
            
            # Sample talk actions
            rng, _rng = jax.random.split(rng)
            talk_dist = distrax.Categorical(logits=talk_logits)
            action_talk = talk_dist.sample(seed=_rng)
            log_prob_talk = talk_dist.log_prob(action_talk)
            
            # Combine log probs
            log_prob = log_prob + log_prob_talk
            
            # Add entropy bonus
            entropy = action_dist.entropy() + talk_dist.entropy()
            
            # Update comm_action for next step
            comm_action = action_talk
            
            # Step environment
            action_dict = {a: action[:, i] for i, a in enumerate(env.agents)}
            rng, _rng = jax.random.split(rng)
            rng_step = jax.random.split(_rng, config["NUM_ENVS"])
            obsv, env_state, reward, done, info = jax.vmap(env.step)(
                rng_step, env_state, action_dict
            )
            
            # Stack rewards
            reward_batch = jnp.stack([reward[a] for a in env.agents], axis=1)
            
            return (train_state, env_state, obsv, comm_action, rng), (value, reward_batch, log_prob, entropy)
        
        # Initialize comm (all agents talk)
        init_comm = jnp.ones((config["NUM_ENVS"], num_agents), dtype=jnp.int32)
        
        # Collect rollout
        init_carry = (train_state, env_state, obs, init_comm, rng)
        final_carry, (values, rewards, log_probs, entropies) = jax.lax.scan(
            env_step, init_carry, None, length=config["NUM_STEPS"]
        )
        
        train_state, env_state, obs, comm_action, rng = final_carry
        
        # Get last value for bootstrapping
        obs_list = []
        for a in env.agents:
            flat_obs = obs[a].reshape(config["NUM_ENVS"], -1)
            obs_list.append(flat_obs)
        last_obs_batch = jnp.stack(obs_list, axis=1)
        
        _, last_value, _ = network.apply(
            train_state.params, last_obs_batch, comm_action=comm_action
        )
        
        # Compute returns with GAE
        gamma = config["GAMMA"]
        
        def compute_gae(carry, transition):
            next_value = carry
            value, reward, done = transition
            returns = reward + gamma * next_value
            return returns, returns
        
        # Scan backwards
        _, returns = jax.lax.scan(
            compute_gae,
            last_value,
            (values, rewards, jnp.zeros_like(rewards)),  # no dones for now
            reverse=True,
        )
        
        # Advantages
        advantages = returns - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Loss function
        def loss_fn(params):
            policy_loss = -jnp.mean(log_probs * advantages)
            value_loss = jnp.mean((values - returns) ** 2)
            entropy_loss = -jnp.mean(entropies)
            
            loss = (
                policy_loss 
                + config["VALUE_COEFF"] * value_loss
                + config["ENTROPY_COEFF"] * entropy_loss
            )
            
            return loss, {
                "loss": loss,
                "policy_loss": policy_loss,
                "value_loss": value_loss,
                "entropy": -entropy_loss,
            }
        
        # Compute gradients and update
        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            train_state.params
        )
        train_state = train_state.apply_gradients(grads=grads)
        
        # Add episode return
        metrics["episode_return"] = env_state.returned_episode_returns[0].mean()
        metrics["episode_length"] = env_state.returned_episode_lengths[0].mean()
        
        return (train_state, env_state, obs, rng), metrics
    
    # Training loop
    print("\nStarting training...")
    rng, _rng = jax.random.split(rng)
    init_carry = (train_state, env_state, obsv, _rng)
    
    # JIT compile
    print("JIT compiling training loop...")
    update_step_jit = jax.jit(lambda carry: jax.lax.scan(
        update_step, carry, None, length=config["NUM_UPDATES"]
    ))
    
    final_carry, metrics = update_step_jit(init_carry)
    
    # Extract final train state
    final_train_state = final_carry[0]
    
    # Report results
    print("\nTraining complete!")
    print(f"Final loss: {float(metrics['loss'][-1]):.4f}")
    print(f"Final policy loss: {float(metrics['policy_loss'][-1]):.4f}")
    print(f"Final value loss: {float(metrics['value_loss'][-1]):.4f}")
    print(f"Final entropy: {float(metrics['entropy'][-1]):.4f}")
    print(f"Final episode return: {float(metrics['episode_return'][-1]):.2f}")
    print(f"Final episode length: {float(metrics['episode_length'][-1]):.1f}")
    
    # Save model
    save_path = "checkpoints/ic3net_overcooked"
    os.makedirs(save_path, exist_ok=True)
    save_file = os.path.join(save_path, "model.msgpack")
    
    from flax import serialization
    with open(save_file, "wb") as f:
        f.write(serialization.to_bytes(final_train_state.params))
    
    print(f"\nModel saved to {save_file}")
    
    print("\n" + "=" * 60)
    print("✓ Training complete!")
    print("=" * 60)
    
    return final_train_state


if __name__ == "__main__":
    try:
        train_overcooked()
    except Exception as e:
        print(f"\n✗ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
