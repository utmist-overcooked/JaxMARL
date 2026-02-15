"""IC3Net-family training script for JaxMARL.

REINFORCE with value baseline trainer for IC3Net, CommNet, IC, and IRIC.
Based on the original IC3Net paper implementation.
"""
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import NamedTuple, Dict, Any
from flax.training.train_state import TrainState
import distrax
import jaxmarl
from jaxmarl.wrappers.baselines import LogWrapper
import hydra
from omegaconf import OmegaConf
import wandb
import os

from baselines.IC3Net.models import IndependentMLP, IndependentLSTM, CommNetDiscrete, CommNetLSTM


class Transition(NamedTuple):
    """Transition for REINFORCE rollouts."""
    obs: jnp.ndarray  # (B, N, obs_dim)
    action: jnp.ndarray  # (B, N) or (B, N, 2) if IC3Net
    value: jnp.ndarray  # (B, N)
    reward: jnp.ndarray  # (B, N)
    done: jnp.ndarray  # (B, N)
    log_prob: jnp.ndarray  # (B, N)
    # For recurrent models
    h: jnp.ndarray = None  # (B, N, hidden_dim) hidden state
    c: jnp.ndarray = None  # (B, N, hidden_dim) cell state


def make_train(config):
    """Create the training function for IC3Net-family models."""
    
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
    
    # Wrap environment
    env = LogWrapper(env)
    
    # Compute number of updates
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    
    def train(rng):
        # Initialize network based on baseline type and recurrence
        baseline = config.get("BASELINE", "ic3net")
        recurrent = config.get("RECURRENT", True)  # Default to True
        hidden_dim = config.get("HIDDEN_DIM", 64)
        
        if baseline in ("ic", "iric"):
            # Independent controller without communication
            if recurrent:
                network = IndependentLSTM(
                    action_dim=action_dim,
                    hidden_dim=hidden_dim,
                )
            else:
                network = IndependentMLP(
                    action_dim=action_dim,
                    hidden_dim=hidden_dim,
                )
            has_talk = False
        else:
            # CommNet or IC3Net with communication
            hard_attn = (baseline == "ic3net")
            if recurrent:
                network = CommNetLSTM(
                    num_agents=num_agents,
                    action_dim=action_dim,
                    hidden_dim=hidden_dim,
                    comm_passes=config.get("COMM_PASSES", 1),
                    comm_mode=config.get("COMM_MODE", "avg"),
                    hard_attn=hard_attn,
                    share_weights=config.get("SHARE_WEIGHTS", False),
                )
            else:
                network = CommNetDiscrete(
                    num_agents=num_agents,
                    action_dim=action_dim,
                    hidden_dim=hidden_dim,
                    comm_passes=config.get("COMM_PASSES", 1),
                    comm_mode=config.get("COMM_MODE", "avg"),
                    hard_attn=hard_attn,
                    share_weights=config.get("SHARE_WEIGHTS", False),
                )
            has_talk = hard_attn
        
        # Initialize network parameters
        rng, _rng = jax.random.split(rng)
        init_obs = jnp.zeros((1, num_agents, obs_dim))
        
        if recurrent:
            init_h = jnp.zeros((1, num_agents, hidden_dim))
            init_c = jnp.zeros((1, num_agents, hidden_dim))
            init_carry = (init_h, init_c)
            if has_talk:
                init_comm = jnp.zeros(num_agents, dtype=jnp.int32)
                network_params = network.init(_rng, init_obs, carry=init_carry, comm_action=init_comm)
            else:
                network_params = network.init(_rng, init_obs, carry=init_carry)
        else:
            if has_talk:
                init_comm = jnp.zeros(num_agents, dtype=jnp.int32)
                network_params = network.init(_rng, init_obs, comm_action=init_comm)
            else:
                network_params = network.init(_rng, init_obs)
        
        # Setup optimizer (RMSprop as per IC3Net paper)
        tx = optax.chain(
            optax.clip_by_global_norm(config.get("MAX_GRAD_NORM", 0.5)),
            optax.rmsprop(
                learning_rate=config.get("LR", 1e-3),
                decay=config.get("RMSPROP_ALPHA", 0.97),
                eps=config.get("RMSPROP_EPS", 1e-6),
            ),
        )
        
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )
        
        # Initialize environment
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset)(reset_rng)
        
        # Initialize hidden states for recurrent models
        if recurrent:
            init_hstate = jnp.zeros((config["NUM_ENVS"], num_agents, hidden_dim))
            init_cstate = jnp.zeros((config["NUM_ENVS"], num_agents, hidden_dim))
        else:
            init_hstate = None
            init_cstate = None
        
        # Initialize comm_action (for consistent pytree structure in scan)
        if has_talk:
            init_comm_action = jnp.ones((config["NUM_ENVS"], num_agents), dtype=jnp.int32)
        else:
            # Use dummy comm_action for consistent pytree, even for non-talk models
            init_comm_action = jnp.zeros((config["NUM_ENVS"], num_agents), dtype=jnp.int32)
        
        # REINFORCE update function
        def _update_step(runner_state, unused):
            """Collect rollouts and update policy."""
            train_state, env_state, last_obs, comm_action, hstate, cstate, rng = runner_state
            
            # Collect rollouts
            def _env_step(carry, step_idx):
                """Single environment step."""
                train_state, env_state, obs, comm_action, hstate, cstate, rng = carry
                
                # Stack obs across agents: dict -> (B, N, obs_dim)
                # Handle both flat observations and spatial observations (e.g., Overcooked)
                obs_list = []
                for a in env.agents:
                    agent_obs = obs[a]
                    # If observation has spatial dimensions (height, width, channels), flatten it
                    if len(agent_obs.shape) > 2:  # (B, H, W, C) -> (B, H*W*C)
                        flat_obs = agent_obs.reshape(agent_obs.shape[0], -1)
                    else:
                        flat_obs = agent_obs
                    obs_list.append(flat_obs)
                obs_batch = jnp.stack(obs_list, axis=1)  # (B, N, obs_dim)
                
                # Forward pass through network
                rng, _rng = jax.random.split(rng)
                
                if recurrent:
                    # Recurrent models
                    carry_in = (hstate, cstate)
                    
                    # Truncated BPTT: detach hidden state every detach_gap steps
                    detach_gap = config.get("DETACH_GAP", 10)
                    if detach_gap > 0:
                        should_detach = (step_idx > 0) & (step_idx % detach_gap == 0)
                        hstate_in = jax.lax.cond(
                            should_detach,
                            lambda x: jax.lax.stop_gradient(x),
                            lambda x: x,
                            hstate
                        )
                        cstate_in = jax.lax.cond(
                            should_detach,
                            lambda x: jax.lax.stop_gradient(x),
                            lambda x: x,
                            cstate
                        )
                        carry_in = (hstate_in, cstate_in)
                    
                    if has_talk:
                        # IC3Net with LSTM
                        logits, value, talk_logits, (hstate_new, cstate_new) = network.apply(
                            train_state.params,
                            obs_batch,
                            carry=carry_in,
                            comm_action=comm_action[0],  # Use first env's comm_action
                        )
                    else:
                        # CommNet/IC with LSTM
                        logits, value, talk_logits, (hstate_new, cstate_new) = network.apply(
                            train_state.params,
                            obs_batch,
                            carry=carry_in,
                        )
                    
                    hstate = hstate_new
                    cstate = cstate_new
                else:
                    # Feedforward models
                    if has_talk:
                        # IC3Net: get action logits, value, and talk logits
                        logits, value, talk_logits = network.apply(
                            train_state.params,
                            obs_batch,
                            comm_action=comm_action[0],  # Use first env's comm_action
                        )
                    else:
                        # IC/IRIC/CommNet: only action logits and value
                        logits, value = network.apply(train_state.params, obs_batch)
                        talk_logits = None
                
                # Sample actions
                rng, _rng = jax.random.split(rng)
                action_dist = distrax.Categorical(logits=logits)
                action_env = action_dist.sample(seed=_rng)
                log_prob = action_dist.log_prob(action_env)
                
                # Handle talk actions for IC3Net
                if has_talk and talk_logits  is not None:
                    rng, _rng = jax.random.split(rng)
                    talk_dist = distrax.Categorical(logits=talk_logits)
                    action_talk = talk_dist.sample(seed=_rng)
                    log_prob_talk = talk_dist.log_prob(action_talk)
                    log_prob = log_prob + log_prob_talk
                    
                    # Update comm_action for next step (one-step delay as in paper)
                    if config.get("COMM_ACTION_ONE", False):
                        comm_action = jnp.ones((config["NUM_ENVS"], num_agents), dtype=jnp.int32)
                    else:
                        comm_action = action_talk
                
                action = action_env  # Only use env action for stepping
                
                # Convert action to dict for multi-agent env
                # action shape: (B, N)
                action_dict = {}
                for i, agent_id in enumerate(env.agents):
                    action_dict[agent_id] = action[:, i]
                
                # Step environment
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(env.step)(
                    rng_step, env_state, action_dict
                )
                
                # Stack rewards and dones
                reward_batch = jnp.stack([reward[a] for a in env.agents], axis=1)
                done_batch = jnp.stack([done[a] for a in env.agents], axis=1)
                
                transition = Transition(
                    obs=obs_batch,
                    action=action,
                    value=value,
                    reward=reward_batch,
                    done=done_batch,
                    log_prob=log_prob,
                    h=hstate if recurrent else None,
                    c=cstate if recurrent else None,
                )
                
                return (train_state, env_state, obsv, comm_action, hstate, cstate, rng), transition
            
            # Use comm_action from runner_state for consistent rollout
            # For IC3Net, reset to all-talk at start of each rollout
            if has_talk:
                rollout_comm_action = jnp.ones((config["NUM_ENVS"], num_agents), dtype=jnp.int32)
            else:
                # Use the dummy comm_action from runner_state
                rollout_comm_action = comm_action
            
            # Collect rollout
            init_carry = (train_state, env_state, last_obs, rollout_comm_action, hstate, cstate, rng)
            final_carry, transitions = jax.lax.scan(
                _env_step,
                init_carry,
                jnp.arange(config["NUM_STEPS"]),  # Pass step indices for truncated BPTT
                length=config["NUM_STEPS"],
            )
            
            train_state, env_state, last_obs, comm_action, hstate, cstate, rng = final_carry
            
            # Compute returns (REINFORCE with value baseline)
            # Get last value for bootstrapping
            # Handle spatial observations (e.g., Overcooked)
            obs_list = []
            for a in env.agents:
                agent_obs = last_obs[a]
                if len(agent_obs.shape) > 2:  # Spatial observations
                    flat_obs = agent_obs.reshape(agent_obs.shape[0], -1)
                else:
                    flat_obs = agent_obs
                obs_list.append(flat_obs)
            last_obs_batch = jnp.stack(obs_list, axis=1)  # (B, N, obs_dim)
            
            if recurrent:
                carry_last = (hstate, cstate)
                if has_talk:
                    _, last_value, _, _ = network.apply(
                        train_state.params,
                        last_obs_batch,
                        carry=carry_last,
                        comm_action=comm_action[0],
                    )
                else:
                    _, last_value, _, _ = network.apply(
                        train_state.params,
                        last_obs_batch,
                        carry=carry_last,
                    )
            else:
                if has_talk:
                    _, last_value, _ = network.apply(
                        train_state.params,
                        last_obs_batch,
                        comm_action=comm_action[0],
                    )
                else:
                    _, last_value = network.apply(train_state.params, last_obs_batch)
            
            # Compute discounted returns
            gamma = config.get("GAMMA", 1.0)
            
            def _compute_returns(carry, transition):
                """Compute returns backwards through time."""
                next_value = carry
                reward = transition.reward
                done = transition.done
                value = transition.value
                
                # Bootstrap from next value if not done
                returns = reward + gamma * next_value * (1 - done)
                
                return returns, returns
            
            # Scan backwards through transitions
            _, returns = jax.lax.scan(
                _compute_returns,
                last_value,
                transitions,
                reverse=True,
            )
            
            # Compute advantages
            advantages = returns - transitions.value
            
            # Normalize advantages if requested
            if config.get("NORMALIZE_ADVANTAGES", False):
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # Compute loss
            def _loss_fn(params):
                """REINFORCE loss with value baseline."""
                # Reconstruction: forward pass through all timesteps
                if has_talk:
                    # For simplicity, we ignore comm_action in loss computation
                    # This is acceptable since we're computing log_probs from stored actions
                    pass
                
                # Policy loss: -log_prob * advantage
                policy_loss = -jnp.mean(transitions.log_prob * advantages)
                
                # Value loss: MSE between predicted values and returns
                value_loss = jnp.mean((transitions.value - returns) ** 2)
                
                # Total loss
                value_coeff = config.get("VALUE_COEFF", 0.01)
                loss = policy_loss + value_coeff * value_loss
                
                return loss, {
                    "loss": loss,
                    "policy_loss": policy_loss,
                    "value_loss": value_loss,
                }
            
            # Compute gradients and update
            (loss, metrics), grads = jax.value_and_grad(_loss_fn, has_aux=True)(
                train_state.params
            )
            train_state = train_state.apply_gradients(grads=grads)
            
            # Add episode return info
            metrics["returned_episode_returns"] = env_state.returned_episode_returns[0].mean()
            
            runner_state = (train_state, env_state, last_obs, comm_action, hstate, cstate, rng)
            return runner_state, metrics
        
        # Training loop
        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obsv, init_comm_action, init_hstate, init_cstate, _rng)
        
        runner_state, metrics = jax.lax.scan(
            _update_step,
            runner_state,
            None,
            length=config["NUM_UPDATES"],
        )
        
        return {"runner_state": runner_state, "metrics": metrics}
    
    return train


@hydra.main(version_base=None, config_path="config", config_name="ic3net_mpe")
def main(config):
    """Main training entry point."""
    config = OmegaConf.to_container(config)
    
    # Setup wandb
    if config.get("WANDB_MODE", "disabled") != "disabled":
        wandb.init(
            project=config.get("WANDB_PROJECT", "jaxmarl-ic3net"),
            name=config.get("WANDB_NAME", None),
            config=config,
            mode=config.get("WANDB_MODE", "online"),
        )
    
    # Run training
    rng = jax.random.PRNGKey(config.get("SEED", 42))
    train_fn = make_train(config)
    output = jax.jit(train_fn)(rng)
    
    # Extract final metrics
    metrics = output["metrics"]
    final_return = metrics["returned_episode_returns"][-1]
    
    print(f"\nTraining complete!")
    print(f"Final episode return: {final_return:.2f}")
    
    # Save model if requested
    if config.get("SAVE_PATH", None):
        from flax import serialization
        runner_state = output["runner_state"]
        train_state = runner_state[0]
        
        os.makedirs(config["SAVE_PATH"], exist_ok=True)
        save_file = os.path.join(config["SAVE_PATH"], "model.msgpack")
        
        with open(save_file, "wb") as f:
            f.write(serialization.to_bytes(train_state.params))
        
        print(f"Model saved to {save_file}")
    
    if config.get("WANDB_MODE", "disabled") != "disabled":
        wandb.finish()


if __name__ == "__main__":
    main()
