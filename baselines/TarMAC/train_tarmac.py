"""
Train TarMAC agents in a specified environment.
"""

import argparse
import os
import time
import json
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import distrax
import numpy as np
import chex
from flax.training.train_state import TrainState
from flax.serialization import to_bytes, from_bytes
from typing import NamedTuple, Dict, Any, Tuple, Callable

from .tarmac import TarMAC, CentralizedCritic, TarMACConfig
from jaxmarl.wrappers.baselines import LogWrapper, CTRolloutManager
from jaxmarl import make

from jaxmarl.environments.traffic_junction.traffic_junction import TrafficJunction

def parse_args():
    parser = argparse.ArgumentParser(description="Train TarMAC agents in a specified environment.")
    parser.add_argument("--env", type=str, default="traffic_junction", help="Environment name")
    parser.add_argument("--update-timestep", type=int, default=16, help="Update policy every n timesteps")
    parser.add_argument("--total-timesteps", type=int, default=100000, help="Total timesteps to train")
    parser.add_argument("--lr", type=float, default=7e-4, help="Learning rate")
    parser.add_argument("--alpha", type=float, default=0.99, help="RMSprop alpha")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--entropy-coef", type=float, default=0.01, help="Entropy regularization coefficient")
    parser.add_argument("--hidden-dim", type=int, default=128, help="Hidden dimension")
    parser.add_argument("--msg-dim", type=int, default=32, help="Message dimension")
    parser.add_argument("--key-dim", type=int, default=16, help="Key/query dimension")
    parser.add_argument("--comm-rounds", type=int, default=1, help="Number of communication rounds")
    parser.add_argument("--seed", type=int, default=42, help="Random Seed")
    parser.add_argument("--num-envs", type=int, default=32, help="Parallel Environments (JAX specific)")
    parser.add_argument("--checkpoint-freq", type=int, default=1000, help="Save checkpoint every N updates")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/", help="Directory to save checkpoints")
    parser.add_argument("--checkpoint-name", type=str, default="tarmac_checkpoint", help="Base name")
    return parser.parse_args()


@chex.dataclass
class RolloutData:
    obs: chex.Array  # [time, num_agents, batch, obs_dim]
    actions: chex.Array  # [time, num_agents, batch]
    rewards: chex.Array  # [time, num_agents, batch]
    dones: chex.Array  # [time, num_agents, batch]
    log_probs: chex.Array  # [time, num_agents, batch]
    values: chex.Array  # [time, batch, 1]


class AgentState(TrainState):
    actor_params: chex.Array
    critic_params: chex.Array


def make_train_step(
        env_manager: Any,
        config: argparse.Namespace
    ) -> Tuple[Callable, nn.Module, nn.Module]:
    """
    Creates the JIT-compilable training step.
    
    Args:
        env_manager: CTRolloutManager instance (wraps JAX environment)
        config: parsed command-line arguments

    Returns:
        train_step: JIT-compilable training step function for one update
        actor_model: TarMAC actor model instance
        critic_model: CentralizedCritic model instance
    """

    # Initialize models
    sample_agent = env_manager.agents[0]
    action_dim = env_manager.action_space(sample_agent).n

    actor = TarMAC(
        action_dim=action_dim,
        config=TarMACConfig(
            hidden_dim=config.hidden_dim,
            msg_dim=config.msg_dim,
            key_dim=config.key_dim,
            num_rounds=config.comm_rounds
        )
    )

    critic = CentralizedCritic()

    def train_step(
            state: AgentState,
            env_state: Any,
            carry: Dict[str, Any],
            rng: chex.PRNGKey
    ) -> Tuple[AgentState, Any, Dict[str, Any], chex.PRNGKey, Dict[str, float]]:
        """
        Performs a single training step (policy and value update).
        1. Rollout: run policy in environment for update_timestep steps
        2. Bootstrap: calculate value for final state
        3. Loss: compute actor and critic losses
        4. Update: apply gradients

        Args:
            state: current AgentState (parameters + optimizer state)
            env_state: current environment state
            carry: Dict containing obs, dones, and RNN carry states
            rng: PRNGKey

        Returns:
            new_state: updated AgentState
            new_env_states: updated environment states
            new_carry: updated RNN carry states
            new_rng: updated PRNGKey
            metrics: dictionary of training metrics
        """

        def env_step(scan_state: Tuple, _: Any) -> Tuple[Tuple, RolloutData]:
        
            env_st, last_obs, last_dones, rnn_state, r_rng = scan_state
            r_rng, rng_act, rng_step = jax.random.split(r_rng, 3)

            # Prepare inputs for TarMAC
            obs_transposed = jnp.swapaxes(last_obs, 0, 1)
            dones_transposed = jnp.swapaxes(last_dones, 0, 1)

            obs_in = obs_transposed[None, ...]
            dones_in = dones_transposed[None, ...]

            # Actor forward pass
            new_rnn_state, (logits_seq, _, _) = state.apply_fn(
                state.actor_params, rnn_state, obs_in, dones_in
            )

            # Remove time for sampling [num_agents, batch, action_dim]
            logits = logits_seq.squeeze(0)

            # Action sampling
            logits_t = logits.transpose((1, 0, 2))  
            
            pi = distrax.Categorical(logits=logits_t)
            action = pi.sample(seed=rng_act)
            log_prob = pi.log_prob(action)
            action_agents = action

            # Step environment
            actions_dict = {a: action_agents[i] for i, a in enumerate(env_manager.agents)}
            obs, env_st, reward, done, info = env_manager.batch_step(rng_step, env_st, actions_dict)

            # Critic Value
            curr_hidden = rnn_state[0]
            action_one_hot = jax.nn.one_hot(action, num_classes=action_dim)
            value = critic.apply(
                state.critic_params,
                curr_hidden,
                action_one_hot
            )
            
            next_obs = jnp.stack([obs[a] for a in env_manager.agents])
            next_dones = jnp.stack([done[a] for a in env_manager.agents]).astype(jnp.float32)
            next_rewards = jnp.stack([reward[a] for a in env_manager.agents])

            step_data = RolloutData(
                obs=last_obs,
                actions=action_agents,
                rewards=next_rewards,
                dones=last_dones,
                log_probs=log_prob,
                values=value
            )

            return (env_st, next_obs, next_dones, new_rnn_state, r_rng), step_data
        
        # Run scan and unpack
        scan_init = (env_state, carry['obs'], carry['dones'], carry['rnn'], rng)
        scan_out, trajectory = jax.lax.scan(env_step, scan_init, None, config.update_timestep)
        new_state, last_obs, last_dones, rnn_state, new_rng = scan_out


        final_hidden = rnn_state[0]
        dummy_act = jnp.zeros((config.num_envs, len(env_manager.agents), action_dim))
        last_val = critic.apply(state.critic_params, final_hidden, dummy_act)

        def loss_fn(actor_params: chex.ArrayTree, critic_params: chex.ArrayTree) -> Tuple[float, Tuple]:
            
            # --- 1. Prepare Data: Transpose EVERYTHING to [Time, Batch, Num_Agents, ...] ---
            # trajectory.obs is [Time, Agents, Batch, Obs] -> Swap to [Time, Batch, Agents, Obs]
            obs_seq = trajectory.obs.transpose((0, 2, 1, 3))
            dones_seq = trajectory.dones.transpose((0, 2, 1))
            
            # FIX: Also transpose actions to [Time, Batch, Agents]
            actions_seq = trajectory.actions.transpose((0, 2, 1)) 
            
            # Re-run Actor
            init_carry = (carry['rnn'][0], carry['rnn'][1]) 
            _, (logits_seq, _, hidden_seq) = state.apply_fn(
                actor_params, init_carry, obs_seq, dones_seq
            )
            
            # --- 2. Critic Section ---
            T, B, N, D = hidden_seq.shape
            flat_hidden = hidden_seq.reshape(T * B, N, -1)
            
            # FIX: Use actions_seq (transposed) here
            flat_actions = jax.nn.one_hot(actions_seq, num_classes=action_dim)
            flat_actions = flat_actions.reshape(T * B, N, -1)
            
            flat_values = critic.apply(critic_params, flat_hidden, flat_actions)
            values = flat_values.reshape(T, B, 1)

            # --- 3. Returns Calculation ---
            returns = []
            gae = last_val 
            for t in reversed(range(config.update_timestep)):
                # Rewards/Dones are [Agents, Batch] -> Sum/Any to [Batch]
                # We need to transpose these too if we access them from trajectory directly
                # But since trajectory.rewards is [Time, Agents, Batch], let's be careful:
                
                r = trajectory.rewards[t] # [Agents, Batch]
                d = trajectory.dones[t]   # [Agents, Batch]

                # We sum over Agents (axis 0), so we get [Batch]
                # This part was ALREADY correct because r is [Agents, Batch]
                team_r = r.sum(axis=0, keepdims=True).T # [Batch, 1]
                team_d = d.any(axis=0, keepdims=True).T # [Batch, 1]

                gae = team_r + config.gamma * gae * (1.0 - team_d)
                returns.insert(0, gae)

            returns = jnp.stack(returns) 
            advantages = returns - values

            # --- 4. Log Probs ---
            pi = distrax.Categorical(logits=logits_seq)
            
            # FIX: Use actions_seq (transposed) here
            # Now both logits and actions are [Time, Batch, Agents], so they match!
            log_probs = pi.log_prob(actions_seq)
            
            # Sum over Agents (axis 2) -> [Time, Batch]
            team_log_probs = log_probs.sum(axis=2) 
            team_log_probs = team_log_probs[..., None] 

            # 5. Compute Losses
            actor_loss = -jnp.mean(jax.lax.stop_gradient(advantages) * team_log_probs)
            critic_loss = jnp.mean((returns - values) ** 2)
            entropy_loss = -jnp.mean(pi.entropy())

            total_loss = actor_loss + 0.5 * critic_loss - config.entropy_coef * entropy_loss
            return total_loss, (actor_loss, critic_loss, entropy_loss, returns.mean())
        
        grad_fn = jax.value_and_grad(loss_fn, argnums=(0, 1), has_aux=True)
        (loss, (a_loss, c_loss, ent, mean_ret)), grads_tuple = grad_fn(state.actor_params, state.critic_params)

        grads = {
            'actor': grads_tuple[0],
            'critic': grads_tuple[1]
        }

        updates, new_opt_state = state.tx.update(grads, state.opt_state, state.params)
        new_params = optax.apply_updates(state.params, updates)

        new_state = state.replace(
            params=new_params,
            opt_state=new_opt_state,
            actor_params=new_params['actor'],
            critic_params=new_params['critic'],
        )

        metrics = {
            'loss': loss,
            'policy_loss': a_loss,
            'critic_loss': c_loss,
            'entropy': ent,
            'mean_reward': trajectory.rewards.mean(),
            'mean_return': mean_ret
        }

        new_carry = {
            'obs': last_obs,
            'dones': last_dones,
            'rnn': rnn_state
        }
        return new_state, env_state, new_carry, rng, metrics
    
    return train_step, actor, critic

def main():
    args = parse_args()
    
    # Create Environment
    print(f"Creating environment: {args.env} with {args.num_envs} parallel envs")

    if args.env == "traffic_junction":
        env = TrafficJunction(
            max_agents=20, 
            spawn_prob=0.3,
        )
    else:
        env = make(args.env, homogenisation_method='max')

    env = LogWrapper(env)
    env_manager = CTRolloutManager(env, batch_size=args.num_envs)

    rng = jax.random.PRNGKey(args.seed)
    rng, rng_init = jax.random.split(rng)

    # Detect Real Observation Dimensions
    print("Detecting observation shapes...")
    rng, rng_reset = jax.random.split(rng)
    obs, env_state = env_manager.batch_reset(rng_reset)
    
    # Access the first agent's observation to check the feature dimension
    sample_obs = obs[env_manager.agents[0]] 
    real_obs_dim = sample_obs.shape[-1] 
    print(f"Detected Observation Dimension: {real_obs_dim}")

    # Initialize Models
    train_step_fn, actor_model, critic_model = make_train_step(env_manager, args)
    
    num_agents = len(env_manager.agents) 
    act_dim = env_manager.action_space(env_manager.agents[0]).n

    # Create Dummy Inputs for Init using the REAL dimension
    dummy_obs = jnp.zeros((1, 1, num_agents, real_obs_dim))
    dummy_dones = jnp.zeros((1, 1, num_agents))
    dummy_carry = actor_model.initialize_carry(1, num_agents)
    
    print("Initializing Models...")
    actor_params = actor_model.init(rng_init, dummy_carry, dummy_obs, dummy_dones)
    
    # Critic needs hidden state [Batch, Num_Agents, Hidden] + Actions
    dummy_hidden = dummy_carry[0] 
    dummy_acts = jnp.zeros((1, num_agents, act_dim))
    critic_params = critic_model.init(rng_init, dummy_hidden, dummy_acts)

    # Initialize Optimizer & Train State
    all_params = {
        'actor': actor_params,
        'critic': critic_params
    }

    # RMSProp is standard for A2C/RNNs
    tx = optax.rmsprop(learning_rate=args.lr, decay=0.99, eps=1e-5)
    
    state = AgentState.create(
        apply_fn=actor_model.apply, # Placeholder
        params=all_params,
        actor_params=actor_params,
        critic_params=critic_params,
        tx=tx
    )

    # JIT Compile
    carry = {
        'obs': jnp.stack([obs[a] for a in env.agents]), # [Num_Agents, Batch, ObsDim]
        'dones': jnp.zeros((num_agents, args.num_envs)),
        'rnn': actor_model.initialize_carry(args.num_envs, num_agents)
    }
    
    print("JIT Compiling Training Step...")
    jit_step = jax.jit(train_step_fn)
    
    # Trigger JIT
    state, env_state, carry, rng, _ = jit_step(state, env_state, carry, rng)
    print("Compilation Complete. Starting Training.")
    
    # Training Loop
    num_updates = args.total_timesteps // args.update_timestep
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.checkpoint_dir, f"{args.checkpoint_name}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    start_time = time.time()
    
    for update in range(1, num_updates + 1):
        state, env_state, carry, rng, metrics = jit_step(state, env_state, carry, rng)
        
        # Logging every 10 updates
        if update % 10 == 0:
            # Move metrics to CPU for printing
            m = jax.device_get(metrics)
            
            elapsed = time.time() - start_time
            total_steps = 10 * args.update_timestep * args.num_envs
            sps = total_steps / elapsed
            start_time = time.time()

            param_sample = state.params['actor']['params']['ScanTarMACCell_0']['obs_encoder_1']['kernel']
            param_mean = jnp.mean(jnp.abs(param_sample))
            
            print(f"Update {update}/{num_updates} | SPS: {int(sps)} | "
              f"Loss: {m['loss']:.4f} | "
              f"Rew: {m['mean_reward']:.4f} | "
              f"Param Mean: {param_mean:.6f}")
            
        # Checkpointing every N updates
        if update % args.checkpoint_freq == 0:
            ckpt_path = os.path.join(run_dir, f"update_{update}.ckpt")
            with open(ckpt_path, 'wb') as f:
                f.write(to_bytes(state.params))
            print(f"Saved checkpoint to {ckpt_path}")

    print(f"Training Complete. Final model saved to {run_dir}")

if __name__ == "__main__":
    main()

