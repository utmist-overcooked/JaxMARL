import jax
import jax.numpy as jnp
import flax.linen as nn
import flax.struct as struct
import numpy as np
import optax
import distrax
import chex
import argparse
import time
import os
from flax.training.train_state import TrainState
from flax.serialization import to_bytes
from typing import Tuple, Dict, Any, NamedTuple
from functools import partial

from jaxmarl import make
from jaxmarl.wrappers.baselines import LogWrapper

from .tarmac import TarMAC, TarMACConfig, TarMACCell, CentralizedCritic


# --- Data Structures ---

class Transition(NamedTuple):
    obs: chex.Array          # [Batch, Agents, ObsDim]
    actions: chex.Array      # [Batch, Agents]
    rewards: chex.Array      # [Batch, Agents]
    dones: chex.Array        # [Batch, Agents]
    logits: chex.Array       # [Batch, Agents, ActDim]
    hidden_states: chex.Array      # [Batch, Agents, HiddenDim] (h_t)
    next_hidden_states: chex.Array # [Batch, Agents, HiddenDim] (h_{t+1})

class AgentState(TrainState):
    critic_net: CentralizedCritic = struct.field(pytree_node=False)


def make_train(args):
    
    # Environment Setup
    env = make(args.env)
    env = LogWrapper(env)
    
    num_agents = len(env.agents)
    action_dim = env.action_space(env.agents[0]).n
    
    # Parallel Rollout Setup
    # vmap the reset/step functions to run 'num_envs' in parallel
    vmap_reset = jax.vmap(env.reset, in_axes=(0,))
    vmap_step = jax.vmap(env.step, in_axes=(0, 0, 0))

    # Model Initialization
    config = TarMACConfig(
        hidden_dim=args.hidden_dim,
        msg_dim=args.msg_dim,
        key_dim=args.key_dim,
        num_rounds=args.comm_rounds
    )
    
    actor = TarMAC(action_dim=action_dim, config=config)
    critic = CentralizedCritic()

    # Optimizer (RMSProp)
    # We assume a single optimizer for both Actor and Critic parameters
    tx = optax.chain(
        optax.clip_by_global_norm(args.max_grad_norm),
        optax.rmsprop(learning_rate=args.lr, decay=args.alpha, eps=1e-4)
    )

    def train_step(state, env_state, last_obs, last_dones, rnn_carry, rng):
        
        # rollout function
        def step_fn(carry, _):
            env_st, obs, dones, rnn, rng_key = carry
            rng_key, rng_act, rng_step = jax.random.split(rng_key, 3)

            # Prepare inputs: [Batch, Agents, ...]
            # obs are dicts {agent: [Batch, Obs]}
            obs_tensor = jnp.stack([obs[a] for a in env.agents], axis=1)
            # Dones are {agent: [Batch]} -> [Batch, Agents, 1]
            dones_tensor = jnp.stack([dones[a] for a in env.agents], axis=1)[..., None]

            h_in, msg_in = rnn
            rnn_masked = (jnp.where(dones_tensor > 0, 0.0, h_in), jnp.where(dones_tensor > 0, 0.0, msg_in))


            # Actor Forward Pass
            obs_seq = obs_tensor[None, ...]     # [1, Batch, Agents, Obs]
            dones_seq = dones_tensor[None, ...] # [1, Batch, Agents, 1]
            
            # Use state.apply_fn
            new_rnn, (logits_seq, _, _) = state.apply_fn(
                state.params['actor'], rnn_masked, obs_seq, dones_seq
            )
            
            # Remove Time dimension: [1, B, N, A] -> [B, N, A]
            logits = logits_seq[0]

            # Action Sampling
            pi = distrax.Categorical(logits=logits)
            actions = pi.sample(seed=rng_act) # [Batch, Agents]

            # Step Environment
            actions_dict = {a: actions[:, i] for i, a in enumerate(env.agents)}
            next_obs, next_env_st, rewards, next_dones, info = vmap_step(
                jax.random.split(rng_step, args.num_envs), env_st, actions_dict
            )

            next_dones_tensor = jnp.stack([next_dones[a] for a in env.agents], axis=1).astype(jnp.float32)[..., None]
            
            mask = (1.0 - next_dones_tensor.astype(jnp.float32))
            next_hidden_states = new_rnn[0] * mask

            trans = Transition(
                obs=obs_tensor,
                actions=actions,
                rewards=jnp.stack([rewards[a] for a in env.agents], axis=1),
                dones=jnp.stack([dones[a] for a in env.agents], axis=1),
                logits=logits,
                hidden_states=rnn_masked[0], # h_t (Post-GRU, Pre-Comm-loop output)
                next_hidden_states=new_rnn[0] * (1.0 - next_dones_tensor)
            )
            
            return (next_env_st, next_obs, next_dones, new_rnn, rng_key), trans

        # Execute Rollout
        rollout_init = (env_state, last_obs, last_dones, rnn_carry, rng)
        (env_state, last_obs, last_dones, rnn_carry, rng), traj = jax.lax.scan(
            step_fn, rollout_init, None, length=args.update_timestep
        )

        # Loss Calculation
        def loss_fn(params):
            actor_params = params['actor']
            critic_params = params['critic']
            
            # Trajectory shapes: [Time, Batch, Agents, ...]
            hidden_batch = traj.hidden_states
            act_batch = traj.actions
            rew_batch = traj.rewards
            done_batch = traj.dones
            
            # Compute Q-Values (Critic)
            # Critic takes [Batch, Agents, Hidden] + [Batch, Agents, Action(OneHot)]
            act_onehot = jax.nn.one_hot(act_batch, action_dim)
            
            def get_q(h, a): return state.critic_net.apply(critic_params, h, a)
            
            # vmap over Time and Batch
            q_values = jax.vmap(get_q)(hidden_batch, act_onehot)
            q_values = q_values.squeeze(-1) # [Time, Batch]

            # Compute Targets (SARSA/Bellman)
            # We need the 'next' hidden state and 'next' action.
            # Next Hidden: Shift hidden_batch by 1 and append the final rnn state from rollout
            # [h1, h2, ..., hT] -> [h2, ..., hT, h_final]
            next_hidden_batch = jnp.concatenate([
                hidden_batch[1:], rnn_carry[0][None, ...]
            ], axis=0)
            
            # Next Action
            next_act_onehot = jnp.concatenate([
                act_onehot[1:], jnp.zeros_like(act_onehot[0:1])
            ], axis=0)
            
            next_q_values = jax.vmap(get_q)(next_hidden_batch, next_act_onehot)
            next_q_values = next_q_values.squeeze(-1)

            # Team Rewards (Sum over agents)
            team_rewards = rew_batch.sum(axis=-1)
            # team_dones = done_batch.squeeze(-1).all(axis=-1)
            team_dones = done_batch.all(axis=-1)
            team_dones = team_dones.reshape(args.update_timestep, args.num_envs)

            # Bellman Target with Clipping
            targets = team_rewards + args.gamma * next_q_values * (1.0 - team_dones)
            targets = jnp.clip(targets, -100.0, 100.0) # Critical stability fix
            
            # Critic Loss (MSE)
            critic_loss = jnp.mean((q_values - jax.lax.stop_gradient(targets)) ** 2)

            # Actor Loss (Policy Gradient with Q-Baseline)
            # Normalize advantages (Q-values) to stabilize gradient
            # Paper accuracy: Gradient should ONLY flow through log_probs
            q_detached = jax.lax.stop_gradient(q_values)
            # Normalize across the Batch dimension (axis 1) for better stability in MPE
            adv_mean = jnp.mean(q_detached, axis=1, keepdims=True)
            adv_std = jnp.std(q_detached, axis=1, keepdims=True)
            advantages = (q_detached - adv_mean) / (adv_std + 1e-8)

            pi = distrax.Categorical(logits=traj.logits)
            log_probs = pi.log_prob(act_batch) # [Time, Batch, Agents]
            pg_loss = -(log_probs * advantages[..., None]).mean()
            
        
            
            # Entropy
            entropy = pi.entropy().mean()

            total_loss = pg_loss + args.value_loss_coef * critic_loss - args.entropy_coef * entropy
            
            return total_loss, {
                'loss': total_loss, 'pg_loss': pg_loss, 
                'v_loss': critic_loss, 'ent': entropy,
                'rew': team_rewards.mean()
            }

        grads, metrics = jax.grad(loss_fn, has_aux=True)(state.params)
        state = state.apply_gradients(grads=grads)

        # Check for NaNs 
        params_are_finite = jnp.all(jnp.isfinite(jax.flatten_util.ravel_pytree(state.params)[0]))
        metrics['is_finite'] = params_are_finite.astype(jnp.float32)
        
        return state, env_state, last_obs, last_dones, rnn_carry, rng, metrics

    return train_step

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="TrafficJunction")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_envs", type=int, default=32)
    parser.add_argument("--update_timestep", type=int, default=200) # Rollout length
    parser.add_argument("--total_timesteps", type=int, default=2_000_000)
    
    # Hyperparams
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--alpha", type=float, default=0.99) # RMSProp alpha
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument("--value_loss_coef", type=float, default=0.5)
    parser.add_argument("--entropy_coef", type=float, default=0.01)
    
    # Model Params
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--msg_dim", type=int, default=32)
    parser.add_argument("--key_dim", type=int, default=16)
    parser.add_argument("--comm_rounds", type=int, default=1)
    parser.add_argument("--ckpt_dir", type=str, default="./checkpoints")

    args = parser.parse_args()

    # Init
    print(f"Initializing {args.env}...")
    rng = jax.random.PRNGKey(args.seed)
    rng, rng_init = jax.random.split(rng)
    
    # Initialize Dummy Environment for Shapes
    dummy_env = make(args.env)
    obs_dim = dummy_env.observation_space(dummy_env.agents[0]).shape[0]
    act_dim = dummy_env.action_space(dummy_env.agents[0]).n
    num_agents = len(dummy_env.agents)
    
    # Init Models & State
    tarmac_config = TarMACConfig(
        hidden_dim=args.hidden_dim, 
        msg_dim=args.msg_dim, 
        key_dim=args.key_dim, 
        num_rounds=args.comm_rounds
    )    
    actor = TarMAC(act_dim, tarmac_config)
    critic = CentralizedCritic()
    
    # Init Params
    dummy_carry = actor.initialize_carry(args.num_envs, num_agents)
    dummy_obs = jnp.zeros((1, args.num_envs, num_agents, obs_dim))
    dummy_dones = jnp.zeros((1, args.num_envs, num_agents, 1))
    
    actor_params = actor.init(rng_init, dummy_carry, dummy_obs, dummy_dones)
    critic_params = critic.init(rng_init, dummy_carry[0], jnp.zeros((args.num_envs, num_agents, act_dim)))
    
    train_state = AgentState.create(
        apply_fn=actor.apply,
        params={'actor': actor_params, 'critic': critic_params},
        tx=optax.chain(optax.clip_by_global_norm(args.max_grad_norm), optax.rmsprop(args.lr, decay=args.alpha)),
        critic_net=critic
    )

    # JIT the step
    train_step = jax.jit(make_train(args))
    
    # Init Running State
    rng, rng_reset = jax.random.split(rng)

    # create env and wrap it
    env = make(args.env)
    env = LogWrapper(env)
    vmap_reset = jax.vmap(env.reset, in_axes=(0,))
    
    obs, env_state = vmap_reset(jax.random.split(rng_reset, args.num_envs))
    
    dones = {a: jnp.zeros(args.num_envs, dtype=bool) for a in dummy_env.agents}
    dones['__all__'] = jnp.zeros(args.num_envs, dtype=bool)
    rnn_carry = actor.initialize_carry(args.num_envs, num_agents)
    
    print("Starting Training...")
    start_time = time.time()
    
    for update in range(1, args.total_timesteps // args.update_timestep + 1):
        train_state, env_state, obs, dones, rnn_carry, rng, metrics = train_step(
            train_state, env_state, obs, dones, rnn_carry, rng
        )
        if not jax.device_get(metrics['is_finite']):
            print(f"FAILURE: NaNs detected at update {update}. Training halted.")
            break
        
        if update % 10 == 0:
            elapsed = time.time() - start_time
            sps = (args.num_envs * args.update_timestep * 10) / elapsed
            print(f"Update {update} | SPS: {sps:.0f} | Rew: {metrics['rew']:.4f} | "
                  f"Loss: {metrics['loss']:.4f} | Val: {metrics['v_loss']:.4f}")
            start_time = time.time()
            
        if update % 500 == 0:
            os.makedirs(args.ckpt_dir, exist_ok=True)
            with open(f"{args.ckpt_dir}/update_{update}.ckpt", "wb") as f:
                f.write(to_bytes(train_state.params))
            print(f"Checkpoint saved to {args.ckpt_dir}")

if __name__ == "__main__":
    main()