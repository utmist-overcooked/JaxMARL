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
import datetime
from flax.training.train_state import TrainState
from flax.serialization import to_bytes
from typing import Tuple, Dict, Any, NamedTuple
from functools import partial

from jaxmarl import make as original_make
from jaxmarl.wrappers.baselines import LogWrapper

from tarmac import TarMAC, TarMACConfig, TarMACCell, CentralizedCritic
import wandb

from jaxmarl.registration import registered_envs
from jaxmarl.environments.traffic_junction.traffic_junction import TrafficJunction


class AutoResetWrapper:
    """Silently resets the environment when done['__all__'] is triggered."""
    def __init__(self, env):
        self._env = env
        self.agents = env.agents
        self.num_agents = getattr(env, "num_agents", len(env.agents))

    def action_space(self, agent):
        return self._env.action_space(agent)

    def observation_space(self, agent):
        return self._env.observation_space(agent)

    def reset(self, key):
        return self._env.reset(key)

    def step(self, key, state, actions):
        obs, next_state, reward, done, info = self._env.step(key, state, actions)
        
        # Generate a fresh environment state alongside the current one
        reset_obs, reset_state = self._env.reset(key)
        
        # If done['__all__'] is True, swap the next_state and obs with the fresh ones
        next_state = jax.tree_util.tree_map(
            lambda fresh, current: jnp.where(done['__all__'], fresh, current), 
            reset_state, next_state
        )
        obs = jax.tree_util.tree_map(
            lambda fresh, current: jnp.where(done['__all__'], fresh, current), 
            reset_obs, obs
        )
        
        return obs, next_state, reward, done, info
    
def make(env_id, **kwargs):
    if env_id == "TrafficJunction":
        env = TrafficJunction(**kwargs)
        return AutoResetWrapper(env)
    
    return original_make(env_id, **kwargs)

# --- Data Structures ---

class Transition(NamedTuple):
    obs: chex.Array          # [Batch, Agents, ObsDim]
    actions: chex.Array      # [Batch, Agents]
    rewards: chex.Array      # [Batch, Agents]
    dones: chex.Array        # [Batch, Agents]
    logits: chex.Array       # [Batch, Agents, ActDim]
    hidden_states: chex.Array      # [Batch, Agents, HiddenDim] (h_t)
    next_hidden_states: chex.Array # [Batch, Agents, HiddenDim] (h_{t+1})
    done_episode: chex.Array # [Batch] (optional, for episode termination)
    episode_returns: chex.Array # [Batch] (optional, for return calculation)
    collisions: chex.Array   # [Batch] NEW: Track collisions per step
    avg_time_grid: chex.Array # [Batch] NEW: Track average time active cars spent on grid

class AgentState(TrainState):
    critic_net: CentralizedCritic = struct.field(pytree_node=False)


def make_train(args):

    env_kwargs = {
        "max_steps": args.max_steps,
        "max_cars": args.max_cars,
        "grid_size": args.grid_size,
        "one_way": args.is_one_way
    } 
    
    # Environment Setup
    env = make(args.env, **env_kwargs)
    env = LogWrapper(env)
    
    num_agents = len(env.agents)
    action_dim = env.action_space(env.agents[0]).n
    
    # Parallel Rollout Setup
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

    def train_step(state, env_state, last_obs, last_dones, rnn_carry, rng, current_spawn_prob):
        
        # new_inner_state = env_state.env_state.replace(spawn_prob=jnp.full((args.num_envs,), current_spawn_prob))
        # env_state = env_state.replace(env_state=new_inner_state)
        
        # rollout function
        def step_fn(carry, _):
            env_st, obs, dones, rnn, rng_key = carry
            rng_key, rng_act, rng_step = jax.random.split(rng_key, 3)

            # 1. Prepare inputs: [Batch, Agents, ...]
            obs_tensor = jnp.stack([obs[a] for a in env.agents], axis=1)
            dones_tensor = jnp.stack([dones[a] for a in env.agents], axis=1)[..., None]

            h_in, msg_in = rnn
            # Mask hidden state for reset agents to prevent stale gradients
            rnn_masked = (jnp.where(dones_tensor > 0, 0.0, h_in), 
                          jnp.where(dones_tensor > 0, 0.0, msg_in))

            # 2. Actor Forward Pass
            obs_seq = obs_tensor[None, ...]     
            dones_seq = dones_tensor[None, ...] 
            
            new_rnn, (logits_seq, _, _) = state.apply_fn(
                state.params['actor'], rnn_masked, obs_seq, dones_seq
            )
            logits = logits_seq[0]

            # 3. Action Sampling
            pi = distrax.Categorical(logits=logits)
            actions = pi.sample(seed=rng_act) 

            actions_dict = {a: actions[:, i] for i, a in enumerate(env.agents)}

            # 4. Step Environment
            next_obs, next_env_st, rewards, next_dones, info = vmap_step(
                jax.random.split(rng_step, args.num_envs), env_st, actions_dict
            )

            # --- Extract Episode Metrics from LogWrapper ---
            done_episode = info.get("returned_episode", jnp.zeros(args.num_envs, dtype=bool))
            if done_episode.ndim > 1:
                done_episode = done_episode.any(axis=-1)
            
            ep_returns = info.get("returned_episode_returns", jnp.zeros(args.num_envs))
            if ep_returns.ndim > 1:
                ep_returns = ep_returns.sum(axis=-1)

            # --- NEW: Extract Collisions and Time on Grid ---
            collided_array = jnp.stack([info[a] for a in env.agents], axis=1) 
            step_collisions = jnp.sum(collided_array, axis=1) 

            active_cars_mask = next_env_st.env_state.active 
            step_counters = next_env_st.env_state.time_on_grid 
            sum_active = jnp.sum(active_cars_mask, axis=1)
            sum_time = jnp.sum(step_counters * active_cars_mask, axis=1)
            avg_time = jnp.where(sum_active > 0, sum_time / sum_active, 0.0) 
            # ------------------------------------------------

            # 5. --- Reward Shaping Implementation ---
            shaped_info = info.get("shaped_reward", {})
            
            if isinstance(shaped_info, dict) and len(shaped_info) > 0:
                shaped_rews = jnp.sum(jnp.stack([v for v in shaped_info.values()]), axis=0)
                if shaped_rews.ndim == 1:
                    shaped_rews = jnp.expand_dims(shaped_rews, axis=-1)
                    shaped_rews = jnp.tile(shaped_rews, (1, num_agents))
            else:
                shaped_rews = jnp.zeros((args.num_envs, num_agents))

            raw_rewards = jnp.stack([rewards[a] for a in env.agents], axis=1)
            total_rewards = raw_rewards + (shaped_rews * args.shaping_coef) 

            # 6. Prepare Transition
            next_dones_tensor = jnp.stack([next_dones[a] for a in env.agents], axis=1).astype(jnp.float32)[..., None]
            
            trans = Transition(
                obs=obs_tensor,
                actions=actions,
                rewards=total_rewards,
                dones=jnp.stack([dones[a] for a in env.agents], axis=1),
                logits=logits,
                hidden_states=rnn_masked[0], 
                next_hidden_states=new_rnn[0] * (1.0 - next_dones_tensor),
                done_episode=done_episode,
                episode_returns=ep_returns,
                collisions=step_collisions,    # NEW
                avg_time_grid=avg_time         # NEW
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
            
            hidden_batch = traj.hidden_states
            act_batch = traj.actions
            rew_batch = traj.rewards
            done_batch = traj.dones
            
            # 1. ACTOR FORWARD PASS
            _, (logits_seq, _, _) = state.apply_fn(
                actor_params, 
                rnn_carry,               
                traj.obs,                
                traj.dones[..., None]    
            )
            
            # 2. CRITIC FORWARD PASS (Joint Q-Value)
            act_onehot = jax.nn.one_hot(act_batch, action_dim)
            
            def get_q(h, a): 
                return state.critic_net.apply(critic_params, h, a)
            
            # q_values shape: [Time, Batch]
            q_values = jax.vmap(get_q)(hidden_batch, act_onehot).squeeze(-1) 

            # ==========================================
            # 3. TRUE A2C TARGETS (Discounted Returns)
            # ==========================================
            team_rewards = rew_batch.sum(axis=-1)
            team_dones = done_batch.all(axis=-1).reshape(args.update_timestep, args.num_envs)

            next_hidden_batch = jnp.concatenate([hidden_batch[1:], rnn_carry[0][None, ...]], axis=0)
            next_act_onehot = jnp.concatenate([act_onehot[1:], jnp.zeros_like(act_onehot[0:1])], axis=0)
            
            next_q_values = jax.vmap(get_q)(next_hidden_batch, next_act_onehot).squeeze(-1)
            bootstrap_val = next_q_values[-1]

            def a2c_scan_fn(carry, transition):
                reward, done = transition
                target = reward + args.gamma * (1.0 - done) * carry
                return target, target

            _, targets = jax.lax.scan(
                a2c_scan_fn,
                bootstrap_val, 
                (team_rewards, team_dones),
                reverse=True
            )
            
            targets = jax.lax.stop_gradient(targets)
            td_errors = q_values - targets
            critic_loss = jnp.mean(jnp.where(jnp.abs(td_errors) < 1.0, 
                                             0.5 * td_errors ** 2, 
                                             jnp.abs(td_errors) - 0.5))

            # ==========================================
            # 4. NORMALIZED JOINT Q-VALUE (TARMAC METHOD)
            # ==========================================
            # Use the Critic's q_values, not the empirical targets
            q_mean = jnp.mean(q_values)
            q_std = jnp.std(q_values)
            normalized_q = (q_values - q_mean) / (q_std + 1e-8)
            
            # Stop gradients so the Actor loss doesn't backprop into the Critic
            normalized_q = jax.lax.stop_gradient(jnp.clip(normalized_q, -5.0, 5.0))

            # ==========================================
            # 5. POLICY GRADIENT & ENTROPY
            # ==========================================
            pi = distrax.Categorical(logits=logits_seq)
            log_probs = pi.log_prob(act_batch) 
            
            pg_loss = -(log_probs * normalized_q[..., None]).mean()            
            entropy = pi.entropy().mean()

            total_loss = pg_loss + args.value_loss_coef * critic_loss - args.entropy_coef * entropy

            # 6. METRICS LOGGING
            mask = traj.done_episode.flatten()
            returns = traj.episode_returns.flatten()
            sum_returns = jnp.sum(returns * mask)
            num_completed = jnp.sum(mask)
            
            mean_episode_return = jax.lax.select(
                num_completed > 0,
                sum_returns / num_completed,
                0.0 
            )
            
            return total_loss, {
                'loss': total_loss, 'pg_loss': pg_loss, 
                'v_loss': critic_loss, 'ent': entropy,
                'rew': team_rewards.mean(), 
                'ep_ret': mean_episode_return, 
                'num_ep_completed': num_completed,
                'collisions': jnp.mean(traj.collisions),    # NEW
                'avg_time_grid': jnp.mean(traj.avg_time_grid) # NEW
            }

        grads, metrics = jax.grad(loss_fn, has_aux=True)(state.params)
        state = state.apply_gradients(grads=grads)

        params_are_finite = jnp.all(jnp.isfinite(jax.flatten_util.ravel_pytree(state.params)[0]))
        metrics['is_finite'] = params_are_finite.astype(jnp.float32)

        return state, env_state, last_obs, last_dones, rnn_carry, rng, metrics

    return train_step

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="TrafficJunction")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_envs", type=int, default=256)
    parser.add_argument("--update_timestep", type=int, default=16) 
    parser.add_argument("--total_timesteps", type=int, default=50_000_000)
    
    # Hyperparams
    parser.add_argument("--lr", type=float, default=7e-4)
    parser.add_argument("--alpha", type=float, default=0.99)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument("--value_loss_coef", type=float, default=0.5)
    parser.add_argument("--entropy_coef", type=float, default=0.01)
    parser.add_argument("--shaping_coef", type=float, default=0.5)
    parser.add_argument("--grid_size", type=int, default=7)
    parser.add_argument("--max_cars", type=int, default=5)
    parser.add_argument("--max_steps", type=int, default=20)
    parser.add_argument("--p_arrive", type=float, default=0.30)
    parser.add_argument("--is_one_way", action="store_true", default=False)
    
    # Model Params
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--msg_dim", type=int, default=32)
    parser.add_argument("--key_dim", type=int, default=16)
    parser.add_argument("--comm_rounds", type=int, default=2)
    parser.add_argument("--ckpt_dir", type=str, default="./checkpoints")



    args = parser.parse_args()
    env_kwargs = {
        "max_steps": args.max_steps,
        "max_cars": args.max_cars,
        "grid_size": args.grid_size,
        "one_way": args.is_one_way
    }

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"TarMAC_{args.env}_H{args.hidden_dim}_R{args.comm_rounds}_{timestamp}"
    run_ckpt_dir = os.path.join(args.ckpt_dir, run_name)

    wandb.init(
        project="JaxMARL-TrafficJunction",
        name=run_name,
        config=vars(args)
    )

    print(f"Initializing {args.env}...")
    rng = jax.random.PRNGKey(args.seed)
    rng, rng_init = jax.random.split(rng)
    
    dummy_env = make(args.env, **env_kwargs)
    num_agents = len(dummy_env.agents)
    act_dim = dummy_env.action_space(dummy_env.agents[0]).n
    
    tarmac_config = TarMACConfig(
        hidden_dim=args.hidden_dim, 
        msg_dim=args.msg_dim, 
        key_dim=args.key_dim, 
        num_rounds=args.comm_rounds
    )    
    actor = TarMAC(act_dim, tarmac_config)
    critic = CentralizedCritic()
    
    dummy_carry = actor.initialize_carry(args.num_envs, num_agents)
    env = make(args.env, **env_kwargs)
    obs_shape = env.observation_space(env.agents[0]).shape 

    dummy_obs = jnp.zeros((1, args.num_envs, num_agents, *obs_shape))
    dummy_dones = jnp.zeros((1, args.num_envs, num_agents, 1))

    actor_params = actor.init(rng_init, dummy_carry, dummy_obs, dummy_dones)    
    critic_params = critic.init(rng_init, dummy_carry[0], jnp.zeros((args.num_envs, num_agents, act_dim)))

    total_updates = args.total_timesteps // (args.num_envs * args.update_timestep)

    lr_schedule = optax.linear_schedule(
        init_value=args.lr,
        end_value=1e-6, 
        transition_steps=total_updates
    )
    
    train_state = AgentState.create(
        apply_fn=actor.apply,
        params={'actor': actor_params, 'critic': critic_params},
        tx=optax.chain(
            optax.clip_by_global_norm(args.max_grad_norm), 
            optax.rmsprop(learning_rate=lr_schedule, decay=args.alpha) 
        ),
        critic_net=critic
    )

    train_step = jax.jit(make_train(args))
    
    rng, rng_reset = jax.random.split(rng)
    env = make(args.env, **env_kwargs)
    env = LogWrapper(env)
    vmap_reset = jax.vmap(env.reset, in_axes=(0,))
    
    obs, env_state = vmap_reset(jax.random.split(rng_reset, args.num_envs))
    
    dones = {a: jnp.zeros(args.num_envs, dtype=bool) for a in dummy_env.agents}
    dones['__all__'] = jnp.zeros(args.num_envs, dtype=bool)
    rnn_carry = actor.initialize_carry(args.num_envs, num_agents)
    
    print("Starting Training...")
    start_time = time.time()

    
    
    for update in range(1, total_updates + 1):

        train_state, env_state, obs, dones, rnn_carry, rng, metrics = train_step(
            train_state, env_state, obs, dones, rnn_carry, rng, jnp.array(args.p_arrive) 
        )
        if not jax.device_get(metrics['is_finite']):
            print(f"FAILURE: NaNs detected at update {update}. Training halted.")
            break
        
        if update % 10 == 0:
            elapsed = time.time() - start_time
            sps = (args.num_envs * args.update_timestep * 10) / elapsed

            # NEW: Print Collisions and Time
            print(f"Update {update} | SPS: {sps:.0f} | Rew: {metrics['rew']:.4f} | "
                  f"Ent: {metrics['ent']:.3f} | Val: {metrics['v_loss']:.4f} | "
                  f"Col: {metrics['collisions']:.3f} | Time: {metrics['avg_time_grid']:.2f}")
            
            wandb.log({
                "train/step_reward": metrics['rew'],          
                "train/episode_return": metrics['ep_ret'],    
                "train/episodes_completed": metrics['num_ep_completed'], 
                "train/loss": metrics['loss'],
                "train/value_loss": metrics['v_loss'],
                "train/policy_entropy": metrics['ent'], 
                "train/policy_loss": metrics['pg_loss'],
                "train/collisions_per_step": metrics['collisions'], # NEW
                "train/avg_time_on_grid": metrics['avg_time_grid'], # NEW
                "charts/SPS": (args.num_envs * args.update_timestep * 10) / (time.time() - start_time),
                "global_step": update * args.num_envs * args.update_timestep
            })
            start_time = time.time()
            
        if update % 500 == 0:
            ckpt_path = f"{run_ckpt_dir}/update_{update}.ckpt"
            os.makedirs(run_ckpt_dir, exist_ok=True)
            with open(ckpt_path, "wb") as f:
                f.write(to_bytes(train_state.params))
            print(f"Checkpoint saved to {run_ckpt_dir}")
            
            artifact = wandb.Artifact(name="tarmac_model", type="model")
            artifact.add_file(ckpt_path)
            wandb.log_artifact(artifact)

if __name__ == "__main__":
    main()