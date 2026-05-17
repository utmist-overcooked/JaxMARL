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

from jaxmarl import make as jaxmarl_make
from jaxmarl.wrappers.baselines import LogWrapper

# Assuming tarmac.py contains your TarMAC, TarMACConfig, and CentralizedCritic
from tarmac import TarMAC, TarMACConfig, CentralizedCritic
from jaxmarl.viz.overcooked_v3_visualizer import OvercookedV3Visualizer
import wandb

# --- Data Structures ---

class Transition(NamedTuple):
    obs: chex.Array          # [Batch, Agents, H, W, C]
    actions: chex.Array      # [Batch, Agents]
    rewards: chex.Array      # [Batch, Agents]
    dones: chex.Array        # [Batch, Agents]
    logits: chex.Array       # [Batch, Agents, ActDim]
    hidden_states: chex.Array      # [Batch, Agents, HiddenDim]
    next_hidden_states: chex.Array # [Batch, Agents, HiddenDim]
    done_episode: chex.Array # [Batch]
    episode_returns: chex.Array # [Batch]
    shaped_rewards: chex.Array

class AgentState(TrainState):
    critic_net: CentralizedCritic = struct.field(pytree_node=False)

def make_train(args):
    env_kwargs = {
        "layout": args.layout,
        "max_steps": args.max_steps,
        "shaped_rewards": args.use_shaped_rewards,
    } 
    
    env = jaxmarl_make(args.env, **env_kwargs)
    env = LogWrapper(env)
    
    num_agents = len(env.agents)
    action_dim = env.action_space(env.agents[0]).n
    
    vmap_step = jax.vmap(env.step, in_axes=(0, 0, 0))

    config = TarMACConfig(
        hidden_dim=args.hidden_dim,
        msg_dim=args.msg_dim,
        key_dim=args.key_dim,
        num_rounds=args.comm_rounds
    )
    
    actor = TarMAC(action_dim=action_dim, config=config)

    def train_step(state, env_state, last_obs, last_dones, rnn_carry, rng):
        
        def step_fn(carry, _):
            env_st, obs, dones, rnn, rng_key = carry
            rng_key, rng_act, rng_step = jax.random.split(rng_key, 3)

            # 1. Align multi-dimensional grid layers dynamically
            obs_tensor = jnp.stack([obs[a] for a in env.agents], axis=1)
            dones_tensor = jnp.stack([dones[a] for a in env.agents], axis=1)[..., None]

            h_in, msg_in = rnn
            rnn_masked = (jnp.where(dones_tensor > 0, 0.0, h_in), 
                          jnp.where(dones_tensor > 0, 0.0, msg_in))

            # 2. Sequence network input processing
            obs_seq = obs_tensor[None, ...]     
            dones_seq = dones_tensor[None, ...] 
            
            new_rnn, (logits_seq, _, _) = state.apply_fn(
                state.params['actor'], rnn_masked, obs_seq, dones_seq
            )
            logits = logits_seq[0]

            # 3. Sample discrete workspace kitchen actions
            pi = distrax.Categorical(logits=logits)
            actions = pi.sample(seed=rng_act) 
            actions_dict = {a: actions[:, i] for i, a in enumerate(env.agents)}

            # 4. Progress execution through parallel engine
            next_obs, next_env_st, rewards, next_dones, info = vmap_step(
                jax.random.split(rng_step, args.num_envs), env_st, actions_dict
            )
            next_dones_carry = {a: next_dones[a] for a in env.agents}

            # Metric updates extracted securely via baseline LogWrapper keys
            done_episode = info.get("returned_episode", jnp.zeros(args.num_envs, dtype=bool))
            if done_episode.ndim > 1:
                done_episode = done_episode.any(axis=-1)
            
            ep_returns = info.get("returned_episode_returns", jnp.zeros(args.num_envs))
            if ep_returns.ndim > 1:
                ep_returns = ep_returns.sum(axis=-1)

            raw_rewards = jnp.stack([rewards[a] for a in env.agents], axis=1)
            
            # 5. --- Reward Shaping Implementation ---
            shaped_info = info.get("shaped_reward", {})
            if isinstance(shaped_info, dict) and len(shaped_info) > 0:
                shaped_rews = jnp.sum(jnp.stack([shaped_info[a] for a in env.agents], axis=0), axis=0)
                if shaped_rews.ndim == 1:
                    shaped_rews = jnp.expand_dims(shaped_rews, axis=-1)
                total_rewards = raw_rewards + (shaped_rews * args.shaping_coef)
            else:
                shaped_rews = jnp.zeros((args.num_envs, num_agents)) 
                total_rewards = raw_rewards

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
                shaped_rewards=shaped_rews,
            )
            
            return (next_env_st, next_obs, next_dones_carry, new_rnn, rng_key), trans

        rollout_init = (env_state, last_obs, last_dones, rnn_carry, rng)
        (env_state, last_obs, last_dones, rnn_carry, rng), traj = jax.lax.scan(
            step_fn, rollout_init, None, length=args.update_timestep
        )

        def loss_fn(params):
            actor_params = params['actor']
            critic_params = params['critic']
            
            hidden_batch = traj.hidden_states
            act_batch = traj.actions
            rew_batch = traj.rewards
            done_batch = traj.dones
            
            _, (logits_seq, _, _) = state.apply_fn(
                actor_params, rnn_carry, traj.obs, traj.dones[..., None]    
            )
            
            act_onehot = jax.nn.one_hot(act_batch, action_dim)
            
            def get_q(h, a): 
                return state.critic_net.apply(critic_params, h, a)
            
            q_values = jax.vmap(get_q)(hidden_batch, act_onehot).squeeze(-1) 

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
                a2c_scan_fn, bootstrap_val, (team_rewards, team_dones), reverse=True
            )
            
            targets = jax.lax.stop_gradient(targets)
            td_errors = q_values - targets
            critic_loss = jnp.mean(jnp.where(jnp.abs(td_errors) < 1.0, 
                                             0.5 * td_errors ** 2, 
                                             jnp.abs(td_errors) - 0.5))

            q_mean = jnp.mean(q_values)
            q_std = jnp.std(q_values)
            normalized_q = (q_values - q_mean) / (q_std + 1e-8)
            normalized_q = jax.lax.stop_gradient(jnp.clip(normalized_q, -5.0, 5.0))

            pi = distrax.Categorical(logits=logits_seq)
            log_probs = pi.log_prob(act_batch) 
            
            pg_loss = -(log_probs * normalized_q[..., None]).mean()            
            entropy = pi.entropy().mean()

            total_loss = pg_loss + args.value_loss_coef * critic_loss - args.entropy_coef * entropy

            scaled_shaped_rews = traj.shaped_rewards * args.shaping_coef
            
            raw_rewards = rew_batch - scaled_shaped_rews
            avg_raw_reward = raw_rewards.sum(axis=-1).mean()
            avg_shaped_reward = scaled_shaped_rews.sum(axis=-1).mean()

            mask = traj.done_episode.flatten()
            returns = traj.episode_returns.flatten()
            sum_returns = jnp.sum(returns * mask)
            num_completed = jnp.sum(mask)
            
            mean_episode_return = jax.lax.select(
                num_completed > 0,
                sum_returns / num_completed,
                0.0 
            )
            
            action_names = {
                0: "move_right", 1: "move_down", 2: "move_left",
                3: "move_up", 4: "stay", 5: "interact"
            }
            
            total_actions_executed = act_batch.size
            act_counts = {
                f"actions/{action_names[i]}": jnp.sum(act_batch == i) / total_actions_executed 
                for i in range(6)
            }
            
 
            metrics_out = {
                'loss': total_loss, 
                'pg_loss': pg_loss, 
                'v_loss': critic_loss, 
                'ent': entropy,
                'rew/total_step_reward': team_rewards.mean(), 
                'rew/raw_delivery_step_reward': avg_raw_reward,
                'rew/shaped_step_reward': avg_shaped_reward,
                'rew/mean_episode_return': mean_episode_return, 
                'num_ep_completed': num_completed
            }
            
            metrics_out.update(act_counts)
            return total_loss, metrics_out

        grads, metrics = jax.grad(loss_fn, has_aux=True)(state.params)
        state = state.apply_gradients(grads=grads)
        metrics['is_finite'] = jnp.all(jnp.isfinite(jax.flatten_util.ravel_pytree(state.params)[0])).astype(jnp.float32)

        return state, env_state, last_obs, last_dones, rnn_carry, rng, metrics

    return train_step

def generate_validation_video(state, params, env_kwargs, max_steps, actor_module, num_envs):
    """Executes parallel validation steps using STOCHASTIC SAMPLING to prevent argmax freezing."""
    from jaxmarl import make as jaxmarl_make
    
    val_env = jaxmarl_make("overcooked_v3", **env_kwargs)
    viz = OvercookedV3Visualizer(val_env)
    
    vmap_reset = jax.vmap(val_env.reset, in_axes=(0,))
    vmap_step = jax.vmap(val_env.step, in_axes=(0, 0, 0))
    
    rng = jax.random.PRNGKey(42)
    rng, rng_reset = jax.random.split(rng)
    obs, env_state = vmap_reset(jax.random.split(rng_reset, num_envs))
    
    num_agents = len(val_env.agents)
    rnn_carry = actor_module.initialize_carry(batch_size=num_envs, num_agents=num_agents)
    dones = {a: jnp.zeros(num_envs, dtype=bool) for a in val_env.agents}
    
    frames = []
    
    for _ in range(max_steps):
        display_state = jax.device_get(jax.tree_util.tree_map(lambda x: x[0], env_state))
        frame = viz.render_state(display_state)
        frames.append(frame)
        
        obs_tensor = jnp.stack([obs[a] for a in val_env.agents], axis=1) 
        dones_tensor = jnp.stack([dones[a] for a in val_env.agents], axis=1)[..., None] 
        
        obs_seq = obs_tensor[None, ...]     
        dones_seq = dones_tensor[None, ...] 
        
        h_in, msg_in = rnn_carry
        rnn_masked = (jnp.where(dones_tensor > 0, 0.0, h_in), 
                      jnp.where(dones_tensor > 0, 0.0, msg_in))
        
        _, (logits_seq, _, _) = actor_module.apply(
            params['actor'], rnn_masked, obs_seq, dones_seq
        )
        logits = logits_seq[0] 
        
        rng, rng_act = jax.random.split(rng)
        actions = jax.random.categorical(rng_act, logits, axis=-1)
        actions_dict = {a: actions[:, i] for i, a in enumerate(val_env.agents)}
        
        rng, rng_step = jax.random.split(rng)
        obs, env_state, _, next_dones, _ = vmap_step(
            jax.random.split(rng_step, num_envs), env_state, actions_dict
        )
        
        env_state = jax.block_until_ready(env_state) 
        
        dones = {a: next_dones[a] for a in val_env.agents}
        if next_dones["__all__"].all():
            break
            
    return frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="overcooked_v3")
    parser.add_argument("--layout", type=str, default="cramped_room")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_envs", type=int, default=32)
    parser.add_argument("--update_timestep", type=int, default=16) 
    parser.add_argument("--total_timesteps", type=int, default=20_000_000)
    
    parser.add_argument("--lr", type=float, default=1.5e-5)
    parser.add_argument("--alpha", type=float, default=0.99)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--max_grad_norm", type=float, default=0.05)
    parser.add_argument("--value_loss_coef", type=float, default=0.5)
    parser.add_argument("--entropy_coef", type=float, default=0.50)
    parser.add_argument("--shaping_coef", type=float, default=8.0)
    parser.add_argument("--max_steps", type=int, default=400)
    parser.add_argument("--use_shaped_rewards", action="store_true", default=True)
    
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--msg_dim", type=int, default=32)
    parser.add_argument("--key_dim", type=int, default=16)
    parser.add_argument("--comm_rounds", type=int, default=2)
    parser.add_argument("--ckpt_dir", type=str, default="./checkpoints")

    args = parser.parse_args()
    env_kwargs = {"layout": args.layout, "max_steps": args.max_steps, "shaped_rewards": args.use_shaped_rewards}

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"TarMAC_{args.env}_{args.layout}_H{args.hidden_dim}_{timestamp}"

    wandb.init(project="JaxMARL-OvercookedV3", name=run_name, config=vars(args))

    rng = jax.random.PRNGKey(args.seed)
    rng, rng_init = jax.random.split(rng)
    
    dummy_env = jaxmarl_make(args.env, **env_kwargs)
    num_agents = len(dummy_env.agents)
    act_dim = dummy_env.action_space(dummy_env.agents[0]).n
    obs_shape = dummy_env.observation_space(dummy_env.agents[0]).shape 

    actor = TarMAC(act_dim, TarMACConfig(hidden_dim=args.hidden_dim, msg_dim=args.msg_dim, key_dim=args.key_dim, num_rounds=args.comm_rounds))
    critic = CentralizedCritic()
    
    dummy_carry = actor.initialize_carry(args.num_envs, num_agents)
    dummy_obs = jnp.zeros((1, args.num_envs, num_agents, *obs_shape))
    dummy_dones = jnp.zeros((1, args.num_envs, num_agents, 1))

    actor_params = actor.init(rng_init, dummy_carry, dummy_obs, dummy_dones)    
    critic_params = critic.init(rng_init, dummy_carry[0], jnp.zeros((args.num_envs, num_agents, act_dim)))

    total_updates = args.total_timesteps // (args.num_envs * args.update_timestep)
    lr_schedule = optax.linear_schedule(init_value=args.lr, end_value=1e-5, transition_steps=total_updates)
    
    train_state = AgentState.create(
        apply_fn=actor.apply,
        params={'actor': actor_params, 'critic': critic_params},
        tx=optax.chain(
            optax.clip_by_global_norm(args.max_grad_norm), 
            optax.rmsprop(
                learning_rate=lr_schedule, 
                decay=args.alpha,
                eps=1e-5 
            ) 
        ),
        critic_net=critic
    )

    train_step = jax.jit(make_train(args))
    
    rng, rng_reset = jax.random.split(rng)
    env = LogWrapper(jaxmarl_make(args.env, **env_kwargs))
    vmap_reset = jax.vmap(env.reset, in_axes=(0,))
    
    obs, env_state = vmap_reset(jax.random.split(rng_reset, args.num_envs))
    dones = {a: jnp.zeros(args.num_envs, dtype=bool) for a in dummy_env.agents}
    rnn_carry = actor.initialize_carry(args.num_envs, num_agents)
    
    print(f"Training on layout: {args.layout}...")
    print("Starting Training...")
    start_time = time.time()
    
    for update in range(1, total_updates + 1):
        train_state, env_state, obs, dones, rnn_carry, rng, metrics = train_step(
            train_state, env_state, obs, dones, rnn_carry, rng
        )
        
        if not jax.device_get(metrics['is_finite']):
            print(f"NaN detected at update {update}. Aborting.")
            break
        
        if update % 10 == 0:
            elapsed = time.time() - start_time
            sps = (args.num_envs * args.update_timestep * 10) / elapsed
            
            # 💥 FIXED: Extract semantic event scores to display directly in stdout logs
            potted = float(metrics.get('events/put_in_pot', 0.0)) * (args.update_timestep * args.num_envs)
            served = float(metrics.get('events/delivery', 0.0)) * (args.update_timestep * args.num_envs)
            burnt  = float(metrics.get('events/burn_pot', 0.0)) * (args.update_timestep * args.num_envs)
            
            # Expanded diagnostic string layout
            print(
                f"Update {update:5d}/{total_updates} | SPS: {sps:4.0f} | "
                f"Total Rew: {metrics['rew/total_step_reward']:6.3f} | "
                f"Potted: {potted:3.0f} | Served: {served:2.0f} | Burnt: {burnt:2.0f}"
            )
            
            wandb_metrics = {
                "train/loss": metrics['loss'],
                "train/value_loss": metrics['v_loss'],
                "train/policy_entropy": metrics['ent'],
                "rewards/total_step_reward": metrics['rew/total_step_reward'],
                "rewards/raw_delivery_step_reward": metrics['rew/raw_delivery_step_reward'],
                "rewards/shaped_step_reward": metrics['rew/shaped_step_reward'],
                "rewards/mean_episode_return": metrics['rew/mean_episode_return'],
                "charts/SPS": sps,
                "global_step": update * args.num_envs * args.update_timestep
            }
            
            for key, val in metrics.items():
                if key.startswith("actions/") or key.startswith("events/"):
                    wandb_metrics[key] = val

            wandb.log(wandb_metrics)
            start_time = time.time()
            
        if update % 5000 == 0:
            import os
            import pickle
            os.makedirs("checkpoints", exist_ok=True)
            checkpoint_path = f"checkpoints/tarmac_cramped_room_update_{update}.pkl"
            with open(checkpoint_path, "wb") as f:
                pickle.dump(jax.device_get(train_state.params), f)
            print(f"💾 Checkpoint safely backed up to {checkpoint_path}")

        if update % 1000 == 0:
            print(f"🎬 Compiling visual rollout GIF at update {update}...")
            import imageio
            
            gif_frames = generate_validation_video(
                state=train_state,
                params=train_state.params,
                env_kwargs=env_kwargs,
                max_steps=args.max_steps,
                actor_module=actor,
                num_envs=args.num_envs
            )
            
            local_gif_path = "current_kitchen_rollout.gif"
            imageio.mimsave(local_gif_path, gif_frames, fps=8)
            
            wandb.log({
                "media/kitchen_coordination_loop": wandb.Video(local_gif_path, format="gif"),
                "global_step": update * args.num_envs * args.update_timestep
            })
            print("Visual kitchen loop successfully synced to dashboard media pane.")

if __name__ == "__main__":
    main()