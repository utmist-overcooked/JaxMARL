"""
MADDPG (Multi-Agent Deep Deterministic Policy Gradient) implementation in JAX.

Centralized training with decentralized execution:
- Each agent has a deterministic actor using only its own observation.
- A centralized critic takes all agents' observations and actions.
- Off-policy learning with replay buffer and soft target network updates.

Reference: Lowe et al., "Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments", 2017.
"""

import os
import copy
import jax
import jax.numpy as jnp
import numpy as np
from typing import Any, Dict, Tuple
from functools import partial

import chex
import optax
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
import flashbax as fbx
import hydra
from omegaconf import OmegaConf
import wandb

import sys as _sys
_sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
try:
    from utils.monitor import TrainingMonitor
    _MONITOR_AVAILABLE = True
except ImportError:
    _MONITOR_AVAILABLE = False

import jaxmarl
from jaxmarl.wrappers.baselines import (
    MPELogWrapper,
    LogWrapper,
    JaxMARLWrapper,
    save_params,
)
from jaxmarl.environments.multi_agent_env import MultiAgentEnv, State


# ───────────────────── World State Wrapper ─────────────────────

class MPEWorldStateWrapper(JaxMARLWrapper):
    """Adds a 'world_state' key to observations: concatenation of all agents' obs."""

    @partial(jax.jit, static_argnums=0)
    def reset(self, key):
        obs, env_state = self._env.reset(key)
        obs["world_state"] = self.world_state(obs)
        return obs, env_state

    @partial(jax.jit, static_argnums=0)
    def step(self, key, state, action):
        obs, env_state, reward, done, info = self._env.step(key, state, action)
        obs["world_state"] = self.world_state(obs)
        return obs, env_state, reward, done, info

    @partial(jax.jit, static_argnums=0)
    def world_state(self, obs):
        all_obs = jnp.concatenate(
            [obs[agent] for agent in self._env.agents], axis=-1
        )
        return all_obs

    def world_state_size(self):
        return sum(
            self._env.observation_space(a).shape[-1] for a in self._env.agents
        )

    def total_action_size(self):
        return sum(
            self._env.action_space(a).shape[0] for a in self._env.agents
        )


# ───────────────────── Networks ─────────────────────

class Actor(nn.Module):
    """Deterministic policy: obs -> action in [0, 1]."""
    action_dim: int
    hidden_size: int = 64
    num_layers: int = 2

    @nn.compact
    def __call__(self, obs: jnp.ndarray):
        x = obs
        for _ in range(self.num_layers):
            x = nn.Dense(
                self.hidden_size,
                kernel_init=orthogonal(np.sqrt(2)),
                bias_init=constant(0.0),
            )(x)
            x = nn.relu(x)
        x = nn.Dense(
            self.action_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
        )(x)
        return nn.sigmoid(x)


class CentralizedCritic(nn.Module):
    """Q(all_obs, all_actions, agent_id) -> scalar Q-value for that agent."""
    hidden_size: int = 64
    num_layers: int = 2

    @nn.compact
    def __call__(
        self,
        world_state: jnp.ndarray,
        all_actions: jnp.ndarray,
        agent_id: jnp.ndarray,
    ):
        x = jnp.concatenate([world_state, all_actions, agent_id], axis=-1)
        for _ in range(self.num_layers):
            x = nn.Dense(
                self.hidden_size,
                kernel_init=orthogonal(np.sqrt(2)),
                bias_init=constant(0.0),
            )(x)
            x = nn.relu(x)
        x = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(x)
        return jnp.squeeze(x, axis=-1)


# ───────────────────── Train State & Transition ─────────────────────

class MADDPGTrainState(TrainState):
    target_network_params: Any
    timesteps: int = 0
    n_updates: int = 0
    grad_steps: int = 0


@chex.dataclass(frozen=True)
class Timestep:
    obs: dict
    actions: dict
    rewards: dict
    dones: dict
    world_state: jnp.ndarray


# ───────────────────── Training ─────────────────────

def make_train(config, env, monitor=None):

    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )

    num_agents = env.num_agents
    agents = env.agents

    def batchify(x: dict):
        return jnp.stack([x[agent] for agent in agents], axis=0)

    def unbatchify(x: jnp.ndarray):
        return {agent: x[i] for i, agent in enumerate(agents)}

    def train(rng):

        original_seed = rng[0]

        # ── INIT ENV ──
        rng, _rng = jax.random.split(rng)
        wrapped_env = MPEWorldStateWrapper(env)
        wrapped_env = MPELogWrapper(wrapped_env)

        test_env = MPEWorldStateWrapper(env)
        test_env = MPELogWrapper(test_env)

        obs_dim = env.observation_space(agents[0]).shape[0]
        action_dim = env.action_space(agents[0]).shape[0]
        world_state_size = sum(
            env.observation_space(a).shape[-1] for a in agents
        )
        total_action_size = sum(
            env.action_space(a).shape[0] for a in agents
        )

        # ── INIT NETWORKS ──
        actor_network = Actor(
            action_dim=action_dim,
            hidden_size=config["ACTOR_HIDDEN_SIZE"],
            num_layers=config["NUM_LAYERS"],
        )
        critic_network = CentralizedCritic(
            hidden_size=config["CRITIC_HIDDEN_SIZE"],
            num_layers=config["NUM_LAYERS"],
        )

        # Agent one-hot IDs for critic conditioning
        agent_onehots = jnp.eye(num_agents)  # (num_agents, num_agents)

        rng, rng_actor, rng_critic = jax.random.split(rng, 3)
        dummy_obs = jnp.zeros((1, obs_dim))
        dummy_world_state = jnp.zeros((1, world_state_size))
        dummy_actions = jnp.zeros((1, total_action_size))
        dummy_agent_id = jnp.zeros((1, num_agents))

        actor_params = actor_network.init(rng_actor, dummy_obs)
        critic_params = critic_network.init(
            rng_critic, dummy_world_state, dummy_actions, dummy_agent_id
        )

        actor_tx = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.adam(config["ACTOR_LR"]),
        )
        critic_tx = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.adam(config["CRITIC_LR"]),
        )

        actor_train_state = MADDPGTrainState.create(
            apply_fn=actor_network.apply,
            params=actor_params,
            target_network_params=actor_params,
            tx=actor_tx,
        )
        critic_train_state = MADDPGTrainState.create(
            apply_fn=critic_network.apply,
            params=critic_params,
            target_network_params=critic_params,
            tx=critic_tx,
        )

        # ── INIT BUFFER ──
        buffer = fbx.make_flat_buffer(
            max_length=int(config["BUFFER_SIZE"]),
            min_length=int(config["BUFFER_BATCH_SIZE"]),
            sample_batch_size=int(config["BUFFER_BATCH_SIZE"]),
            add_sequences=False,
            add_batch_size=int(config["NUM_ENVS"] * config["NUM_STEPS"]),
        )
        buffer = buffer.replace(
            init=jax.jit(buffer.init),
            add=jax.jit(buffer.add, donate_argnums=0),
            sample=jax.jit(buffer.sample),
            can_sample=jax.jit(buffer.can_sample),
        )

        # Create dummy timestep for buffer init
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        init_obs, init_env_state = jax.vmap(wrapped_env.reset)(reset_rng)

        dummy_actions_dict = {
            agent: jnp.zeros((config["NUM_ENVS"], action_dim))
            for agent in agents
        }
        rng, _rng = jax.random.split(rng)
        step_rng = jax.random.split(_rng, config["NUM_ENVS"])
        next_obs, _, rewards, dones, _ = jax.vmap(wrapped_env.step)(
            step_rng, init_env_state, dummy_actions_dict
        )

        init_agent_obs = {agent: init_obs[agent] for agent in agents}
        dummy_timestep = Timestep(
            obs=init_agent_obs,
            actions=dummy_actions_dict,
            rewards=rewards,
            dones=dones,
            world_state=init_obs["world_state"],
        )
        dummy_timestep_single = jax.tree.map(lambda x: x[0], dummy_timestep)
        buffer_state = buffer.init(dummy_timestep_single)

        # Re-init env for actual training
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obs, env_state = jax.vmap(wrapped_env.reset)(reset_rng)
        expl_state = (obs, env_state)

        # ── TEST FUNCTION ──
        def get_greedy_metrics(rng, actor_train_state):
            if not config.get("TEST_DURING_TRAINING", True):
                return None

            def _greedy_env_step(step_state, unused):
                env_state, last_obs, rng = step_state
                rng, key_s = jax.random.split(rng)
                obs_batch = batchify(last_obs)  # (num_agents, test_num_envs, obs_dim)
                actions = jax.vmap(actor_network.apply, in_axes=(None, 0))(
                    actor_train_state.params, obs_batch
                )  # (num_agents, test_num_envs, action_dim)
                actions_dict = unbatchify(actions)
                rng_step = jax.random.split(key_s, config["TEST_NUM_ENVS"])
                obs, env_state, rewards, dones, infos = jax.vmap(test_env.step)(
                    rng_step, env_state, actions_dict
                )
                return (env_state, obs, rng), (rewards, dones, infos)

            rng, _rng = jax.random.split(rng)
            test_reset_rng = jax.random.split(_rng, config["TEST_NUM_ENVS"])
            init_obs, env_state = jax.vmap(test_env.reset)(test_reset_rng)
            rng, _rng = jax.random.split(rng)
            step_state = (env_state, init_obs, _rng)
            step_state, (rewards, dones, infos) = jax.lax.scan(
                _greedy_env_step, step_state, None, config["TEST_NUM_STEPS"]
            )
            metrics = jax.tree.map(
                lambda x: jnp.nanmean(
                    jnp.where(infos["returned_episode"], x, jnp.nan)
                ),
                infos,
            )
            return metrics

        rng, _rng = jax.random.split(rng)
        test_state = get_greedy_metrics(_rng, actor_train_state)

        # ── MAIN TRAINING LOOP ──
        def _update_step(runner_state, unused):
            (
                actor_train_state,
                critic_train_state,
                buffer_state,
                expl_state,
                test_state,
                rng,
            ) = runner_state

            # ── SAMPLE PHASE ──
            def _step_env(carry, _):
                last_obs, env_state, rng = carry
                rng, rng_a, rng_s, rng_noise = jax.random.split(rng, 4)

                # Deterministic actions from actor
                obs_batch = batchify(last_obs)  # (num_agents, num_envs, obs_dim)
                actions = jax.vmap(actor_network.apply, in_axes=(None, 0))(
                    actor_train_state.params, obs_batch
                )  # (num_agents, num_envs, action_dim)

                # Gaussian exploration noise
                noise = (
                    jax.random.normal(rng_noise, shape=actions.shape)
                    * config["NOISE_STD"]
                )
                noise = jnp.clip(noise, -config["NOISE_CLIP"], config["NOISE_CLIP"])
                noisy_actions = jnp.clip(actions + noise, 0.0, 1.0)

                actions_dict = unbatchify(noisy_actions)

                # Step environment
                rng_step = jax.random.split(rng_s, config["NUM_ENVS"])
                new_obs, new_env_state, reward, done, info = jax.vmap(
                    wrapped_env.step
                )(rng_step, env_state, actions_dict)

                # Exclude "world_state" from obs to avoid storing it twice
                agent_obs = {agent: last_obs[agent] for agent in agents}
                timestep = Timestep(
                    obs=agent_obs,
                    actions=actions_dict,
                    rewards=reward,
                    dones=done,
                    world_state=last_obs["world_state"],
                )
                return (new_obs, new_env_state, rng), (timestep, info)

            rng, _rng = jax.random.split(rng)
            carry, (timesteps, infos) = jax.lax.scan(
                _step_env,
                (*expl_state, _rng),
                None,
                config["NUM_STEPS"],
            )
            expl_state = carry[:2]

            actor_train_state = actor_train_state.replace(
                timesteps=actor_train_state.timesteps
                + config["NUM_STEPS"] * config["NUM_ENVS"]
            )

            # ── BUFFER UPDATE ──
            timesteps_flat = jax.tree.map(
                lambda x: x.reshape(-1, *x.shape[2:]), timesteps
            )
            buffer_state = buffer.add(buffer_state, timesteps_flat)

            # ── LEARN PHASE ──
            def _learn_phase(carry, _):
                actor_ts, critic_ts, rng = carry
                rng, _rng = jax.random.split(rng)
                minibatch = buffer.sample(buffer_state, _rng).experience

                batch_size = config["BUFFER_BATCH_SIZE"]

                # Current timestep data
                curr_obs_batch = batchify(
                    minibatch.first.obs
                )  # (num_agents, batch, obs_dim)
                curr_actions_batch = batchify(
                    minibatch.first.actions
                )  # (num_agents, batch, act_dim)
                curr_world_state = (
                    minibatch.first.world_state
                )  # (batch, world_state_size)
                curr_actions_flat = curr_actions_batch.swapaxes(0, 1).reshape(
                    batch_size, -1
                )  # (batch, num_agents * act_dim)
                rewards_all = batchify(
                    minibatch.first.rewards
                )  # (num_agents, batch)
                dones_all = minibatch.first.dones["__all__"]  # (batch,)

                # Next timestep data
                next_obs_batch = batchify(
                    minibatch.second.obs
                )  # (num_agents, batch, obs_dim)
                next_world_state = (
                    minibatch.second.world_state
                )  # (batch, world_state_size)

                # Target next actions from target actor
                target_next_actions = jax.vmap(
                    actor_network.apply, in_axes=(None, 0)
                )(
                    actor_ts.target_network_params, next_obs_batch
                )  # (num_agents, batch, act_dim)
                target_next_actions_flat = target_next_actions.swapaxes(0, 1).reshape(
                    batch_size, -1
                )

                # Per-agent one-hot IDs broadcast to batch:
                # agent_ids_batch: (num_agents, batch, num_agents)
                agent_ids_batch = jnp.broadcast_to(
                    agent_onehots[:, None, :],
                    (num_agents, batch_size, num_agents),
                )

                # Target Q-value per agent (vmap critic over agents)
                def _target_q_per_agent(agent_id_batch):
                    return critic_network.apply(
                        critic_ts.target_network_params,
                        next_world_state,
                        target_next_actions_flat,
                        agent_id_batch,
                    )

                target_q = jax.vmap(_target_q_per_agent)(
                    agent_ids_batch
                )  # (num_agents, batch)

                # Per-agent targets: y_i = r_i + gamma * (1-done) * Q_target_i
                targets = rewards_all + config["GAMMA"] * (
                    1 - dones_all[None, :]
                ) * jax.lax.stop_gradient(target_q)  # (num_agents, batch)

                # ── CRITIC LOSS ──
                def _critic_loss_fn(critic_params):
                    def _q_per_agent(agent_id_batch):
                        return critic_network.apply(
                            critic_params,
                            curr_world_state,
                            curr_actions_flat,
                            agent_id_batch,
                        )

                    q_pred = jax.vmap(_q_per_agent)(
                        agent_ids_batch
                    )  # (num_agents, batch)
                    loss = jnp.mean((q_pred - jax.lax.stop_gradient(targets)) ** 2)
                    return loss, q_pred.mean()

                (critic_loss, q_mean), critic_grads = jax.value_and_grad(
                    _critic_loss_fn, has_aux=True
                )(critic_ts.params)

                # Save pre-update critic params for actor loss
                critic_params_for_actor = critic_ts.params
                critic_ts = critic_ts.apply_gradients(grads=critic_grads)

                # ── ACTOR LOSS ──
                def _actor_loss_fn(actor_params):
                    new_actions = jax.vmap(
                        actor_network.apply, in_axes=(None, 0)
                    )(
                        actor_params, curr_obs_batch
                    )  # (num_agents, batch, act_dim)
                    new_actions_flat = new_actions.swapaxes(0, 1).reshape(
                        batch_size, -1
                    )

                    # Q-value per agent with pre-update critic
                    def _q_per_agent(agent_id_batch):
                        return critic_network.apply(
                            critic_params_for_actor,
                            curr_world_state,
                            new_actions_flat,
                            agent_id_batch,
                        )

                    q_vals = jax.vmap(_q_per_agent)(
                        agent_ids_batch
                    )  # (num_agents, batch)
                    actor_loss = -jnp.mean(q_vals)
                    reg_loss = config["ACTION_REG"] * jnp.mean(new_actions ** 2)
                    return actor_loss + reg_loss, -actor_loss

                (actor_loss, q_actor), actor_grads = jax.value_and_grad(
                    _actor_loss_fn, has_aux=True
                )(actor_ts.params)
                actor_ts = actor_ts.apply_gradients(grads=actor_grads)
                actor_ts = actor_ts.replace(grad_steps=actor_ts.grad_steps + 1)
                critic_ts = critic_ts.replace(grad_steps=critic_ts.grad_steps + 1)

                return (actor_ts, critic_ts, rng), (critic_loss, actor_loss, q_mean)

            rng, _rng = jax.random.split(rng)
            is_learn_time = buffer.can_sample(buffer_state) & (
                actor_train_state.timesteps > config["LEARNING_STARTS"]
            )

            (actor_train_state, critic_train_state, rng), (
                critic_loss,
                actor_loss,
                qvals,
            ) = jax.lax.cond(
                is_learn_time,
                lambda a, c, r: jax.lax.scan(
                    _learn_phase, (a, c, r), None, config["NUM_EPOCHS"]
                ),
                lambda a, c, r: (
                    (a, c, r),
                    (
                        jnp.zeros(config["NUM_EPOCHS"]),
                        jnp.zeros(config["NUM_EPOCHS"]),
                        jnp.zeros(config["NUM_EPOCHS"]),
                    ),
                ),
                actor_train_state,
                critic_train_state,
                _rng,
            )

            # ── TARGET NETWORK UPDATES ──
            actor_train_state = jax.lax.cond(
                actor_train_state.n_updates % config["TARGET_UPDATE_INTERVAL"] == 0,
                lambda ts: ts.replace(
                    target_network_params=optax.incremental_update(
                        ts.params, ts.target_network_params, config["TAU"]
                    )
                ),
                lambda ts: ts,
                operand=actor_train_state,
            )
            critic_train_state = jax.lax.cond(
                critic_train_state.n_updates % config["TARGET_UPDATE_INTERVAL"] == 0,
                lambda ts: ts.replace(
                    target_network_params=optax.incremental_update(
                        ts.params, ts.target_network_params, config["TAU"]
                    )
                ),
                lambda ts: ts,
                operand=critic_train_state,
            )

            # ── METRICS ──
            actor_train_state = actor_train_state.replace(
                n_updates=actor_train_state.n_updates + 1
            )
            critic_train_state = critic_train_state.replace(
                n_updates=critic_train_state.n_updates + 1
            )

            metrics = {
                "env_step": actor_train_state.timesteps,
                "update_steps": actor_train_state.n_updates,
                "grad_steps": actor_train_state.grad_steps,
                "critic_loss": critic_loss.mean(),
                "actor_loss": actor_loss.mean(),
                "qvals": qvals.mean(),
            }
            metrics.update(jax.tree.map(lambda x: x.mean(), infos))

            if config.get("TEST_DURING_TRAINING", True):
                rng, _rng = jax.random.split(rng)
                test_state = jax.lax.cond(
                    actor_train_state.n_updates
                    % int(config["NUM_UPDATES"] * config["TEST_INTERVAL"])
                    == 0,
                    lambda _: get_greedy_metrics(_rng, actor_train_state),
                    lambda _: test_state,
                    operand=None,
                )
                metrics.update({"test_" + k: v for k, v in test_state.items()})

            # log to wandb every step, but only print to stdout periodically
            def _log_callback(metrics, original_seed):
                step = int(metrics["env_step"])
                updates = int(metrics["update_steps"])
                num_updates = int(config["NUM_UPDATES"])
                c_loss = float(metrics["critic_loss"])
                a_loss = float(metrics["actor_loss"])
                q = float(metrics["qvals"])
                ret = float(metrics.get("returned_episode_returns", 0.0))
                test_ret = float(metrics.get("test_returned_episode_returns", 0.0))

                # ── Rich monitor update ──
                if monitor is not None:
                    monitor.update(
                        step=updates,
                        metrics={
                            "env_step": step,
                            "update": f"{updates}/{num_updates}",
                            "critic_loss": c_loss,
                            "actor_loss": a_loss,
                            "qvals": q,
                            "train_return": ret,
                            "test_return": test_ret,
                        },
                        seed=int(original_seed),
                    )

                # ── stdout fallback (only when monitor is not active) ──
                steps_per_update = config["NUM_STEPS"] * config["NUM_ENVS"]
                print_interval = int(config.get("PRINT_INTERVAL", 1_000_000))
                should_print = (
                    config["WANDB_MODE"] == "disabled"
                    or step % print_interval < steps_per_update
                    or updates == num_updates
                )
                if monitor is None and should_print:
                    print(
                        f"  step={step:>8d}  update={updates:>5d}/{num_updates}"
                        f"  critic_loss={c_loss:>8.4f}  actor_loss={a_loss:>8.4f}"
                        f"  qvals={q:>8.4f}  train_ret={ret:>8.2f}  test_ret={test_ret:>8.2f}"
                    )

                # ── W&B logging (unchanged) ──
                if config["WANDB_MODE"] != "disabled":
                    if config.get("WANDB_LOG_ALL_SEEDS", False):
                        metrics.update(
                            {
                                f"rng{int(original_seed)}/{k}": v
                                for k, v in metrics.items()
                            }
                        )
                    wandb.log(metrics, step=metrics["update_steps"])

            jax.debug.callback(_log_callback, metrics, original_seed)

            runner_state = (
                actor_train_state,
                critic_train_state,
                buffer_state,
                expl_state,
                test_state,
                rng,
            )
            return runner_state, None

        # ── RUN ──
        rng, _rng = jax.random.split(rng)
        runner_state = (
            actor_train_state,
            critic_train_state,
            buffer_state,
            expl_state,
            test_state,
            _rng,
        )
        runner_state, _ = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state}

    return train


# ───────────────────── Entry Point ─────────────────────

def env_from_config(config):
    env_name = config["ENV_NAME"]
    if "mpe" in env_name.lower():
        env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    else:
        env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    return env, env_name


def single_run(config):
    config = {**config, **config["alg"]}
    print("Config:\n", OmegaConf.to_yaml(config))

    alg_name = config["ALG_NAME"]
    env, env_name = env_from_config(copy.deepcopy(config))

    wandb.init(
        entity=config.get("ENTITY", "zacharytang24-"),
        project=config.get("PROJECT", "jaxmarl"),
        tags=[
            alg_name.upper(),
            env_name.upper(),
            f"jax_{jax.__version__}",
        ],
        name=f"{alg_name}_{env_name}",
        config=config,
        mode=config["WANDB_MODE"],
    )

    rng = jax.random.PRNGKey(config["SEED"])
    rngs = jax.random.split(rng, config["NUM_SEEDS"])

    # ── Rich monitor setup ──
    use_monitor = config.get("USE_RICH_MONITOR", True) and _MONITOR_AVAILABLE
    num_updates = int(
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    monitor = None
    if use_monitor:
        monitor = TrainingMonitor(
            total_updates=num_updates,
            config_dict={
                "env": config["ENV_NAME"],
                "total_timesteps": int(config["TOTAL_TIMESTEPS"]),
                "num_updates": num_updates,
                "num_envs": config["NUM_ENVS"],
                "num_seeds": config["NUM_SEEDS"],
                "actor_lr": config["ACTOR_LR"],
                "critic_lr": config["CRITIC_LR"],
                "gamma": config["GAMMA"],
            },
            title=f"MADDPG - {env_name}",
        )

    print("Creating training function...")
    train_vjit = jax.jit(jax.vmap(make_train(config, env, monitor=monitor)))
    print("JIT compiling and running training (this may take several minutes)...")
    try:
        if monitor is not None:
            with monitor:
                outs = jax.block_until_ready(train_vjit(rngs))
        else:
            outs = jax.block_until_ready(train_vjit(rngs))
    except Exception as e:
        print(f"Training failed: {e}")
        raise
    print("Training complete!")

    # save params
    if config.get("SAVE_PATH", None) is not None:
        actor_state = outs["runner_state"][0]
        critic_state = outs["runner_state"][1]
        save_dir = os.path.join(config["SAVE_PATH"], env_name)
        os.makedirs(save_dir, exist_ok=True)
        OmegaConf.save(
            config,
            os.path.join(
                save_dir,
                f'{alg_name}_{env_name}_seed{config["SEED"]}_config.yaml',
            ),
        )
        for i, seed_rng in enumerate(rngs):
            actor_params = jax.tree.map(lambda x: x[i], actor_state.params)
            save_path = os.path.join(
                save_dir,
                f'{alg_name}_{env_name}_seed{config["SEED"]}_vmap{i}_actor.safetensors',
            )
            save_params(actor_params, save_path)

            critic_params = jax.tree.map(lambda x: x[i], critic_state.params)
            save_path = os.path.join(
                save_dir,
                f'{alg_name}_{env_name}_seed{config["SEED"]}_vmap{i}_critic.safetensors',
            )
            save_params(critic_params, save_path)


def tune(default_config):
    """Hyperparameter sweep with wandb."""
    default_config = {**default_config, **default_config["alg"]}
    env_name = default_config["ENV_NAME"]
    alg_name = default_config["ALG_NAME"]
    env, env_name = env_from_config(default_config)

    def wrapped_make_train():
        wandb.init(project=default_config["PROJECT"])
        config = copy.deepcopy(default_config)
        for k, v in dict(wandb.config).items():
            config[k] = v
        print("running experiment with params:", config)
        rng = jax.random.PRNGKey(config["SEED"])
        rngs = jax.random.split(rng, config["NUM_SEEDS"])
        train_vjit = jax.jit(jax.vmap(make_train(config, env)))
        outs = jax.block_until_ready(train_vjit(rngs))

    sweep_config = {
        "name": f"{alg_name}_{env_name}",
        "method": "bayes",
        "metric": {
            "name": "test_returned_episode_returns",
            "goal": "maximize",
        },
        "parameters": {
            "ACTOR_LR": {"values": [0.01, 0.005, 0.001, 0.0005, 0.0001]},
            "CRITIC_LR": {"values": [0.01, 0.005, 0.001, 0.0005, 0.0001]},
            "NUM_ENVS": {"values": [8, 32, 64, 128]},
        },
    }

    wandb.login()
    sweep_id = wandb.sweep(
        sweep_config,
        entity=default_config["ENTITY"],
        project=default_config["PROJECT"],
    )
    wandb.agent(sweep_id, wrapped_make_train, count=300)


@hydra.main(version_base=None, config_path="./config", config_name="config")
def main(config):
    config = OmegaConf.to_container(config)
    print("Config:\n", OmegaConf.to_yaml(config))
    if config["HYP_TUNE"]:
        tune(config)
    else:
        single_run(config)


if __name__ == "__main__":
    main()
