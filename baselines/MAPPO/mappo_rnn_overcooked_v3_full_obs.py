"""
MAPPO (Multi-Agent PPO) with RNN + CNN for full-observation Overcooked V3.
Actor and critic both use CNN+RNN over each agent's full grid observation.
"""

import datetime
import os
import sys
import time
import types
from pathlib import Path
import jax
import jax.api_util
import jax.numpy as jnp
import flax.linen as nn
import optax
from flax.linen.initializers import constant, orthogonal
from flax import struct
from flax.traverse_util import flatten_dict
from typing import Callable, Sequence, NamedTuple, Any, Dict
from flax.training.train_state import TrainState
import distrax
import hydra
from omegaconf import OmegaConf
import copy
import wandb
import functools
from safetensors.flax import save_file

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "baselines"))


def _install_jaxmarl_import_shims():
    """Import Overcooked without importing optional env stacks."""
    if "jaxmarl" not in sys.modules:
        pkg = types.ModuleType("jaxmarl")
        pkg.__path__ = [str(REPO_ROOT / "jaxmarl")]
        sys.modules["jaxmarl"] = pkg

    if "jaxmarl.environments" not in sys.modules:
        env_pkg = types.ModuleType("jaxmarl.environments")
        env_pkg.__path__ = [str(REPO_ROOT / "jaxmarl" / "environments")]
        sys.modules["jaxmarl.environments"] = env_pkg

        from jaxmarl.environments import spaces
        from jaxmarl.environments.multi_agent_env import MultiAgentEnv, State

        env_pkg.spaces = spaces
        env_pkg.MultiAgentEnv = MultiAgentEnv
        env_pkg.State = State


_install_jaxmarl_import_shims()

from jaxmarl.environments.overcooked_v3 import OvercookedV3
from jaxmarl.viz.overcooked_v3_visualizer import OvercookedV3Visualizer


def save_params(params: Dict, filename: os.PathLike | str) -> None:
    save_file(flatten_dict(params, sep=","), filename)


@struct.dataclass
class LogEnvState:
    env_state: Any
    episode_returns: float
    episode_lengths: int
    returned_episode_returns: float
    returned_episode_lengths: int


class LogWrapper:
    """Minimal logging wrapper used by this standalone Overcooked baseline."""

    def __init__(self, env, replace_info: bool = False):
        self._env = env
        self.replace_info = replace_info

    def __getattr__(self, name: str):
        return getattr(self._env, name)

    def _batchify_floats(self, x: dict):
        return jnp.stack([x[a] for a in self._env.agents])

    @functools.partial(jax.jit, static_argnums=(0,))
    def reset(self, key):
        obs, env_state = self._env.reset(key)
        state = LogEnvState(
            env_state,
            jnp.zeros((self._env.num_agents,)),
            jnp.zeros((self._env.num_agents,)),
            jnp.zeros((self._env.num_agents,)),
            jnp.zeros((self._env.num_agents,)),
        )
        return obs, state

    @functools.partial(jax.jit, static_argnums=(0,))
    def step(self, key, state, action):
        obs, env_state, reward, done, info = self._env.step(
            key, state.env_state, action
        )
        ep_done = done["__all__"]
        new_episode_return = state.episode_returns + self._batchify_floats(reward)
        new_episode_length = state.episode_lengths + 1
        state = LogEnvState(
            env_state=env_state,
            episode_returns=new_episode_return * (1 - ep_done),
            episode_lengths=new_episode_length * (1 - ep_done),
            returned_episode_returns=state.returned_episode_returns * (1 - ep_done)
            + new_episode_return * ep_done,
            returned_episode_lengths=state.returned_episode_lengths * (1 - ep_done)
            + new_episode_length * ep_done,
        )
        if self.replace_info:
            info = {}
        info["returned_episode_returns"] = state.returned_episode_returns
        info["returned_episode_lengths"] = state.returned_episode_lengths
        info["returned_episode"] = jnp.full((self._env.num_agents,), ep_done)
        return obs, state, reward, done, info

try:
    from utils.monitor import TrainingMonitor

    _MONITOR_AVAILABLE = True
except ImportError:
    _MONITOR_AVAILABLE = False


class ScannedRNN(nn.Module):
    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        """Applies the module."""
        rnn_state = carry
        ins, resets = x

        new_carry = self.initialize_carry(ins.shape[0], ins.shape[1])

        rnn_state = jnp.where(
            resets[:, None],
            new_carry,
            rnn_state,
        )
        new_rnn_state, y = nn.GRUCell(features=ins.shape[1])(rnn_state, ins)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        cell = nn.GRUCell(features=hidden_size)
        return cell.initialize_carry(jax.random.PRNGKey(0), (batch_size, hidden_size))


class CNN(nn.Module):
    output_size: int = 64
    activation: Callable[..., Any] = nn.relu

    @nn.compact
    def __call__(self, x, train=False):
        x = nn.Conv(
            features=128,
            kernel_size=(1, 1),
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = self.activation(x)
        x = nn.Conv(
            features=128,
            kernel_size=(1, 1),
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = self.activation(x)
        x = nn.Conv(
            features=8,
            kernel_size=(1, 1),
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = self.activation(x)

        x = nn.Conv(
            features=16,
            kernel_size=(3, 3),
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = self.activation(x)

        x = nn.Conv(
            features=32,
            kernel_size=(3, 3),
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = self.activation(x)

        x = nn.Conv(
            features=32,
            kernel_size=(3, 3),
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = self.activation(x)

        x = x.reshape((x.shape[0], -1))

        x = nn.Dense(
            features=self.output_size,
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = self.activation(x)

        return x


class ActorRNN(nn.Module):
    action_dim: Sequence[int]
    config: Dict

    @nn.compact
    def __call__(self, hidden, x):
        obs, dones = x

        if self.config["ACTIVATION"] == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        embed_model = nn.vmap(
            CNN,
            variable_axes={"params": None},
            split_rngs={"params": False},
            in_axes=0,
            out_axes=0,
        )(
            output_size=self.config["GRU_HIDDEN_DIM"],
            activation=activation,
        )
        embedding = embed_model(obs)

        embedding = nn.LayerNorm()(embedding)

        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)

        actor_mean = nn.Dense(
            self.config["FC_DIM_SIZE"],
            kernel_init=orthogonal(2),
            bias_init=constant(0.0),
        )(embedding)
        actor_mean = nn.relu(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)

        pi = distrax.Categorical(logits=actor_mean)

        return hidden, pi


class CriticRNN(nn.Module):
    config: Dict

    @nn.compact
    def __call__(self, hidden, x):
        obs, dones = x

        if self.config["ACTIVATION"] == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        embed_model = nn.vmap(
            CNN,
            variable_axes={"params": None},
            split_rngs={"params": False},
            in_axes=0,
            out_axes=0,
        )(
            output_size=self.config["GRU_HIDDEN_DIM"],
            activation=activation,
        )
        embedding = embed_model(obs)

        embedding = nn.LayerNorm()(embedding)

        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)

        critic = nn.Dense(
            self.config["FC_DIM_SIZE"],
            kernel_init=orthogonal(2),
            bias_init=constant(0.0),
        )(embedding)
        critic = nn.relu(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return hidden, jnp.squeeze(critic, axis=-1)


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    action_mask: jnp.ndarray
    info: jnp.ndarray


def batchify(x: dict, agent_list, num_actors):
    x = jnp.stack([x[a] for a in agent_list])
    return x.reshape((num_actors, -1))


def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}


def run_eval_episode(actor_params, actor_network, env, rng, config, max_steps):
    """Roll out one deterministic episode with the given actor params.

    Runs as a plain Python loop (not jit/scan/vmap) since this executes once per
    training run, against a single unwrapped env, not once per training step
    against `NUM_ENVS` vectorized envs. Mirrors the obs-batching, action-masking,
    and hidden-state conventions used in `_env_step` inside `make_train`, except
    actions are taken greedily (argmax over logits) instead of sampled, for a
    reproducible eval rollout.

    Returns a list of env states suitable for `OvercookedV3Visualizer.animate`.
    """
    num_actors = env.num_agents  # single env, so NUM_ACTORS == num_agents

    print(f"[run_eval_episode] resetting env (max_steps={max_steps})...", flush=True)
    rng, _rng = jax.random.split(rng)
    obs, env_state = env.reset(_rng)
    print("[run_eval_episode] env reset done, starting rollout loop", flush=True)

    hstate = ScannedRNN.initialize_carry(num_actors, config["GRU_HIDDEN_DIM"])
    done = jnp.zeros((num_actors,), dtype=bool)

    state_seq = [env_state]

    for step in range(max_steps):
        if step == 0 or (step + 1) % 50 == 0:
            print(f"[run_eval_episode] step {step + 1}/{max_steps}", flush=True)

        obs_batch = jnp.stack([obs[a] for a in env.agents]).reshape(
            -1, *env.observation_space().shape
        )
        ac_in = (
            obs_batch[None, :],
            done[None, :],
        )
        hstate, pi = actor_network.apply(actor_params, hstate, ac_in)

        if config.get("USE_ACTION_MASK", False):
            env_action_mask = env.action_mask(env_state)
            action_mask = env_action_mask.reshape((1, num_actors, env.num_actions))
            pi = distrax.Categorical(logits=jnp.where(action_mask, pi.logits, -1e9))

        action = jnp.argmax(pi.logits, axis=-1)  # shape (1, num_actors)

        env_act = unbatchify(action, env.agents, 1, env.num_agents)
        # unbatchify reshapes to (num_actors, 1, 1); for a single env, step_agents
        # expects a true scalar per agent, not a length-1 array (unlike training,
        # where NUM_ENVS > 1 makes flatten() produce a genuine (NUM_ENVS,) batch).
        env_act = {k: v.reshape(()) for k, v in env_act.items()}

        if step < 5:
            print(f"[run_eval_episode] step {step}: env_act = {env_act}", flush=True)

        rng, _rng = jax.random.split(rng)
        obs, env_state, reward, done_dict, info = env.step(_rng, env_state, env_act)

        if step < 5:
            print(
                f"[run_eval_episode] step {step}: done_dict = {done_dict}, "
                f"reward = {reward}",
                flush=True,
            )

        state_seq.append(env_state)

        done = jnp.full((num_actors,), done_dict["__all__"])

        if bool(done_dict["__all__"]):
            print(
                f"[run_eval_episode] episode ended early at step {step + 1}/{max_steps}",
                flush=True,
            )
            break
    else:
        print(
            f"[run_eval_episode] reached max_steps={max_steps} without termination",
            flush=True,
        )

    print(
        f"[run_eval_episode] rollout complete, collected {len(state_seq)} states",
        flush=True,
    )

    return state_seq


def make_train(config, monitor=None):
    env = OvercookedV3(**config["ENV_KWARGS"])

    config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ACTORS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )

    env = LogWrapper(env, replace_info=False)

    def create_learning_rate_fn():
        base_learning_rate = config["LR"]

        lr_warmup = config["LR_WARMUP"]
        update_steps = config["NUM_UPDATES"]
        warmup_steps = int(lr_warmup * update_steps)

        steps_per_epoch = config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]

        warmup_fn = optax.linear_schedule(
            init_value=0.0,
            end_value=base_learning_rate,
            transition_steps=warmup_steps * steps_per_epoch,
        )
        cosine_epochs = max(update_steps - warmup_steps, 1)

        cosine_fn = optax.cosine_decay_schedule(
            init_value=base_learning_rate, decay_steps=cosine_epochs * steps_per_epoch
        )
        schedule_fn = optax.join_schedules(
            schedules=[warmup_fn, cosine_fn],
            boundaries=[warmup_steps * steps_per_epoch],
        )
        return schedule_fn

    rew_shaping_anneal = optax.linear_schedule(
        init_value=1.0, end_value=0.0, transition_steps=config["REW_SHAPING_HORIZON"]
    )

    checkpoint_interval = max(int(config["NUM_UPDATES"]) // 10, 1)
    checkpoint_dir = os.path.join(config["WANDB_DIR"], "models")
    layout_name = config["ENV_KWARGS"]["layout"]

    def train(rng):
        original_seed = rng[0]

        # INIT NETWORKS
        actor_network = ActorRNN(env.action_space(env.agents[0]).n, config=config)
        critic_network = CriticRNN(config=config)

        rng, _rng_actor, _rng_critic = jax.random.split(rng, 3)

        # Actor init: grid observations
        ac_init_x = (
            jnp.zeros((1, config["NUM_ENVS"], *env.observation_space().shape)),
            jnp.zeros((1, config["NUM_ENVS"])),
        )
        ac_init_hstate = ScannedRNN.initialize_carry(
            config["NUM_ENVS"], config["GRU_HIDDEN_DIM"]
        )
        actor_network_params = actor_network.init(
            _rng_actor, ac_init_hstate, ac_init_x
        )

        # Critic init: full grid observations
        cr_init_x = (
            jnp.zeros((1, config["NUM_ENVS"], *env.observation_space().shape)),
            jnp.zeros((1, config["NUM_ENVS"])),
        )
        cr_init_hstate = ScannedRNN.initialize_carry(
            config["NUM_ENVS"], config["GRU_HIDDEN_DIM"]
        )
        critic_network_params = critic_network.init(
            _rng_critic, cr_init_hstate, cr_init_x
        )

        if config["ANNEAL_LR"]:
            lr_schedule = create_learning_rate_fn()
            actor_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(lr_schedule, eps=1e-5),
            )
            critic_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(lr_schedule, eps=1e-5),
            )
        else:
            actor_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )
            critic_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )
        actor_train_state = TrainState.create(
            apply_fn=actor_network.apply,
            params=actor_network_params,
            tx=actor_tx,
        )
        critic_train_state = TrainState.create(
            apply_fn=critic_network.apply,
            params=critic_network_params,
            tx=critic_tx,
        )

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)
        ac_init_hstate = ScannedRNN.initialize_carry(
            config["NUM_ACTORS"], config["GRU_HIDDEN_DIM"]
        )
        cr_init_hstate = ScannedRNN.initialize_carry(
            config["NUM_ACTORS"], config["GRU_HIDDEN_DIM"]
        )

        # TRAIN LOOP
        def _update_step(runner_state, unused):
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                (
                    train_states,
                    env_state,
                    last_obs,
                    last_done,
                    update_step,
                    hstates,
                    rng,
                ) = runner_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)

                obs_batch = jnp.stack([last_obs[a] for a in env.agents]).reshape(
                    -1, *env.observation_space().shape
                )
                ac_in = (
                    obs_batch[None, :],
                    last_done[None, :],
                )

                ac_hstate, pi = actor_network.apply(
                    train_states[0].params, hstates[0], ac_in
                )
                action_mask = jnp.ones(
                    (config["NUM_ACTORS"], env.num_actions), dtype=jnp.bool_
                )
                if config.get("USE_ACTION_MASK", False):
                    env_action_mask = jax.vmap(env.action_mask)(env_state.env_state)
                    action_mask = jnp.swapaxes(env_action_mask, 0, 1).reshape(
                        (config["NUM_ACTORS"], env.num_actions)
                    )
                    pi = distrax.Categorical(
                        logits=jnp.where(action_mask, pi.logits, -1e9)
                    )
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)

                cr_in = (
                    obs_batch[None, :],
                    last_done[None, :],
                )
                cr_hstate, value = critic_network.apply(
                    train_states[1].params, hstates[1], cr_in
                )

                env_act = unbatchify(
                    action, env.agents, config["NUM_ENVS"], env.num_agents
                )
                env_act = {k: v.flatten() for k, v in env_act.items()}

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])

                obsv, env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0)
                )(rng_step, env_state, env_act)
                original_reward = jnp.array([reward[a] for a in env.agents])

                current_timestep = (
                    update_step * config["NUM_STEPS"] * config["NUM_ENVS"]
                )
                anneal_factor = rew_shaping_anneal(current_timestep)
                reward = jax.tree_util.tree_map(
                    lambda x, y: x + y * anneal_factor * config["SHAPED_REWARD_SCALE"],
                    reward,
                    info["shaped_reward"],
                )

                shaped_reward = jnp.array(
                    [info["shaped_reward"][a] for a in env.agents]
                )
                combined_reward = jnp.array([reward[a] for a in env.agents])

                info["shaped_reward"] = shaped_reward
                info["original_reward"] = original_reward
                info["anneal_factor"] = jnp.full_like(shaped_reward, anneal_factor)
                info["combined_reward"] = combined_reward

                info = jax.tree_util.tree_map(
                    lambda x: x.reshape((config["NUM_ACTORS"])), info
                )
                done_batch = batchify(done, env.agents, config["NUM_ACTORS"]).squeeze()
                transition = Transition(
                    jnp.tile(done["__all__"], env.num_agents),
                    action.squeeze(),
                    value.squeeze(),
                    batchify(reward, env.agents, config["NUM_ACTORS"]).squeeze(),
                    log_prob.squeeze(),
                    obs_batch,
                    action_mask,
                    info,
                )
                runner_state = (
                    train_states,
                    env_state,
                    obsv,
                    done_batch,
                    update_step,
                    (ac_hstate, cr_hstate),
                    rng,
                )
                return runner_state, transition

            initial_hstates = runner_state[-2]
            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            # CALCULATE ADVANTAGE
            train_states, env_state, last_obs, last_done, update_step, hstates, rng = (
                runner_state
            )

            last_obs_batch = jnp.stack([last_obs[a] for a in env.agents]).reshape(
                -1, *env.observation_space().shape
            )

            cr_in = (
                last_obs_batch[None, :],
                last_done[None, :],
            )
            _, last_val = critic_network.apply(
                train_states[1].params, hstates[1], cr_in
            )
            last_val = last_val.squeeze()

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = (
                        delta
                        + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    )
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_states, batch_info):
                    actor_train_state, critic_train_state = train_states
                    ac_init_hstate, cr_init_hstate, traj_batch, advantages, targets = (
                        batch_info
                    )

                    def _actor_loss_fn(actor_params, init_hstate, traj_batch, gae):
                        # RERUN ACTOR
                        _, pi = actor_network.apply(
                            actor_params,
                            init_hstate.squeeze(),
                            (traj_batch.obs, traj_batch.done),
                        )
                        if config.get("USE_ACTION_MASK", False):
                            pi = distrax.Categorical(
                                logits=jnp.where(
                                    traj_batch.action_mask, pi.logits, -1e9
                                )
                            )
                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE ACTOR LOSS
                        logratio = log_prob - traj_batch.log_prob
                        ratio = jnp.exp(logratio)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config["CLIP_EPS"],
                                1.0 + config["CLIP_EPS"],
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        # Diagnostic metrics
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clip_frac = jnp.mean(
                            jnp.abs(ratio - 1.0) > config["CLIP_EPS"]
                        )

                        actor_loss = loss_actor - config["ENT_COEF"] * entropy
                        return actor_loss, (
                            loss_actor,
                            entropy,
                            approx_kl,
                            clip_frac,
                        )

                    def _critic_loss_fn(
                        critic_params, init_hstate, traj_batch, targets
                    ):
                        # RERUN CRITIC
                        _, value = critic_network.apply(
                            critic_params,
                            init_hstate.squeeze(),
                            (traj_batch.obs, traj_batch.done),
                        )

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(
                            value_pred_clipped - targets
                        )
                        value_loss = (
                            0.5
                            * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )
                        critic_loss = config["VF_COEF"] * value_loss
                        return critic_loss, value_loss

                    actor_grad_fn = jax.value_and_grad(_actor_loss_fn, has_aux=True)
                    actor_loss, actor_grads = actor_grad_fn(
                        actor_train_state.params,
                        ac_init_hstate,
                        traj_batch,
                        advantages,
                    )
                    critic_grad_fn = jax.value_and_grad(_critic_loss_fn, has_aux=True)
                    critic_loss, critic_grads = critic_grad_fn(
                        critic_train_state.params,
                        cr_init_hstate,
                        traj_batch,
                        targets,
                    )

                    actor_train_state = actor_train_state.apply_gradients(
                        grads=actor_grads
                    )
                    critic_train_state = critic_train_state.apply_gradients(
                        grads=critic_grads
                    )

                    total_loss = actor_loss[0] + critic_loss[0]
                    loss_info = {
                        "total_loss": total_loss,
                        "actor_loss": actor_loss[1][0],
                        "value_loss": critic_loss[1],
                        "entropy": actor_loss[1][1],
                        "approx_kl": actor_loss[1][2],
                        "clip_frac": actor_loss[1][3],
                    }

                    return (actor_train_state, critic_train_state), loss_info

                (
                    train_states,
                    init_hstates,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                ) = update_state
                rng, _rng = jax.random.split(rng)

                init_hstates = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(x, (1, config["NUM_ACTORS"], -1)),
                    init_hstates,
                )

                batch = (
                    init_hstates[0],
                    init_hstates[1],
                    traj_batch,
                    advantages.squeeze(),
                    targets.squeeze(),
                )
                permutation = jax.random.permutation(_rng, config["NUM_ACTORS"])

                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=1), batch
                )

                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.swapaxes(
                        jnp.reshape(
                            x,
                            [x.shape[0], config["NUM_MINIBATCHES"], -1]
                            + list(x.shape[2:]),
                        ),
                        1,
                        0,
                    ),
                    shuffled_batch,
                )

                train_states, loss_info = jax.lax.scan(
                    _update_minbatch, train_states, minibatches
                )
                update_state = (
                    train_states,
                    jax.tree_util.tree_map(lambda x: x.squeeze(), init_hstates),
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                )
                return update_state, loss_info

            update_state = (
                train_states,
                initial_hstates,
                traj_batch,
                advantages,
                targets,
                rng,
            )
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            train_states = update_state[0]
            metric = traj_batch.info
            rng = update_state[-1]

            def callback(metric, original_seed, actor_params, critic_params):
                step = int(metric["env_step"])
                updates = int(metric["update_step"])
                num_updates = int(config["NUM_UPDATES"])
                ret = float(metric.get("returned_episode_returns", 0.0))
                print_interval = int(config.get("PRINT_METRIC_INTERVAL", 0) or 0)

                if monitor is not None:
                    monitor.update(
                        step=updates,
                        metrics={
                            "env_step": step,
                            "update": f"{updates}/{num_updates}",
                            "train_return": ret,
                            "shaped_reward": float(metric.get("shaped_reward", 0.0)),
                            "original_reward": float(
                                metric.get("original_reward", 0.0)
                            ),
                            "anneal_factor": float(metric.get("anneal_factor", 0.0)),
                        },
                        seed=int(original_seed),
                    )

                if print_interval > 0 and (
                    updates == 1
                    or updates == num_updates
                    or updates % print_interval == 0
                ):
                    print(
                        "METRIC "
                        f"update={updates}/{num_updates} "
                        f"env_step={step} "
                        f"returned_episode_returns={ret:.6f} "
                        f"original_reward={float(metric.get('original_reward', 0.0)):.6f} "
                        f"shaped_reward={float(metric.get('shaped_reward', 0.0)):.6f} "
                        f"combined_reward={float(metric.get('combined_reward', 0.0)):.6f} "
                        f"anneal_factor={float(metric.get('anneal_factor', 0.0)):.6f} "
                        f"entropy={float(metric.get('entropy', 0.0)):.6f} "
                        f"approx_kl={float(metric.get('approx_kl', 0.0)):.6f} "
                        f"clip_frac={float(metric.get('clip_frac', 0.0)):.6f}",
                        flush=True,
                    )

                if config["WANDB_MODE"] != "disabled":
                    wandb.log(metric)

                # Periodic checkpointing
                if (
                    not config.get("DISABLE_CHECKPOINTS", False)
                    and updates % checkpoint_interval == 0
                ):
                    run_name = wandb.run.name if wandb.run else "offline"
                    date_str = datetime.datetime.now().strftime("%Y%m%d")
                    ckpt_subdir = os.path.join(checkpoint_dir, f"{run_name}_{date_str}")
                    os.makedirs(ckpt_subdir, exist_ok=True)
                    save_params(
                        actor_params,
                        os.path.join(ckpt_subdir, f"{updates}_actor.safetensors"),
                    )
                    save_params(
                        critic_params,
                        os.path.join(ckpt_subdir, f"{updates}_critic.safetensors"),
                    )
                    print(f"Checkpoint saved: {ckpt_subdir}/{updates}_*.safetensors")

            update_step = update_step + 1
            loss_info = jax.tree_util.tree_map(lambda x: x.mean(), loss_info)
            metric = jax.tree_util.tree_map(lambda x: x.mean(), metric)
            metric["total_loss"] = loss_info["total_loss"]
            metric["value_loss"] = loss_info["value_loss"]
            metric["actor_loss"] = loss_info["actor_loss"]
            metric["entropy"] = loss_info["entropy"]
            metric["approx_kl"] = loss_info["approx_kl"]
            metric["clip_frac"] = loss_info["clip_frac"]
            metric["update_step"] = update_step
            metric["env_step"] = update_step * config["NUM_STEPS"] * config["NUM_ENVS"]
            if (
                monitor is not None
                or config["WANDB_MODE"] != "disabled"
                or not config.get("DISABLE_CHECKPOINTS", False)
            ):
                jax.debug.callback(
                    callback,
                    metric,
                    original_seed,
                    train_states[0].params,
                    train_states[1].params,
                )

            runner_state = (
                train_states,
                env_state,
                last_obs,
                last_done,
                update_step,
                hstates,
                rng,
            )
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (
            (actor_train_state, critic_train_state),
            env_state,
            obsv,
            jnp.zeros((config["NUM_ACTORS"]), dtype=bool),
            0,
            (ac_init_hstate, cr_init_hstate),
            _rng,
        )
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metrics": metric}

    return train


def single_run(config):
    """Execute a single training run."""
    layout_name = config["ENV_KWARGS"]["layout"]
    num_seeds = config["NUM_SEEDS"]

    wandb_dir = config["WANDB_DIR"]
    os.makedirs(wandb_dir, exist_ok=True)

    wandb.init(
        dir=wandb_dir,
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=["MAPPO", "RNN", "OvercookedV3", "FullObs"],
        config=copy.deepcopy(config),
        mode=config["WANDB_MODE"],
        name=config["WANDB_RUN_NAME"] or f"mappo_rnn_overcooked_v3_full_obs_{layout_name}",
    )

    num_updates = int(
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    use_monitor = config.get("USE_RICH_MONITOR", True) and _MONITOR_AVAILABLE
    monitor = None
    if use_monitor:
        monitor = TrainingMonitor(
            total_updates=num_updates,
            config_dict={
                "env": "overcooked_v3",
                "algo": "MAPPO",
                "layout": layout_name,
                "total_timesteps": int(config["TOTAL_TIMESTEPS"]),
                "num_updates": num_updates,
                "num_envs": config["NUM_ENVS"],
                "num_seeds": num_seeds,
                "lr": config["LR"],
                "gamma": config["GAMMA"],
            },
            title=f"MAPPO-RNN Full Obs - OvercookedV3 ({layout_name})",
        )

    rng = jax.random.PRNGKey(config["SEED"])
    rngs = jax.random.split(rng, num_seeds)
    train_jit = jax.jit(make_train(config, monitor=monitor))
    if monitor is not None:
        with monitor:
            out = jax.block_until_ready(jax.vmap(train_jit)(rngs))
    else:
        out = jax.block_until_ready(jax.vmap(train_jit)(rngs))

    # Save final model params
    save_dir = os.path.join(wandb_dir, "models")
    os.makedirs(save_dir, exist_ok=True)

    actor_state, critic_state = out["runner_state"][0]
    OmegaConf.save(
        config,
        os.path.join(
            save_dir,
            f"mappo_rnn_overcooked_v3_full_obs_{layout_name}_seed{config['SEED']}_config.yaml",
        ),
    )

    for i, rng in enumerate(rngs):
        actor_params = jax.tree.map(lambda x: x[i], actor_state.params)
        critic_params = jax.tree.map(lambda x: x[i], critic_state.params)
        actor_path = os.path.join(
            save_dir,
            f"mappo_rnn_overcooked_v3_full_obs_{layout_name}_seed{config['SEED']}_vmap{i}_actor.safetensors",
        )
        critic_path = os.path.join(
            save_dir,
            f"mappo_rnn_overcooked_v3_full_obs_{layout_name}_seed{config['SEED']}_vmap{i}_critic.safetensors",
        )
        save_params(actor_params, actor_path)
        save_params(critic_params, critic_path)
        print(f"Saved actor params to {actor_path}")
        print(f"Saved critic params to {critic_path}")

        if config["WANDB_MODE"] != "disabled":
            print(f"[eval seed {i}] building unwrapped eval env...", flush=True)
            eval_env = OvercookedV3(**config["ENV_KWARGS"])  # unwrapped, fresh instance
            actor_network = ActorRNN(
                eval_env.action_space(eval_env.agents[0]).n, config=config
            )
            print(f"[eval seed {i}] constructing OvercookedV3Visualizer...", flush=True)
            viz = OvercookedV3Visualizer(eval_env)
            print(f"[eval seed {i}] visualizer ready, starting eval rollout...", flush=True)
            eval_rng = jax.random.PRNGKey(config["SEED"] + 1000 + i)
            state_seq = run_eval_episode(
                actor_params,
                actor_network,
                eval_env,
                eval_rng,
                config,
                max_steps=config["ENV_KWARGS"]["max_steps"],
            )
            gif_path = os.path.join(save_dir, f"final_episode_seed{i}.gif")
            print(f"[eval seed {i}] stacking {len(state_seq)} states into one pytree...", flush=True)
            stacked_state_seq = jax.tree_util.tree_map(
                lambda *leaves: jnp.stack(leaves), *state_seq
            )
            print(f"[eval seed {i}] rendering frames and writing gif to {gif_path}...", flush=True)
            viz.animate(stacked_state_seq, filename=gif_path)
            print(f"[eval seed {i}] gif written, logging to wandb...", flush=True)
            wandb.log({f"eval/final_episode_seed{i}": wandb.Video(gif_path, format="gif")})
            print(f"Logged eval gif for seed {i} to wandb: {gif_path}")


def tune(config):
    """Hyperparameter sweep with CARBS."""
    from carbs_sweep import CARBSSweep

    layout_name = config["ENV_KWARGS"]["layout"]
    sweep = CARBSSweep(config)

    print(f"Starting CARBS sweep: {sweep.num_trials} trials, layout={layout_name}")

    for trial in range(sweep.num_trials):
        suggestion = sweep.suggest()
        trial_config = sweep.apply_suggestion(suggestion)
        trial_config["WANDB_MODE"] = "disabled"

        print(f"\n{'='*60}")
        print(f"Trial {trial+1}/{sweep.num_trials}")
        print(f"  {CARBSSweep.format_suggestion(suggestion)}")

        start_time = time.time()
        try:
            rng = jax.random.PRNGKey(trial_config["SEED"])
            rngs = jax.random.split(rng, trial_config["NUM_SEEDS"])
            train_fn = make_train(trial_config, monitor=None)
            outs = jax.block_until_ready(jax.jit(jax.vmap(train_fn))(rngs))

            final_return = float(
                outs["metrics"]["returned_episode_returns"][:, -1].mean()
            )
            elapsed = time.time() - start_time

            sweep.observe(suggestion, output=final_return, cost=elapsed)
            print(
                f"  Return: {final_return:.2f}  Time: {elapsed:.1f}s  "
                f"Best: {sweep.best_return:.2f}"
            )

        except Exception as e:
            elapsed = time.time() - start_time
            print(f"  FAILED: {e}")
            sweep.observe_failure(suggestion, cost=elapsed)

    sweep.print_summary()


@hydra.main(
    version_base=None, config_path="config", config_name="mappo_rnn_overcooked_v3_full_obs"
)
def main(config):
    config = OmegaConf.to_container(config, resolve=True)
    if config.get("TUNE", False):
        tune(config)
    else:
        single_run(config)


if __name__ == "__main__":
    main()