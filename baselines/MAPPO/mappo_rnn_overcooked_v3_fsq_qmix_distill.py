"""
MAPPO with RNN + CNN for partially observed Overcooked V3 plus FSQ communication
and actor distillation from a privileged full-observation QMIX teacher.

The teacher is a trained QMIX agent Q-network (baselines/QLearning/qmix_rnn.py)
that was trained on the full grid observation (agent_view_size=None), fed to the
QMIX MLP+GRU as CTRolloutManager preprocesses it: flatten(full_obs) concatenated
with a per-agent one-hot id. The student sees only a partial view and must learn
an FSQ communication channel to recover the teacher's behaviour.

QMIX is value-based, so the teacher emits per-agent Q-values rather than policy
logits. We treat the Q-values as logits of a Boltzmann policy (softmax(Q / temp))
and distill the student toward that distribution via the same KL objective. The
QMIX mixer is only used for centralized training and is ignored here.
"""

import datetime
import time
import os
import sys
import jax
import jax.api_util
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import Callable, Sequence, NamedTuple, Any, Dict
from flax.training.train_state import TrainState
from flax import serialization
import distrax

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import jaxmarl
from jaxmarl.wrappers.baselines import LogWrapper, load_params, save_params
from jaxmarl.environments.overcooked_v3 import OvercookedV3, overcooked_v3_layouts
from jaxmarl.environments.overcooked_v3.common import DynamicObject
import hydra
from omegaconf import OmegaConf
import copy
import wandb
import functools

from FSQ import FSQ

try:
    from utils.monitor import TrainingMonitor

    _MONITOR_AVAILABLE = True
except ImportError:
    _MONITOR_AVAILABLE = False


def cosine_distill_weight(update_step, config):
    current_timestep = update_step * config["NUM_STEPS"] * config["NUM_ENVS"]
    decay_steps = jnp.maximum(
        config["DISTILL_DECAY_FRACTION"] * config["TOTAL_TIMESTEPS"], 1.0
    )
    progress = jnp.clip(current_timestep / decay_steps, 0.0, 1.0)
    weight = config["DISTILL_COEF"] * 0.5 * (1.0 + jnp.cos(jnp.pi * progress))
    return jnp.where(progress >= 1.0, 0.0, weight)


def categorical_kl_from_logits(teacher_logits, student_logits, temperature):
    teacher_log_probs = jax.nn.log_softmax(teacher_logits / temperature, axis=-1)
    student_log_probs = jax.nn.log_softmax(student_logits / temperature, axis=-1)
    teacher_probs = jnp.exp(teacher_log_probs)
    return jnp.sum(
        teacher_probs * (teacher_log_probs - student_log_probs), axis=-1
    )


def load_qmix_teacher_params(path):
    """Load a trained QMIX agent Q-network from a safetensors checkpoint.

    `path` is the ..._vmap0.safetensors file (a directory is also accepted if it
    contains exactly one .safetensors). The checkpoint stores both `agent` and
    `mixer` params; only the agent network is needed for distillation. Its GRU
    is a stock flax `GRUCell` (ir/iz/in/hr/hz/hn); we remap it into the flat
    param layout consumed by QMixTeacherAgentRNN below.
    """
    if os.path.isdir(path):
        cands = [f for f in os.listdir(path) if f.endswith(".safetensors")]
        if len(cands) != 1:
            raise ValueError(
                f"Expected exactly one .safetensors in {path}, found {cands}"
            )
        path = os.path.join(path, cands[0])
    raw = load_params(path)
    agent = raw["agent"]["params"]
    g = agent["ScannedRNN_0"]["GRUCell_0"]
    mapped = {
        "embed": agent["Dense_0"],
        "qhead": agent["Dense_1"],
        "ir_kernel": g["ir"]["kernel"],
        "ir_bias": g["ir"]["bias"],
        "iz_kernel": g["iz"]["kernel"],
        "iz_bias": g["iz"]["bias"],
        "in_kernel": g["in"]["kernel"],
        "in_bias": g["in"]["bias"],
        "hr_kernel": g["hr"]["kernel"],
        "hz_kernel": g["hz"]["kernel"],
        "hn_kernel": g["hn"]["kernel"],
        "hn_bias": g["hn"]["bias"],
    }
    return jax.tree_util.tree_map(jnp.asarray, {"params": mapped})


class ScannedRNN(nn.Module):
    """GRU over a (T, batch, feat) sequence with per-step done resets.

    Drop-in replacement for the Flax `nn.scan` + `nn.GRUCell` version, which is
    broken on jax 0.4.38 / flax 0.10.4 (flax `axes_scan` calls the nonexistent
    `jax.api_util.debug_info`). Mirrors the manual-GRU approach used in
    baselines/IPPO/ippo_rnn_overcooked_v3.py: input projections (Dense) are
    computed for all timesteps outside the scan, recurrent weights are raw
    params applied inside `jax.lax.scan`. Interface is unchanged:
    `ScannedRNN()(hidden, (ins, resets)) -> (final_hidden, ys)`.
    """

    @nn.compact
    def __call__(self, carry, x):
        ins, resets = x  # ins: (T, batch, feat), resets: (T, batch)
        hidden_dim = carry.shape[-1]
        T, batch = ins.shape[0], ins.shape[1]
        feat = ins.shape[-1]

        flat = ins.reshape(-1, feat)
        Wi_z = nn.Dense(hidden_dim, use_bias=False, name="gru_Wi_z")(flat)
        Wi_r = nn.Dense(hidden_dim, use_bias=False, name="gru_Wi_r")(flat)
        Wi_h = nn.Dense(hidden_dim, use_bias=False, name="gru_Wi_h")(flat)
        Wi_z = Wi_z.reshape(T, batch, hidden_dim)
        Wi_r = Wi_r.reshape(T, batch, hidden_dim)
        Wi_h = Wi_h.reshape(T, batch, hidden_dim)

        Wh_z = self.param("gru_Wh_z", nn.initializers.orthogonal(), (hidden_dim, hidden_dim))
        Wh_r = self.param("gru_Wh_r", nn.initializers.orthogonal(), (hidden_dim, hidden_dim))
        Wh_h = self.param("gru_Wh_h", nn.initializers.orthogonal(), (hidden_dim, hidden_dim))
        b_z = self.param("gru_b_z", nn.initializers.zeros_init(), (hidden_dim,))
        b_r = self.param("gru_b_r", nn.initializers.zeros_init(), (hidden_dim,))
        b_h = self.param("gru_b_h", nn.initializers.zeros_init(), (hidden_dim,))

        def _step(h, inp):
            wiz_t, wir_t, wih_t, reset_t = inp
            h = jnp.where(reset_t[:, None], jnp.zeros_like(h), h)
            z = jax.nn.sigmoid(wiz_t + h @ Wh_z + b_z)
            r = jax.nn.sigmoid(wir_t + h @ Wh_r + b_r)
            h_hat = jnp.tanh(wih_t + (r * h) @ Wh_h + b_h)
            new_h = (1 - z) * h + z * h_hat
            return new_h, new_h

        final_hidden, ys = jax.lax.scan(_step, carry, (Wi_z, Wi_r, Wi_h, resets))
        return final_hidden, ys

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        return jnp.zeros((batch_size, hidden_size))


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

        fsq = FSQ(levels=self.config["FSQ_LEVELS"])
        message_pre = nn.Dense(
            fsq.num_dimensions,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
        )(embedding)
        message = fsq.quantize(message_pre)
        message_code = fsq.codes_to_indexes(message)
        message_levels = fsq._scale_and_shift(message).round().astype(jnp.int32)

        num_agents = self.config["NUM_AGENTS"]
        num_envs = embedding.shape[1] // num_agents
        grouped_message = message.reshape(
            (message.shape[0], num_agents, num_envs, fsq.num_dimensions)
        )
        partner_message = jnp.flip(grouped_message, axis=1).reshape(message.shape)
        actor_features = jnp.concatenate([embedding, partner_message], axis=-1)

        actor_mean = nn.Dense(
            self.config["FC_DIM_SIZE"],
            kernel_init=orthogonal(2),
            bias_init=constant(0.0),
        )(actor_features)
        actor_mean = nn.relu(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)

        pi = distrax.Categorical(logits=actor_mean)
        comm_info = {
            "message": message,
            "code": message_code,
            "levels": message_levels,
        }

        return hidden, pi, comm_info


class QMixTeacherAgentRNN(nn.Module):
    """Privileged QMIX teacher agent Q-network.

    Reproduces baselines/QLearning/qmix_rnn.py's RNNQNetwork (Dense embed → GRU →
    Dense Q-head) but with a manual GRU inside jax.lax.scan, because that repo's
    ScannedRNN uses Flax `nn.scan`, which is broken on jax 0.4.38 / flax 0.10.4.
    The GRU math matches Flax `GRUCell` exactly (ir/iz/in with bias, hr/hz no
    bias, hn with bias); `load_qmix_teacher_params` remaps a trained checkpoint
    into the flat param layout used here (embed, qhead, {ir,iz,in,hr,hz,hn}_*).

    Input `obs` is the QMIX-preprocessed observation: flatten(full_obs) followed
    by a per-agent one-hot id. Output is per-agent Q-values (frozen teacher);
    the caller turns them into a Boltzmann distribution for distillation.
    """

    action_dim: int
    config: Dict

    @nn.compact
    def __call__(self, hidden, x):
        obs, dones = x  # obs: (T, batch, obs_flat), dones: (T, batch)
        hidden_dim = self.config.get("HIDDEN_SIZE", 256)
        T, batch = obs.shape[0], obs.shape[1]

        embedding = nn.Dense(hidden_dim, name="embed")(obs)
        embedding = nn.relu(embedding)  # (T, batch, hidden_dim)

        ir_k = self.param("ir_kernel", nn.initializers.zeros_init(), (hidden_dim, hidden_dim))
        ir_b = self.param("ir_bias", nn.initializers.zeros_init(), (hidden_dim,))
        iz_k = self.param("iz_kernel", nn.initializers.zeros_init(), (hidden_dim, hidden_dim))
        iz_b = self.param("iz_bias", nn.initializers.zeros_init(), (hidden_dim,))
        in_k = self.param("in_kernel", nn.initializers.zeros_init(), (hidden_dim, hidden_dim))
        in_b = self.param("in_bias", nn.initializers.zeros_init(), (hidden_dim,))
        hr_k = self.param("hr_kernel", nn.initializers.zeros_init(), (hidden_dim, hidden_dim))
        hz_k = self.param("hz_kernel", nn.initializers.zeros_init(), (hidden_dim, hidden_dim))
        hn_k = self.param("hn_kernel", nn.initializers.zeros_init(), (hidden_dim, hidden_dim))
        hn_b = self.param("hn_bias", nn.initializers.zeros_init(), (hidden_dim,))

        def _gru_step(h, inp):
            x_t, done_t = inp
            h = jnp.where(done_t[:, None], jnp.zeros_like(h), h)
            r = jax.nn.sigmoid(x_t @ ir_k + ir_b + h @ hr_k)
            z = jax.nn.sigmoid(x_t @ iz_k + iz_b + h @ hz_k)
            n = jnp.tanh(x_t @ in_k + in_b + r * (h @ hn_k + hn_b))
            new_h = (1.0 - z) * n + z * h
            return new_h, new_h

        final_hidden, embedding = jax.lax.scan(
            _gru_step, hidden, (embedding, dones)
        )

        q_vals = nn.Dense(self.action_dim, name="qhead")(embedding)
        return final_hidden, q_vals


class CriticRNN(nn.Module):
    config: Dict

    @nn.compact
    def __call__(self, hidden, x):
        world_state, dones = x

        embedding = nn.Dense(
            self.config["GRU_HIDDEN_DIM"],
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(world_state)
        embedding = nn.relu(embedding)

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
    world_state: jnp.ndarray
    teacher_logits: jnp.ndarray
    info: jnp.ndarray


def batchify(x: dict, agent_list, num_actors):
    x = jnp.stack([x[a] for a in agent_list])
    return x.reshape((num_actors, -1))


def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}


def make_train(config, monitor=None, teacher_actor_params=None):
    env = OvercookedV3(**config["ENV_KWARGS"])
    teacher_env_kwargs = copy.deepcopy(config["ENV_KWARGS"])
    teacher_env_kwargs["agent_view_size"] = None
    teacher_env = OvercookedV3(**teacher_env_kwargs)

    if env.num_agents != 2:
        raise ValueError("FSQ communication distillation currently supports 2 agents.")
    if teacher_actor_params is None:
        raise ValueError("TEACHER_ACTOR_PATH is required for distillation training.")

    # Teacher GRU width must match the trained QMIX checkpoint (HIDDEN_SIZE),
    # independent of the (possibly smaller) student network dims.
    teacher_config = {
        "HIDDEN_SIZE": config.get("TEACHER_HIDDEN_SIZE", 256),
    }
    config["TEACHER_HIDDEN_SIZE"] = teacher_config["HIDDEN_SIZE"]
    # Per-agent one-hot ids appended to the flattened full obs, matching
    # CTRolloutManager's preprocessing during QMIX training (agent-major order).
    teacher_agent_one_hot = jnp.eye(env.num_agents)

    config["NUM_AGENTS"] = env.num_agents
    config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    if config["NUM_ENVS"] % config["NUM_MINIBATCHES"] != 0:
        raise ValueError("NUM_ENVS must be divisible by NUM_MINIBATCHES.")
    config["MINIBATCH_ENVS"] = config["NUM_ENVS"] // config["NUM_MINIBATCHES"]
    config["MINIBATCH_SIZE"] = (
        env.num_agents * config["MINIBATCH_ENVS"] * config["NUM_STEPS"]
    )

    world_state_size = env.num_agents * int(np.prod(env.observation_space().shape))
    # QMIX teacher sees flatten(full_obs) + agent one-hot (CTRolloutManager style).
    teacher_full_obs_flat = int(np.prod(teacher_env.observation_space().shape))
    teacher_obs_dim = teacher_full_obs_flat + env.num_agents
    comm_codebook_size = int(np.prod(np.asarray(config["FSQ_LEVELS"])))
    comm_num_dims = len(config["FSQ_LEVELS"])
    comm_num_levels = int(max(config["FSQ_LEVELS"]))

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
        teacher_actor_network = QMixTeacherAgentRNN(
            env.action_space(env.agents[0]).n, config=teacher_config
        )

        rng, _rng_actor, _rng_critic = jax.random.split(rng, 3)

        # Actor init: grid observations
        ac_init_x = (
            jnp.zeros((1, config["NUM_ACTORS"], *env.observation_space().shape)),
            jnp.zeros((1, config["NUM_ACTORS"])),
        )
        ac_init_hstate = ScannedRNN.initialize_carry(
            config["NUM_ACTORS"], config["GRU_HIDDEN_DIM"]
        )
        actor_network_params = actor_network.init(
            _rng_actor, ac_init_hstate, ac_init_x
        )

        # Critic init: flat world state
        cr_init_x = (
            jnp.zeros((1, config["NUM_ACTORS"], world_state_size)),
            jnp.zeros((1, config["NUM_ACTORS"])),
        )
        cr_init_hstate = ScannedRNN.initialize_carry(
            config["NUM_ACTORS"], config["GRU_HIDDEN_DIM"]
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
        # QMIX teacher uses a plain zeros GRU carry (manual GRU, not ScannedRNN).
        teacher_init_hstate = jnp.zeros(
            (config["NUM_ACTORS"], teacher_config["HIDDEN_SIZE"])
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
                    obs_batch[np.newaxis, :],
                    last_done[np.newaxis, :],
                )

                ac_hstate, pi, comm_info = actor_network.apply(
                    train_states[0].params, hstates[0], ac_in
                )
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)

                # QMIX teacher input: flatten(full_obs) + per-agent one-hot id,
                # matching CTRolloutManager's preprocessing, in agent-major order
                # (first NUM_ENVS rows are agent_0, next NUM_ENVS are agent_1).
                full_obs = jax.vmap(teacher_env.get_obs)(env_state.env_state)
                teacher_obs_parts = []
                for i, a in enumerate(env.agents):
                    flat_a = full_obs[a].reshape(config["NUM_ENVS"], -1)
                    oh_a = jnp.broadcast_to(
                        teacher_agent_one_hot[i],
                        (config["NUM_ENVS"], env.num_agents),
                    )
                    teacher_obs_parts.append(
                        jnp.concatenate([flat_a, oh_a], axis=-1)
                    )
                teacher_obs_batch = jnp.concatenate(teacher_obs_parts, axis=0)
                teacher_in = (
                    teacher_obs_batch[np.newaxis, :],
                    last_done[np.newaxis, :],
                )
                teacher_hstate, teacher_q = teacher_actor_network.apply(
                    teacher_actor_params, hstates[2], teacher_in
                )
                # QMIX Q-values have a tiny, uncalibrated spread across actions
                # (~0.01), so softmax(Q/temp) at temp=1 is nearly uniform and
                # carries no signal even though argmax(Q) is a good policy.
                # Standardize per action-vector (zero mean, unit std) so the KL
                # target reflects the teacher's action *ranking*; DISTILL_TEMPERATURE
                # then controls sharpness on a calibrated scale.
                if config.get("TEACHER_Q_STANDARDIZE", True):
                    teacher_q = (
                        teacher_q - teacher_q.mean(axis=-1, keepdims=True)
                    ) / (teacher_q.std(axis=-1, keepdims=True) + 1e-8)

                # WORLD STATE for critic
                obs_flat = obs_batch.reshape(env.num_agents, config["NUM_ENVS"], -1)
                world_state_per_env = jnp.concatenate(
                    [obs_flat[i] for i in range(env.num_agents)], axis=-1
                )
                world_state_batch = jnp.tile(
                    world_state_per_env, (env.num_agents, 1)
                )

                cr_in = (
                    world_state_batch[np.newaxis, :],
                    last_done[np.newaxis, :],
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
                info["distill_weight"] = jnp.full_like(
                    shaped_reward, cosine_distill_weight(update_step, config)
                )
                comm_code = comm_info["code"].squeeze()
                comm_levels = comm_info["levels"].squeeze()
                info["comm_code"] = comm_code
                for dim in range(comm_num_dims):
                    info[f"comm_dim{dim}"] = comm_levels[:, dim]

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
                    world_state_batch,
                    teacher_q.squeeze(),
                    info,
                )
                runner_state = (
                    train_states,
                    env_state,
                    obsv,
                    done_batch,
                    update_step,
                    (ac_hstate, cr_hstate, teacher_hstate),
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

            # Build world state for last obs
            last_obs_batch = jnp.stack([last_obs[a] for a in env.agents]).reshape(
                -1, *env.observation_space().shape
            )
            last_obs_flat = last_obs_batch.reshape(
                env.num_agents, config["NUM_ENVS"], -1
            )
            last_world_state = jnp.concatenate(
                [last_obs_flat[i] for i in range(env.num_agents)], axis=-1
            )
            last_world_state_batch = jnp.tile(
                last_world_state, (env.num_agents, 1)
            )

            cr_in = (
                last_world_state_batch[np.newaxis, :],
                last_done[np.newaxis, :],
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
                        _, pi, _ = actor_network.apply(
                            actor_params,
                            init_hstate.squeeze(),
                            (traj_batch.obs, traj_batch.done),
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

                        teacher_student_kl = categorical_kl_from_logits(
                            traj_batch.teacher_logits,
                            pi.logits,
                            config["DISTILL_TEMPERATURE"],
                        ).mean()
                        distill_weight = traj_batch.info["distill_weight"].mean()
                        distill_loss = (
                            distill_weight
                            * (config["DISTILL_TEMPERATURE"] ** 2)
                            * teacher_student_kl
                        )

                        actor_loss = (
                            loss_actor
                            - config["ENT_COEF"] * entropy
                            + distill_loss
                        )
                        return actor_loss, (
                            loss_actor,
                            entropy,
                            approx_kl,
                            clip_frac,
                            teacher_student_kl,
                            distill_loss,
                            distill_weight,
                        )

                    def _critic_loss_fn(
                        critic_params, init_hstate, traj_batch, targets
                    ):
                        # RERUN CRITIC
                        _, value = critic_network.apply(
                            critic_params,
                            init_hstate.squeeze(),
                            (traj_batch.world_state, traj_batch.done),
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
                        "teacher_student_kl": actor_loss[1][4],
                        "distill_loss": actor_loss[1][5],
                        "distill_weight": actor_loss[1][6],
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

                grouped_init_hstates = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(
                        x, (1, env.num_agents, config["NUM_ENVS"], -1)
                    ),
                    init_hstates,
                )

                def group_actor_axis(x):
                    return jnp.reshape(
                        x,
                        (x.shape[0], env.num_agents, config["NUM_ENVS"])
                        + x.shape[2:],
                    )

                batch = (
                    grouped_init_hstates[0],
                    grouped_init_hstates[1],
                    jax.tree_util.tree_map(group_actor_axis, traj_batch),
                    group_actor_axis(advantages),
                    group_actor_axis(targets),
                )
                permutation = jax.random.permutation(_rng, config["NUM_ENVS"])

                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=2), batch
                )

                def make_minibatches(x):
                    x = jnp.reshape(
                        x,
                        (x.shape[0], env.num_agents, config["NUM_MINIBATCHES"], -1)
                        + x.shape[3:],
                    )
                    x = jnp.moveaxis(x, 2, 0)
                    return jnp.reshape(
                        x,
                        (config["NUM_MINIBATCHES"], x.shape[1], -1)
                        + x.shape[4:],
                    )

                minibatches = jax.tree_util.tree_map(make_minibatches, shuffled_batch)

                train_states, loss_info = jax.lax.scan(
                    _update_minbatch, train_states, minibatches
                )
                update_state = (
                    train_states,
                    init_hstates,
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
            comm_code_flat = traj_batch.info["comm_code"].astype(jnp.int32).reshape(-1)
            comm_code_hist = jnp.bincount(
                comm_code_flat, length=comm_codebook_size
            )
            comm_code_total = jnp.maximum(comm_code_hist.sum(), 1)
            comm_code_unique = jnp.sum(comm_code_hist > 0)
            comm_code_top1_frac = jnp.max(comm_code_hist) / comm_code_total
            comm_dim_hists = jnp.stack(
                [
                    jnp.bincount(
                        traj_batch.info[f"comm_dim{dim}"]
                        .astype(jnp.int32)
                        .reshape(-1),
                        length=comm_num_levels,
                    )
                    for dim in range(comm_num_dims)
                ]
            )

            def callback(metric, original_seed, actor_params, critic_params):
                step = int(metric["env_step"])
                updates = int(metric["update_step"])
                num_updates = int(config["NUM_UPDATES"])
                ret = float(metric.get("returned_episode_returns", 0.0))

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

                if config["WANDB_MODE"] != "disabled":
                    log_metric = dict(metric)
                    if "comm_code_hist" in log_metric:
                        code_counts = np.asarray(log_metric["comm_code_hist"])
                        log_metric["comm_code_hist"] = wandb.Histogram(
                            np_histogram=(
                                code_counts,
                                np.arange(code_counts.shape[0] + 1),
                            )
                        )
                    if "comm_dim_hists" in log_metric:
                        dim_counts = np.asarray(log_metric.pop("comm_dim_hists"))
                        for dim, counts in enumerate(dim_counts):
                            log_metric[f"comm_dim{dim}_hist"] = wandb.Histogram(
                                np_histogram=(
                                    counts,
                                    np.arange(counts.shape[0] + 1),
                                )
                            )
                    wandb.log(log_metric)

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
            metric["teacher_student_kl"] = loss_info["teacher_student_kl"]
            metric["distill_loss"] = loss_info["distill_loss"]
            metric["distill_weight"] = loss_info["distill_weight"]
            metric["comm_code_unique"] = comm_code_unique
            metric["comm_code_top1_frac"] = comm_code_top1_frac
            metric["comm_code_hist"] = comm_code_hist
            metric["comm_dim_hists"] = comm_dim_hists
            metric["update_step"] = update_step
            metric["env_step"] = update_step * config["NUM_STEPS"] * config["NUM_ENVS"]
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
            (ac_init_hstate, cr_init_hstate, teacher_init_hstate),
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
    teacher_actor_path = config.get("TEACHER_ACTOR_PATH", "")
    if not teacher_actor_path:
        raise ValueError(
            "TEACHER_ACTOR_PATH must point to a trained QMIX agent checkpoint "
            "(a ..._vmap0.safetensors file, or a dir containing exactly one)."
        )
    teacher_actor_params = load_qmix_teacher_params(teacher_actor_path)
    print(f"[distill] QMIX teacher loaded from {teacher_actor_path}", flush=True)

    wandb_dir = config["WANDB_DIR"]
    os.makedirs(wandb_dir, exist_ok=True)

    wandb.init(
        dir=wandb_dir,
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=["MAPPO", "RNN", "OvercookedV3", "FSQComm", "Distill", "QMIXTeacher"],
        config=copy.deepcopy(config),
        mode=config["WANDB_MODE"],
        name=config["WANDB_RUN_NAME"]
        or f"mappo_rnn_overcooked_v3_fsq_qmix_distill_{layout_name}",
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
            title=f"MAPPO-RNN FSQ Distill - OvercookedV3 ({layout_name})",
        )

    with jax.disable_jit(False):
        rng = jax.random.PRNGKey(config["SEED"])
        rngs = jax.random.split(rng, num_seeds)
        train_jit = jax.jit(
            make_train(
                config,
                monitor=monitor,
                teacher_actor_params=teacher_actor_params,
            )
        )
        if monitor is not None:
            with monitor:
                out = jax.block_until_ready(jax.vmap(train_jit)(rngs))
        else:
            out = jax.vmap(train_jit)(rngs)

    # Save final model params
    save_dir = os.path.join(wandb_dir, "models")
    os.makedirs(save_dir, exist_ok=True)

    actor_state, critic_state = out["runner_state"][0]
    OmegaConf.save(
        config,
        os.path.join(
            save_dir,
            f"mappo_rnn_overcooked_v3_fsq_qmix_distill_{layout_name}_seed{config['SEED']}_config.yaml",
        ),
    )

    for i, rng in enumerate(rngs):
        actor_params = jax.tree.map(lambda x: x[i], actor_state.params)
        critic_params = jax.tree.map(lambda x: x[i], critic_state.params)
        actor_path = os.path.join(
            save_dir,
            f"mappo_rnn_overcooked_v3_fsq_qmix_distill_{layout_name}_seed{config['SEED']}_vmap{i}_actor.safetensors",
        )
        critic_path = os.path.join(
            save_dir,
            f"mappo_rnn_overcooked_v3_fsq_qmix_distill_{layout_name}_seed{config['SEED']}_vmap{i}_critic.safetensors",
        )
        save_params(actor_params, actor_path)
        save_params(critic_params, critic_path)
        print(f"Saved actor params to {actor_path}")
        print(f"Saved critic params to {critic_path}")


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
    version_base=None,
    config_path="config",
    config_name="mappo_rnn_overcooked_v3_fsq_qmix_distill",
)
def main(config):
    config = OmegaConf.to_container(config, resolve=True)
    if config.get("TUNE", False):
        raise NotImplementedError("CARBS tuning is not wired for teacher distillation.")
    else:
        single_run(config)


if __name__ == "__main__":
    main()
