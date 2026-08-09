"""MAPPO (RNN) training script for Overcooked V3, primitive actions.

This is the centralised-critic counterpart of
`baselines/IPPO/ippo_rnn_overcooked_v3.py`, and is deliberately kept
step-for-step comparable with it: same primitive action space, same CNN+GRU
encoder, same reward-shaping anneal, same event/order metrics and W&B logging.
The only structural difference is the one that makes it MAPPO:

  IPPO   one network per agent view; the critic sees the acting agent's obs.
  MAPPO  separate actor and critic networks. The actor still sees only its own
         observation, but the critic sees a *world state* built by stacking
         every agent's observation along the channel axis, so value estimates
         are conditioned on the joint state rather than one agent's view.

Because Overcooked V3 observations are grids (H, W, C), the world state is
formed by channel concatenation rather than the flatten-and-roll used by the
MPE MAPPO baseline - flattening would destroy the spatial structure the CNN
encoder relies on.

The GRU works around the same Flax 0.10.4 / JAX 0.4.38 `nn.scan` bug as the
IPPO script: Dense input projections are precomputed outside `lax.scan` and the
recurrent matrices are raw `self.param` arrays used inside it.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from functools import partial
from typing import Any, Dict, NamedTuple, Sequence

import distrax
import flax.linen as nn
import hydra
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import serialization
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from omegaconf import OmegaConf

import jaxmarl
import wandb
from baselines.IC3Net.monitor import TrainingMonitorInterface
from baselines.IPPO.ippo_cnn_overcooked_v3 import (
    EVENT_METRIC_NAMES,
    _log_reward_structure_to_wandb,
)
from baselines.IPPO.ippo_rnn_overcooked_v3 import (
    CNN,
    _flatten_metric_dict,
    _first_scalar,
    _monitor_payload,
    unbatchify,
)
from jaxmarl.viz.overcooked_v3_visualizer import OvercookedV3Visualizer
from jaxmarl.wrappers.baselines import JaxMARLWrapper, LogWrapper

_ACTIVE_MONITOR = None


def _save_params(params, save_path, filename):
    """Serialise one network's parameters to `save_path/filename`."""
    os.makedirs(save_path, exist_ok=True)
    model_path = os.path.join(save_path, filename)
    with open(model_path, "wb") as f:
        f.write(serialization.to_bytes({"params": params}))
    return model_path


def _log_training_metrics(metric: Dict[str, Any]) -> None:
    """Push one update's metrics to W&B and the terminal monitor."""
    payload = _flatten_metric_dict(metric)
    env_step = _first_scalar(payload.get("env_step", payload.get("update_step", 0)))
    update_step = _first_scalar(payload.get("update_step", 0))
    payload["env_step"] = env_step
    payload["update_step"] = update_step

    if wandb.run is not None:
        wandb.log(payload, step=env_step)

    if _ACTIVE_MONITOR is not None:
        _ACTIVE_MONITOR.update(update_step, _monitor_payload(payload))


# ── World state ────────────────────────────────────────────────────────


class OvercookedV3WorldStateWrapper(JaxMARLWrapper):
    """Adds `obs["world_state"]`: every agent's grid observation, stacked.

    The result has shape (num_agents, H, W, C * num_agents) - one copy per
    agent so the critic can be batched exactly like the actor, with all agents'
    channels concatenated. Channel concatenation (rather than flattening) keeps
    the grid layout intact for the CNN encoder.
    """

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
        """Concatenate all agent observations along the channel axis."""
        joint = jnp.concatenate(
            [obs[agent] for agent in self._env.agents], axis=-1
        )  # (H, W, C * num_agents)
        return jnp.broadcast_to(joint, (self._env.num_agents, *joint.shape))

    def world_state_shape(self):
        """Shape of a single agent's slice of the world state."""
        base = self._env.observation_space(self._env.agents[0]).shape
        return (*base[:-1], base[-1] * self._env.num_agents)


# ── Network architecture ───────────────────────────────────────────────


class RecurrentEncoder(nn.Module):
    """CNN encoder + GRU over time, shared in form by the actor and critic.

    Returns the final hidden state and the per-timestep GRU outputs. The GRU is
    hand-rolled so the recurrent matmuls happen inside `lax.scan` on raw params
    while the input projections are computed for all timesteps at once outside
    it - the workaround for the Flax `nn.scan` bug noted in the module docstring.
    """

    config: Dict

    @nn.compact
    def __call__(self, hidden, x):
        obs, dones = x
        T = obs.shape[0]
        num_actors = obs.shape[1]
        hidden_dim = self.config.get("GRU_HIDDEN_DIM", 128)
        activation = (
            nn.relu if self.config.get("ACTIVATION", "relu") == "relu" else nn.tanh
        )

        embedding = jax.vmap(CNN(output_size=hidden_dim, activation=activation))(obs)
        embedding = nn.LayerNorm()(embedding)

        flat_emb = embedding.reshape(-1, hidden_dim)
        Wi_z = nn.Dense(hidden_dim, use_bias=False, name="gru_Wi_z")(flat_emb)
        Wi_r = nn.Dense(hidden_dim, use_bias=False, name="gru_Wi_r")(flat_emb)
        Wi_h = nn.Dense(hidden_dim, use_bias=False, name="gru_Wi_h")(flat_emb)
        Wi_z = Wi_z.reshape(T, num_actors, hidden_dim)
        Wi_r = Wi_r.reshape(T, num_actors, hidden_dim)
        Wi_h = Wi_h.reshape(T, num_actors, hidden_dim)

        Wh_z = self.param("gru_Wh_z", nn.initializers.orthogonal(), (hidden_dim, hidden_dim))
        Wh_r = self.param("gru_Wh_r", nn.initializers.orthogonal(), (hidden_dim, hidden_dim))
        Wh_h = self.param("gru_Wh_h", nn.initializers.orthogonal(), (hidden_dim, hidden_dim))
        b_z = self.param("gru_b_z", nn.initializers.zeros_init(), (hidden_dim,))
        b_r = self.param("gru_b_r", nn.initializers.zeros_init(), (hidden_dim,))
        b_h = self.param("gru_b_h", nn.initializers.zeros_init(), (hidden_dim,))

        def _gru_step(h, inp):
            wiz_t, wir_t, wih_t, done_t = inp
            # Reset the hidden state on episode boundaries.
            h = jnp.where(done_t[:, None], jnp.zeros_like(h), h)
            z = jax.nn.sigmoid(wiz_t + h @ Wh_z + b_z)
            r = jax.nn.sigmoid(wir_t + h @ Wh_r + b_r)
            h_hat = jnp.tanh(wih_t + (r * h) @ Wh_h + b_h)
            new_h = (1 - z) * h + z * h_hat
            return new_h, new_h

        return jax.lax.scan(_gru_step, hidden, (Wi_z, Wi_r, Wi_h, dones))


class ActorRNN(nn.Module):
    """Policy network over one agent's own observation."""

    action_dim: Sequence[int]
    config: Dict

    @nn.compact
    def __call__(self, hidden, x):
        final_hidden, embedding = RecurrentEncoder(config=self.config)(hidden, x)
        actor_mean = nn.Dense(
            self.config.get("FC_DIM_SIZE", 128),
            kernel_init=orthogonal(2), bias_init=constant(0.0),
        )(embedding)
        actor_mean = nn.relu(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0),
        )(actor_mean)
        return final_hidden, distrax.Categorical(logits=actor_mean)


class CriticRNN(nn.Module):
    """Centralised value network over the joint world state."""

    config: Dict

    @nn.compact
    def __call__(self, hidden, x):
        final_hidden, embedding = RecurrentEncoder(config=self.config)(hidden, x)
        critic = nn.Dense(
            self.config.get("FC_DIM_SIZE", 128),
            kernel_init=orthogonal(2), bias_init=constant(0.0),
        )(embedding)
        critic = nn.relu(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(critic)
        return final_hidden, jnp.squeeze(critic, axis=-1)


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    world_state: jnp.ndarray
    info: jnp.ndarray


def get_rollout(actor_params, config):
    """Run one sampled episode with the trained actor, for GIF generation."""
    env = jaxmarl.make(config["ENV_NAME"], **config.get("ENV_KWARGS", {}))
    hidden_dim = config.get("GRU_HIDDEN_DIM", 128)
    actor = ActorRNN(env.action_space(env.agents[0]).n, config=config)

    key = jax.random.PRNGKey(0)
    key, key_r = jax.random.split(key)
    obs, state = env.reset(key_r)
    hstate = jnp.zeros((env.num_agents, hidden_dim))
    done_batch = jnp.zeros((env.num_agents,), dtype=bool)
    state_seq = [state]
    done = False

    while not done:
        key, key_a, key_s = jax.random.split(key, 3)
        obs_batch = jnp.stack([obs[a] for a in env.agents]).reshape(
            -1, *env.observation_space(env.agents[0]).shape
        )
        hstate, pi = actor.apply(
            actor_params, hstate, (obs_batch[np.newaxis, :], done_batch[np.newaxis, :])
        )
        action = pi.sample(seed=key_a).squeeze(0)
        obs, state, reward, done, info = env.step(
            key_s, state, {a: action[i] for i, a in enumerate(env.agents)}
        )
        done_batch = jnp.array([done[a] for a in env.agents])
        done = done["__all__"]
        state_seq.append(state)

    return state_seq


# ── Training ───────────────────────────────────────────────────────────


def make_train(config):
    """Create the fully JIT'd MAPPO training function for overcooked_v3."""
    env = jaxmarl.make(config["ENV_NAME"], **config.get("ENV_KWARGS", {}))

    config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ACTORS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )

    obs_shape = env.observation_space(env.agents[0]).shape
    action_dim = env.action_space(env.agents[0]).n
    num_agents = env.num_agents

    env = OvercookedV3WorldStateWrapper(env)
    world_state_shape = env.world_state_shape()
    env = LogWrapper(env, replace_info=False)

    # v3 shaped rewards are 0.1-0.3 where the v1 baseline used 3.0-5.0, so
    # COEFF=30 restores a comparable gradient signal.
    shaped_reward_coeff = config.get("SHAPED_REWARD_COEFF", 30.0)
    rew_shaping_min_coeff = config.get("REW_SHAPING_MIN_COEFF", 0.0)
    rew_shaping_anneal = optax.linear_schedule(
        init_value=1.0, end_value=0.0, transition_steps=config["REW_SHAPING_HORIZON"],
    )

    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    def train(rng):
        hidden_dim = config.get("GRU_HIDDEN_DIM", 128)
        actor_network = ActorRNN(action_dim, config=config)
        critic_network = CriticRNN(config=config)

        rng, _rng_actor, _rng_critic = jax.random.split(rng, 3)
        ac_init_x = (
            jnp.zeros((1, config["NUM_ENVS"], *obs_shape)),
            jnp.zeros((1, config["NUM_ENVS"])),
        )
        cr_init_x = (
            jnp.zeros((1, config["NUM_ENVS"], *world_state_shape)),
            jnp.zeros((1, config["NUM_ENVS"])),
        )
        init_hstate_single = jnp.zeros((config["NUM_ENVS"], hidden_dim))
        actor_params = actor_network.init(_rng_actor, init_hstate_single, ac_init_x)
        critic_params = critic_network.init(_rng_critic, init_hstate_single, cr_init_x)

        if config.get("ANNEAL_LR", True):
            lr = linear_schedule
        else:
            lr = config["LR"]
        actor_tx = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.adam(learning_rate=lr, eps=1e-5),
        )
        critic_tx = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.adam(learning_rate=lr, eps=1e-5),
        )
        actor_train_state = TrainState.create(
            apply_fn=actor_network.apply, params=actor_params, tx=actor_tx,
        )
        critic_train_state = TrainState.create(
            apply_fn=critic_network.apply, params=critic_params, tx=critic_tx,
        )

        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset)(reset_rng)
        init_hstate = jnp.zeros((config["NUM_ACTORS"], hidden_dim))

        def _batch_world_state(obs):
            """(NUM_ENVS, num_agents, ...) -> (NUM_ACTORS, ...) agent-major."""
            ws = obs["world_state"].swapaxes(0, 1)
            return ws.reshape((config["NUM_ACTORS"], *world_state_shape))

        def _update_step(runner_state, unused):
            (actor_train_state, critic_train_state, env_state, last_obs, last_done,
             update_step, ac_hstate, cr_hstate, rng) = runner_state

            def _env_step(runner_state, unused):
                (actor_train_state, critic_train_state, env_state, last_obs,
                 last_done, update_step, ac_hstate, cr_hstate, rng) = runner_state

                rng, _rng = jax.random.split(rng)
                obs_batch = jnp.stack(
                    [last_obs[a] for a in env.agents]
                ).reshape(-1, *obs_shape)
                world_state = _batch_world_state(last_obs)

                ac_hstate, pi = actor_network.apply(
                    actor_train_state.params, ac_hstate,
                    (obs_batch[np.newaxis, :], last_done[np.newaxis, :]),
                )
                cr_hstate, value = critic_network.apply(
                    critic_train_state.params, cr_hstate,
                    (world_state[np.newaxis, :], last_done[np.newaxis, :]),
                )
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)
                env_act = unbatchify(
                    action, env.agents, config["NUM_ENVS"], num_agents
                )
                env_act = {k: v.flatten() for k, v in env_act.items()}

                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(env.step)(
                    rng_step, env_state, env_act
                )

                # Shaped reward with anneal, identical to the IPPO trainer.
                original_reward = jnp.array([reward[a] for a in env.agents])
                shaped_reward = info.pop("shaped_reward")
                current_timestep = (
                    update_step * config["NUM_STEPS"] * config["NUM_ENVS"]
                )
                anneal_factor = (
                    rew_shaping_min_coeff
                    + (1.0 - rew_shaping_min_coeff)
                    * rew_shaping_anneal(current_timestep)
                )
                reward = jax.tree.map(
                    lambda x, y: x + shaped_reward_coeff * y * anneal_factor,
                    reward, shaped_reward,
                )

                shaped_reward_arr = jnp.array([shaped_reward[a] for a in env.agents])
                combined_reward = jnp.array([reward[a] for a in env.agents])
                info["shaped_reward"] = shaped_reward_arr
                info["original_reward"] = original_reward
                info["combined_reward"] = combined_reward
                info["anneal_factor"] = jnp.full_like(shaped_reward_arr, anneal_factor)

                agent_major_info_keys = {
                    "shaped_reward", "original_reward", "combined_reward",
                    "anneal_factor",
                }

                def _flatten_info_value(key, value):
                    if (
                        key not in agent_major_info_keys
                        and len(value.shape) >= 2
                        and value.shape[0] == config["NUM_ENVS"]
                        and value.shape[1] == num_agents
                    ):
                        value = jnp.swapaxes(value, 0, 1)
                    return value.reshape((config["NUM_ACTORS"],) + value.shape[2:])

                info = {k: _flatten_info_value(k, v) for k, v in info.items()}
                done_batch = jnp.stack(
                    [done[a] for a in env.agents]
                ).reshape(config["NUM_ACTORS"])

                transition = Transition(
                    jnp.tile(done["__all__"], num_agents),
                    action.squeeze(),
                    value.squeeze(),
                    jnp.stack([reward[a] for a in env.agents]).reshape(
                        config["NUM_ACTORS"]
                    ),
                    log_prob.squeeze(),
                    obs_batch,
                    world_state,
                    info,
                )
                runner_state = (
                    actor_train_state, critic_train_state, env_state, obsv,
                    done_batch, update_step, ac_hstate, cr_hstate, rng,
                )
                return runner_state, transition

            initial_ac_hstate, initial_cr_hstate = ac_hstate, cr_hstate
            runner_state, traj_batch = jax.lax.scan(
                _env_step,
                (actor_train_state, critic_train_state, env_state, last_obs,
                 last_done, update_step, ac_hstate, cr_hstate, rng),
                None,
                config["NUM_STEPS"],
            )
            (actor_train_state, critic_train_state, env_state, last_obs, last_done,
             update_step, ac_hstate, cr_hstate, rng) = runner_state

            # CALCULATE ADVANTAGE (GAE) from the centralised critic
            last_world_state = _batch_world_state(last_obs)
            _, last_val = critic_network.apply(
                critic_train_state.params, cr_hstate,
                (last_world_state[np.newaxis, :], last_done[np.newaxis, :]),
            )
            last_val = last_val.squeeze()

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done, transition.value, transition.reward,
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

            # Normalise at BATCH level. Per-minibatch normalisation collapses on
            # sparse-reward maps where most minibatches contain no reward.
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

            def _update_epoch(update_state, unused):
                def _update_minbatch(train_states, batch_info):
                    actor_train_state, critic_train_state = train_states
                    ac_init_h, cr_init_h, traj_batch, gae, targets = batch_info

                    def _actor_loss_fn(actor_params):
                        _, pi = actor_network.apply(
                            actor_params, ac_init_h[0],
                            (traj_batch.obs, traj_batch.done),
                        )
                        log_prob = pi.log_prob(traj_batch.action)
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config["CLIP_EPS"],
                                1.0 + config["CLIP_EPS"],
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2).mean()
                        entropy = pi.entropy().mean()

                        ent_coef = jnp.maximum(
                            config["ENT_COEF"], config.get("ENT_COEF_MIN", 0.0)
                        )
                        entropy_floor = config.get("ENTROPY_FLOOR", 0.0)
                        entropy_floor_coef = config.get("ENTROPY_FLOOR_COEF", 0.0)
                        entropy_deficit = jnp.maximum(0.0, entropy_floor - entropy)

                        total = (
                            loss_actor
                            - ent_coef * entropy
                            + entropy_floor_coef * entropy_deficit
                        )
                        return total, (loss_actor, entropy)

                    def _critic_loss_fn(critic_params):
                        _, value = critic_network.apply(
                            critic_params, cr_init_h[0],
                            (traj_batch.world_state, traj_batch.done),
                        )
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = 0.5 * jnp.maximum(
                            value_losses, value_losses_clipped
                        ).mean()
                        return config["VF_COEF"] * value_loss, (value_loss,)

                    actor_grad_fn = jax.value_and_grad(_actor_loss_fn, has_aux=True)
                    actor_loss, actor_grads = actor_grad_fn(actor_train_state.params)
                    critic_grad_fn = jax.value_and_grad(_critic_loss_fn, has_aux=True)
                    critic_loss, critic_grads = critic_grad_fn(critic_train_state.params)

                    actor_train_state = actor_train_state.apply_gradients(
                        grads=actor_grads
                    )
                    critic_train_state = critic_train_state.apply_gradients(
                        grads=critic_grads
                    )

                    loss_info = (
                        actor_loss[0] + critic_loss[0],   # total
                        critic_loss[1][0],                # value loss
                        actor_loss[1][0],                 # policy loss
                        actor_loss[1][1],                 # entropy
                    )
                    return (actor_train_state, critic_train_state), loss_info

                (actor_train_state, critic_train_state, ac_init_h, cr_init_h,
                 traj_batch, advantages, targets, rng) = update_state
                rng, _rng = jax.random.split(rng)

                ac_init_h_r = jnp.reshape(ac_init_h, (1, config["NUM_ACTORS"], -1))
                cr_init_h_r = jnp.reshape(cr_init_h, (1, config["NUM_ACTORS"], -1))
                batch = (
                    ac_init_h_r, cr_init_h_r, traj_batch,
                    advantages.squeeze(), targets.squeeze(),
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
                        1, 0,
                    ),
                    shuffled_batch,
                )

                (actor_train_state, critic_train_state), loss_info = jax.lax.scan(
                    _update_minbatch,
                    (actor_train_state, critic_train_state),
                    minibatches,
                )
                update_state = (
                    actor_train_state, critic_train_state, ac_init_h, cr_init_h,
                    traj_batch, advantages, targets, rng,
                )
                return update_state, loss_info

            update_state = (
                actor_train_state, critic_train_state, initial_ac_hstate,
                initial_cr_hstate, traj_batch, advantages, targets, rng,
            )
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            actor_train_state, critic_train_state = update_state[0], update_state[1]
            rng = update_state[-1]

            metric = traj_batch.info
            update_step = update_step + 1
            metric = jax.tree_util.tree_map(lambda x: x.mean(), metric)
            for event_name in EVENT_METRIC_NAMES:
                event_key = f"event/{event_name}"
                if event_key in traj_batch.info:
                    event_values = traj_batch.info[event_key]
                    metric[event_key] = event_values.sum()
                    metric[f"event_rate/{event_name}"] = event_values.mean()
            if "event/delivery" in traj_batch.info:
                delivery_values = traj_batch.info["event/delivery"]
                metric["delivery"] = delivery_values.sum()
                metric["delivery_count.agent_0"] = delivery_values[
                    :, : config["NUM_ENVS"]
                ].sum()
                metric["delivery_count.agent_1"] = delivery_values[
                    :, config["NUM_ENVS"] :
                ].sum()
            metric["update_step"] = update_step
            metric["env_step"] = update_step * config["NUM_STEPS"] * config["NUM_ENVS"]
            metric["base_reward"] = traj_batch.info["original_reward"].sum()
            metric["base_reward_per_step"] = traj_batch.info["original_reward"].mean()
            metric["combined_reward"] = traj_batch.reward.sum()
            metric["combined_reward_per_step"] = traj_batch.reward.mean()
            metric["mean_reward"] = traj_batch.reward.mean()
            metric["max_reward"] = traj_batch.reward.max()
            metric["reward_sum"] = traj_batch.reward.sum()
            metric["loss/total"] = loss_info[0].mean()
            metric["loss/value"] = loss_info[1].mean()
            metric["loss/policy"] = loss_info[2].mean()
            metric["loss/entropy"] = loss_info[3].mean()
            jax.debug.callback(_log_training_metrics, metric)

            runner_state = (
                actor_train_state, critic_train_state, env_state, last_obs,
                last_done, update_step, ac_hstate, cr_hstate, rng,
            )
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (
            actor_train_state,
            critic_train_state,
            env_state,
            obsv,
            jnp.zeros((config["NUM_ACTORS"]), dtype=bool),
            0,
            init_hstate,
            init_hstate,
            _rng,
        )
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metrics": metric}

    return train


@hydra.main(
    version_base=None, config_path="config", config_name="mappo_rnn_overcooked_v3"
)
def main(config):
    """Main training entry point."""
    global _ACTIVE_MONITOR

    config = OmegaConf.to_container(config, resolve=True)
    layout_name = config.get("ENV_KWARGS", {}).get("layout", "unknown")

    wandb.init(
        entity=config.get("ENTITY", ""),
        project=config.get("WANDB_PROJECT", "jaxmarl-mappo"),
        tags=["MAPPO", "RNN", "OvercookedV3"],
        config=config,
        mode=config.get("WANDB_MODE", "disabled"),
        name=config.get("WANDB_NAME") or f"mappo_rnn_overcooked_v3_{layout_name}",
    )
    if wandb.run is not None:
        wandb.define_metric("env_step")
        wandb.define_metric("*", step_metric="env_step")

    rng = jax.random.PRNGKey(config.get("SEED", 42))
    train_fn = make_train(config)
    if wandb.run is not None:
        wandb.config.update(
            {
                "NUM_UPDATES": config["NUM_UPDATES"],
                "NUM_ACTORS": config["NUM_ACTORS"],
                "MINIBATCH_SIZE": config["MINIBATCH_SIZE"],
            },
            allow_val_change=True,
        )
        _log_reward_structure_to_wandb(config)

    monitor_config = {
        "layout": layout_name,
        "total_timesteps": config["TOTAL_TIMESTEPS"],
        "completed_env_steps": config["NUM_UPDATES"] * config["NUM_STEPS"] * config["NUM_ENVS"],
        "num_updates": config["NUM_UPDATES"],
        "num_envs": config["NUM_ENVS"],
        "num_steps": config["NUM_STEPS"],
        "wandb_step": "env_step",
    }

    try:
        with TrainingMonitorInterface(config["NUM_UPDATES"], monitor_config) as monitor:
            _ACTIVE_MONITOR = monitor
            monitor.log(
                "Compiling + training MAPPO RNN v3 on "
                f"{layout_name} for {config['TOTAL_TIMESTEPS']:,} env steps "
                f"({config['NUM_UPDATES']:,} PPO updates)."
            )
            train_jit = jax.jit(train_fn)
            out = jax.block_until_ready(train_jit(rng))

            monitor.log("Training finished; saving checkpoints and GIF.")
            runner_state = out["runner_state"]
            actor_params = runner_state[0].params
            critic_params = runner_state[1].params
            save_path = config.get("SAVE_PATH", "checkpoints/mappo_overcooked_v3")
            # The actor keeps the IPPO checkpoint filename so rollout tooling
            # that looks for model.msgpack keeps working; the critic is only
            # needed to resume training.
            model_path = _save_params(actor_params, save_path, "model.msgpack")
            critic_path = _save_params(critic_params, save_path, "critic.msgpack")
            print(f"Saved actor checkpoint to: {model_path}", flush=True)
            print(f"Saved critic checkpoint to: {critic_path}", flush=True)

            completed_env_steps = (
                config["NUM_UPDATES"] * config["NUM_STEPS"] * config["NUM_ENVS"]
            )
            if wandb.run is not None:
                wandb.log({"saved_model_path": model_path}, step=int(completed_env_steps))

            gif_path = config.get("SAVE_GIF_PATH")
            if gif_path:
                state_seq_list = get_rollout(actor_params, config)
                state_seq = jax.tree.map(lambda *xs: jnp.stack(xs), *state_seq_list)
                env_viz = jaxmarl.make(config["ENV_NAME"], **config.get("ENV_KWARGS", {}))
                viz = OvercookedV3Visualizer(env_viz)
                viz.animate(state_seq, filename=gif_path)
                print(f"Saved GIF to: {gif_path}", flush=True)
    finally:
        _ACTIVE_MONITOR = None
        wandb.finish()


if __name__ == "__main__":
    main()
