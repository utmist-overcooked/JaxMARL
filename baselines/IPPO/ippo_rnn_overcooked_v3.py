"""IPPO (RNN) training script for Overcooked V3.

Closely follows ippo_cnn_overcooked.py (the original, working IPPO):
  - Fully JIT'd training loop via jax.lax.scan over all updates
  - Reward shaping anneal via optax.linear_schedule (REW_SHAPING_HORIZON)
  - jax.debug.callback for W&B logging inside the scan
  - update_step counter tracked in runner_state

GRU replaces nn.scan (broken in Flax 0.10.4 / JAX 0.4.38) with:
  - Pre-computed Dense input projections outside lax.scan
  - Raw weight matrices (self.param) for recurrent ops inside lax.scan
  - W&B sweep support with Bayesian optimization
"""
import sys
import os
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import Callable, Sequence, NamedTuple, Any, Dict
from flax.training.train_state import TrainState
from flax import serialization

if not hasattr(jax.interpreters.xla, "pytype_aval_mappings"):
    jax.interpreters.xla.pytype_aval_mappings = jax.core.pytype_aval_mappings

import distrax
import jaxmarl
from jaxmarl.wrappers.baselines import LogWrapper
from jaxmarl.environments.overcooked_v3.settings import SHAPED_REWARDS, DELIVERY_REWARD
import hydra
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf
import wandb


def _resolve_overcooked_v3_shaped_rewards(config):
    env_kwargs = config.get("ENV_KWARGS", {})
    return {
        "PLACEMENT_IN_POT": float(
            env_kwargs.get(
                "placement_in_pot_reward",
                SHAPED_REWARDS.get("PLACEMENT_IN_POT", 0.0),
            )
        ),
        "POT_START_COOKING": float(
            env_kwargs.get(
                "pot_start_cooking_reward",
                SHAPED_REWARDS["POT_START_COOKING"],
            )
        ),
    }


def _save_model_params(params, save_path):
    os.makedirs(save_path, exist_ok=True)
    model_path = os.path.join(save_path, "model.msgpack")
    with open(model_path, "wb") as f:
        f.write(serialization.to_bytes({"params": params}))
    return model_path


def _restore_model_params(checkpoint_path, template_params=None):
    with open(checkpoint_path, "rb") as f:
        restored = serialization.msgpack_restore(f.read())
    params = restored["params"] if isinstance(restored, dict) and "params" in restored else restored
    if template_params is None:
        return params
    return serialization.from_state_dict(template_params, params)


# ── Network Architecture ───────────────────────────────────────────────


class CNN(nn.Module):
    """CNN encoder for grid observations (matches v2 architecture)."""
    output_size: int = 64
    activation: Callable[..., Any] = nn.relu

    @nn.compact
    def __call__(self, x, train=False):
        # 1x1 convs for channel mixing
        x = nn.Conv(features=128, kernel_size=(1, 1),
                     kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(x)
        x = self.activation(x)
        x = nn.Conv(features=128, kernel_size=(1, 1),
                     kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(x)
        x = self.activation(x)
        x = nn.Conv(features=8, kernel_size=(1, 1),
                     kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(x)
        x = self.activation(x)
        # Spatial convs
        x = nn.Conv(features=16, kernel_size=(3, 3),
                     kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(x)
        x = self.activation(x)
        x = nn.Conv(features=32, kernel_size=(3, 3),
                     kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(x)
        x = self.activation(x)
        x = nn.Conv(features=32, kernel_size=(3, 3),
                     kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(x)
        x = self.activation(x)
        # Flatten: (batch, H', W', C) -> (batch, H'*W'*C)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(features=self.output_size,
                      kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(x)
        x = self.activation(x)
        return x


class ActorCriticRNN(nn.Module):
    """Actor-Critic with CNN encoder and GRU via jax.lax.scan.

    Avoids Flax nn.scan bug by pre-computing input projections (Dense) for
    all timesteps OUTSIDE scan and using raw weight matrices inside scan.
    """
    action_dim: Sequence[int]
    config: Dict

    @nn.compact
    def __call__(self, hidden, x):
        obs, dones = x
        T = obs.shape[0]
        hidden_dim = self.config.get("GRU_HIDDEN_DIM", 128)

        if self.config.get("ACTIVATION", "relu") == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        # CNN embed: vmap over T, CNN handles actor batch dim
        embed_model = CNN(output_size=hidden_dim, activation=activation)
        embedding = jax.vmap(embed_model)(obs)  # (T, num_actors, hidden_dim)
        embedding = nn.LayerNorm()(embedding)

        # GRU input projections — Dense applied to all timesteps at once (outside scan)
        num_actors = obs.shape[1]
        flat_emb = embedding.reshape(-1, hidden_dim)

        Wi_z = nn.Dense(hidden_dim, use_bias=False, name='gru_Wi_z')(flat_emb)
        Wi_r = nn.Dense(hidden_dim, use_bias=False, name='gru_Wi_r')(flat_emb)
        Wi_h = nn.Dense(hidden_dim, use_bias=False, name='gru_Wi_h')(flat_emb)

        Wi_z = Wi_z.reshape(T, num_actors, hidden_dim)
        Wi_r = Wi_r.reshape(T, num_actors, hidden_dim)
        Wi_h = Wi_h.reshape(T, num_actors, hidden_dim)

        # Recurrent weight matrices as raw params (safe inside lax.scan)
        Wh_z = self.param('gru_Wh_z', nn.initializers.orthogonal(), (hidden_dim, hidden_dim))
        Wh_r = self.param('gru_Wh_r', nn.initializers.orthogonal(), (hidden_dim, hidden_dim))
        Wh_h = self.param('gru_Wh_h', nn.initializers.orthogonal(), (hidden_dim, hidden_dim))
        b_z = self.param('gru_b_z', nn.initializers.zeros_init(), (hidden_dim,))
        b_r = self.param('gru_b_r', nn.initializers.zeros_init(), (hidden_dim,))
        b_h = self.param('gru_b_h', nn.initializers.zeros_init(), (hidden_dim,))

        def _gru_step(h, inp):
            wiz_t, wir_t, wih_t, done_t = inp
            # Reset hidden on episode boundaries
            h = jnp.where(done_t[:, None], jnp.zeros_like(h), h)
            z = jax.nn.sigmoid(wiz_t + h @ Wh_z + b_z)
            r = jax.nn.sigmoid(wir_t + h @ Wh_r + b_r)
            h_hat = jnp.tanh(wih_t + (r * h) @ Wh_h + b_h)
            new_h = (1 - z) * h + z * h_hat
            return new_h, new_h

        final_hidden, embedding = jax.lax.scan(
            _gru_step, hidden, (Wi_z, Wi_r, Wi_h, dones)
        )  # embedding: (T, num_actors, hidden_dim)

        # Actor head — applied directly to (T, num_actors, hidden_dim)
        actor_mean = nn.Dense(
            self.config.get("FC_DIM_SIZE", 128),
            kernel_init=orthogonal(2), bias_init=constant(0.0),
        )(embedding)
        actor_mean = nn.relu(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0),
        )(actor_mean)

        pi = distrax.Categorical(logits=actor_mean)

        # Critic head — applied directly to (T, num_actors, hidden_dim)
        critic = nn.Dense(
            self.config.get("FC_DIM_SIZE", 128),
            kernel_init=orthogonal(2), bias_init=constant(0.0),
        )(embedding)
        critic = nn.relu(critic)
        critic = nn.Dense(
            1, kernel_init=orthogonal(1.0), bias_init=constant(0.0),
        )(critic)

        return final_hidden, pi, jnp.squeeze(critic, axis=-1)


# ── Utilities ──────────────────────────────────────────────────────────


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray


def batchify(x: dict, agent_list, num_actors):
    x = jnp.stack([x[a] for a in agent_list])
    return x.reshape((num_actors, -1))


def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}


def flatten_info_leaf(x, num_envs, num_agents, num_actors):
    x = jnp.asarray(x)
    if x.size == num_actors:
        return x.reshape((num_actors,))
    if x.size == num_envs:
        return jnp.repeat(x.reshape((num_envs,)), num_agents, axis=0)
    if x.size == 1:
        return jnp.broadcast_to(x.reshape(()), (num_actors,))
    return x.reshape((num_actors,))


# ── Training ───────────────────────────────────────────────────────────


def make_train(config):
    """Create the fully JIT'd IPPO training function for overcooked_v3.

    Mirrors the original ippo_cnn_overcooked.py structure:
      - jax.lax.scan over _update_step (all updates compiled)
      - optax.linear_schedule for reward shaping anneal
      - jax.debug.callback for W&B logging
    """
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

    env = LogWrapper(env, replace_info=False)

    # Shaped reward coefficient:
    # v3 shaped rewards are 0.1-0.3; v1 baseline uses 3.0-5.0.
    # COEFF=30 brings v3 to parity (30x0.1=3.0), matching v1 gradient signal.
    shaped_reward_coeff = config.get("SHAPED_REWARD_COEFF", 30.0)

    # Reward shaping anneal: linearly decay from 1.0 to 0.0 (matches original)
    rew_shaping_anneal = optax.linear_schedule(
        init_value=1.0,
        end_value=0.0,
        transition_steps=config["REW_SHAPING_HORIZON"],
    )

    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    def train(rng):
        # INIT NETWORK
        hidden_dim = config.get("GRU_HIDDEN_DIM", 128)
        network = ActorCriticRNN(action_dim, config=config)

        rng, _rng = jax.random.split(rng)
        init_x = (
            jnp.zeros((1, config["NUM_ENVS"], *obs_shape)),
            jnp.zeros((1, config["NUM_ENVS"])),
        )
        init_hstate = jnp.zeros((config["NUM_ENVS"], hidden_dim))
        network_params = network.init(_rng, init_hstate, init_x)
        pretrained_checkpoint = config.get("PRETRAINED_CHECKPOINT")
        if pretrained_checkpoint:
            network_params = _restore_model_params(pretrained_checkpoint, network_params)

        if config.get("ANNEAL_LR", True):
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )
        train_state = TrainState.create(
            apply_fn=network.apply, params=network_params, tx=tx,
        )

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset)(reset_rng)
        init_hstate = jnp.zeros((config["NUM_ACTORS"], hidden_dim))

        # TRAIN LOOP — fully inside jax.lax.scan (matches original)
        def _update_step(runner_state, unused):
            train_state, env_state, last_obs, last_done, update_step, hstate, rng = (
                runner_state
            )

            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, last_done, update_step, hstate, rng = (
                    runner_state
                )

                rng, _rng = jax.random.split(rng)
                obs_batch = jnp.stack(
                    [last_obs[a] for a in env.agents]
                ).reshape(-1, *obs_shape)

                ac_in = (obs_batch[np.newaxis, :], last_done[np.newaxis, :])
                hstate, pi, value = network.apply(
                    train_state.params, hstate, ac_in
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

                # Shaped reward with anneal (matches original ippo_cnn_overcooked)
                original_reward = jnp.array([reward[a] for a in env.agents])
                shaped_reward = info.pop("shaped_reward")
                current_timestep = (
                    update_step * config["NUM_STEPS"] * config["NUM_ENVS"]
                )
                anneal_factor = rew_shaping_anneal(current_timestep)
                reward = jax.tree.map(
                    lambda x, y: x + shaped_reward_coeff * y * anneal_factor,
                    reward,
                    shaped_reward,
                )

                shaped_reward_arr = jnp.array([shaped_reward[a] for a in env.agents])
                combined_reward = jnp.array([reward[a] for a in env.agents])
                info["shaped_reward"] = shaped_reward_arr
                info["original_reward"] = original_reward
                info["combined_reward"] = combined_reward
                info["anneal_factor"] = jnp.full_like(shaped_reward_arr, anneal_factor)

                info = jax.tree.map(
                    lambda x: flatten_info_leaf(
                        x,
                        config["NUM_ENVS"],
                        num_agents,
                        config["NUM_ACTORS"],
                    ),
                    info,
                )
                done_batch = jnp.stack(
                    [done[a] for a in env.agents]
                ).reshape(config["NUM_ACTORS"])

                transition = Transition(
                    jnp.tile(done["__all__"], num_agents),
                    action.squeeze(),
                    value.squeeze(),
                    batchify(reward, env.agents, config["NUM_ACTORS"]).squeeze(),
                    log_prob.squeeze(),
                    obs_batch,
                    info,
                )
                runner_state = (
                    train_state, env_state, obsv, done_batch,
                    update_step, hstate, rng,
                )
                return runner_state, transition

            initial_hstate = hstate
            runner_state, traj_batch = jax.lax.scan(
                _env_step,
                (train_state, env_state, last_obs, last_done,
                 update_step, hstate, rng),
                None,
                config["NUM_STEPS"],
            )
            train_state, env_state, last_obs, last_done, update_step, hstate, rng = (
                runner_state
            )

            # CALCULATE ADVANTAGE (GAE)
            last_obs_batch = jnp.stack(
                [last_obs[a] for a in env.agents]
            ).reshape(-1, *obs_shape)
            ac_in = (last_obs_batch[np.newaxis, :], last_done[np.newaxis, :])
            _, _, last_val = network.apply(train_state.params, hstate, ac_in)
            last_val = last_val.squeeze()

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    delta = (
                        reward + config["GAMMA"] * next_value * (1 - done) - value
                    )
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

            # Normalise advantages at BATCH level (before minibatch split).
            # Per-minibatch normalization causes catastrophic collapse on
            # sparse-reward maps where most minibatches have zero rewards.
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

            # UPDATE NETWORK (PPO epochs)
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    init_hstate, traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, init_hstate, traj_batch, gae, targets):
                        _, pi, value = network.apply(
                            params,
                            init_hstate.squeeze(),
                            (traj_batch.obs, traj_batch.done),
                        )
                        log_prob = pi.log_prob(traj_batch.action)

                        # Value loss (clipped)
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(
                            value_pred_clipped - targets
                        )
                        value_loss = (
                            0.5
                            * jnp.maximum(
                                value_losses, value_losses_clipped
                            ).mean()
                        )

                        # Actor loss (clipped)
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        # NOTE: gae is already normalised at batch level
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

                        ent_coef = jnp.maximum(
                            config["ENT_COEF"], config.get("ENT_COEF_MIN", 0.0)
                        )
                        entropy_floor = config.get("ENTROPY_FLOOR", 0.0)
                        entropy_floor_coef = config.get("ENTROPY_FLOOR_COEF", 0.0)
                        entropy_deficit = jnp.maximum(0.0, entropy_floor - entropy)

                        total_loss = (
                            loss_actor
                            + config["VF_COEF"] * value_loss
                            - ent_coef * entropy
                            + entropy_floor_coef * entropy_deficit
                        )
                        return total_loss, (value_loss, loss_actor, entropy)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params,
                        init_hstate,
                        traj_batch,
                        advantages,
                        targets,
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                (
                    train_state,
                    init_hstate,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                ) = update_state
                rng, _rng = jax.random.split(rng)

                init_hstate_r = jnp.reshape(
                    init_hstate, (1, config["NUM_ACTORS"], -1)
                )
                batch = (
                    init_hstate_r,
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

                train_state, total_loss = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )
                update_state = (
                    train_state,
                    init_hstate,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                )
                return update_state, total_loss

            update_state = (
                train_state,
                initial_hstate,
                traj_batch,
                advantages,
                targets,
                rng,
            )
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            train_state = update_state[0]
            metric = traj_batch.info
            rng = update_state[-1]

            # Logging via jax.debug.callback (matches original)
            def callback(metric):
                wandb.log(metric)

            update_step = update_step + 1
            metric = jax.tree_util.tree_map(lambda x: x.mean(), metric)
            metric["update_step"] = update_step
            metric["env_step"] = (
                update_step * config["NUM_STEPS"] * config["NUM_ENVS"]
            )
            metric["base_reward_per_step"] = traj_batch.info["original_reward"].mean()
            metric["shaped_reward_per_step"] = traj_batch.info["shaped_reward"].mean()
            metric["combined_reward_per_step"] = traj_batch.info["combined_reward"].mean()
            metric["delivery_rate_per_step"] = jnp.stack(
                [metric["delivery_count"][agent] for agent in env.agents]
            ).mean()
            metric["mean_reward"] = traj_batch.reward.mean()
            metric["max_reward"] = traj_batch.reward.max()
            metric["reward_sum"] = traj_batch.reward.sum()

            # Event counters (per-step rates across rollout)
            for event_name in ["pickup", "pot_placement", "pot_start_cooking",
                               "dish_pickup", "drop", "delivery"]:
                metric[f"event/{event_name}"] = traj_batch.info[event_name].mean()

            # Loss components from last epoch
            metric["loss/total"] = loss_info[0].mean()
            metric["loss/value"] = loss_info[1][0].mean()
            metric["loss/actor"] = loss_info[1][1].mean()
            metric["loss/entropy"] = loss_info[1][2].mean()

            jax.debug.callback(callback, metric)

            runner_state = (
                train_state,
                env_state,
                last_obs,
                last_done,
                update_step,
                hstate,
                rng,
            )
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (
            train_state,
            env_state,
            obsv,
            jnp.zeros((config["NUM_ACTORS"]), dtype=bool),
            0,
            init_hstate,
            _rng,
        )
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metrics": metric}

    return train


# ── Sweep Configuration ───────────────────────────────────────────────


def _build_overcooked_v3_sweep_configuration():
    """W&B sweep config for IPPO on overcooked_v3."""
    return {
        "method": "bayes",
        "metric": {"goal": "maximize", "name": "returned_episode_returns"},
        "parameters": {
            "LR": {"distribution": "log_uniform_values", "min": 5e-5, "max": 1e-3},
            "ENT_COEF": {
                "distribution": "log_uniform_values",
                "min": 1e-3,
                "max": 1e-2,
            },
            "VF_COEF": {"values": [0.25, 0.5, 0.75, 1.0]},
            "MAX_GRAD_NORM": {"values": [0.25, 0.5]},
            "GAMMA": {"values": [0.99, 0.995]},
            "GAE_LAMBDA": {"values": [0.95, 0.98]},
            "CLIP_EPS": {"values": [0.1, 0.2]},
            "UPDATE_EPOCHS": {"values": [2, 4]},
            "GRU_HIDDEN_DIM": {"values": [128, 256]},
            "REW_SHAPING_HORIZON": {"values": [10000000, 20000000, 40000000]},
            "SHAPED_REWARD_COEFF": {"values": [0.5, 1.0, 2.0, 3.0]},
        },
    }


def run_wandb_sweep(base_config):
    """Run a W&B sweep using current config as base defaults."""
    project = base_config.get("WANDB_PROJECT", "jaxmarl-ippo")
    sweep_count = int(base_config.get("WANDB_SWEEP_COUNT", 20))
    sweep_configuration = _build_overcooked_v3_sweep_configuration()

    def _objective():
        with wandb.init(project=project, config=base_config, mode="online") as run:
            run_config = dict(run.config)
            train_config = dict(base_config)
            train_config.update(run_config)
            train_config["WANDB_MODE"] = "online"

            train_jit = jax.jit(make_train(train_config))
            rng = jax.random.PRNGKey(train_config.get("SEED", 42))
            output = train_jit(rng)

            runner_state = output["runner_state"]
            train_state = runner_state[0]
            params = train_state.params

            base_save_path = train_config.get("SAVE_PATH", "checkpoints/ippo_overcooked_v3")
            run_save_path = os.path.join(base_save_path, f"sweep_{run.id}")
            model_path = _save_model_params(params, run_save_path)
            wandb.log({"saved_model_path": model_path})

    sweep_id = wandb.sweep(sweep=sweep_configuration, project=project)
    wandb.agent(sweep_id, function=_objective, count=sweep_count)


@hydra.main(
    version_base=None, config_path="config", config_name="ippo_rnn_overcooked_v3"
)
def main(config):
    """Main training entry point."""
    config = OmegaConf.to_container(config, resolve=True)
    if config.get("PRETRAINED_CHECKPOINT"):
        config["PRETRAINED_CHECKPOINT"] = to_absolute_path(config["PRETRAINED_CHECKPOINT"])

    if config.get("WANDB_SWEEP", False):
        run_wandb_sweep(config)
        return

    layout_name = config.get("ENV_KWARGS", {}).get("layout", "unknown")

    # Log shaped rewards config alongside hyperparams
    config["SHAPED_REWARDS"] = _resolve_overcooked_v3_shaped_rewards(config)
    config["DELIVERY_REWARD"] = float(
        config.get("ENV_KWARGS", {}).get("delivery_reward", DELIVERY_REWARD)
    )

    wandb.init(
        entity=config.get("ENTITY", ""),
        project=config.get("WANDB_PROJECT", "jaxmarl-ippo"),
        tags=["IPPO", "RNN", "OvercookedV3"],
        config=config,
        mode=config.get("WANDB_MODE", "disabled"),
        name=config.get("WANDB_NAME") or f"ippo_rnn_overcooked_v3_{layout_name}",
    )

    if config.get("PRETRAINED_CHECKPOINT"):
        print(f"Warm-starting IPPO from checkpoint: {config['PRETRAINED_CHECKPOINT']}")

    rng = jax.random.PRNGKey(config.get("SEED", 42))
    train_jit = jax.jit(make_train(config))
    out = train_jit(rng)

    runner_state = out["runner_state"]
    train_state = runner_state[0]
    params = train_state.params
    save_path = config.get("SAVE_PATH", "checkpoints/ippo_overcooked_v3")
    model_path = _save_model_params(params, save_path)
    print(f"Saved model checkpoint to: {model_path}")

    if config.get("WANDB_MODE", "disabled") != "disabled":
        wandb.finish()


if __name__ == "__main__":
    main()
