"""IPPO with CNN on Overcooked V3.

Exact same model architecture and training loop as ippo_cnn_overcooked.py (v1).
Only differences:
  - ENV_NAME: "overcooked_v3"  (layout passed as string, not numpy array)
  - SHAPED_REWARD_COEFF=30 to compensate v3's smaller raw shaped reward values
    (v3 raw: 0.1-0.3 vs v1 raw: 3.0-5.0 → coeff=30 brings them to parity)
  - Uses OvercookedV3Visualizer for gif output
"""

import os
import copy
import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any, Dict
from flax.training.train_state import TrainState
import distrax
from flax import serialization
import jaxmarl
from jaxmarl.environments.overcooked_v3.settings import (
    BURN_PENALTY,
    DEFAULT_ORDER_EXPIRATION_TIME,
    DEFAULT_ORDER_GENERATION_RATE,
    DELIVERY_REWARD,
    ORDER_EXPIRED_PENALTY,
    POT_BURN_TIME,
    POT_COOK_TIME,
    SHAPED_REWARDS,
)
from jaxmarl.wrappers.baselines import LogWrapper
import hydra
from omegaconf import OmegaConf
import wandb
from baselines.IC3Net.monitor import TrainingMonitorInterface
from baselines.overcooked_v3.models.ippo import IPPOCNNRolloutPolicy
from baselines.overcooked_v3.rollout import rollout_episode
from baselines.overcooked_v3.training import OvercookedV3Training


def _save_model_params(params, save_path):
    """Save model params to a msgpack checkpoint."""
    os.makedirs(save_path, exist_ok=True)
    model_path = os.path.join(save_path, "model.msgpack")
    with open(model_path, "wb") as f:
        f.write(serialization.to_bytes({"params": params}))
    print(f"** Checkpoint saved to: {model_path} **", flush=True)
    return model_path


def _load_model_params(checkpoint_path, template_params):
    """Load model params from a msgpack checkpoint."""
    with open(checkpoint_path, "rb") as f:
        data = f.read()
    restored = serialization.from_bytes({"params": template_params}, data)
    return restored["params"]


EVENT_METRIC_NAMES = (
    "pot_start_cooking",
    "pot_placement",
    "pickup",
    "drop",
    "dish_pickup",
    "dish_to_goal_progress",
    "delivery",
    "pot_burn",
    "order_expired",
    "order_added",
)
_ACTIVE_MONITOR = None
_ACTIVE_SHAPED_REWARD_KEYS = (
    "INGREDIENT_PICKUP",
    "PLACEMENT_IN_POT",
    "SOUP_IN_DISH",
    "PLATE_PICKUP",
    "PLATE_PICKUP_DURING_COOKING",
    "DISH_TO_GOAL_PROGRESS",
)


def _env_kwarg(config: Dict[str, Any], name: str, default: Any) -> Any:
    return (config.get("ENV_KWARGS", {}) or {}).get(name, default)


def _build_reward_structure(config: Dict[str, Any]) -> Dict[str, Any]:
    """Describe the reward setup that will actually be used for learning."""
    shaped_enabled = bool(_env_kwarg(config, "shaped_rewards", True))
    shaped_reward_coeff = float(config.get("SHAPED_REWARD_COEFF", 1.0))
    rew_shaping_min_coeff = float(config.get("REW_SHAPING_MIN_COEFF", 0.0))
    delivery_reward = float(_env_kwarg(config, "delivery_reward", DELIVERY_REWARD))
    pot_burn_time = int(_env_kwarg(config, "pot_burn_time", POT_BURN_TIME))
    order_queue_enabled = bool(_env_kwarg(config, "enable_order_queue", False))
    order_expiration_time = int(
        _env_kwarg(config, "order_expiration_time", DEFAULT_ORDER_EXPIRATION_TIME)
    )
    raw_shaped_rewards = {
        reward_name: float(reward_value)
        for reward_name, reward_value in SHAPED_REWARDS.items()
    }

    reward_rows = [
        {
            "category": "base",
            "name": "DELIVERY_REWARD",
            "raw_value": delivery_reward,
            "effective_at_anneal_1": delivery_reward,
            "effective_at_floor": delivery_reward,
            "active_in_learning": True,
            "note": "Sparse reward for a correct delivery.",
        }
    ]

    for reward_name, reward_value in raw_shaped_rewards.items():
        used_by_env = reward_name in _ACTIVE_SHAPED_REWARD_KEYS
        active = shaped_enabled and used_by_env and reward_value != 0.0
        reward_rows.append(
            {
                "category": "shaped",
                "name": reward_name,
                "raw_value": reward_value,
                "effective_at_anneal_1": (
                    reward_value * shaped_reward_coeff if active else 0.0
                ),
                "effective_at_floor": (
                    reward_value * shaped_reward_coeff * rew_shaping_min_coeff
                    if active
                    else 0.0
                ),
                "active_in_learning": active,
                "note": (
                    "Configured but not currently added by the environment."
                    if not used_by_env
                    else "Weight is zero; event may still be logged."
                    if reward_value == 0.0
                    else "Added to shaped_reward before the trainer coefficient/anneal."
                ),
            }
        )

    burn_enabled = pot_burn_time > 0 and float(BURN_PENALTY) != 0.0
    reward_rows.append(
        {
            "category": "penalty",
            "name": "BURN_PENALTY",
            "raw_value": float(BURN_PENALTY),
            "effective_at_anneal_1": float(BURN_PENALTY) if burn_enabled else 0.0,
            "effective_at_floor": float(BURN_PENALTY) if burn_enabled else 0.0,
            "active_in_learning": burn_enabled,
            "note": "Inactive when pot_burn_time is 0 or the penalty is 0.",
        }
    )
    reward_rows.append(
        {
            "category": "penalty",
            "name": "ORDER_EXPIRED_PENALTY",
            "raw_value": float(ORDER_EXPIRED_PENALTY),
            "effective_at_anneal_1": (
                float(ORDER_EXPIRED_PENALTY) if order_queue_enabled else 0.0
            ),
            "effective_at_floor": (
                float(ORDER_EXPIRED_PENALTY) if order_queue_enabled else 0.0
            ),
            "active_in_learning": order_queue_enabled,
            "note": "Inactive when enable_order_queue is false.",
        }
    )

    return {
        "layout": _env_kwarg(config, "layout", "unknown"),
        "env_name": config.get("ENV_NAME", "unknown"),
        "shaped_rewards_enabled": shaped_enabled,
        "shaped_reward_coeff": shaped_reward_coeff,
        "rew_shaping_horizon": int(config.get("REW_SHAPING_HORIZON", 0)),
        "rew_shaping_min_coeff": rew_shaping_min_coeff,
        "anneal_formula": (
            "sparse_reward + SHAPED_REWARD_COEFF * shaped_reward * "
            "(REW_SHAPING_MIN_COEFF + (1 - REW_SHAPING_MIN_COEFF) * "
            "linear_decay(env_step, REW_SHAPING_HORIZON))"
        ),
        "mechanics": {
            "pot_cook_time": int(_env_kwarg(config, "pot_cook_time", POT_COOK_TIME)),
            "pot_burn_time": pot_burn_time,
            "burn_enabled": burn_enabled,
            "order_queue_enabled": order_queue_enabled,
            "order_generation_rate": float(
                _env_kwarg(config, "order_generation_rate", DEFAULT_ORDER_GENERATION_RATE)
            ),
            "order_expiration_time": order_expiration_time,
            "max_steps": int(_env_kwarg(config, "max_steps", 400)),
        },
        "shaped_rewards": raw_shaped_rewards,
        "rewards": reward_rows,
    }


def _log_reward_structure_to_wandb(config: Dict[str, Any]) -> None:
    """Persist the reward setup into W&B config, summary, and a run-start table."""
    if wandb.run is None:
        return

    reward_structure = _build_reward_structure(config)
    wandb.config.update({"reward_structure": reward_structure}, allow_val_change=True)
    wandb.run.summary["reward_structure"] = reward_structure

    table = wandb.Table(
        columns=[
            "category",
            "name",
            "raw_value",
            "effective_at_anneal_1",
            "effective_at_floor",
            "active_in_learning",
            "note",
        ]
    )
    for row in reward_structure["rewards"]:
        table.add_data(
            row["category"],
            row["name"],
            row["raw_value"],
            row["effective_at_anneal_1"],
            row["effective_at_floor"],
            row["active_in_learning"],
            row["note"],
        )

    flat_metrics = {
        "env_step": 0,
        "reward_structure/shaped_reward_coeff": reward_structure["shaped_reward_coeff"],
        "reward_structure/rew_shaping_horizon": reward_structure["rew_shaping_horizon"],
        "reward_structure/rew_shaping_min_coeff": reward_structure["rew_shaping_min_coeff"],
        "reward_structure/pot_cook_time": reward_structure["mechanics"]["pot_cook_time"],
        "reward_structure/pot_burn_time": reward_structure["mechanics"]["pot_burn_time"],
        "reward_structure/burn_enabled": int(reward_structure["mechanics"]["burn_enabled"]),
        "reward_structure/order_queue_enabled": int(
            reward_structure["mechanics"]["order_queue_enabled"]
        ),
    }
    for row in reward_structure["rewards"]:
        metric_name = row["name"].lower()
        flat_metrics[f"reward_structure/raw/{metric_name}"] = row["raw_value"]
        flat_metrics[f"reward_structure/effective_start/{metric_name}"] = row[
            "effective_at_anneal_1"
        ]
        flat_metrics[f"reward_structure/effective_floor/{metric_name}"] = row[
            "effective_at_floor"
        ]
        flat_metrics[f"reward_structure/active/{metric_name}"] = int(
            row["active_in_learning"]
        )

    wandb.log({"reward_structure/table": table, **flat_metrics}, step=0)


def _to_python_value(value):
    arr = np.asarray(value)
    if arr.shape == () or arr.size == 1:
        return arr.reshape(()).item()
    return arr.tolist()


def _flatten_metric_dict(metric: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    flat = {}
    for key, value in metric.items():
        name = f"{prefix}/{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(_flatten_metric_dict(value, name))
        else:
            flat[name] = _to_python_value(value)
    return flat


def _first_scalar(value, default=0):
    if isinstance(value, (list, tuple)):
        return _first_scalar(value[0], default) if value else default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _monitor_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    keys = (
        ("env_step", "env_step"),
        ("base_reward_per_step", "base_rew/step"),
        ("combined_reward_per_step", "combined_rew/step"),
        ("combined_reward", "combined_rew"),
        ("delivery", "delivery"),
        ("event/pickup", "pickup"),
        ("event/drop", "drop"),
        ("event/pot_placement", "pot_place"),
        ("event/pot_start_cooking", "pot_start"),
        ("event/dish_pickup", "dish_pickup"),
        ("event/dish_to_goal_progress", "dish_to_goal"),
        ("event/pot_burn", "pot_burn"),
        ("event/order_expired", "order_expired"),
        ("event/order_added", "order_added"),
        ("order/front_type", "order_front"),
        ("order/active_count", "orders_active"),
        ("loss/total", "loss"),
        ("loss/value", "value_loss"),
        ("loss/entropy", "entropy"),
        ("anneal_factor", "anneal"),
    )
    return {label: payload[key] for key, label in keys if key in payload}


def _log_training_metrics(metric: Dict[str, Any]) -> None:
    """Move JAX-side metrics to Python for W&B and terminal progress logging.

    The training loop is JIT-compiled, so ordinary Python logging cannot run
    inside it. jax.debug.callback calls this function after each PPO update.
    """
    payload = _flatten_metric_dict(metric)
    env_step = _first_scalar(payload.get("env_step", payload.get("update_step", 0)))
    update_step = _first_scalar(payload.get("update_step", 0))
    payload["env_step"] = env_step
    payload["update_step"] = update_step

    if wandb.run is not None:
        wandb.log(payload, step=env_step)

    if _ACTIVE_MONITOR is not None:
        _ACTIVE_MONITOR.update(update_step, _monitor_payload(payload))


# ──────────────────────────────────────────────────────────────────────────────
# Network
# ──────────────────────────────────────────────────────────────────────────────

class CNN(nn.Module):
    """Small visual encoder shared by the actor and critic.

    Input observations are Overcooked grid tensors. The three convolutions learn
    local spatial features (walls, pots, agents, objects), then the dense layer
    compresses the grid into a compact embedding for policy/value heads.
    """
    activation: str = "relu"
    channels: int = 128
    embed_dim: int = 128

    @nn.compact
    def __call__(self, x):
        activation = nn.relu if self.activation == "relu" else nn.tanh
        x = nn.Conv(features=self.channels, kernel_size=(5, 5),
                    kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = activation(x)
        x = nn.Conv(features=self.channels, kernel_size=(3, 3),
                    kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = activation(x)
        x = nn.Conv(features=self.channels, kernel_size=(3, 3),
                    kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = activation(x)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(features=self.embed_dim, kernel_init=orthogonal(np.sqrt(2)),
                     bias_init=constant(0.0))(x)
        x = activation(x)
        return x


class ActorCritic(nn.Module):
    """Policy/value network used by IPPO.

    IPPO trains one shared network for both agents. Each agent observation is
    evaluated independently, but all agents contribute gradients to the same
    actor and critic parameters.
    """
    action_dim: Sequence[int]
    activation: str = "relu"
    cnn_channels: int = 128
    cnn_embed_dim: int = 128
    fc_dim_size: int = 128

    @nn.compact
    def __call__(self, x):
        activation = nn.relu if self.activation == "relu" else nn.tanh
        embedding = CNN(
            activation=self.activation,
            channels=self.cnn_channels,
            embed_dim=self.cnn_embed_dim,
        )(x)

        # Actor head: logits over discrete actions, wrapped as a categorical
        # distribution so we can sample actions and compute PPO log-probs.
        actor_mean = nn.Dense(self.fc_dim_size, kernel_init=orthogonal(np.sqrt(2)),
                              bias_init=constant(0.0))(embedding)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01),
                              bias_init=constant(0.0))(actor_mean)
        pi = distrax.Categorical(logits=actor_mean)

        # Critic head: scalar state-value estimate for GAE and value loss.
        critic = nn.Dense(self.fc_dim_size, kernel_init=orthogonal(np.sqrt(2)),
                          bias_init=constant(0.0))(embedding)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0),
                          bias_init=constant(0.0))(critic)

        return pi, jnp.squeeze(critic, axis=-1)


class Transition(NamedTuple):
    """One flattened actor-time transition stored during rollout collection."""
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def batchify(x: dict, agent_list, num_actors):
    x = jnp.stack([x[a] for a in agent_list])
    return x.reshape((num_actors, -1))


def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}


def _make_network(action_dim, config):
    """Build the CNN policy/value network from config-controlled widths."""
    return ActorCritic(
        action_dim,
        activation=config["ACTIVATION"],
        cnn_channels=int(config.get("CNN_CHANNELS", 128)),
        cnn_embed_dim=int(config.get("CNN_EMBED_DIM", 128)),
        fc_dim_size=int(config.get("FC_DIM_SIZE", 128)),
    )


def get_rollout(params, config):
    """Return states from one deterministic episode for compatibility."""

    env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    episode = rollout_episode(
        env,
        IPPOCNNRolloutPolicy(env, config, _make_network),
        params,
        seed=int(config.get("ROLLOUT_GIF_ENV_SEED", 0)),
        max_steps=int(
            config.get(
                "ROLLOUT_GIF_MAX_STEPS",
                config.get("ENV_KWARGS", {}).get("max_steps", 400),
            )
        ),
    )
    return list(episode.states)


# ──────────────────────────────────────────────────────────────────────────────
# Training (identical loop to v1)
# ──────────────────────────────────────────────────────────────────────────────

def make_train(config):
    """Create the fully JIT-compiled IPPO CNN training function."""

    env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])

    # Derived PPO sizes. NUM_ACTORS is agents x parallel envs. Each PPO update
    # collects NUM_STEPS transitions for every actor, then shuffles that rollout
    # into NUM_MINIBATCHES for UPDATE_EPOCHS passes.
    config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]
    config["NUM_UPDATES"] = (
        int(config["TOTAL_TIMESTEPS"]) // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ACTORS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )

    shaped_reward_coeff = config.get("SHAPED_REWARD_COEFF", 30.0)
    rew_shaping_min_coeff = config.get("REW_SHAPING_MIN_COEFF", 0.0)

    # Keep the environment's shaped_reward info while adding episode logging.
    env = LogWrapper(env, replace_info=False)

    # Reward used for learning:
    #   sparse_delivery_reward + coeff * shaped_reward * anneal_factor
    # anneal_factor linearly decays over REW_SHAPING_HORIZON env steps. A small
    # optional floor keeps curriculum rewards alive during late sparse-reward
    # fine-tuning instead of making every non-delivery transition exactly zero.
    rew_shaping_anneal = optax.linear_schedule(
        init_value=1.0,
        end_value=0.0,
        transition_steps=config["REW_SHAPING_HORIZON"],
    )

    def linear_schedule(count):
        # Optional PPO learning-rate decay. count is optimizer minibatch count,
        # so divide by minibatches-per-update to recover PPO update progress.
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    def train(rng):
        """Train one random seed and return its final runner state."""

        # Build and initialize the shared actor-critic network.
        network = _make_network(env.action_space("agent_0").n, config)
        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros((1, *env.observation_space("agent_0").shape))
        network_params = network.init(_rng, init_x)

        # Optionally warm-start from a pre-trained checkpoint
        init_ckpt = config.get("INIT_CHECKPOINT", None)
        if init_ckpt:
            print(f"** Loading init weights from: {init_ckpt} **", flush=True)
            network_params = _load_model_params(init_ckpt, network_params)

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

        # Reset every parallel environment with an independent RNG key.
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)

        def _update_step(runner_state, unused):
            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, update_step, rng = runner_state
                rng, _rng = jax.random.split(rng)

                # Flatten agent-major observations into one actor batch:
                # shape = (num_agents * num_envs, H, W, C).
                obs_batch = jnp.stack(
                    [last_obs[a] for a in env.agents]
                ).reshape(-1, *env.observation_space("agent_0").shape)

                # Sample actions from the current policy and keep log-probs for
                # the PPO ratio calculation during the update phase.
                pi, value = network.apply(train_state.params, obs_batch)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)
                env_act = unbatchify(action, env.agents, config["NUM_ENVS"], env.num_agents)
                env_act = {k: v.flatten() for k, v in env_act.items()}

                # Step every parallel env in one vectorized call.
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0)
                )(rng_step, env_state, env_act)

                original_reward = jnp.array([reward[a] for a in env.agents])
                shaped_reward = info.pop("shaped_reward")
                current_timestep = update_step * config["NUM_STEPS"] * config["NUM_ENVS"]
                anneal_factor = (
                    rew_shaping_min_coeff
                    + (1.0 - rew_shaping_min_coeff)
                    * rew_shaping_anneal(current_timestep)
                )
                # Combine sparse reward and shaped reward here, not inside the
                # environment, so we can sweep coeff/horizon without changing env.
                reward = jax.tree.map(
                    lambda x, y: x + shaped_reward_coeff * y * anneal_factor,
                    reward, shaped_reward,
                )

                shaped_reward_arr = jnp.array([shaped_reward[a] for a in env.agents])
                combined_reward_arr = jnp.array([reward[a] for a in env.agents])
                info["shaped_reward"] = shaped_reward_arr
                info["original_reward"] = original_reward
                info["combined_reward"] = combined_reward_arr
                info["anneal_factor"] = jnp.full_like(shaped_reward_arr, anneal_factor)

                agent_major_info_keys = {
                    "shaped_reward",
                    "original_reward",
                    "combined_reward",
                    "anneal_factor",
                }

                def _flatten_info_value(key, value):
                    # Env event metrics arrive env-major: (num_envs, num_agents).
                    # Training batches are actor-major, so swap to
                    # (num_agents, num_envs) before flattening when needed.
                    if (
                        key not in agent_major_info_keys
                        and len(value.shape) >= 2
                        and value.shape[0] == config["NUM_ENVS"]
                        and value.shape[1] == env.num_agents
                    ):
                        value = jnp.swapaxes(value, 0, 1)
                    return value.reshape((config["NUM_ACTORS"],) + value.shape[2:])

                info = {key: _flatten_info_value(key, value) for key, value in info.items()}
                transition = Transition(
                    batchify(done, env.agents, config["NUM_ACTORS"]).squeeze(),
                    action,
                    value,
                    batchify(reward, env.agents, config["NUM_ACTORS"]).squeeze(),
                    log_prob,
                    obs_batch,
                    info,
                )
                runner_state = (train_state, env_state, obsv, update_step, rng)
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            # Bootstrap from the critic value of the last observation after the
            # rollout; GAE uses this as V(s_{t+1}) for the final transition.
            train_state, env_state, last_obs, update_step, rng = runner_state
            last_obs_batch = jnp.stack(
                [last_obs[a] for a in env.agents]
            ).reshape(-1, *env.observation_space("agent_0").shape)
            _, last_val = network.apply(train_state.params, last_obs_batch)

            def _calculate_gae(traj_batch, last_val):
                # Reverse scan computes Generalized Advantage Estimation over
                # the collected rollout. targets = advantages + old values.
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = transition.done, transition.value, transition.reward
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = delta + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    return (gae, value), gae
                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch, reverse=True, unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)

            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, traj_batch, gae, targets):
                        pi, value = network.apply(params, traj_batch.obs)
                        log_prob = pi.log_prob(traj_batch.action)

                        # PPO clipped value loss prevents the critic from
                        # changing too far from the rollout-time estimate.
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()

                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        # PPO clipped policy objective: improve actions with
                        # positive advantage while limiting destructive updates.
                        loss_actor = -jnp.minimum(
                            ratio * gae,
                            jnp.clip(ratio, 1.0 - config["CLIP_EPS"], 1.0 + config["CLIP_EPS"]) * gae,
                        ).mean()
                        entropy = pi.entropy().mean()

                        total_loss = (
                            loss_actor
                            + config["VF_COEF"] * value_loss
                            - config["ENT_COEF"] * entropy
                        )
                        return total_loss, (value_loss, loss_actor, entropy)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(train_state.params, traj_batch, advantages, targets)
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                train_state, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)
                batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
                assert batch_size == config["NUM_STEPS"] * config["NUM_ACTORS"]
                # Shuffle the entire rollout before slicing minibatches. This
                # mixes agents, envs, and timesteps for a standard PPO update.
                permutation = jax.random.permutation(_rng, batch_size)
                batch = jax.tree.map(lambda x: x.reshape((batch_size,) + x.shape[2:]),
                                     (traj_batch, advantages, targets))
                shuffled_batch = jax.tree.map(lambda x: jnp.take(x, permutation, axis=0), batch)
                minibatches = jax.tree.map(
                    lambda x: jnp.reshape(x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])),
                    shuffled_batch,
                )
                train_state, total_loss = jax.lax.scan(_update_minbatch, train_state, minibatches)
                return (train_state, traj_batch, advantages, targets, rng), total_loss

            update_state = (train_state, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            train_state = update_state[0]
            metric = traj_batch.info
            rng = update_state[-1]

            # Aggregate metrics per PPO update. Event metrics are sums over the
            # rollout so W&B can show concrete counts like dish pickups.
            update_step = update_step + 1
            metric = jax.tree.map(lambda x: x.mean(), metric)
            for event_name in EVENT_METRIC_NAMES:
                event_key = f"event/{event_name}"
                if event_key in traj_batch.info:
                    event_values = traj_batch.info[event_key]
                    metric[event_key] = event_values.sum()
                    metric[f"event_rate/{event_name}"] = event_values.mean()
            if "event/delivery" in traj_batch.info:
                delivery_values = traj_batch.info["event/delivery"]
                metric["delivery"] = delivery_values.sum()
                metric["delivery_count.agent_0"] = delivery_values[:, :config["NUM_ENVS"]].sum()
                metric["delivery_count.agent_1"] = delivery_values[:, config["NUM_ENVS"]:].sum()
            metric["base_reward"] = traj_batch.info["original_reward"].sum()
            metric["base_reward_per_step"] = traj_batch.info["original_reward"].mean()
            metric["combined_reward"] = traj_batch.reward.sum()
            metric["combined_reward_per_step"] = traj_batch.reward.mean()
            metric["mean_reward"] = traj_batch.reward.mean()
            metric["max_reward"] = traj_batch.reward.max()
            metric["reward_sum"] = traj_batch.reward.sum()
            metric["loss/total"] = loss_info[0].mean()
            metric["loss/value"] = loss_info[1][0].mean()
            metric["loss/policy"] = loss_info[1][1].mean()
            metric["loss/entropy"] = loss_info[1][2].mean()
            metric["update_step"] = update_step
            metric["env_step"] = update_step * config["NUM_STEPS"] * config["NUM_ENVS"]
            # This callback is also what drives the terminal training monitor.
            jax.debug.callback(_log_training_metrics, metric)

            runner_state = (train_state, env_state, last_obs, update_step, rng)
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obsv, 0, _rng)
        # Outer scan: repeat rollout collection + PPO update NUM_UPDATES times.
        runner_state, metric = jax.lax.scan(_update_step, runner_state, None, config["NUM_UPDATES"])
        return {"runner_state": runner_state, "metrics": metric}

    return train


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

@hydra.main(version_base=None, config_path="config", config_name="ippo_cnn_overcooked_v3")
def main(config):
    """Run IPPO CNN training, checkpointing, and rollout GIF logging."""

    global _ACTIVE_MONITOR

    config = OmegaConf.to_container(config)

    # W&B receives both config and per-update metrics. The actual x-axis is
    # env_step rather than W&B's implicit row number, which makes runs with
    # different NUM_ENVS/NUM_STEPS comparable.
    config["RUN_NAME"] = config.get("WANDB_NAME") or (
        f'ippo_cnn_v3_{config["ENV_KWARGS"]["layout"]}'
    )
    wandb.init(
        entity=config.get("ENTITY", ""),
        project=config.get("PROJECT", "ippo_v3_cnn"),
        config=config,
        mode=config.get("WANDB_MODE", "online"),
        name=config["RUN_NAME"],
    )
    if wandb.run is not None:
        wandb.define_metric("env_step")
        wandb.define_metric("*", step_metric="env_step")

    rng = jax.random.PRNGKey(config["SEED"])
    rngs = jax.random.split(rng, config["NUM_SEEDS"])
    train_fn = make_train(config)
    config["NUM_CHECKPOINTS"] = 1
    checkpoint_logging = OvercookedV3Training(
        config,
        lambda rollout_env: IPPOCNNRolloutPolicy(
            rollout_env,
            config,
            _make_network,
        ),
    )
    if wandb.run is not None:
        # make_train mutates config with derived dimensions after env creation,
        # so push those values back into W&B for easier audit/debugging.
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
        "layout": config["ENV_KWARGS"]["layout"],
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
                "Compiling + training IPPO CNN v3 on "
                f"{config['ENV_KWARGS']['layout']} for {config['TOTAL_TIMESTEPS']:,} "
                f"env steps ({config['NUM_UPDATES']:,} PPO updates)."
            )
            # JIT compile the whole training function. vmap supports NUM_SEEDS
            # independent seeds; our current runs use one seed, so x[0] below is
            # the trained state for that single seed.
            train_jit = jax.jit(train_fn)
            out = jax.block_until_ready(jax.vmap(train_jit)(rngs))

            monitor.log("Training finished; saving checkpoint and GIF.")
            train_state = jax.tree.map(lambda x: x[0], out["runner_state"][0])

            # The checkpoint stores only model params. To resume training exactly
            # we would also need optimizer/env state; this file is sufficient for
            # inference and warm-starting a new run.
            save_path = config.get("SAVE_CHECKPOINT_PATH") or os.path.join(
                "/workspace/JaxMARL/checkpoints",
                f'ippo_cnn_v3_{config["ENV_KWARGS"]["layout"]}',
            )
            model_path = _save_model_params(train_state.params, save_path)
            checkpoint_logging.checkpoint_saved(
                train_state.params,
                checkpoint_index=1,
                update_step=config["NUM_UPDATES"],
                env_step=(
                    config["NUM_UPDATES"]
                    * config["NUM_STEPS"]
                    * config["NUM_ENVS"]
                ),
                training_seed=int(rngs[0][0]),
                run_name=config["RUN_NAME"],
            )
            if wandb.run is not None:
                completed_env_steps = config["NUM_UPDATES"] * config["NUM_STEPS"] * config["NUM_ENVS"]
                wandb.log({"saved_model_path": model_path}, step=int(completed_env_steps))

    finally:
        _ACTIVE_MONITOR = None
        wandb.finish()


if __name__ == "__main__":
    main()
