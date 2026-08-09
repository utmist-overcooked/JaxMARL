"""MAPPO (RNN) with a CommNet communication block, Overcooked V3, primitive actions.

The communication counterpart of `mappo_rnn_overcooked_v3.py`. Everything about
the training loop, reward shaping, metrics and the centralised critic is shared
with that script; the difference is in the actor:

  mappo_rnn_overcooked_v3          each agent's policy reads only its own
                                   CNN+GRU embedding.
  mappo_rnn_overcooked_v3_commnet  after the CNN+GRU, agents exchange hidden
                                   states through CommNet passes before the
                                   policy head, so an agent's action depends on
                                   what its partner is perceiving.

The CommNet update follows the same formulation as the repo's IC3Net baseline
(`baselines/IC3Net/models.py`):

    h <- tanh(x + f(h) + c(mean of the *other* agents' h))

repeated for COMM_PASSES hops, with x the agent's own encoding used as a skip
connection.

Pairing this with `ENV_KWARGS.agent_view_size` is the intended experiment: with
full observability communication is largely redundant, whereas under a narrow
view each agent must rely on its partner's messages to know the kitchen state.

IMPORTANT - minibatching differs from the non-communicating MAPPO. Because a
CommNet pass mixes the agents that share an environment, every agent of an env
must land in the same minibatch. This script therefore shuffles and splits along
the *environment* axis, keeping agent groups intact, instead of shuffling the
flat actor axis. That requires NUM_ENVS % NUM_MINIBATCHES == 0.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from typing import Dict, Sequence

import distrax
import flax.linen as nn
import hydra
import jax
import jax.numpy as jnp
import numpy as np
import optax
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
from baselines.IPPO.ippo_rnn_overcooked_v3 import unbatchify
from baselines.MAPPO import mappo_rnn_overcooked_v3 as base
from baselines.MAPPO.mappo_rnn_overcooked_v3 import (
    CriticRNN,
    OvercookedV3WorldStateWrapper,
    RecurrentEncoder,
    Transition,
    _save_params,
)
from jaxmarl.viz.overcooked_v3_visualizer import OvercookedV3Visualizer
from jaxmarl.wrappers.baselines import LogWrapper


class CommNetBlock(nn.Module):
    """CommNet message passing over the agents that share an environment.

    Input and output are (T, num_agents, num_envs, hidden). Each hop replaces an
    agent's hidden state with

        tanh(x + f(h) + c(mean of the other agents' h))

    where x is the agent's own pre-communication encoding, kept as a skip
    connection. The mean over "other agents" is computed as (sum - self) /
    (num_agents - 1), which is exactly the self-masked average the IC3Net
    baseline builds with an (N, N) mask, without materialising that matrix.
    """

    num_agents: int
    comm_passes: int = 2
    comm_mode: str = "avg"

    @nn.compact
    def __call__(self, x):
        hidden_dim = x.shape[-1]
        h = x
        for hop in range(self.comm_passes):
            # Messages from every other agent sharing this environment.
            others = jnp.sum(h, axis=1, keepdims=True) - h
            if self.comm_mode == "avg":
                others = others / max(self.num_agents - 1, 1)
            c = nn.Dense(
                hidden_dim,
                kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0),
                name=f"c_{hop}",
            )(others)
            f_h = nn.Dense(
                hidden_dim,
                kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0),
                name=f"f_{hop}",
            )(h)
            h = nn.tanh(x + f_h + c)
        return h


class CommNetActorRNN(nn.Module):
    """Policy network whose agents exchange GRU hidden states before acting."""

    action_dim: Sequence[int]
    config: Dict
    num_agents: int

    @nn.compact
    def __call__(self, hidden, x):
        final_hidden, embedding = RecurrentEncoder(config=self.config)(hidden, x)

        # (T, num_agents * num_envs, H) -> (T, num_agents, num_envs, H). The
        # actor axis is agent-major everywhere in this script, so this recovers
        # which agents share an environment.
        T, num_actors, hidden_dim = embedding.shape
        num_envs = num_actors // self.num_agents
        grouped = embedding.reshape(T, self.num_agents, num_envs, hidden_dim)
        grouped = CommNetBlock(
            num_agents=self.num_agents,
            comm_passes=self.config.get("COMM_PASSES", 2),
            comm_mode=self.config.get("COMM_MODE", "avg"),
        )(grouped)
        embedding = grouped.reshape(T, num_actors, hidden_dim)

        actor_mean = nn.Dense(
            self.config.get("FC_DIM_SIZE", 128),
            kernel_init=orthogonal(2), bias_init=constant(0.0),
        )(embedding)
        actor_mean = nn.relu(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0),
        )(actor_mean)
        return final_hidden, distrax.Categorical(logits=actor_mean)


def env_major_minibatches(x, permutation, num_agents, num_envs, num_minibatches):
    """Split a batch along environments, keeping each env's agents together.

    `x` has shape (L, num_agents * num_envs, ...) with the actor axis
    agent-major (actor index = agent_idx * num_envs + env_idx). Environments are
    shuffled by `permutation` and then cut into `num_minibatches` groups, so the
    result has shape
        (num_minibatches, L, num_agents * envs_per_minibatch, ...)
    and every minibatch still factors cleanly into (num_agents, num_envs) for a
    CommNet pass. Shuffling the flat actor axis instead - as the
    non-communicating MAPPO does - would scatter an environment's agents across
    different minibatches and break communication during the update.
    """
    envs_per_minibatch = num_envs // num_minibatches
    rest = x.shape[2:]
    x = x.reshape(x.shape[0], num_agents, num_envs, *rest)
    x = jnp.take(x, permutation, axis=2)
    x = x.reshape(x.shape[0], num_agents, num_minibatches, envs_per_minibatch, *rest)
    x = jnp.moveaxis(x, 2, 0)
    return x.reshape(
        num_minibatches, x.shape[1], num_agents * envs_per_minibatch, *rest
    )


def get_rollout(actor_params, config):
    """Run one sampled episode with the trained actor, for GIF generation."""
    env = jaxmarl.make(config["ENV_NAME"], **config.get("ENV_KWARGS", {}))
    hidden_dim = config.get("GRU_HIDDEN_DIM", 128)
    actor = CommNetActorRNN(
        env.action_space(env.agents[0]).n, config=config, num_agents=env.num_agents,
    )

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


def make_train(config):
    """Create the fully JIT'd CommNet-MAPPO training function for overcooked_v3."""
    env = jaxmarl.make(config["ENV_NAME"], **config.get("ENV_KWARGS", {}))

    config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ACTORS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )

    # Communication groups agents by environment, so minibatches are cut along
    # the environment axis and must divide it evenly.
    if config["NUM_ENVS"] % config["NUM_MINIBATCHES"] != 0:
        raise ValueError(
            "CommNet MAPPO splits minibatches over environments so that agents "
            "which communicate stay together: NUM_ENVS "
            f"({config['NUM_ENVS']}) must be divisible by NUM_MINIBATCHES "
            f"({config['NUM_MINIBATCHES']})."
        )

    obs_shape = env.observation_space(env.agents[0]).shape
    action_dim = env.action_space(env.agents[0]).n
    num_agents = env.num_agents
    envs_per_minibatch = config["NUM_ENVS"] // config["NUM_MINIBATCHES"]

    env = OvercookedV3WorldStateWrapper(env)
    world_state_shape = env.world_state_shape()
    env = LogWrapper(env, replace_info=False)

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
        actor_network = CommNetActorRNN(
            action_dim, config=config, num_agents=num_agents,
        )
        critic_network = CriticRNN(config=config)

        rng, _rng_actor, _rng_critic = jax.random.split(rng, 3)
        # The actor is initialised with a full NUM_ACTORS batch: CommNet needs
        # the actor axis to factor into (num_agents, num_envs).
        ac_init_x = (
            jnp.zeros((1, config["NUM_ACTORS"], *obs_shape)),
            jnp.zeros((1, config["NUM_ACTORS"])),
        )
        cr_init_x = (
            jnp.zeros((1, config["NUM_ACTORS"], *world_state_shape)),
            jnp.zeros((1, config["NUM_ACTORS"])),
        )
        actor_params = actor_network.init(
            _rng_actor, jnp.zeros((config["NUM_ACTORS"], hidden_dim)), ac_init_x,
        )
        critic_params = critic_network.init(
            _rng_critic, jnp.zeros((config["NUM_ACTORS"], hidden_dim)), cr_init_x,
        )

        lr = linear_schedule if config.get("ANNEAL_LR", True) else config["LR"]
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
                        actor_loss[0] + critic_loss[0],
                        critic_loss[1][0],
                        actor_loss[1][0],
                        actor_loss[1][1],
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
                # Permute environments, not actors, so communicating agents stay
                # in the same minibatch.
                permutation = jax.random.permutation(_rng, config["NUM_ENVS"])
                minibatches = jax.tree_util.tree_map(
                    lambda x: env_major_minibatches(
                        x, permutation, num_agents,
                        config["NUM_ENVS"], config["NUM_MINIBATCHES"],
                    ),
                    batch,
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
            jax.debug.callback(base._log_training_metrics, metric)

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
    version_base=None,
    config_path="config",
    config_name="mappo_rnn_overcooked_v3_commnet",
)
def main(config):
    """Main training entry point."""
    config = OmegaConf.to_container(config, resolve=True)
    layout_name = config.get("ENV_KWARGS", {}).get("layout", "unknown")

    wandb.init(
        entity=config.get("ENTITY", ""),
        project=config.get("WANDB_PROJECT", "jaxmarl-mappo"),
        tags=["MAPPO", "RNN", "CommNet", "OvercookedV3"],
        config=config,
        mode=config.get("WANDB_MODE", "disabled"),
        name=config.get("WANDB_NAME")
        or f"mappo_commnet_overcooked_v3_{layout_name}",
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
            base._ACTIVE_MONITOR = monitor
            monitor.log(
                "Compiling + training CommNet-MAPPO RNN v3 on "
                f"{layout_name} for {config['TOTAL_TIMESTEPS']:,} env steps "
                f"({config['NUM_UPDATES']:,} PPO updates, "
                f"{config.get('COMM_PASSES', 2)} comm passes)."
            )
            train_jit = jax.jit(train_fn)
            out = jax.block_until_ready(train_jit(rng))

            monitor.log("Training finished; saving checkpoints and GIF.")
            runner_state = out["runner_state"]
            actor_params = runner_state[0].params
            critic_params = runner_state[1].params
            save_path = config.get(
                "SAVE_PATH", "checkpoints/mappo_commnet_overcooked_v3"
            )
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
        base._ACTIVE_MONITOR = None
        wandb.finish()


if __name__ == "__main__":
    main()
