"""MAPPO RNN trainer for Overcooked V3.

Adapted from the generic MAPPO RNN trainer with a global-state wrapper that
flattens each agent's grid observation and concatenates all agents' views for a
centralised critic.
"""

import os
from functools import partial
from typing import NamedTuple

import hydra
import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb
from flax.training.train_state import TrainState
from omegaconf import DictConfig, OmegaConf

import jaxmarl
from baselines.overcooked_v3.models.mappo_rnn import (
    ActorRNN,
    CriticRNN,
    MAPPORNNPolicy,
    ScannedRNN,
    batchify,
    unbatchify,
)
from baselines.overcooked_v3.training import OvercookedV3Training
from jaxmarl.wrappers.baselines import JaxMARLWrapper, LogWrapper, save_params

class OvercookedWorldStateWrapper(JaxMARLWrapper):
    """Add a flattened joint observation for MAPPO's centralized critic."""

    @partial(jax.jit, static_argnums=0)
    def reset(self, key):
        """Reset the environment and attach the initial joint observation."""

        obs, env_state = self._env.reset(key)
        obs["world_state"] = self.world_state(obs)
        return obs, env_state

    @partial(jax.jit, static_argnums=0)
    def step(self, key, state, action):
        """Step the environment and attach the next joint observation."""

        obs, env_state, reward, done, info = self._env.step(key, state, action)
        obs["world_state"] = self.world_state(obs)
        return obs, env_state, reward, done, info

    @partial(jax.jit, static_argnums=0)
    def world_state(self, obs):
        """Repeat the flattened joint observation once for every agent."""

        all_obs = jnp.stack([jnp.ravel(obs[agent]) for agent in self._env.agents], axis=0)
        world = all_obs.reshape(-1)
        return jnp.repeat(world[jnp.newaxis, :], self._env.num_agents, axis=0)

    def world_state_size(self):
        """Return the flattened joint-observation width."""

        obs_space = self._env.observation_space(self._env.agents[0])
        return int(np.prod(obs_space.shape)) * self._env.num_agents


class Transition(NamedTuple):
    """Store one vectorized MAPPO environment transition."""

    global_done: jnp.ndarray
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    world_state: jnp.ndarray
    info: jnp.ndarray


def flatten_info_leaf(x, num_envs, num_agents, num_actors):
    """Normalize an environment-info leaf to one value per actor."""

    x = jnp.asarray(x)
    if x.size == num_actors:
        return x.reshape((num_actors,))
    if x.size == num_envs:
        return jnp.repeat(x.reshape((num_envs,)), num_agents, axis=0)
    if x.size == 1:
        return jnp.broadcast_to(x.reshape(()), (num_actors,))
    return x.reshape((num_actors,))


def make_train(config):
    """Build the compiled MAPPO training function for one random seed."""

    env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]
    config["NUM_TEST_ACTORS"] = env.num_agents * config["NUM_TEST_ENVS"]
    config["NUM_UPDATES"] = (
        int(config["TOTAL_TIMESTEPS"]) // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ACTORS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )
    config["CLIP_EPS"] = (
        config["CLIP_EPS"] / env.num_agents
        if config["SCALE_CLIP_EPS"]
        else config["CLIP_EPS"]
    )

    env = OvercookedWorldStateWrapper(env)
    env = LogWrapper(env)
    test_env = LogWrapper(env)
    test_interval = max(int(config["NUM_UPDATES"] * config["TEST_INTERVAL"]), 1)

    def linear_schedule(count):
        """Linearly anneal the optimizer learning rate across PPO updates."""

        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    def train(rng):
        """Initialize model and environment state, then run all updates."""

        original_seed = rng[0]
        actor_network = ActorRNN(env.action_space(env.agents[0]).n, config=config)
        critic_network = CriticRNN(config=config)
        rng, _rng_actor, _rng_critic = jax.random.split(rng, 3)

        obs_dim = int(np.prod(env.observation_space(env.agents[0]).shape))
        ac_init_x = (
            jnp.zeros((1, config["NUM_ENVS"], obs_dim)),
            jnp.zeros((1, config["NUM_ENVS"])),
        )
        ac_init_hstate = ScannedRNN.initialize_carry(
            config["NUM_ENVS"], config["GRU_HIDDEN_DIM"]
        )
        actor_network_params = actor_network.init(_rng_actor, ac_init_hstate, ac_init_x)

        cr_init_x = (
            jnp.zeros((1, config["NUM_ENVS"], env.world_state_size())),
            jnp.zeros((1, config["NUM_ENVS"])),
        )
        cr_init_hstate = ScannedRNN.initialize_carry(
            config["NUM_ENVS"], config["GRU_HIDDEN_DIM"]
        )
        critic_network_params = critic_network.init(_rng_critic, cr_init_hstate, cr_init_x)

        if config["ANNEAL_LR"]:
            actor_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
            critic_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
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
            apply_fn=actor_network.apply, params=actor_network_params, tx=actor_tx
        )
        critic_train_state = TrainState.create(
            apply_fn=critic_network.apply, params=critic_network_params, tx=critic_tx
        )

        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)
        ac_init_hstate = ScannedRNN.initialize_carry(
            config["NUM_ACTORS"], config["GRU_HIDDEN_DIM"]
        )
        cr_init_hstate = ScannedRNN.initialize_carry(
            config["NUM_ACTORS"], config["GRU_HIDDEN_DIM"]
        )

        def _update_step(update_runner_state, unused):
            """Collect one batch, update actor and critic, and report metrics."""

            runner_state, update_steps = update_runner_state

            def _env_step(runner_state, unused):
                """Sample actions and advance all vectorized environments once."""

                train_states, env_state, last_obs, last_done, hstates, rng, test_metrics = runner_state
                rng, _rng = jax.random.split(rng)
                obs_batch = batchify(last_obs, env.agents, config["NUM_ACTORS"])
                ac_in = (obs_batch[np.newaxis, :], last_done[np.newaxis, :])
                ac_hstate, pi = actor_network.apply(train_states[0].params, hstates[0], ac_in)
                action = pi.sample(seed=_rng).squeeze(0)
                log_prob = pi.log_prob(action)
                env_act = unbatchify(action, env.agents, config["NUM_ENVS"], env.num_agents)

                world_state = last_obs["world_state"].swapaxes(0, 1)
                world_state = world_state.reshape((config["NUM_ACTORS"], -1))
                cr_in = (world_state[None, :], last_done[np.newaxis, :])
                cr_hstate, value = critic_network.apply(train_states[1].params, hstates[1], cr_in)

                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0))(
                    rng_step, env_state, env_act
                )
                info = jax.tree.map(
                    lambda x: flatten_info_leaf(
                        x,
                        config["NUM_ENVS"],
                        env.num_agents,
                        config["NUM_ACTORS"],
                    ),
                    info,
                )
                done_batch = batchify(done, env.agents, config["NUM_ACTORS"]).squeeze()
                transition = Transition(
                    jnp.tile(done["__all__"], env.num_agents),
                    last_done,
                    action.squeeze(),
                    value.squeeze(),
                    batchify(reward, env.agents, config["NUM_ACTORS"]).squeeze(),
                    log_prob.squeeze(),
                    obs_batch,
                    world_state,
                    info,
                )
                runner_state = (
                    train_states,
                    env_state,
                    obsv,
                    done_batch,
                    (ac_hstate, cr_hstate),
                    rng,
                    test_metrics,
                )
                return runner_state, transition

            initial_hstates = runner_state[-3]
            runner_state, traj_batch = jax.lax.scan(_env_step, runner_state, None, config["NUM_STEPS"])

            train_states, env_state, last_obs, last_done, hstates, rng, test_metrics = runner_state
            last_world_state = last_obs["world_state"].swapaxes(0, 1)
            last_world_state = last_world_state.reshape((config["NUM_ACTORS"], -1))
            cr_in = (last_world_state[None, :], last_done[np.newaxis, :])
            _, last_val = critic_network.apply(train_states[1].params, hstates[1], cr_in)
            last_val = last_val.squeeze()

            def _calculate_gae(traj_batch, last_val):
                """Compute generalized advantages and value targets."""

                def _get_advantages(gae_and_next_value, transition):
                    """Accumulate one reverse-time generalized-advantage step."""

                    gae, next_value = gae_and_next_value
                    done, value, reward = transition.global_done, transition.value, transition.reward
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = delta + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
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

            def _update_epoch(update_state, unused):
                """Shuffle the rollout and apply one PPO epoch."""

                def _update_minbatch(train_states, batch_info):
                    """Apply actor and critic gradients for one minibatch."""

                    actor_train_state, critic_train_state = train_states
                    ac_init_hstate, cr_init_hstate, traj_batch, advantages, targets = batch_info

                    def _actor_loss_fn(actor_params, init_hstate, traj_batch, gae):
                        """Compute the clipped PPO actor objective."""

                        _, pi = actor_network.apply(actor_params, init_hstate.squeeze(), (traj_batch.obs, traj_batch.done))
                        log_prob = pi.log_prob(traj_batch.action)
                        logratio = log_prob - traj_batch.log_prob
                        ratio = jnp.exp(logratio)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = jnp.clip(ratio, 1.0 - config["CLIP_EPS"], 1.0 + config["CLIP_EPS"]) * gae
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2).mean()
                        entropy = pi.entropy().mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clip_frac = jnp.mean(jnp.abs(ratio - 1) > config["CLIP_EPS"])
                        actor_loss = loss_actor - config["ENT_COEF"] * entropy
                        return actor_loss, (loss_actor, entropy, ratio, approx_kl, clip_frac)

                    def _critic_loss_fn(critic_params, init_hstate, traj_batch, targets):
                        """Compute the clipped value-function objective."""

                        _, value = critic_network.apply(critic_params, init_hstate.squeeze(), (traj_batch.world_state, traj_batch.done))
                        value_pred_clipped = traj_batch.value + (value - traj_batch.value).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        critic_loss = config["VF_COEF"] * value_loss
                        return critic_loss, value_loss

                    actor_grad_fn = jax.value_and_grad(_actor_loss_fn, has_aux=True)
                    actor_loss, actor_grads = actor_grad_fn(actor_train_state.params, ac_init_hstate, traj_batch, advantages)
                    critic_grad_fn = jax.value_and_grad(_critic_loss_fn, has_aux=True)
                    critic_loss, critic_grads = critic_grad_fn(critic_train_state.params, cr_init_hstate, traj_batch, targets)

                    actor_train_state = actor_train_state.apply_gradients(grads=actor_grads)
                    critic_train_state = critic_train_state.apply_gradients(grads=critic_grads)

                    total_loss = actor_loss[0] + critic_loss[0]
                    loss_info = {
                        "total_loss": total_loss,
                        "actor_loss": actor_loss[0],
                        "value_loss": critic_loss[0],
                        "entropy": actor_loss[1][1],
                        "ratio": actor_loss[1][2],
                        "approx_kl": actor_loss[1][3],
                        "clip_frac": actor_loss[1][4],
                    }
                    return (actor_train_state, critic_train_state), loss_info

                train_states, init_hstates, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)
                init_hstates = jax.tree.map(lambda x: jnp.reshape(x, (1, config["NUM_ACTORS"], -1)), init_hstates)
                batch = (init_hstates[0], init_hstates[1], traj_batch, advantages.squeeze(), targets.squeeze())
                permutation = jax.random.permutation(_rng, config["NUM_ACTORS"])
                shuffled_batch = jax.tree.map(lambda x: jnp.take(x, permutation, axis=1), batch)
                minibatches = jax.tree.map(
                    lambda x: jnp.swapaxes(
                        jnp.reshape(x, [x.shape[0], config["NUM_MINIBATCHES"], -1] + list(x.shape[2:])),
                        1,
                        0,
                    ),
                    shuffled_batch,
                )
                train_states, loss_info = jax.lax.scan(_update_minbatch, train_states, minibatches)
                update_state = (train_states, jax.tree.map(lambda x: x.squeeze(), init_hstates), traj_batch, advantages, targets, rng)
                return update_state, loss_info

            update_state = (train_states, initial_hstates, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(_update_epoch, update_state, None, config["UPDATE_EPOCHS"])
            loss_info["ratio_0"] = loss_info["ratio"].at[0, 0].get()
            loss_info = jax.tree.map(lambda x: x.mean(), loss_info)

            train_states = update_state[0]
            metric = traj_batch.info
            metric["loss"] = loss_info
            rng = update_state[-1]
            update_steps = update_steps + 1
            metric["update_steps"] = update_steps
            metrics = {
                "returns": metric["returned_episode_returns"][-1, :].mean(),
                "update_steps": metric["update_steps"],
                "env_step": metric["update_steps"] * config["NUM_ENVS"] * config["NUM_STEPS"],
                **metric["loss"],
            }

            if config.get("TEST_DURING_TRAINING", True):
                rng, _rng = jax.random.split(rng)
                test_metrics = jax.lax.cond(
                    update_steps % test_interval == 0,
                    lambda _: _get_greedy_metrics(_rng, train_states[0].params),
                    lambda _: test_metrics,
                    operand=None,
                )
                metrics.update({"test_" + k: v for k, v in test_metrics.items()})

            if config["WANDB_MODE"] != "disabled":
                def callback(metrics, original_seed):
                    """Send one update's metrics from JAX to W&B on the host."""

                    seed = int(np.asarray(original_seed).reshape(-1)[0])
                    if config.get("WANDB_LOG_ALL_SEEDS", False):
                        metrics.update({f"rng{seed}/{k}": v for k, v in metrics.items()})
                    wandb.log(metrics)

                jax.debug.callback(callback, metrics, original_seed)

            runner_state = (train_states, env_state, last_obs, last_done, hstates, rng, test_metrics)
            return (runner_state, update_steps), metric

        def _get_greedy_metrics(rng, actor_params):
            """Evaluate the highest-probability actor actions."""

            def _greedy_env_step(step_state, unused):
                """Advance all evaluation environments by one policy step."""

                actor_params, env_state, last_obs, last_done, ac_hstate, rng = step_state
                rng, _rng = jax.random.split(rng)
                obs_batch = batchify(last_obs, env.agents, config["NUM_TEST_ACTORS"])
                ac_in = (obs_batch[np.newaxis, :], last_done[np.newaxis, :])
                ac_hstate, pi = actor_network.apply(actor_params, ac_hstate, ac_in)
                action = pi.mode().squeeze(0)
                env_act = unbatchify(action, env.agents, config["NUM_TEST_ENVS"], env.num_agents)

                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_TEST_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(test_env.step, in_axes=(0, 0, 0))(
                    rng_step, env_state, env_act
                )
                info = jax.tree.map(
                    lambda x: flatten_info_leaf(
                        x,
                        config["NUM_TEST_ENVS"],
                        env.num_agents,
                        config["NUM_TEST_ACTORS"],
                    ),
                    info,
                )
                done_batch = batchify(done, env.agents, config["NUM_TEST_ACTORS"]).squeeze()
                reward_batch = batchify(reward, env.agents, config["NUM_TEST_ACTORS"]).squeeze()
                step_state = (actor_params, env_state, obsv, done_batch, ac_hstate, rng)
                return step_state, (reward_batch, done_batch, info)

            rng, _rng = jax.random.split(rng)
            reset_rng = jax.random.split(_rng, config["NUM_TEST_ENVS"])
            init_obsv, env_state = jax.vmap(test_env.reset, in_axes=(0,))(reset_rng)
            init_dones = jnp.zeros((config["NUM_TEST_ACTORS"]), dtype=bool)
            ac_hstate = ScannedRNN.initialize_carry(config["NUM_TEST_ACTORS"], config["GRU_HIDDEN_DIM"])
            step_state = (actor_params, env_state, init_obsv, init_dones, ac_hstate, rng)
            _, (_, _, infos) = jax.lax.scan(_greedy_env_step, step_state, None, config["TEST_NUM_STEPS"])
            return jax.tree.map(
                lambda x: jnp.nanmean(jnp.where(infos["returned_episode"], x, jnp.nan)),
                infos,
            )

        rng, _rng = jax.random.split(rng)
        test_metrics = _get_greedy_metrics(_rng, actor_train_state.params)
        rng, _rng = jax.random.split(rng)
        runner_state = (
            (actor_train_state, critic_train_state),
            env_state,
            obsv,
            jnp.zeros((config["NUM_ACTORS"]), dtype=bool),
            (ac_init_hstate, cr_init_hstate),
            _rng,
            test_metrics,
        )
        runner_state, metric = jax.lax.scan(_update_step, (runner_state, 0), None, config["NUM_UPDATES"])
        return {"runner_state": runner_state, "metrics": metric}

    return train


@hydra.main(version_base=None, config_path="../config", config_name="mappo_rnn")
def main(config: DictConfig):
    """Train MAPPO, save each seed's final checkpoint, and log one rollout."""

    config = OmegaConf.to_container(config)
    alg_name = "mappo"
    env_name = config["ENV_NAME"]
    run_name = config.get("WANDB_NAME") or f"{alg_name}_{config['ENV_KWARGS'].get('layout', env_name)}"
    config["RUN_NAME"] = run_name
    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=["MAPPO", "RNN", config["ENV_NAME"], config["ENV_KWARGS"].get("layout", "unknown")],
        name=run_name,
        config=config,
        mode=config["WANDB_MODE"],
    )

    rng = jax.random.PRNGKey(config["SEED"])
    rngs = jax.random.split(rng, config["NUM_SEEDS"])
    train = make_train(config)
    checkpoint_logging = OvercookedV3Training(
        config,
        lambda rollout_env: MAPPORNNPolicy.create(rollout_env, config),
    )
    train_vjit = jax.jit(jax.vmap(train))
    outs = jax.block_until_ready(train_vjit(rngs))

    save_root = config.get("SAVE_PATH")
    if save_root:
        save_dir = os.path.join(save_root, alg_name, env_name)
        os.makedirs(save_dir, exist_ok=True)
        model_state = outs["runner_state"][0][0][0]
        OmegaConf.save(
            OmegaConf.create(config),
            os.path.join(save_dir, f"{run_name}_config.yaml"),
        )
        for i, seed_rng in enumerate(rngs):
            params = jax.tree.map(lambda x: x[i], model_state.params)
            save_path = os.path.join(save_dir, f"{run_name}_vmap{i}_rng{int(seed_rng[0])}.safetensors")
            save_params(params, save_path)
            if i == checkpoint_logging.gif_hook.training_seed_index:
                checkpoint_logging.checkpoint_saved(
                    params,
                    checkpoint_index=1,
                    update_step=config["NUM_UPDATES"],
                    env_step=(
                        config["NUM_UPDATES"]
                        * config["NUM_STEPS"]
                        * config["NUM_ENVS"]
                    ),
                    training_seed=int(seed_rng[0]),
                    run_name=run_name,
                )

    wandb.finish()


if __name__ == "__main__":
    main()
