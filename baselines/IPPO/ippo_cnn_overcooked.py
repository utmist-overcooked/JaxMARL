"""
Based on PureJaxRL Implementation of PPO
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any
from flax.training.train_state import TrainState
import distrax
from gymnax.wrappers.purerl import LogWrapper, FlattenObservationWrapper
import jaxmarl
from jaxmarl.wrappers.baselines import LogWrapper
from jaxmarl.environments.overcooked import overcooked_layouts
from jaxmarl.viz.overcooked_visualizer import OvercookedVisualizer
import hydra
from omegaconf import OmegaConf
import wandb
import copy
import os
import sys

import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
try:
    from utils.monitor import TrainingMonitor

    _MONITOR_AVAILABLE = True
except ImportError:
    _MONITOR_AVAILABLE = False


class CNN(nn.Module):
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
        x = nn.Conv(
            features=32,
            kernel_size=(5, 5),
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = activation(x)
        x = nn.Conv(
            features=32,
            kernel_size=(3, 3),
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = activation(x)
        x = nn.Conv(
            features=32,
            kernel_size=(3, 3),
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = activation(x)
        x = x.reshape((x.shape[0], -1))  # Flatten

        x = nn.Dense(
            features=64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        x = activation(x)

        return x


class ActorCritic(nn.Module):
    action_dim: Sequence[int]
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        embedding = CNN(self.activation)(x)

        actor_mean = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(embedding)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(embedding)
        pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(embedding)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return pi, jnp.squeeze(critic, axis=-1)


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray


def get_rollout(params, config):
    env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])

    network = ActorCritic(
        env.action_space(env.agents[0]).n, activation=config["ACTIVATION"]
    )
    key = jax.random.PRNGKey(0)
    key, key_r, key_a = jax.random.split(key, 3)

    done = False

    obs, state = env.reset(key_r)
    state_seq = [state]
    while not done:
        key, key_a0, key_a1, key_s = jax.random.split(key, 4)

        obs_batch = jnp.stack([obs[a] for a in env.agents]).reshape(
            -1, *env.observation_space("agent_0").shape
        )

        pi, value = network.apply(params, obs_batch)
        action = pi.sample(seed=key_a0)
        env_act = unbatchify(action, env.agents, 1, env.num_agents)

        env_act = {k: v.squeeze() for k, v in env_act.items()}

        # STEP ENV
        obs, state, reward, done, info = env.step(key_s, state, env_act)
        done = done["__all__"]

        state_seq.append(state)

    return state_seq


def batchify(x: dict, agent_list, num_actors):
    x = jnp.stack([x[a] for a in agent_list])
    return x.reshape((num_actors, -1))


def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}


def make_train(config, monitor=None):
    env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])

    config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ACTORS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )

    env = LogWrapper(env, replace_info=False)

    rew_shaping_anneal = optax.linear_schedule(
        init_value=1.0, end_value=0.0, transition_steps=config["REW_SHAPING_HORIZON"]
    )

    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    def train(rng):
        original_seed = rng[0]

        # INIT NETWORK
        network = ActorCritic(
            env.action_space(env.agents[0]).n, activation=config["ACTIVATION"]
        )
        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros((1, *env.observation_space("agent_0").shape))

        network_params = network.init(_rng, init_x)
        if config["ANNEAL_LR"]:
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
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)

        # TRAIN LOOP
        def _update_step(runner_state, unused):
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, update_step, rng = runner_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)

                obs_batch = jnp.stack([last_obs[a] for a in env.agents]).reshape(
                    -1, *env.observation_space("agent_0").shape
                )

                print("input_obs_shape", obs_batch.shape)

                pi, value = network.apply(train_state.params, obs_batch)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)
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

                shaped_reward = info.pop("shaped_reward")
                current_timestep = (
                    update_step * config["NUM_STEPS"] * config["NUM_ENVS"]
                )
                reward = jax.tree.map(
                    lambda x, y: x + y * rew_shaping_anneal(current_timestep),
                    reward,
                    shaped_reward,
                )

                info = jax.tree.map(lambda x: x.reshape((config["NUM_ACTORS"])), info)
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

            # CALCULATE ADVANTAGE
            train_state, env_state, last_obs, update_step, rng = runner_state
            last_obs_batch = jnp.stack([last_obs[a] for a in env.agents]).reshape(
                -1, *env.observation_space("agent_0").shape
            )
            _, last_val = network.apply(train_state.params, last_obs_batch)

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
                def _update_minbatch(train_state, batch_info):
                    traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, traj_batch, gae, targets):
                        # RERUN NETWORK
                        pi, value = network.apply(params, traj_batch.obs)
                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )

                        # CALCULATE ACTOR LOSS
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
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

                        total_loss = (
                            loss_actor
                            + config["VF_COEF"] * value_loss
                            - config["ENT_COEF"] * entropy
                        )
                        return total_loss, (value_loss, loss_actor, entropy)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params, traj_batch, advantages, targets
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                train_state, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)
                batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
                assert batch_size == config["NUM_STEPS"] * config["NUM_ACTORS"], (
                    "batch size must be equal to number of steps * number of actors"
                )
                permutation = jax.random.permutation(_rng, batch_size)
                batch = (traj_batch, advantages, targets)
                batch = jax.tree.map(
                    lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
                )
                shuffled_batch = jax.tree.map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
                minibatches = jax.tree.map(
                    lambda x: jnp.reshape(
                        x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
                    ),
                    shuffled_batch,
                )
                train_state, total_loss = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )
                update_state = (train_state, traj_batch, advantages, targets, rng)
                return update_state, total_loss

            update_state = (train_state, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            train_state = update_state[0]
            metric = traj_batch.info
            rng = update_state[-1]

            def callback(metric, original_seed):
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
                        },
                        seed=int(original_seed),
                    )

                if config["WANDB_MODE"] != "disabled":
                    wandb.log(metric)

            update_step = update_step + 1
            metric = jax.tree.map(lambda x: x.mean(), metric)
            metric["update_step"] = update_step
            metric["env_step"] = update_step * config["NUM_STEPS"] * config["NUM_ENVS"]
            jax.debug.callback(callback, metric, original_seed)

            runner_state = (train_state, env_state, last_obs, update_step, rng)
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obsv, 0, _rng)
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metrics": metric}

    return train


def single_run(config):
    config = OmegaConf.to_container(config)
    layout_name = copy.deepcopy(config["ENV_KWARGS"]["layout"])
    config["ENV_KWARGS"]["layout"] = overcooked_layouts[layout_name]

    os.environ["WANDB_MODE"] = config["WANDB_MODE"]
    wandb_dir = os.path.join(os.environ.get("SCRATCH", "."), "jaxmarl", "wandb")
    os.makedirs(wandb_dir, exist_ok=True)

    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=["IPPO", "FF"],
        config=config,
        mode=config["WANDB_MODE"],
        name=f"ippo_cnn_overcooked_tuned_{layout_name}",
        dir=wandb_dir,
    )

    num_seeds = config["NUM_SEEDS"]
    num_updates = int(
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    use_monitor = config.get("USE_RICH_MONITOR", True) and _MONITOR_AVAILABLE
    monitor = None
    if use_monitor:
        monitor = TrainingMonitor(
            total_updates=num_updates,
            config_dict={
                "env": config["ENV_NAME"],
                "layout": layout_name,
                "total_timesteps": int(config["TOTAL_TIMESTEPS"]),
                "num_updates": num_updates,
                "num_envs": config["NUM_ENVS"],
                "num_seeds": num_seeds,
                "lr": config["LR"],
                "gamma": config["GAMMA"],
            },
            title=f"IPPO-CNN - Overcooked ({layout_name})",
        )

    rng = jax.random.PRNGKey(config["SEED"])
    rngs = jax.random.split(rng, num_seeds)
    train_jit = jax.jit(make_train(config, monitor=monitor))
    if monitor is not None:
        with monitor:
            out = jax.block_until_ready(jax.vmap(train_jit)(rngs))
    else:
        out = jax.vmap(train_jit)(rngs)

    print("** Saving Results **")
    save_dir = os.path.join(os.environ.get("SCRATCH", "."), "jaxmarl", "results")
    os.makedirs(save_dir, exist_ok=True)
    filename = f"{config['ENV_NAME']}_{layout_name}_seed{config['SEED']}"
    filepath = os.path.join(save_dir, f"{filename}.gif")
    train_state = jax.tree.map(lambda x: x[0], out["runner_state"][0])
    state_seq = get_rollout(train_state.params, config)
    viz = OvercookedVisualizer()
    # agent_view_size is hardcoded as it determines the padding around the layout.
    viz.animate(state_seq, agent_view_size=5, filename=filepath)
    print(f"Saved to {filepath}")


def tune(default_config):
    """Hyperparameter sweep with CARBS (local Bayesian optimization)."""
    import copy
    import pickle
    import time
    from carbs import (
        CARBS,
        CARBSParams,
        Param,
        LogSpace,
        LinearSpace,
        ObservationInParam,
    )

    default_config = OmegaConf.to_container(default_config)
    layout_name = default_config["ENV_KWARGS"]["layout"]

    num_trials = int(default_config.get("CARBS_NUM_TRIALS", 50))

    # Define parameter search spaces
    param_spaces = [
        Param(
            name="LR",
            space=LogSpace(scale=0.5, min=1e-5, max=1e-2),
            search_center=default_config["LR"],
        ),
        Param(
            name="NUM_ENVS",
            space=LogSpace(is_integer=True, min=32, max=2048),
            search_center=default_config["NUM_ENVS"],
        ),
        Param(
            name="NUM_STEPS",
            space=LogSpace(is_integer=True, min=32, max=512),
            search_center=default_config["NUM_STEPS"],
        ),
        Param(
            name="UPDATE_EPOCHS",
            space=LinearSpace(scale=4, is_integer=True, min=1, max=16),
            search_center=default_config["UPDATE_EPOCHS"],
        ),
        Param(
            name="NUM_MINIBATCHES",
            space=LogSpace(is_integer=True, min=2, max=32),
            search_center=default_config["NUM_MINIBATCHES"],
        ),
        Param(
            name="CLIP_EPS",
            space=LinearSpace(scale=0.1, min=0.05, max=0.5),
            search_center=default_config["CLIP_EPS"],
        ),
        Param(
            name="ENT_COEF",
            space=LogSpace(scale=0.5, min=1e-5, max=0.1),
            search_center=default_config["ENT_COEF"],
        ),
        Param(
            name="GAE_LAMBDA",
            space=LinearSpace(scale=0.05, min=0.8, max=1.0),
            search_center=default_config["GAE_LAMBDA"],
        ),
    ]

    save_dir = os.path.join(os.environ.get("SCRATCH", "."), "jaxmarl", "carbs_sweep")
    os.makedirs(save_dir, exist_ok=True)

    carbs_params = CARBSParams(
        better_direction_sign=1,  # maximize return
        is_wandb_logging_enabled=False,
        resample_frequency=0,
        num_random_samples=4,
        initial_search_radius=0.3,
        checkpoint_dir=os.path.join(save_dir, "checkpoints"),
        is_saved_on_every_observation=True,
    )
    carbs = CARBS(carbs_params, param_spaces)

    def nearest_power_of_2(x):
        """Round to nearest power of 2."""
        return int(2 ** round(np.log2(x)))

    def sanitize_suggestion(suggestion):
        """Round integer params to powers of 2 for batch size compatibility."""
        s = dict(suggestion)
        for key in ["NUM_ENVS", "NUM_STEPS", "NUM_MINIBATCHES"]:
            if key in s:
                s[key] = nearest_power_of_2(s[key])
        return s

    print(f"Starting CARBS sweep with {num_trials} trials")
    print(f"Results will be saved to {save_dir}")

    all_results = []
    best_return = float("-inf")
    best_config = None

    for trial in range(num_trials):
        raw_suggestion = carbs.suggest().suggestion
        suggestion = sanitize_suggestion(raw_suggestion)
        config = copy.deepcopy(default_config)
        for k, v in suggestion.items():
            config[k] = v
        config["ENV_KWARGS"]["layout"] = overcooked_layouts[layout_name]

        print(f"\n{'=' * 60}")
        print(f"Trial {trial + 1}/{num_trials}")
        print(
            f"  LR={config['LR']:.6f}  NUM_ENVS={config['NUM_ENVS']}  "
            f"NUM_STEPS={config['NUM_STEPS']}  UPDATE_EPOCHS={config['UPDATE_EPOCHS']}"
        )
        print(
            f"  NUM_MINIBATCHES={config['NUM_MINIBATCHES']}  CLIP_EPS={config['CLIP_EPS']:.3f}  "
            f"ENT_COEF={config['ENT_COEF']:.5f}  GAE_LAMBDA={config['GAE_LAMBDA']:.3f}"
        )

        # Disable wandb logging during sweep trials (no wandb.init called)
        config["WANDB_MODE"] = "disabled"

        start_time = time.time()
        try:
            rng = jax.random.PRNGKey(config["SEED"])
            rngs = jax.random.split(rng, config["NUM_SEEDS"])
            train_vjit = jax.jit(jax.vmap(make_train(config)))
            outs = jax.block_until_ready(train_vjit(rngs))

            # Extract final mean return across seeds
            final_return = float(
                outs["metrics"]["returned_episode_returns"][:, -1].mean()
            )
            elapsed = time.time() - start_time

            obs_out = carbs.observe(
                ObservationInParam(input=suggestion, output=final_return, cost=elapsed)
            )

            if final_return > best_return:
                best_return = final_return
                best_config = {k: v for k, v in config.items() if k != "ENV_KWARGS"}

            result = {
                "trial": trial,
                "suggestion": suggestion,
                "return": final_return,
                "cost": elapsed,
            }
            all_results.append(result)
            print(
                f"  Return: {final_return:.2f}  Time: {elapsed:.1f}s  Best: {best_return:.2f}"
            )

        except Exception as e:
            elapsed = time.time() - start_time
            print(f"  FAILED: {e}")
            carbs.observe(
                ObservationInParam(
                    input=suggestion, output=0.0, cost=elapsed, is_failure=True
                )
            )
            all_results.append(
                {
                    "trial": trial,
                    "suggestion": suggestion,
                    "return": None,
                    "cost": elapsed,
                    "error": str(e),
                }
            )

        # Save checkpoint after each trial
        with open(os.path.join(save_dir, "carbs_results.pkl"), "wb") as f:
            pickle.dump(
                {
                    "results": all_results,
                    "best_config": best_config,
                    "best_return": best_return,
                },
                f,
            )

    print(f"\n{'=' * 60}")
    print(f"Sweep complete. Best return: {best_return:.2f}")
    print(f"Best config: {best_config}")
    print(f"Full results saved to {save_dir}/carbs_results.pkl")


scratch_dir = os.environ.get("SCRATCH", ".")
hydra_output_dir = os.path.join(
    scratch_dir, "jaxmarl", "outputs", "ippo_cnn_overcooked"
)
os.makedirs(hydra_output_dir, exist_ok=True)


@hydra.main(version_base=None, config_path="config", config_name="ippo_cnn_overcooked")
def main(config):
    if config["TUNE"]:
        tune(config)
    else:
        single_run(config)


if __name__ == "__main__":
    main()
