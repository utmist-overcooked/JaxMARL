"""IPPO (RNN) training script for Overcooked V3.

PPO with GRU-based recurrence, CNN observation encoder, and W&B sweep support.
Adapted from ippo_rnn_overcooked_v2.py for overcooked_v3 environments.

Key differences from v2:
  - Uses standard LogWrapper (overcooked_v3 doesn't need OvercookedV2LogWrapper)
  - Shaped reward annealing via configurable coefficient (not horizon-based)
  - W&B sweep support with Bayesian optimization
  - Python-level training loop for live metric reporting
"""
import sys
import os
import json
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import Callable, Sequence, NamedTuple, Any, Dict, List
from flax.training.train_state import TrainState
from flax import serialization
import distrax
import jaxmarl
from jaxmarl.wrappers.baselines import LogWrapper
import hydra
from omegaconf import OmegaConf
import wandb


# ── Network Architecture ───────────────────────────────────────────────


class CNN(nn.Module):
    """CNN encoder for grid-based observations (matches IPPO v2 architecture)."""
    output_size: int = 64
    activation: Callable[..., Any] = nn.relu

    @nn.compact
    def __call__(self, x, train=False):
        # x: (batch, H, W, C)
        # 1x1 convs for channel mixing
        x = nn.Conv(128, (1, 1), kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(x)
        x = self.activation(x)
        x = nn.Conv(128, (1, 1), kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(x)
        x = self.activation(x)
        x = nn.Conv(8, (1, 1), kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(x)
        x = self.activation(x)
        # Spatial convs
        x = nn.Conv(16, (3, 3), kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(x)
        x = self.activation(x)
        x = nn.Conv(32, (3, 3), kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(x)
        x = self.activation(x)
        x = nn.Conv(32, (3, 3), kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(x)
        x = self.activation(x)
        # Flatten: (batch, H', W', 32) -> (batch, H'*W'*32)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(self.output_size, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(x)
        x = self.activation(x)
        return x


class ActorCriticRNN(nn.Module):
    """Actor-Critic with CNN encoder and GRU via jax.lax.scan.

    Avoids Flax 0.10.4 / JAX 0.4.38 nn.scan bug by:
      - Pre-computing input projections (Dense) for all timesteps OUTSIDE scan
      - Using raw weight matrices (self.param) for recurrent ops INSIDE scan
      - No Flax module calls inside scan body → no tracer leak
    """
    action_dim: int
    config: Dict

    @nn.compact
    def __call__(self, hidden, x):
        """
        hidden: (num_actors, hidden_dim)
        x: (obs, dones) where obs is (T, num_actors, *obs_shape), dones is (T, num_actors)
        Returns: (new_hidden, pi, value) where pi and value are over (T, num_actors)
        """
        obs, dones = x
        T = obs.shape[0]
        hidden_dim = self.config.get("GRU_HIDDEN_DIM", 128)
        fc_dim = self.config.get("FC_DIM_SIZE", 128)

        # CNN embed: vmap over T, CNN handles actor batch dim internally
        cnn = CNN(output_size=hidden_dim)
        embedding = jax.vmap(cnn)(obs)  # (T, num_actors, hidden_dim)
        embedding = nn.LayerNorm()(embedding)  # (T, num_actors, hidden_dim)

        # GRU via lax.scan with pre-computed input projections
        flat_emb = embedding.reshape(-1, hidden_dim)  # (T*actors, hidden_dim)

        # Input projections — Dense layers applied to all timesteps at once (outside scan)
        Wi_z = nn.Dense(hidden_dim, use_bias=False, name='gru_Wi_z')(flat_emb)
        Wi_r = nn.Dense(hidden_dim, use_bias=False, name='gru_Wi_r')(flat_emb)
        Wi_h = nn.Dense(hidden_dim, use_bias=False, name='gru_Wi_h')(flat_emb)

        # Reshape back to (T, actors, hidden_dim)
        num_actors = obs.shape[1]
        Wi_z = Wi_z.reshape(T, num_actors, hidden_dim)
        Wi_r = Wi_r.reshape(T, num_actors, hidden_dim)
        Wi_h = Wi_h.reshape(T, num_actors, hidden_dim)

        # Recurrent weight matrices as raw params (safe to use inside lax.scan)
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
            # GRU gates
            z = jax.nn.sigmoid(wiz_t + h @ Wh_z + b_z)
            r = jax.nn.sigmoid(wir_t + h @ Wh_r + b_r)
            h_hat = jnp.tanh(wih_t + (r * h) @ Wh_h + b_h)
            new_h = (1 - z) * h + z * h_hat
            return new_h, new_h

        final_hidden, gru_out = jax.lax.scan(
            _gru_step, hidden, (Wi_z, Wi_r, Wi_h, dones)
        )  # gru_out: (T, num_actors, hidden_dim)

        # Actor/critic heads
        flat = gru_out.reshape(-1, hidden_dim)
        actor_mean = nn.relu(nn.Dense(fc_dim, kernel_init=orthogonal(2), bias_init=constant(0.0), name='actor_fc')(flat))
        actor_mean = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0), name='actor_out')(actor_mean)
        actor_mean = actor_mean.reshape(T, -1, self.action_dim)

        critic = nn.relu(nn.Dense(fc_dim, kernel_init=orthogonal(2), bias_init=constant(0.0), name='critic_fc')(flat))
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0), name='critic_out')(critic)
        critic = critic.reshape(T, -1)

        pi = distrax.Categorical(logits=actor_mean)
        return final_hidden, pi, critic


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


# ── Training ───────────────────────────────────────────────────────────


def make_train(config):
    """Create the IPPO training function for overcooked_v3."""
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

    use_shaped_reward = config.get("USE_SHAPED_REWARD", True)
    shaped_reward_coeff = config.get("SHAPED_REWARD_COEFF", 1.0)

    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    num_updates = config["NUM_UPDATES"]
    save_path = config.get("SAVE_PATH", None)
    checkpoint_every = config.get("CHECKPOINT_EVERY", max(1, num_updates // 10))
    log_every = config.get("LOG_EVERY", max(1, num_updates // 100))

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

        # JIT-compiled update step
        def _update_step(runner_state):
            train_state, env_state, last_obs, last_done, hstate, rng = runner_state

            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, last_done, hstate, rng = runner_state

                rng, _rng = jax.random.split(rng)
                obs_batch = jnp.stack([last_obs[a] for a in env.agents]).reshape(
                    -1, *obs_shape
                )
                ac_in = (obs_batch[np.newaxis, :], last_done[np.newaxis, :])
                hstate, pi, value = network.apply(train_state.params, hstate, ac_in)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)
                env_act = unbatchify(action, env.agents, config["NUM_ENVS"], num_agents)
                env_act = {k: v.flatten() for k, v in env_act.items()}

                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(env.step)(
                    rng_step, env_state, env_act
                )

                # Add shaped reward if available
                if use_shaped_reward and "shaped_reward" in info:
                    shaped = info["shaped_reward"]
                    reward = {
                        a: reward[a] + shaped_reward_coeff * shaped[a]
                        for a in env.agents
                    }

                # Only keep LogWrapper fields (shape NUM_ENVS × num_agents)
                # to avoid reshape issues with env-specific info like shaped_reward dict
                log_info = {
                    "returned_episode_returns": info["returned_episode_returns"],
                    "returned_episode_lengths": info["returned_episode_lengths"],
                    "returned_episode": info["returned_episode"],
                }
                log_info = jax.tree_util.tree_map(
                    lambda x: x.reshape((config["NUM_ACTORS"])), log_info
                )
                done_batch = jnp.stack(
                    [done[a] for a in env.agents]
                ).reshape(config["NUM_ACTORS"])
                transition = Transition(
                    jnp.tile(done["__all__"], num_agents),
                    action.squeeze(),
                    value.squeeze(),
                    jnp.stack([reward[a] for a in env.agents]).reshape(config["NUM_ACTORS"]),
                    log_prob.squeeze(),
                    obs_batch,
                    log_info,
                )
                runner_state = (train_state, env_state, obsv, done_batch, hstate, rng)
                return runner_state, transition

            initial_hstate = hstate
            scan_carry = (train_state, env_state, last_obs, last_done, hstate, rng)
            runner_state_out, traj_batch = jax.lax.scan(
                _env_step, scan_carry, None, config["NUM_STEPS"]
            )
            train_state, env_state, last_obs, last_done, hstate, rng = runner_state_out

            # CALCULATE ADVANTAGE (GAE)
            last_obs_batch = jnp.stack([last_obs[a] for a in env.agents]).reshape(
                -1, *obs_shape
            )
            ac_in = (last_obs_batch[np.newaxis, :], last_done[np.newaxis, :])
            _, _, last_val = network.apply(train_state.params, hstate, ac_in)
            last_val = last_val.squeeze()

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = transition.done, transition.value, transition.reward
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

            # UPDATE NETWORK (PPO epochs)
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    init_hstate, traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, init_hstate, traj_batch, gae, targets):
                        _, pi, value = network.apply(
                            params, init_hstate.squeeze(), (traj_batch.obs, traj_batch.done)
                        )
                        log_prob = pi.log_prob(traj_batch.action)

                        # Value loss (clipped)
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()

                        # Actor loss (clipped)
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = jnp.clip(ratio, 1.0 - config["CLIP_EPS"], 1.0 + config["CLIP_EPS"]) * gae
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2).mean()
                        entropy = pi.entropy().mean()

                        total_loss = (
                            loss_actor
                            + config["VF_COEF"] * value_loss
                            - config["ENT_COEF"] * entropy
                        )
                        return total_loss, (value_loss, loss_actor, entropy)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params, init_hstate, traj_batch, advantages, targets
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                train_state, init_hstate, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)

                init_hstate_r = jnp.reshape(init_hstate, (1, config["NUM_ACTORS"], -1))
                batch = (init_hstate_r, traj_batch, advantages.squeeze(), targets.squeeze())
                permutation = jax.random.permutation(_rng, config["NUM_ACTORS"])
                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=1), batch
                )
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.swapaxes(
                        jnp.reshape(x, [x.shape[0], config["NUM_MINIBATCHES"], -1] + list(x.shape[2:])),
                        1, 0,
                    ),
                    shuffled_batch,
                )
                train_state, total_loss = jax.lax.scan(_update_minbatch, train_state, minibatches)
                update_state = (train_state, init_hstate, traj_batch, advantages, targets, rng)
                return update_state, total_loss

            update_state = (train_state, initial_hstate, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            train_state = update_state[0]
            rng = update_state[-1]

            # Metrics
            metric = traj_batch.info
            loss_info_mean = jax.tree_util.tree_map(lambda x: x.mean(), loss_info)
            metrics = {
                "returned_episode_returns": metric["returned_episode_returns"][-1, :].mean(),
                "mean_reward_per_step": traj_batch.reward.mean(),
                "max_reward_per_step": traj_batch.reward.max(),
                "reward_sum": traj_batch.reward.sum(),
                "total_loss": loss_info_mean[0],
                "value_loss": loss_info_mean[1][0],
                "actor_loss": loss_info_mean[1][1],
                "entropy": loss_info_mean[1][2],
            }

            runner_state = (train_state, env_state, last_obs, last_done, hstate, rng)
            return runner_state, metrics

        update_step_jit = jax.jit(_update_step)

        # Python-level training loop
        rng, _rng = jax.random.split(rng)
        runner_state = (
            train_state, env_state, obsv,
            jnp.zeros((config["NUM_ACTORS"]), dtype=bool),
            init_hstate, _rng,
        )

        if save_path:
            os.makedirs(save_path, exist_ok=True)
            with open(os.path.join(save_path, "run_config.json"), "w") as f:
                json.dump({
                    "config": {k: v for k, v in config.items()},
                    "env_info": {
                        "num_agents": num_agents,
                        "obs_shape": list(obs_shape),
                        "action_dim": action_dim,
                    },
                }, f, indent=2, default=str)

        all_metrics: Dict[str, List[float]] = {}
        print(f"Starting IPPO training: {num_updates} updates | "
              f"{config['NUM_ENVS']} envs | {config['NUM_STEPS']} steps/rollout")

        for step in range(1, num_updates + 1):
            runner_state, metrics = update_step_jit(runner_state)
            jax.block_until_ready(metrics)

            step_metrics = {k: float(v) for k, v in metrics.items()}
            for k, v in step_metrics.items():
                all_metrics.setdefault(k, []).append(v)

            if step == 1 or step % log_every == 0 or step == num_updates:
                ep_ret = step_metrics.get("returned_episode_returns", 0.0)
                mean_rew = step_metrics.get("mean_reward_per_step", 0.0)
                max_rew = step_metrics.get("max_reward_per_step", 0.0)
                rew_sum = step_metrics.get("reward_sum", 0.0)
                print(f"[{step}/{num_updates}] EpRet={ep_ret:.4f} MeanRew={mean_rew:.6f} "
                      f"MaxRew={max_rew:.4f} RewSum={rew_sum:.2f} "
                      f"Loss={step_metrics.get('total_loss', 0):.4f} "
                      f"Entropy={step_metrics.get('entropy', 0):.4f}")

            if config.get("WANDB_MODE", "disabled") != "disabled":
                wandb.log({"update": step, **step_metrics})

            if save_path and (step % checkpoint_every == 0 or step == num_updates):
                ts = runner_state[0]
                ckpt_file = os.path.join(save_path, f"model_update_{step}.msgpack")
                with open(ckpt_file, "wb") as f:
                    f.write(serialization.to_bytes(ts.params))
                latest_file = os.path.join(save_path, "model.msgpack")
                with open(latest_file, "wb") as f:
                    f.write(serialization.to_bytes(ts.params))

        final_return = all_metrics.get("returned_episode_returns", [0.0])[-1]
        final_rew = all_metrics.get("mean_reward_per_step", [0.0])[-1]
        print(f"\nTraining complete! EpRet={final_return:.2f} MeanRew={final_rew:.4f}")

        return {
            "runner_state": runner_state,
            "metrics": all_metrics,
        }

    return train


# ── Sweep Configuration ───────────────────────────────────────────────


def _build_overcooked_v3_sweep_configuration():
    """W&B sweep config for IPPO on overcooked_v3.

    Based on IC3Net sweep analysis + standard IPPO hyperparameters.
    """
    return {
        "method": "bayes",
        "metric": {"goal": "maximize", "name": "mean_reward_per_step"},
        "parameters": {
            "LR": {"distribution": "log_uniform_values", "min": 5e-5, "max": 1e-3},
            "ENT_COEF": {
                "distribution": "log_uniform_values",
                "min": 1e-4,
                "max": 5e-2,
            },
            "VF_COEF": {"values": [0.25, 0.5, 0.75, 1.0]},
            "MAX_GRAD_NORM": {"values": [0.25, 0.5, 1.0]},
            "GAMMA": {"values": [0.99, 0.995]},
            "GAE_LAMBDA": {"values": [0.9, 0.95, 0.98]},
            "CLIP_EPS": {"values": [0.1, 0.2, 0.3]},
            "UPDATE_EPOCHS": {"values": [2, 4, 8]},
            "GRU_HIDDEN_DIM": {"values": [128, 256]},
            "SHAPED_REWARD_COEFF": {"values": [1.0, 2.0, 3.0]},
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
            train_config["WANDB_NAME"] = run.name
            train_config["SAVE_PATH"] = os.path.join(
                base_config.get("SAVE_PATH", "checkpoints/ippo_overcooked_v3"),
                f"sweep_{run.id}",
            )

            train_fn = make_train(train_config)
            rng = jax.random.PRNGKey(train_config.get("SEED", 42))
            output = train_fn(rng)

            final_mean_reward = output["metrics"].get("mean_reward_per_step", [0.0])[-1]
            final_episode_return = output["metrics"].get("returned_episode_returns", [0.0])[-1]
            wandb.log({
                "mean_reward_per_step": float(final_mean_reward),
                "final_episode_return": float(final_episode_return),
            })

    sweep_id = wandb.sweep(sweep=sweep_configuration, project=project)
    wandb.agent(sweep_id, function=_objective, count=sweep_count)


@hydra.main(version_base=None, config_path="config", config_name="ippo_rnn_overcooked_v3")
def main(config):
    """Main training entry point."""
    config = OmegaConf.to_container(config, resolve=True)

    if config.get("WANDB_SWEEP", False):
        run_wandb_sweep(config)
        return

    # Setup wandb
    if config.get("WANDB_MODE", "disabled") != "disabled":
        wandb.init(
            project=config.get("WANDB_PROJECT", "jaxmarl-ippo"),
            name=config.get("WANDB_NAME", None),
            tags=["IPPO", "RNN", "OvercookedV3"],
            config=config,
            mode=config.get("WANDB_MODE", "online"),
        )

    train_fn = make_train(config)
    rng = jax.random.PRNGKey(config.get("SEED", 42))
    output = train_fn(rng)

    if config.get("WANDB_MODE", "disabled") != "disabled":
        wandb.finish()


if __name__ == "__main__":
    main()
