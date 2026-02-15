"""IC3Net-family training script for JaxMARL.

REINFORCE with value baseline trainer for IC3Net, CommNet, IC, and IRIC.
Uses a Python-level training loop with live TrainingMonitor (rich UI) and
periodic checkpoint saving, following the reference CoGrid runner pattern.
"""
import sys
import os
import time
import json

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import NamedTuple, Dict, Any, List
from flax.training.train_state import TrainState
from flax import serialization
import distrax
import jaxmarl
from jaxmarl.wrappers.baselines import LogWrapper
import hydra
from omegaconf import OmegaConf
import wandb

from baselines.IC3Net.models import IndependentMLP, IndependentLSTM, CommNetDiscrete, CommNetLSTM
from baselines.IC3Net.monitor import TrainingMonitorInterface


class Transition(NamedTuple):
    """Transition for REINFORCE rollouts."""
    obs: jnp.ndarray       # (T, B, N, obs_dim)
    action: jnp.ndarray    # (T, B, N)
    value: jnp.ndarray     # (T, B, N)
    reward: jnp.ndarray    # (T, B, N)
    done: jnp.ndarray      # (T, B, N)
    log_prob: jnp.ndarray  # (T, B, N)
    h: jnp.ndarray         # (T, B, N, hidden_dim) or None
    c: jnp.ndarray         # (T, B, N, hidden_dim) or None


def _build_network(config, num_agents, action_dim):
    """Build network based on baseline type and recurrence setting."""
    baseline = config.get("BASELINE", "ic3net")
    recurrent = config.get("RECURRENT", True)
    hidden_dim = config.get("HIDDEN_DIM", 64)

    if baseline in ("ic", "iric"):
        if recurrent:
            network = IndependentLSTM(action_dim=action_dim, hidden_dim=hidden_dim)
        else:
            network = IndependentMLP(action_dim=action_dim, hidden_dim=hidden_dim)
        has_talk = False
    else:
        hard_attn = (baseline == "ic3net")
        kw = dict(
            num_agents=num_agents,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            comm_passes=config.get("COMM_PASSES", 1),
            comm_mode=config.get("COMM_MODE", "avg"),
            hard_attn=hard_attn,
            share_weights=config.get("SHARE_WEIGHTS", False),
        )
        network = CommNetLSTM(**kw) if recurrent else CommNetDiscrete(**kw)
        has_talk = hard_attn

    return network, has_talk


def _init_params(rng, network, config, num_agents, obs_dim, has_talk):
    """Initialize network parameters."""
    hidden_dim = config.get("HIDDEN_DIM", 64)
    recurrent = config.get("RECURRENT", True)
    init_obs = jnp.zeros((1, num_agents, obs_dim))

    if recurrent:
        init_carry = (
            jnp.zeros((1, num_agents, hidden_dim)),
            jnp.zeros((1, num_agents, hidden_dim)),
        )
        if has_talk:
            return network.init(rng, init_obs, carry=init_carry,
                                comm_action=jnp.zeros(num_agents, dtype=jnp.int32))
        return network.init(rng, init_obs, carry=init_carry)
    else:
        if has_talk:
            return network.init(rng, init_obs,
                                comm_action=jnp.zeros(num_agents, dtype=jnp.int32))
        return network.init(rng, init_obs)


def make_reinforce_step(config, env, network, has_talk, num_agents, obs_dim):
    """Create a JIT-compiled single REINFORCE update step.

    Returns a function: (runner_state) -> (runner_state, metrics_dict)
    """
    recurrent = config.get("RECURRENT", True)
    hidden_dim = config.get("HIDDEN_DIM", 64)

    def _update_step(runner_state):
        train_state, env_state, last_obs, comm_action, hstate, cstate, rng = runner_state

        # -- Rollout collection ------------------------------------------
        def _env_step(carry, step_idx):
            train_state, env_state, obs, comm_action, hstate, cstate, rng = carry

            # Stack & flatten obs: dict -> (B, N, obs_dim)
            obs_list = []
            for a in env.agents:
                agent_obs = obs[a]
                if len(agent_obs.shape) > 2:
                    flat_obs = agent_obs.reshape(agent_obs.shape[0], -1)
                else:
                    flat_obs = agent_obs
                obs_list.append(flat_obs)
            obs_batch = jnp.stack(obs_list, axis=1)

            rng, _rng = jax.random.split(rng)

            if recurrent:
                carry_in = (hstate, cstate)
                detach_gap = config.get("DETACH_GAP", 10)
                if detach_gap > 0:
                    should_detach = (step_idx > 0) & (step_idx % detach_gap == 0)
                    hstate_in = jax.lax.cond(
                        should_detach, jax.lax.stop_gradient, lambda x: x, hstate)
                    cstate_in = jax.lax.cond(
                        should_detach, jax.lax.stop_gradient, lambda x: x, cstate)
                    carry_in = (hstate_in, cstate_in)
                if has_talk:
                    logits, value, talk_logits, (hstate_new, cstate_new) = network.apply(
                        train_state.params, obs_batch,
                        carry=carry_in, comm_action=comm_action[0])
                else:
                    logits, value, talk_logits, (hstate_new, cstate_new) = network.apply(
                        train_state.params, obs_batch, carry=carry_in)
                hstate, cstate = hstate_new, cstate_new
            else:
                if has_talk:
                    logits, value, talk_logits = network.apply(
                        train_state.params, obs_batch, comm_action=comm_action[0])
                else:
                    logits, value = network.apply(train_state.params, obs_batch)
                    talk_logits = None

            # Sample actions
            rng, _rng = jax.random.split(rng)
            action_dist = distrax.Categorical(logits=logits)
            action_env = action_dist.sample(seed=_rng)
            log_prob = action_dist.log_prob(action_env)

            # Talk actions for IC3Net
            if has_talk and talk_logits is not None:
                rng, _rng = jax.random.split(rng)
                talk_dist = distrax.Categorical(logits=talk_logits)
                action_talk = talk_dist.sample(seed=_rng)
                log_prob = log_prob + talk_dist.log_prob(action_talk)
                if config.get("COMM_ACTION_ONE", False):
                    comm_action = jnp.ones((config["NUM_ENVS"], num_agents), dtype=jnp.int32)
                else:
                    comm_action = action_talk

            action_dict = {a: action_env[:, i] for i, a in enumerate(env.agents)}
            rng, _rng = jax.random.split(rng)
            rng_step = jax.random.split(_rng, config["NUM_ENVS"])
            obsv, env_state, reward, done, info = jax.vmap(env.step)(
                rng_step, env_state, action_dict)

            reward_batch = jnp.stack([reward[a] for a in env.agents], axis=1)
            done_batch = jnp.stack([done[a] for a in env.agents], axis=1)

            transition = Transition(
                obs=obs_batch, action=action_env, value=value,
                reward=reward_batch, done=done_batch, log_prob=log_prob,
                h=hstate if recurrent else jnp.zeros(()),
                c=cstate if recurrent else jnp.zeros(()),
            )
            return (train_state, env_state, obsv, comm_action, hstate, cstate, rng), transition

        if has_talk:
            rollout_comm = jnp.ones((config["NUM_ENVS"], num_agents), dtype=jnp.int32)
        else:
            rollout_comm = comm_action

        init_carry = (train_state, env_state, last_obs, rollout_comm, hstate, cstate, rng)
        final_carry, transitions = jax.lax.scan(
            _env_step, init_carry,
            jnp.arange(config["NUM_STEPS"]),
            length=config["NUM_STEPS"],
        )
        train_state, env_state, last_obs, comm_action, hstate, cstate, rng = final_carry

        # -- Bootstrap last value ----------------------------------------
        obs_list = []
        for a in env.agents:
            agent_obs = last_obs[a]
            if len(agent_obs.shape) > 2:
                flat_obs = agent_obs.reshape(agent_obs.shape[0], -1)
            else:
                flat_obs = agent_obs
            obs_list.append(flat_obs)
        last_obs_batch = jnp.stack(obs_list, axis=1)

        if recurrent:
            carry_last = (hstate, cstate)
            if has_talk:
                _, last_value, _, _ = network.apply(
                    train_state.params, last_obs_batch,
                    carry=carry_last, comm_action=comm_action[0])
            else:
                _, last_value, _, _ = network.apply(
                    train_state.params, last_obs_batch, carry=carry_last)
        else:
            if has_talk:
                _, last_value, _ = network.apply(
                    train_state.params, last_obs_batch, comm_action=comm_action[0])
            else:
                _, last_value = network.apply(train_state.params, last_obs_batch)

        # -- Compute discounted returns ----------------------------------
        gamma = config.get("GAMMA", 1.0)

        def _compute_returns(next_value, transition):
            returns = transition.reward + gamma * next_value * (1 - transition.done)
            return returns, returns

        _, returns = jax.lax.scan(
            _compute_returns, last_value, transitions, reverse=True)

        # -- Advantages --------------------------------------------------
        advantages = returns - transitions.value
        if config.get("NORMALIZE_ADVANTAGES", False):
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # -- REINFORCE loss ----------------------------------------------
        def _loss_fn(params):
            policy_loss = -jnp.mean(transitions.log_prob * advantages)
            value_loss = jnp.mean((transitions.value - returns) ** 2)
            value_coeff = config.get("VALUE_COEFF", 0.01)
            entropy_coeff = config.get("ENTROPY_COEFF", 0.0)

            loss = policy_loss + value_coeff * value_loss
            # Entropy bonus (if configured): recompute from stored logits is
            # expensive; instead approximate from log_prob for IC3Net.
            return loss, {
                "loss": loss,
                "policy_loss": policy_loss,
                "value_loss": value_loss,
            }

        (loss, metrics), grads = jax.value_and_grad(_loss_fn, has_aux=True)(
            train_state.params)
        train_state = train_state.apply_gradients(grads=grads)

        metrics["returned_episode_returns"] = env_state.returned_episode_returns[0].mean()

        runner_state = (train_state, env_state, last_obs, comm_action, hstate, cstate, rng)
        return runner_state, metrics

    return jax.jit(_update_step)


def make_train(config):
    """Create and run the full REINFORCE training loop with TrainingMonitor.

    This uses a Python-level loop around a JIT-compiled update step, enabling
    live metric reporting via the TrainingMonitor and periodic checkpoint saves.
    """
    # -- Environment setup -----------------------------------------------
    env_kwargs = dict(config.get("ENV_KWARGS", {}))

    # Handle layout for overcooked v1 (v3 handles string layouts internally)
    if config["ENV_NAME"] == "overcooked" and "layout" in env_kwargs:
        layout_name = env_kwargs["layout"]
        if isinstance(layout_name, str):
            from jaxmarl.environments.overcooked.layouts import overcooked_layouts
            from flax.core.frozen_dict import FrozenDict
            env_kwargs["layout"] = FrozenDict(overcooked_layouts[layout_name])

    env = jaxmarl.make(config["ENV_NAME"], **env_kwargs)
    num_agents = env.num_agents
    obs_shape = env.observation_space(env.agents[0]).shape
    obs_dim = int(np.prod(obs_shape))
    action_dim = env.action_space(env.agents[0]).n
    env = LogWrapper(env)

    num_updates = config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    config["NUM_UPDATES"] = num_updates

    baseline = config.get("BASELINE", "ic3net")
    recurrent = config.get("RECURRENT", True)
    hidden_dim = config.get("HIDDEN_DIM", 64)

    # -- Build network & optimizer ---------------------------------------
    network, has_talk = _build_network(config, num_agents, action_dim)

    rng = jax.random.PRNGKey(config.get("SEED", 42))
    rng, init_rng = jax.random.split(rng)
    network_params = _init_params(init_rng, network, config, num_agents, obs_dim, has_talk)

    tx = optax.chain(
        optax.clip_by_global_norm(config.get("MAX_GRAD_NORM", 0.5)),
        optax.rmsprop(
            learning_rate=config.get("LR", 1e-3),
            decay=config.get("RMSPROP_ALPHA", 0.97),
            eps=config.get("RMSPROP_EPS", 1e-6),
        ),
    )
    train_state = TrainState.create(apply_fn=network.apply, params=network_params, tx=tx)

    # -- Initialize environment ------------------------------------------
    rng, _rng = jax.random.split(rng)
    reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
    obsv, env_state = jax.vmap(env.reset)(reset_rng)

    if recurrent:
        init_hstate = jnp.zeros((config["NUM_ENVS"], num_agents, hidden_dim))
        init_cstate = jnp.zeros((config["NUM_ENVS"], num_agents, hidden_dim))
    else:
        init_hstate = None
        init_cstate = None

    if has_talk:
        init_comm_action = jnp.ones((config["NUM_ENVS"], num_agents), dtype=jnp.int32)
    else:
        init_comm_action = jnp.zeros((config["NUM_ENVS"], num_agents), dtype=jnp.int32)

    # -- JIT-compiled update step ----------------------------------------
    update_step = make_reinforce_step(config, env, network, has_talk, num_agents, obs_dim)

    rng, _rng = jax.random.split(rng)
    runner_state = (train_state, env_state, obsv, init_comm_action, init_hstate, init_cstate, _rng)

    # -- Checkpoint setup ------------------------------------------------
    save_path = config.get("SAVE_PATH", None)
    checkpoint_every = config.get("CHECKPOINT_EVERY", max(1, num_updates // 10))
    log_every = config.get("LOG_EVERY", max(1, num_updates // 100))

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        # Save run config
        config_file = os.path.join(save_path, "run_config.json")
        with open(config_file, "w") as f:
            json.dump({
                "config": {k: v for k, v in config.items()},
                "env_info": {
                    "num_agents": num_agents,
                    "obs_dim": obs_dim,
                    "obs_shape": list(obs_shape),
                    "action_dim": action_dim,
                },
            }, f, indent=2, default=str)

    # -- Monitor config --------------------------------------------------
    monitor_config = {
        "Env": config["ENV_NAME"],
        "Baseline": baseline,
        "Hidden Dim": hidden_dim,
        "Recurrent": recurrent,
        "Num Envs": config["NUM_ENVS"],
        "Num Steps": config["NUM_STEPS"],
        "Total Updates": num_updates,
        "LR": config.get("LR", 1e-3),
        "Gamma": config.get("GAMMA", 1.0),
        "Device": str(jax.default_backend()),
    }

    # -- Training loop with monitor -------------------------------------
    monitor = TrainingMonitorInterface(
        total_updates=num_updates,
        config_dict=monitor_config,
    )

    all_metrics: Dict[str, List[float]] = {}

    with monitor:
        monitor.log(f"Starting REINFORCE training: {baseline.upper()} "
                     f"| {num_updates} updates | {config['NUM_ENVS']} envs "
                     f"| {config['NUM_STEPS']} steps/rollout")
        monitor.log(f"JIT-compiling update step (first call will be slow)...")

        for step in range(1, num_updates + 1):
            # Execute one REINFORCE update
            runner_state, metrics = update_step(runner_state)

            # Block until metrics are ready (device->host transfer)
            jax.block_until_ready(metrics)

            # Extract scalar metrics
            step_metrics = {
                k: float(v) for k, v in metrics.items()
            }

            # Accumulate
            for k, v in step_metrics.items():
                all_metrics.setdefault(k, []).append(v)

            # Log to monitor
            if step == 1 or step % log_every == 0 or step == num_updates:
                display = {"Update": step}
                display["EpRet"] = step_metrics.get("returned_episode_returns", 0.0)
                display["Loss"] = step_metrics.get("loss", 0.0)
                display["Pi Loss"] = step_metrics.get("policy_loss", 0.0)
                display["V Loss"] = step_metrics.get("value_loss", 0.0)
                monitor.update(step, display)

            # Log to wandb
            if config.get("WANDB_MODE", "disabled") != "disabled":
                wandb.log({"update": step, **step_metrics})

            # Checkpoint
            if save_path and (step % checkpoint_every == 0 or step == num_updates):
                ts = runner_state[0]  # train_state
                ckpt_file = os.path.join(save_path, f"model_update_{step}.msgpack")
                with open(ckpt_file, "wb") as f:
                    f.write(serialization.to_bytes(ts.params))
                # Also save as latest
                latest_file = os.path.join(save_path, "model.msgpack")
                with open(latest_file, "wb") as f:
                    f.write(serialization.to_bytes(ts.params))
                monitor.log(f"Checkpoint saved: {ckpt_file}")

        monitor.log("Training complete!")

    # -- Final summary ---------------------------------------------------
    final_return = all_metrics.get("returned_episode_returns", [0.0])[-1]
    print(f"\nTraining complete! Final episode return: {final_return:.2f}")

    return {
        "runner_state": runner_state,
        "metrics": all_metrics,
    }


@hydra.main(version_base=None, config_path="config", config_name="ic3net_mpe")
def main(config):
    """Main training entry point."""
    config = OmegaConf.to_container(config)

    # Setup wandb
    if config.get("WANDB_MODE", "disabled") != "disabled":
        wandb.init(
            project=config.get("WANDB_PROJECT", "jaxmarl-ic3net"),
            name=config.get("WANDB_NAME", None),
            config=config,
            mode=config.get("WANDB_MODE", "online"),
        )

    output = make_train(config)

    if config.get("WANDB_MODE", "disabled") != "disabled":
        wandb.finish()


if __name__ == "__main__":
    main()
