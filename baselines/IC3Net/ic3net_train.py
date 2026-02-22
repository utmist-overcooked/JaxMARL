"""IC3Net-family training script for JaxMARL.

REINFORCE with value baseline trainer for IC3Net, CommNet, IC, and IRIC.
Uses a Python-level training loop with live TrainingMonitor (rich UI) and
periodic checkpoint saving, following the reference CoGrid runner pattern.

Key design:
  - Network is re-evaluated inside the loss function so that gradients
    flow through the parameters (JAX requires explicit dependency).
  - LSTM hidden states are reset at episode boundaries.
  - For IC3Net, talk actions from the rollout are stored and replayed
    during re-evaluation.
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
    obs: jnp.ndarray         # (T, B, N, obs_dim)
    action: jnp.ndarray      # (T, B, N)
    talk_action: jnp.ndarray # (T, B, N)  talk actions for IC3Net
    reward: jnp.ndarray      # (T, B, N)
    done: jnp.ndarray        # (T, B, N)


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
            encoder_layers=config.get("ENCODER_LAYERS", 1),
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
                                comm_action=jnp.zeros((1, num_agents), dtype=jnp.int32))
        return network.init(rng, init_obs, carry=init_carry)
    else:
        if has_talk:
            return network.init(rng, init_obs,
                                comm_action=jnp.zeros((1, num_agents), dtype=jnp.int32))
        return network.init(rng, init_obs)


def make_reinforce_step(config, env, network, has_talk, num_agents, obs_dim):
    """Create a JIT-compiled single REINFORCE update step.

    Returns a function: (runner_state) -> (runner_state, metrics_dict)
    """
    recurrent = config.get("RECURRENT", True)
    hidden_dim = config.get("HIDDEN_DIM", 64)
    gamma = config.get("GAMMA", 1.0)
    detach_gap = config.get("DETACH_GAP", 10)
    value_coeff = config.get("VALUE_COEFF", 0.01)
    entropy_coeff = config.get("ENTROPY_COEFF", 0.0)
    comm_action_one = config.get("COMM_ACTION_ONE", False)
    num_envs = config["NUM_ENVS"]
    num_steps = config["NUM_STEPS"]
    is_independent = config.get("BASELINE", "ic3net") in ("ic", "iric")
    use_shaped_reward = config.get("USE_SHAPED_REWARD", False)
    shaped_reward_coeff = config.get("SHAPED_REWARD_COEFF", 1.0)

    def _stack_obs(obs):
        """Stack obs dict -> (B, N, obs_dim)."""
        obs_list = []
        for a in env.agents:
            agent_obs = obs[a]
            if len(agent_obs.shape) > 2:
                flat_obs = agent_obs.reshape(agent_obs.shape[0], -1)
            else:
                flat_obs = agent_obs
            obs_list.append(flat_obs)
        return jnp.stack(obs_list, axis=1)

    def _update_step(runner_state):
        train_state, env_state, last_obs, comm_action, hstate, cstate, rng = runner_state

        # Save initial hidden state for loss re-evaluation
        init_h_eval = hstate
        init_c_eval = cstate
        init_comm_eval = comm_action

        # ── ROLLOUT COLLECTION (stop_gradient on params) ────────────────
        def _env_step(carry, step_idx):
            train_state, env_state, obs, comm_action, hstate, cstate, rng = carry

            obs_batch = _stack_obs(obs)
            rng, _rng = jax.random.split(rng)

            # Forward pass for action sampling only (no gradient needed)
            stopped_params = jax.lax.stop_gradient(train_state.params)

            if recurrent:
                carry_in = (hstate, cstate)
                if is_independent:
                    logits, _, (h_new, c_new) = network.apply(
                        stopped_params, obs_batch, carry=carry_in)
                    talk_logits = None
                elif has_talk:
                    logits, _, talk_logits, (h_new, c_new) = network.apply(
                        stopped_params, obs_batch, carry=carry_in, comm_action=comm_action)
                else:
                    logits, _, talk_logits, (h_new, c_new) = network.apply(
                        stopped_params, obs_batch, carry=carry_in)
                hstate, cstate = h_new, c_new
            else:
                if has_talk:
                    logits, _, talk_logits = network.apply(
                        stopped_params, obs_batch, comm_action=comm_action)
                else:
                    logits, _ = network.apply(stopped_params, obs_batch)
                    talk_logits = None

            # Sample environment actions
            rng, _rng = jax.random.split(rng)
            action_env = distrax.Categorical(logits=logits).sample(seed=_rng)

            # Sample talk actions
            talk_action = jnp.zeros((num_envs, num_agents), dtype=jnp.int32)
            if has_talk and talk_logits is not None:
                rng, _rng = jax.random.split(rng)
                talk_action = distrax.Categorical(logits=talk_logits).sample(seed=_rng)
                if comm_action_one:
                    comm_action = jnp.ones((num_envs, num_agents), dtype=jnp.int32)
                else:
                    comm_action = talk_action

            # Step environment
            action_dict = {a: action_env[:, i] for i, a in enumerate(env.agents)}
            rng, _rng = jax.random.split(rng)
            rng_step = jax.random.split(_rng, num_envs)
            obsv, env_state, reward, done, info = jax.vmap(env.step)(
                rng_step, env_state, action_dict)

            reward_batch = jnp.stack([reward[a] for a in env.agents], axis=1)
            if use_shaped_reward and "shaped_reward" in info:
                shaped = jnp.stack([info["shaped_reward"][a] for a in env.agents], axis=1)
                reward_batch = reward_batch + shaped_reward_coeff * shaped
            done_batch = jnp.stack([done[a] for a in env.agents], axis=1)

            transition = Transition(
                obs=obs_batch, action=action_env,
                talk_action=talk_action,
                reward=reward_batch, done=done_batch,
            )

            # Reset hidden state at episode boundaries
            if recurrent:
                done_all = done_batch[:, 0]  # (B,)
                reset_h = done_all[:, None, None]
                hstate = jnp.where(reset_h, jnp.zeros_like(hstate), hstate)
                cstate = jnp.where(reset_h, jnp.zeros_like(cstate), cstate)
                if has_talk:
                    comm_action = jnp.where(
                        done_all[:, None], jnp.zeros_like(comm_action), comm_action)

            return (train_state, env_state, obsv, comm_action, hstate, cstate, rng), transition

        init_carry = (train_state, env_state, last_obs, comm_action, hstate, cstate, rng)
        final_carry, transitions = jax.lax.scan(
            _env_step, init_carry, jnp.arange(num_steps), length=num_steps)
        train_state, env_state, last_obs, comm_action, hstate, cstate, rng = final_carry

        # ── LOSS with re-evaluation (gradients flow through params) ─────
        def _loss_fn(params):
            if recurrent:
                # Re-run LSTM forward pass to get proper gradients
                def _fwd_step(carry, transition_t):
                    h, c, ca, step_count = carry

                    # Truncated BPTT: detach gap
                    if detach_gap > 0:
                        should_det = (step_count > 0) & (step_count % detach_gap == 0)
                        h = jax.lax.cond(should_det, jax.lax.stop_gradient, lambda x: x, h)
                        c = jax.lax.cond(should_det, jax.lax.stop_gradient, lambda x: x, c)

                    if is_independent:
                        logits, value, (h_new, c_new) = network.apply(
                            params, transition_t.obs, carry=(h, c))
                        talk_logits = None
                    elif has_talk:
                        logits, value, talk_logits, (h_new, c_new) = network.apply(
                            params, transition_t.obs, carry=(h, c), comm_action=ca)
                    else:
                        logits, value, _, (h_new, c_new) = network.apply(
                            params, transition_t.obs, carry=(h, c))
                        talk_logits = None

                    action_dist = distrax.Categorical(logits=logits)
                    log_prob = action_dist.log_prob(transition_t.action)
                    entropy = action_dist.entropy()

                    if has_talk and talk_logits is not None:
                        talk_lp = distrax.Categorical(logits=talk_logits).log_prob(
                            transition_t.talk_action)
                        log_prob = log_prob + talk_lp
                        if comm_action_one:
                            ca = jnp.ones_like(ca)
                        else:
                            ca = transition_t.talk_action

                    # Reset hidden state at episode boundaries
                    done_all = transition_t.done[:, 0]
                    h_new = jnp.where(done_all[:, None, None], jnp.zeros_like(h_new), h_new)
                    c_new = jnp.where(done_all[:, None, None], jnp.zeros_like(c_new), c_new)
                    if has_talk:
                        ca = jnp.where(done_all[:, None], jnp.zeros_like(ca), ca)

                    return (h_new, c_new, ca, step_count + 1), (log_prob, value, entropy)

                init_fwd = (init_h_eval, init_c_eval, init_comm_eval, jnp.int32(0))
                _, (log_probs, values, entropies) = jax.lax.scan(_fwd_step, init_fwd, transitions)

                # Bootstrap last value
                last_obs_batch = _stack_obs(last_obs)
                if is_independent:
                    _, last_val, _ = network.apply(
                        params, last_obs_batch, carry=(hstate, cstate))
                elif has_talk:
                    _, last_val, _, _ = network.apply(
                        params, last_obs_batch, carry=(hstate, cstate), comm_action=comm_action)
                else:
                    _, last_val, _, _ = network.apply(
                        params, last_obs_batch, carry=(hstate, cstate))

            else:
                # Feedforward: vmap re-evaluation
                def _eval_step(transition_t):
                    if has_talk:
                        logits, value, talk_logits = network.apply(
                            params, transition_t.obs, comm_action=transition_t.talk_action)
                    else:
                        logits, value = network.apply(params, transition_t.obs)
                        talk_logits = None
                    action_dist = distrax.Categorical(logits=logits)
                    log_prob = action_dist.log_prob(transition_t.action)
                    entropy = action_dist.entropy()
                    if has_talk and talk_logits is not None:
                        talk_lp = distrax.Categorical(logits=talk_logits).log_prob(
                            transition_t.talk_action)
                        log_prob = log_prob + talk_lp
                    return log_prob, value, entropy

                log_probs, values, entropies = jax.vmap(_eval_step)(transitions)

                last_obs_batch = _stack_obs(last_obs)
                if has_talk:
                    _, last_val, _ = network.apply(
                        params, last_obs_batch, comm_action=comm_action)
                else:
                    _, last_val = network.apply(params, last_obs_batch)

            # Discounted returns
            def _compute_returns(next_value, transition_t):
                ret = transition_t.reward + gamma * next_value * (1 - transition_t.done)
                return ret, ret

            _, returns = jax.lax.scan(_compute_returns, last_val, transitions, reverse=True)

            # Advantages
            advantages = returns - values
            if config.get("NORMALIZE_ADVANTAGES", False):
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # REINFORCE loss
            policy_loss = -jnp.mean(log_probs * jax.lax.stop_gradient(advantages))
            value_loss = jnp.mean((values - jax.lax.stop_gradient(returns)) ** 2)
            entropy_bonus = jnp.mean(entropies)
            loss = policy_loss + value_coeff * value_loss - entropy_coeff * entropy_bonus

            return loss, {
                "loss": loss,
                "policy_loss": policy_loss,
                "value_loss": value_loss,
                "entropy": entropy_bonus,
            }

        (loss, metrics), grads = jax.value_and_grad(_loss_fn, has_aux=True)(
            train_state.params)
        train_state = train_state.apply_gradients(grads=grads)

        metrics["returned_episode_returns"] = env_state.returned_episode_returns[0].mean()
        # Also track mean reward per step from the rollout (useful when episodes span multiple rollouts)
        metrics["mean_reward_per_step"] = transitions.reward.mean()

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
        init_comm_action = jnp.zeros((config["NUM_ENVS"], num_agents), dtype=jnp.int32)
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
                ep_ret = step_metrics.get("returned_episode_returns", 0.0)
                if ep_ret != 0.0:
                    display["EpRet"] = ep_ret
                else:
                    # Episodes don't finish in one rollout; show mean reward/step instead
                    display["MeanRew"] = step_metrics.get("mean_reward_per_step", 0.0)
                display["Loss"] = step_metrics.get("loss", 0.0)
                display["Pi Loss"] = step_metrics.get("policy_loss", 0.0)
                display["V Loss"] = step_metrics.get("value_loss", 0.0)
                display["Entropy"] = step_metrics.get("entropy", 0.0)
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
    if final_return == 0.0:
        final_rew = all_metrics.get("mean_reward_per_step", [0.0])[-1]
        print(f"\nTraining complete! Final mean reward/step: {final_rew:.4f}")
    else:
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
