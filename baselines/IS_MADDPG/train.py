import os
import copy
import time
import pickle
import numpy as np
import jax
import jax.numpy as jnp
import optax
import wandb
from functools import partial
from typing import Dict, Any, Optional

from networks import ISAgentNet, ISCriticNet
from buffer import BufferState, Batch, buffer_init, buffer_add, buffer_sample, buffer_is_ready
from update import TrainState, UpdateMetrics, init_train_state, train_step, polyak_update
from loss import received_messages

try:
    from utils.monitor import TrainingMonitor
    _MONITOR_AVAILABLE = True
except ImportError:
    _MONITOR_AVAILABLE = False


# ---------------------------------------------------------------------------
# Default config  (overridden by Hydra yaml)
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    # Environment
    "ENV_NAME":          "overcooked",
    "ENV_KWARGS":        {},
    "NUM_ENVS":          8,
    "NUM_STEPS":         128,        # steps per env per update cycle
    "TOTAL_TIMESTEPS":   5_000_000,

    # Algorithm
    "ALG_NAME":          "is_maddpg",
    "NUM_AGENTS":        2,
    "OBS_DIM":           96,
    "ACT_DIM":           6,
    "MSG_DIM":           3,
    "HIDDEN_DIM":        128,
    "HORIZON_H":         5,

    # Training
    "ACTOR_LR":          1e-3,
    "CRITIC_LR":         1e-3,
    "GAMMA":             0.99,
    "TAU":               0.005,
    "BATCH_SIZE":        256,
    "BUFFER_SIZE":       100_000,
    "LEARNING_STARTS":   2_000,
    "UPDATE_EVERY":      1,
    "UPDATES_PER_STEP":  1,
    "NUM_EPOCHS":        1,
    "GRAD_CLIP":         10.0,
    "GUMBEL_TAU":        1.0,
    "GUMBEL_HARD":       True,
    "PRED_LOSS_COEF":    0.5,

    # Exploration
    "EPSILON_START":     1.0,
    "EPSILON_END":       0.05,
    "EPSILON_DECAY":     50_000,

    # Logging / saving
    "SEED":              0,
    "LOG_EVERY":         1_000,
    "TEST_INTERVAL":     0.05,       # fraction of total updates
    "SAVE_PATH":         None,
    "WANDB_MODE":        "disabled",
    "WANDB_PROJECT":     "jaxmarl",
    "WANDB_ENTITY":      "",
    "USE_RICH_MONITOR":  True,
}


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(train_state: TrainState, path: str, config: dict) -> None:
    """Save actor/critic params and config to a pickle checkpoint.

    Only params are saved (not optimizer state) to keep files small.
    To resume training you would also need to save opt_state — add that
    if you need warm restarts.

    Args:
        train_state: current TrainState
        path:        file path (should end in .pkl)
        config:      experiment config dict (saved alongside params)
    """
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    payload = {
        "actor_params":         train_state.actor_params,
        "target_actor_params":  train_state.target_actor_params,
        "critic_params":        train_state.critic_params,
        "target_critic_params": train_state.target_critic_params,
        "config":               config,
    }
    with open(path, "wb") as f:
        pickle.dump(payload, f)
    print(f"[IS-MADDPG] Checkpoint saved → {path}")


def load_checkpoint(path: str) -> dict:
    """Load a checkpoint saved by save_checkpoint().

    Returns:
        dict with keys: actor_params, target_actor_params,
                        critic_params, target_critic_params, config
    """
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Env step helpers (vmap-compatible)
# ---------------------------------------------------------------------------

def batchify_obs(obs_dict: dict, agent_ids: list, num_envs: int, obs_dim: int) -> np.ndarray:
    """Convert per-agent obs dict from vectorised env to (num_envs, N, obs_dim).

    JaxMARL vectorised envs return obs as dict[agent_id -> (num_envs, obs_dim)].
    We stack agents on axis=1 for the IS-MADDPG joint format.

    Args:
        obs_dict:  dict mapping agent_id -> (num_envs, obs_dim)
        agent_ids: ordered list of agent ids
        num_envs:  number of parallel environments
        obs_dim:   per-agent observation dimension

    Returns:
        (num_envs, N, obs_dim)
    """
    return np.stack(
        [np.asarray(obs_dict[aid]).reshape(num_envs, obs_dim) for aid in agent_ids],
        axis=1,
    ).astype(np.float32)


def batchify_rewards(rewards_dict: dict, agent_ids: list, num_envs: int) -> np.ndarray:
    """Stack per-agent rewards into (num_envs, N).

    Args:
        rewards_dict: dict mapping agent_id -> (num_envs,)
        agent_ids:    ordered list of agent ids
        num_envs:     number of parallel environments

    Returns:
        (num_envs, N)
    """
    return np.stack(
        [np.asarray(rewards_dict[aid]).reshape(num_envs) for aid in agent_ids],
        axis=1,
    ).astype(np.float32)


def batchify_dones(dones_dict: dict, agent_ids: list, num_envs: int) -> np.ndarray:
    """Compute per-environment done flag (True if ALL agents done).

    Args:
        dones_dict: dict mapping agent_id -> (num_envs,) bool
        agent_ids:  ordered list of agent ids
        num_envs:   number of parallel environments

    Returns:
        (num_envs,) float32 — 1.0 if episode ended, 0.0 otherwise
    """
    per_agent = np.stack(
        [np.asarray(dones_dict[aid]).reshape(num_envs) for aid in agent_ids],
        axis=1,
    )  # (num_envs, N)
    return per_agent.all(axis=1).astype(np.float32)  # (num_envs,)


# ---------------------------------------------------------------------------
# Action selection (outside lax.scan — numpy/JAX boundary)
# ---------------------------------------------------------------------------

def select_actions(
    train_state: TrainState,
    actor:       ISAgentNet,
    obs_all:     np.ndarray,   # (num_envs, N, obs_dim)
    prev_msgs:   np.ndarray,   # (num_envs, N, msg_dim)
    epsilon:     float,
    rng:         Any,
    num_agents:  int,
    act_dim:     int,
    gumbel_tau:  float,
) -> tuple:
    """Select actions for all envs and agents with epsilon-greedy exploration.

    Runs the actor forward pass for each agent across all envs in parallel.
    Exploration uses epsilon-greedy: with probability epsilon, a random
    action is chosen; otherwise the actor's argmax is used.

    Args:
        train_state: current TrainState (contains actor_params)
        actor:       ISAgentNet module
        obs_all:     observations  (num_envs, N, obs_dim)
        prev_msgs:   previous messages (num_envs, N, msg_dim)
        epsilon:     exploration probability
        rng:         JAX PRNG key
        num_agents:  number of agents N
        act_dim:     number of discrete actions
        gumbel_tau:  Gumbel temperature

    Returns:
        actions_onehot: (num_envs, N, act_dim)
        actions_idx:    (num_envs, N) integer actions
        msgs:           (num_envs, N, msg_dim)
        rng:            updated PRNG key
    """
    num_envs = obs_all.shape[0]

    obs_jax      = jnp.array(obs_all)       # (num_envs, N, obs_dim)
    prev_msgs_jax= jnp.array(prev_msgs)     # (num_envs, N, msg_dim)

    # Build received messages: (num_envs, N, N-1, msg_dim)
    received = received_messages(prev_msgs_jax)

    actions_onehot = np.zeros((num_envs, num_agents, act_dim),  dtype=np.float32)
    actions_idx    = np.zeros((num_envs, num_agents),            dtype=np.int32)
    msgs_out       = np.zeros((num_envs, num_agents, prev_msgs.shape[-1]), dtype=np.float32)

    for j in range(num_agents):
        rng, subkey = jax.random.split(rng)

        # Forward pass for agent j across all envs simultaneously
        logits, _, msg, _ = actor.apply(
            train_state.actor_params,
            obs_jax[:, j, :],          # (num_envs, obs_dim)
            received[:, j, :, :],      # (num_envs, N-1, msg_dim)
            rng=subkey,
            gumbel_tau=gumbel_tau,
            gumbel_hard=True,
        )

        # Greedy actions from logits
        greedy_acts = np.array(jnp.argmax(logits, axis=-1))  # (num_envs,)

        # Epsilon-greedy exploration
        rng, eps_key = jax.random.split(rng)
        random_acts  = np.array(
            jax.random.randint(eps_key, (num_envs,), 0, act_dim)
        )
        explore_mask = np.random.random(num_envs) < epsilon
        final_acts   = np.where(explore_mask, random_acts, greedy_acts)

        # Convert to one-hot
        onehot = np.zeros((num_envs, act_dim), dtype=np.float32)
        onehot[np.arange(num_envs), final_acts] = 1.0

        actions_onehot[:, j, :] = onehot
        actions_idx[:, j]       = final_acts
        msgs_out[:, j, :]       = np.array(msg)

    return actions_onehot, actions_idx, msgs_out, rng


# ---------------------------------------------------------------------------
# Greedy evaluation
# ---------------------------------------------------------------------------

def evaluate(
    train_state:  TrainState,
    actor:        ISAgentNet,
    env,
    rng:          Any,
    config:       dict,
    num_episodes: int = 10,
) -> Dict[str, float]:
    """Run greedy rollouts and return mean episode return.

    Uses a single env (not vectorised) for clean episode boundaries.
    Called periodically during training to track policy quality.

    Args:
        train_state:  current TrainState
        actor:        ISAgentNet module
        env:          JaxMARL environment (single, not vectorised)
        rng:          PRNG key
        config:       experiment config
        num_episodes: number of evaluation episodes

    Returns:
        dict with "test_return_mean" and "test_return_std"
    """
    num_agents = config["NUM_AGENTS"]
    msg_dim    = config["MSG_DIM"]
    obs_dim    = config["OBS_DIM"]
    act_dim    = config["ACT_DIM"]
    gumbel_tau = config["GUMBEL_TAU"]

    agent_ids  = sorted(env.agents)
    returns    = []

    for _ in range(num_episodes):
        rng, reset_key = jax.random.split(rng)
        obs_dict, env_state = env.reset(reset_key)

        prev_msgs = np.zeros((1, num_agents, msg_dim), dtype=np.float32)
        ep_return = 0.0
        done      = False

        while not done:
            obs_all = batchify_obs(obs_dict, agent_ids, 1, obs_dim)

            # Greedy: epsilon=0.0
            _, acts_idx, msgs, rng = select_actions(
                train_state, actor, obs_all, prev_msgs,
                epsilon=0.0, rng=rng,
                num_agents=num_agents, act_dim=act_dim,
                gumbel_tau=gumbel_tau,
            )

            # Build action dict for single env (index 0)
            action_dict = {
                aid: int(acts_idx[0, i])
                for i, aid in enumerate(agent_ids)
            }

            rng, step_key = jax.random.split(rng)
            obs_dict, env_state, rewards_dict, dones_dict, _ = env.step(
                step_key, env_state, action_dict
            )

            rewards = batchify_rewards(rewards_dict, agent_ids, 1)
            ep_return += float(rewards.sum())
            done = bool(batchify_dones(dones_dict, agent_ids, 1)[0])
            prev_msgs = msgs

        returns.append(ep_return)

    return {
        "test_return_mean": float(np.mean(returns)),
        "test_return_std":  float(np.std(returns)),
    }


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def make_train(config: dict, env_vec, env_eval, monitor=None):
    """Build and return the training function for IS-MADDPG.

    Separation of concerns:
        env_vec:  vectorised env (num_envs parallel)  — for data collection
        env_eval: single env                          — for evaluation

    The returned train() function:
        1. Initialises networks, optimizer, buffer
        2. Runs a Python for-loop over env steps (buffer boundary)
        3. Inside each step: calls jit-compiled train_step (lax.scan epochs)
        4. Logs to W&B and/or monitor
        5. Saves checkpoints periodically

    Args:
        config:   experiment config dict
        env_vec:  vectorised JaxMARL environment
        env_eval: single JaxMARL environment for evaluation
        monitor:  optional TrainingMonitor instance

    Returns:
        train function that takes a JAX PRNGKey and returns metrics dict
    """

    def train(rng: Any) -> Dict[str, Any]:
        num_agents  = config["NUM_AGENTS"]
        num_envs    = config["NUM_ENVS"]
        obs_dim     = config["OBS_DIM"]
        act_dim     = config["ACT_DIM"]
        msg_dim     = config["MSG_DIM"]
        batch_size  = config["BATCH_SIZE"]
        total_steps = config["TOTAL_TIMESTEPS"]
        log_every   = config["LOG_EVERY"]
        update_every= config["UPDATE_EVERY"]
        updates_per = config["UPDATES_PER_STEP"]
        learn_start = config["LEARNING_STARTS"]
        test_frac   = config["TEST_INTERVAL"]
        num_epochs  = config["NUM_EPOCHS"]

        eps_start   = config["EPSILON_START"]
        eps_end     = config["EPSILON_END"]
        eps_decay   = config["EPSILON_DECAY"]

        agent_ids   = sorted(env_vec.agents)

        # ------------------------------------------------------------------
        # 1. Initialise networks
        # ------------------------------------------------------------------
        actor  = ISAgentNet(
            obs_dim=obs_dim, act_dim=act_dim, msg_dim=msg_dim,
            hidden_dim=config["HIDDEN_DIM"], num_agents=num_agents,
            horizon_H=config["HORIZON_H"],
        )
        critic = ISCriticNet(
            num_agents=num_agents, obs_dim=obs_dim,
            act_dim=act_dim, msg_dim=msg_dim,
            hidden_dim=config["HIDDEN_DIM"],
        )

        rng, init_rng = jax.random.split(rng)
        train_state = init_train_state(
            actor=actor, critic=critic,
            actor_lr=config["ACTOR_LR"], critic_lr=config["CRITIC_LR"],
            obs_dim=obs_dim, num_agents=num_agents,
            msg_dim=msg_dim, act_dim=act_dim,
            batch_size=batch_size, rng=init_rng,
        )

        # ------------------------------------------------------------------
        # 2. Initialise buffer
        # ------------------------------------------------------------------
        buffer_state = buffer_init(
            capacity=config["BUFFER_SIZE"],
            num_agents=num_agents,
            obs_dim=obs_dim,
            act_dim=act_dim,
            msg_dim=msg_dim,
        )

        # ------------------------------------------------------------------
        # 3. JIT-compile train_step (called every update_every steps)
        #    lax.scan over num_epochs gradient updates lives inside here
        # ------------------------------------------------------------------
        @partial(jax.jit, static_argnums=())
        def jit_train_step(state, batch):
            # Scan over NUM_EPOCHS gradient steps per train_step call
            def epoch_step(carry, _):
                s = carry
                s, metrics = train_step(
                    s, batch, actor, critic,
                    gamma=config["GAMMA"],
                    tau=config["TAU"],
                    gumbel_tau=config["GUMBEL_TAU"],
                    gumbel_hard=config["GUMBEL_HARD"],
                    pred_loss_coef=config["PRED_LOSS_COEF"],
                    grad_clip=config["GRAD_CLIP"],
                    num_agents=num_agents,
                    actor_lr=config["ACTOR_LR"],
                    critic_lr=config["CRITIC_LR"],
                )
                return s, metrics

            final_state, all_metrics = jax.lax.scan(
                epoch_step, state, None, length=num_epochs
            )
            # Average metrics over epochs
            avg_metrics = UpdateMetrics(
                critic_loss=jnp.mean(all_metrics.critic_loss),
                actor_loss= jnp.mean(all_metrics.actor_loss),
                pred_loss=  jnp.mean(all_metrics.pred_loss),
                q_mean=     jnp.mean(all_metrics.q_mean),
            )
            return final_state, avg_metrics

        # ------------------------------------------------------------------
        # 4. Reset vectorised env
        # ------------------------------------------------------------------
        rng, reset_rng = jax.random.split(rng)
        reset_rngs = jax.random.split(reset_rng, num_envs)
        obs_dict, env_states = jax.vmap(env_vec.reset)(reset_rngs)

        prev_msgs = np.zeros((num_envs, num_agents, msg_dim), dtype=np.float32)

        # ------------------------------------------------------------------
        # 5. Metrics tracking
        # ------------------------------------------------------------------
        ep_returns  = np.zeros((num_envs, num_agents), dtype=np.float32)
        all_returns = []
        last_metrics: Optional[UpdateMetrics] = None
        total_updates  = 0
        num_updates_target = int(total_steps // (num_envs))
        test_interval  = max(1, int(num_updates_target * test_frac))
        ckpt_dir       = config.get("SAVE_PATH", None)
        t_start        = time.time()

        # ------------------------------------------------------------------
        # 6. Main loop  (Python for-loop — buffer lives here)
        # ------------------------------------------------------------------
        for t in range(1, int(total_steps // num_envs) + 1):

            # --- Epsilon schedule ---
            global_step = t * num_envs
            frac    = min(1.0, global_step / max(1, eps_decay))
            epsilon = eps_start + frac * (eps_end - eps_start)

            # --- Action selection across all envs and agents ---
            obs_all = batchify_obs(obs_dict, agent_ids, num_envs, obs_dim)

            actions_onehot, actions_idx, msgs, rng = select_actions(
                train_state, actor, obs_all, prev_msgs,
                epsilon=epsilon, rng=rng,
                num_agents=num_agents, act_dim=act_dim,
                gumbel_tau=config["GUMBEL_TAU"],
            )

            # --- Step all envs in parallel ---
            # JaxMARL vectorised env: action_dict values are (num_envs,) arrays
            action_dict = {
                aid: jnp.array(actions_idx[:, i])
                for i, aid in enumerate(agent_ids)
            }
            rng, step_rng = jax.random.split(rng)
            step_rngs = jax.random.split(step_rng, num_envs)
            next_obs_dict, env_states, rewards_dict, dones_dict, _ = jax.vmap(
                env_vec.step
            )(step_rngs, env_states, action_dict)

            # --- Convert to numpy arrays ---
            next_obs_all = batchify_obs(next_obs_dict, agent_ids, num_envs, obs_dim)
            rewards_all  = batchify_rewards(rewards_dict, agent_ids, num_envs)
            dones_all    = batchify_dones(dones_dict, agent_ids, num_envs)

            # --- Add each env's transition to the shared buffer ---
            # Sequential loop over envs — numpy buffer can't be vmapped
            for e in range(num_envs):
                buffer_state = buffer_add(
                    buffer_state,
                    obs=           obs_all[e],
                    prev_msgs=     prev_msgs[e],
                    actions=       actions_onehot[e],
                    msgs=          msgs[e],
                    rewards=       rewards_all[e],
                    next_obs=      next_obs_all[e],
                    next_prev_msgs=msgs[e],       # current msgs become next prev_msgs
                    done=          bool(dones_all[e]),
                )

            # Track episode returns
            ep_returns += rewards_all
            for e in range(num_envs):
                if dones_all[e]:
                    all_returns.append(float(ep_returns[e].sum()))
                    ep_returns[e] = 0.0

            obs_dict  = next_obs_dict
            prev_msgs = msgs

            # Reset prev_msgs for done envs
            for e in range(num_envs):
                if dones_all[e]:
                    prev_msgs[e] = 0.0

            # --- Gradient updates ---
            if (buffer_is_ready(buffer_state, batch_size)
                    and global_step >= learn_start
                    and t % update_every == 0):

                for _ in range(updates_per):
                    rng, sample_rng = jax.random.split(rng)
                    batch, rng = buffer_sample(buffer_state, batch_size, rng)
                    train_state, last_metrics = jit_train_step(train_state, batch)
                    total_updates += 1

            # --- Logging ---
            if t % log_every == 0:
                recent = all_returns[-100:] if all_returns else [0.0]
                metrics_dict = {
                    "env_step":    global_step,
                    "update_step": total_updates,
                    "epsilon":     epsilon,
                    "return_mean": float(np.mean(recent)),
                    "return_std":  float(np.std(recent)) if len(recent) > 1 else 0.0,
                    "critic_loss": float(last_metrics.critic_loss) if last_metrics else 0.0,
                    "actor_loss":  float(last_metrics.actor_loss)  if last_metrics else 0.0,
                    "pred_loss":   float(last_metrics.pred_loss)   if last_metrics else 0.0,
                    "q_mean":      float(last_metrics.q_mean)      if last_metrics else 0.0,
                    "steps_per_sec": global_step / max(1.0, time.time() - t_start),
                }

                if monitor is not None:
                    monitor.update(total_updates, metrics_dict)
                else:
                    print(
                        f"[IS-MADDPG] step={global_step:>8d} "
                        f"upd={total_updates:>5d} "
                        f"ret={metrics_dict['return_mean']:>7.2f} "
                        f"c_loss={metrics_dict['critic_loss']:>7.4f} "
                        f"a_loss={metrics_dict['actor_loss']:>7.4f} "
                        f"pred={metrics_dict['pred_loss']:>7.4f} "
                        f"q={metrics_dict['q_mean']:>7.4f} "
                        f"eps={epsilon:.3f}"
                    )

                if config["WANDB_MODE"] != "disabled":
                    wandb.log(metrics_dict, step=global_step)

            # --- Periodic evaluation ---
            if total_updates > 0 and total_updates % test_interval == 0:
                rng, eval_rng = jax.random.split(rng)
                eval_metrics = evaluate(
                    train_state, actor, env_eval, eval_rng, config
                )
                if config["WANDB_MODE"] != "disabled":
                    wandb.log(eval_metrics, step=global_step)
                print(
                    f"[EVAL] step={global_step} "
                    f"test_return={eval_metrics['test_return_mean']:.2f} "
                    f"± {eval_metrics['test_return_std']:.2f}"
                )

            # --- Checkpoint ---
            if (ckpt_dir is not None
                    and total_updates > 0
                    and total_updates % max(1, num_updates_target // 5) == 0):
                ckpt_path = os.path.join(
                    ckpt_dir,
                    f"{config['ALG_NAME']}_{config['ENV_NAME']}"
                    f"_step{global_step}.pkl"
                )
                save_checkpoint(train_state, ckpt_path, config)

        # Final checkpoint
        if ckpt_dir is not None:
            save_checkpoint(
                train_state,
                os.path.join(ckpt_dir, f"{config['ALG_NAME']}_{config['ENV_NAME']}_final.pkl"),
                config,
            )

        return {
            "train_state": train_state,
            "returns":     all_returns,
            "total_updates": total_updates,
        }

    return train


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

# train.py  — replace everything from the @hydra.main decorator to the end of file

# ---------------------------------------------------------------------------
# Entry point (plain argparse — no Hydra dependency)
# ---------------------------------------------------------------------------

def main():
    import argparse
    import wandb

    parser = argparse.ArgumentParser(description="IS-MADDPG generic training entry point")
    parser.add_argument("--env_name",          type=str,   default="overcooked_v3")
    parser.add_argument("--total_timesteps",   type=int,   default=2_000_000)
    parser.add_argument("--num_envs",          type=int,   default=8)
    parser.add_argument("--seed",              type=int,   default=0)
    parser.add_argument("--save_path",         type=str,   default=None)
    parser.add_argument("--wandb",             action="store_true")
    parser.add_argument("--wandb_entity",      type=str,   default="")
    parser.add_argument("--wandb_project",     type=str,   default="jaxmarl")
    args = parser.parse_args()

    config = {
        **DEFAULT_CONFIG,
        "ENV_NAME":        args.env_name,
        "TOTAL_TIMESTEPS": args.total_timesteps,
        "NUM_ENVS":        args.num_envs,
        "SEED":            args.seed,
        "SAVE_PATH":       args.save_path,
        "WANDB_MODE":      "online" if args.wandb else "disabled",
        "WANDB_PROJECT":   args.wandb_project,
        "WANDB_ENTITY":    args.wandb_entity,
    }

    wandb.init(
        project= config["WANDB_PROJECT"],
        entity=  config["WANDB_ENTITY"],
        name=    f"{config['ALG_NAME']}_{config['ENV_NAME']}_seed{config['SEED']}",
        config=  config,
        mode=    config["WANDB_MODE"],
    )

    rng    = jax.random.PRNGKey(config["SEED"])
    import jaxmarl
    env_vec  = jaxmarl.make(config["ENV_NAME"])
    env_eval = jaxmarl.make(config["ENV_NAME"])
    train  = make_train(config, env_vec, env_eval)
    results = train(rng)

    print(f"\nDone. {results['total_updates']} updates.")
    wandb.finish()


if __name__ == "__main__":
    main()