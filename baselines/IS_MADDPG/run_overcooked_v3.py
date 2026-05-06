# run_overcooked_v3.py
"""
Entry point for IS-MADDPG on OvercookedV3 (custom fork).

Usage:
    python run_overcooked_v3.py                          # cramped_room default
    python run_overcooked_v3.py --layout asymmetric_advantages
    python run_overcooked_v3.py --layout coordination_ring --num_envs 16 --wandb
    python run_overcooked_v3.py --total_timesteps 50000  # quick smoke test
"""

import argparse
import os
import time
import numpy as np
import jax
import jax.numpy as jnp

from jaxmarl.environments.overcooked_v3.overcooked import OvercookedV3, State

from networks import ISAgentNet, ISCriticNet
from buffer import buffer_init, buffer_add, buffer_is_ready, buffer_sample
from update import TrainState, UpdateMetrics, init_train_state, train_step
from loss import received_messages
from train import save_checkpoint, DEFAULT_CONFIG


# ---------------------------------------------------------------------------
# Available Layouts
# ---------------------------------------------------------------------------

LAYOUTS = [
    "cramped_room",
    "asymmetric_advantages",
    "coordination_ring",
    "forced_coordination",
    "counter_circuit",
]


# ---------------------------------------------------------------------------
# Env helpers specific to OvercookedV3 API
# ---------------------------------------------------------------------------

def probe_env(layout: str) -> dict:
    """Instantiate OvercookedV3 once to read dims without hardcoding.

    OvercookedV3 obs shape is (height, width, 26 + 5*num_ingredients).
    We flatten it for the MLP actor. The exact size depends on the layout
    grid dimensions so we always read it from the env directly.

    Args:
        layout: layout name string

    Returns:
        dict with obs_dim, act_dim, num_agents, obs_shape
    """
    env = OvercookedV3(layout=layout)
    rng = jax.random.PRNGKey(0)
    obs_dict, _ = env.reset(rng)

    agent_ids  = sorted(env.agents)                       # ["agent_0", "agent_1"]
    num_agents = len(agent_ids)
    sample_obs = obs_dict[agent_ids[0]]                   # (H, W, C)
    obs_shape  = sample_obs.shape
    obs_dim    = int(np.prod(obs_shape))                  # flatten H*W*C

    # OvercookedV3 action space: left, right, up, down, interact, no-op = 6
    act_dim = int(env.action_space(agent_ids[0]).n)

    print(f"\n[OvercookedV3 probe / {layout}]")
    print(f"  num_agents : {num_agents}")
    print(f"  obs_shape  : {obs_shape}  →  obs_dim: {obs_dim}")
    print(f"  act_dim    : {act_dim}")
    print(f"  agent_ids  : {agent_ids}")

    return {
        "obs_dim":    obs_dim,
        "obs_shape":  obs_shape,
        "act_dim":    act_dim,
        "num_agents": num_agents,
        "agent_ids":  agent_ids,
    }


def make_overcooked_config(layout: str, args: argparse.Namespace, env_info: dict) -> dict:
    """Build full experiment config from DEFAULT_CONFIG + env-probed dims + CLI args.

    Args:
        layout:   layout name
        args:     parsed CLI args
        env_info: output of probe_env()

    Returns:
        Complete config dict for make_train() / the manual train loop below
    """
    return {
        **DEFAULT_CONFIG,

        # ── Environment ──────────────────────────────────────────────────
        "ENV_NAME":   "overcooked_v3",
        "LAYOUT":     layout,

        # Read from env — never hardcode
        "NUM_AGENTS": env_info["num_agents"],
        "OBS_DIM":    env_info["obs_dim"],
        "ACT_DIM":    env_info["act_dim"],

        # ── IS-MADDPG hyperparameters ────────────────────────────────────
        # msg_dim=32: lightweight intention signal
        # horizon_H=5: ~one pick-up+place cycle in Overcooked timing
        "MSG_DIM":          16,
        "HORIZON_H":        5,
        "HIDDEN_DIM":       128,
        "ACTOR_LR":         1e-4,
        "CRITIC_LR":        1e-4,
        "GAMMA":            0.90, # lower gamma = smaller Bellman targets = more stable
        "TAU":              0.01,
        "GRAD_CLIP":        0.5, # tight clip
        "GUMBEL_TAU":       1.0,
        "GUMBEL_HARD":      True,
        "PRED_LOSS_COEF":   0.05,

        # ── Training schedule ────────────────────────────────────────────
        "TOTAL_TIMESTEPS":  args.total_timesteps,
        "NUM_ENVS":         args.num_envs,
        "BATCH_SIZE":       512,
        "BUFFER_SIZE":      100_000,
        "LEARNING_STARTS":  5_000,
        "UPDATE_EVERY":     1,
        "UPDATES_PER_STEP": 1,
        "NUM_EPOCHS":       1,

        # ── Exploration ──────────────────────────────────────────────────
        # Decay epsilon over first 30% of training — Overcooked is dense
        # reward so the policy picks up signal quickly
        "EPSILON_START":    1.0,
        "EPSILON_END":      0.1, # higher minimum epsilon — don't go fully greedy
        "EPSILON_DECAY":    int(args.total_timesteps * 0.7),

        # ── Logging / saving ─────────────────────────────────────────────
        "SEED":             args.seed,
        "LOG_EVERY":        1_000,
        "TEST_INTERVAL":    0.05,
        "SAVE_PATH":        args.save_path,
        "WANDB_MODE":       "online" if args.wandb else "disabled",
        "WANDB_PROJECT":    "is-maddpg-overcooked-v3",
        "WANDB_ENTITY":     args.wandb_entity,
        "USE_RICH_MONITOR": True,
        "ALG_NAME":         "is_maddpg",
    }


# ---------------------------------------------------------------------------
# OvercookedV3-specific data conversion
# The env returns shared reward (same scalar for all agents) and
# obs as (H, W, C) arrays — both need reshaping for the buffer.
# ---------------------------------------------------------------------------

def obs_dict_to_array(obs_dict: dict, agent_ids: list,
                      num_envs: int, obs_dim: int) -> np.ndarray:
    """Stack per-agent obs into (num_envs, N, obs_dim).

    OvercookedV3 reset/step returns:
        obs_dict[agent_id] : (num_envs, H, W, C)   when vmapped
        obs_dict[agent_id] : (H, W, C)              for single env

    We flatten spatial dims into obs_dim for the MLP actor.

    Args:
        obs_dict:  raw obs dict from env
        agent_ids: sorted agent id list
        num_envs:  number of parallel envs (1 for eval)
        obs_dim:   H * W * C (probed at init)

    Returns:
        (num_envs, N, obs_dim) float32
    """
    return np.stack(
        [
            np.asarray(obs_dict[aid]).reshape(num_envs, obs_dim)
            for aid in agent_ids
        ],
        axis=1,
    ).astype(np.float32)


def rewards_dict_to_array(rewards_dict: dict, agent_ids: list,
                          num_envs: int) -> np.ndarray:
    """Stack rewards into (num_envs, N).

    OvercookedV3 returns the same shared scalar for all agents.
    We broadcast it across the agent axis so the buffer format is
    consistent — the critic indexes rewards[:, agent_idx] per agent.

    Args:
        rewards_dict: dict agent_id -> (num_envs,) or scalar
        agent_ids:    sorted agent id list
        num_envs:     number of parallel envs

    Returns:
        (num_envs, N) float32
    """
    return np.stack(
        [
            np.asarray(rewards_dict[aid]).reshape(num_envs)
            for aid in agent_ids
        ],
        axis=1,
    ).astype(np.float32)


def dones_dict_to_array(dones_dict: dict, agent_ids: list,
                        num_envs: int) -> np.ndarray:
    """Compute per-env done flag as float32 (1.0 = done).

    Episode ends when __all__ is True (all agents done simultaneously
    in Overcooked since it's a cooperative task with shared termination).

    Args:
        dones_dict: dict with agent keys + "__all__"
        agent_ids:  sorted agent id list
        num_envs:   number of parallel envs

    Returns:
        (num_envs,) float32
    """

    # episode done if all agents are done
    per_agent = np.stack(
        [np.asarray(dones_dict[aid]).reshape(num_envs) for aid in agent_ids],
        axis=1,
    )
    return per_agent.all(axis=1).astype(np.float32)


# ---------------------------------------------------------------------------
# Action selection
# ---------------------------------------------------------------------------

def select_actions(
    train_state: TrainState,
    actor:       ISAgentNet,
    obs_all:     np.ndarray,    # (num_envs, N, obs_dim)
    prev_msgs:   np.ndarray,    # (num_envs, N, msg_dim)
    epsilon:     float,
    rng,
    *,
    num_agents:  int,
    act_dim:     int,
    gumbel_tau:  float,
) -> tuple:
    """Epsilon-greedy action selection for all envs and agents.

    Runs the IS-MADDPG actor for each agent across all envs simultaneously.
    With probability epsilon picks a random action (exploration); otherwise
    uses the actor's argmax (exploitation).

    Args:
        train_state: current TrainState (actor_params used)
        actor:       ISAgentNet module
        obs_all:     (num_envs, N, obs_dim)
        prev_msgs:   (num_envs, N, msg_dim)
        epsilon:     exploration probability
        rng:         JAX PRNG key
        num_agents:  N
        act_dim:     number of discrete actions
        gumbel_tau:  temperature for actor's Gumbel sampling

    Returns:
        actions_onehot: (num_envs, N, act_dim)  one-hot for buffer
        actions_idx:    (num_envs, N)            int for env.step
        msgs_out:       (num_envs, N, msg_dim)
        rng:            updated key
    """
    num_envs = obs_all.shape[0]
    msg_dim  = prev_msgs.shape[-1]

    obs_jax       = jnp.array(obs_all)
    prev_msgs_jax = jnp.array(prev_msgs)

    # (num_envs, N, N-1, msg_dim)
    received = received_messages(prev_msgs_jax)

    actions_onehot = np.zeros((num_envs, num_agents, act_dim),  dtype=np.float32)
    actions_idx    = np.zeros((num_envs, num_agents),            dtype=np.int32)
    msgs_out       = np.zeros((num_envs, num_agents, msg_dim),   dtype=np.float32)

    for j in range(num_agents):
        rng, subkey = jax.random.split(rng)

        logits, _, msg, _ = actor.apply(
            train_state.actor_params,
            obs_jax[:, j, :],        # (num_envs, obs_dim)
            received[:, j, :, :],    # (num_envs, N-1, msg_dim)
            rng=subkey,
            gumbel_tau=gumbel_tau,
            gumbel_hard=True,
        )

        greedy_acts = np.array(jnp.argmax(logits, axis=-1))   # (num_envs,)

        rng, eps_key = jax.random.split(rng)
        random_acts  = np.array(
            jax.random.randint(eps_key, (num_envs,), 0, act_dim)
        )
        explore = np.random.random(num_envs) < epsilon
        final_acts = np.where(explore, random_acts, greedy_acts)

        onehot = np.zeros((num_envs, act_dim), dtype=np.float32)
        onehot[np.arange(num_envs), final_acts] = 1.0

        actions_onehot[:, j, :] = onehot
        actions_idx[:, j]       = final_acts
        msgs_out[:, j, :]       = np.array(msg)

    return actions_onehot, actions_idx, msgs_out, rng


# ---------------------------------------------------------------------------
# Greedy evaluation (single env, no exploration)
# ---------------------------------------------------------------------------

def evaluate(
    train_state:  TrainState,
    actor:        ISAgentNet,
    env:          OvercookedV3,
    rng,
    *,
    config:       dict,
    num_episodes: int = 10,
) -> dict:
    """Run greedy rollouts on a single env and return mean episode return.

    Uses a single (non-vmapped) env for clean episode boundaries.
    Called periodically during training to track policy quality
    without epsilon noise contaminating the measurement.

    Args:
        train_state:  current TrainState
        actor:        ISAgentNet module
        env:          single OvercookedV3 instance (not vectorised)
        rng:          PRNG key
        config:       experiment config dict
        num_episodes: how many episodes to average over

    Returns:
        dict with "test_return_mean" and "test_return_std"
    """
    num_agents = config["NUM_AGENTS"]
    msg_dim    = config["MSG_DIM"]
    obs_dim    = config["OBS_DIM"]
    act_dim    = config["ACT_DIM"]
    gumbel_tau = config["GUMBEL_TAU"]
    agent_ids  = [f"agent_{i}" for i in range(num_agents)]

    returns = []

    for _ in range(num_episodes):
        rng, reset_key = jax.random.split(rng)
        obs_dict, env_state = env.reset(reset_key)

        # Single env: add batch dim of 1 for select_actions compatibility
        prev_msgs = np.zeros((1, num_agents, msg_dim), dtype=np.float32)
        ep_return = 0.0
        max_steps = getattr(env, 'max_steps', 400)

        for _step in range(max_steps):   # bounded — never hangs
            obs_all = obs_dict_to_array(obs_dict, agent_ids, num_envs=1, obs_dim=obs_dim)

            _, acts_idx, msgs, rng = select_actions(
                train_state, actor, obs_all, prev_msgs,
                epsilon=0.0,   # greedy — no exploration
                rng=rng,
                num_agents=num_agents,
                act_dim=act_dim,
                gumbel_tau=gumbel_tau,
            )

            # step_env expects scalar int actions per agent
            action_dict = {
                f"agent_{i}": int(acts_idx[0, i])
                for i in range(num_agents)
            }

            rng, step_key = jax.random.split(rng)
            obs_dict, env_state, rewards_dict, dones_dict, _ = env.step_env(
                step_key, env_state, action_dict
            )

            rewards = rewards_dict_to_array(rewards_dict, agent_ids, num_envs=1)
            ep_return += float(rewards.sum())

            dones = dones_dict_to_array(dones_dict, agent_ids, num_envs=1)
            if bool(dones[0]):
                break

            prev_msgs = msgs

        returns.append(ep_return)

    return {
        "test_return_mean": float(np.mean(returns)),
        "test_return_std":  float(np.std(returns)),
    }


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def run(config: dict, env_vec: OvercookedV3,
        monitor=None) -> dict:
    """IS-MADDPG training loop for OvercookedV3.

    Structure:
        Python for-loop (env steps + buffer adds)  ← numpy boundary
            ├── jax.vmap(env.step_env)              ← parallel envs
            ├── buffer_add() × num_envs             ← numpy, sequential
            └── jit(train_step) when ready          ← fully compiled
                    └── lax.scan over NUM_EPOCHS

    Args:
        config:   experiment config dict
        env_vec:  OvercookedV3 instance (will be vmapped)
        env_eval: OvercookedV3 instance for greedy eval (single env)
        monitor:  optional TrainingMonitor

    Returns:
        dict with train_state, returns, total_updates
    """
    import wandb
    from functools import partial

    num_agents  = config["NUM_AGENTS"]
    num_envs    = config["NUM_ENVS"]
    obs_dim     = config["OBS_DIM"]
    act_dim     = config["ACT_DIM"]
    msg_dim     = config["MSG_DIM"]
    batch_size  = config["BATCH_SIZE"]
    learn_start = config["LEARNING_STARTS"]
    update_every= config["UPDATE_EVERY"]
    updates_per = config["UPDATES_PER_STEP"]
    num_epochs  = config["NUM_EPOCHS"]
    log_every   = config["LOG_EVERY"]
    eps_decay   = config["EPSILON_DECAY"]
    eps_start   = config["EPSILON_START"]
    eps_end     = config["EPSILON_END"]
    ckpt_dir    = config.get("SAVE_PATH", None)
    agent_ids   = [f"agent_{i}" for i in range(num_agents)]

    rng = jax.random.PRNGKey(config["SEED"])

    # ------------------------------------------------------------------
    # 1. Networks
    # ------------------------------------------------------------------
    actor = ISAgentNet(
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
        actor_lr=config["ACTOR_LR"],
        critic_lr=config["CRITIC_LR"],
        grad_clip=config["GRAD_CLIP"],
        obs_dim=obs_dim, num_agents=num_agents,
        msg_dim=msg_dim, act_dim=act_dim,
        batch_size=batch_size, rng=init_rng,
    )

    jit_step = jax.jit(jax.vmap(env_vec.step_env))
    jit_reset = jax.jit(jax.vmap(env_vec.reset))

    # ------------------------------------------------------------------
    # 2. Buffer
    # ------------------------------------------------------------------
    buffer_state = buffer_init(
        capacity=config["BUFFER_SIZE"],
        num_agents=num_agents,
        obs_dim=obs_dim,
        act_dim=act_dim,
        msg_dim=msg_dim,
    )

    # ------------------------------------------------------------------
    # 3. JIT-compile train_step with lax.scan over epochs
    # ------------------------------------------------------------------
    @jax.jit
    def jit_train_step(state, batch):
        """One call = NUM_EPOCHS gradient updates via lax.scan."""
        def epoch_step(carry, _):
            s, metrics = train_step(
                carry, batch, actor, critic,
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
        return final_state, UpdateMetrics(
            critic_loss=jnp.mean(all_metrics.critic_loss),
            actor_loss= jnp.mean(all_metrics.actor_loss),
            pred_loss=  jnp.mean(all_metrics.pred_loss),
            q_mean=     jnp.mean(all_metrics.q_mean),
        )

    # ------------------------------------------------------------------
    # 4. Reset vectorised envs
    #    jax.vmap(env.reset) requires splitting one key per env
    # ------------------------------------------------------------------
    rng, reset_rng = jax.random.split(rng)
    reset_rngs = jax.random.split(reset_rng, num_envs)
    obs_dict, env_states = jit_reset(reset_rngs)

    prev_msgs = np.zeros((num_envs, num_agents, msg_dim), dtype=np.float32)

    # ------------------------------------------------------------------
    # 5. Metrics
    # ------------------------------------------------------------------
    ep_returns   = np.zeros((num_envs, num_agents), dtype=np.float32)
    all_returns  = []
    all_deliveries = []
    ep_deliveries  = np.zeros((num_envs,), dtype=np.float32)
    last_metrics = None
    total_updates = 0
    total_steps_target = config["TOTAL_TIMESTEPS"] // num_envs

    # TEST_INTERVAL is a fraction of total_timesteps e.g. 0.05 = every 5%.
    test_interval_steps  = max(1, int(config["TOTAL_TIMESTEPS"] * config["TEST_INTERVAL"]))
    
    # Convert to loop iterations (how many for-loop steps between evals)
    test_interval = max(1, test_interval_steps // num_envs)    
    t_start = time.time()

    reward_type_counts = {
    "placement_in_pot": 0,   # shaped reward = 3
    # "pot_start_cooking": 0,  # shaped reward = 4
    "soup_in_dish":      0,  # shaped reward = 6
    "plate_pickup":      0,  # shaped reward = 2
    "delivery":          0,  # raw reward = 20
    # "burn_penalty":      0,  # raw reward = -5
    }

    # Track rewards
    reward_type_history = {k: [] for k in reward_type_counts}  # per-episode counts
    ep_reward_types = {k: np.zeros(num_envs) for k in reward_type_counts}

    print(f"\n[IS-MADDPG] Starting training on OvercookedV3 / {config['LAYOUT']}")
    print(f"  total_timesteps : {config['TOTAL_TIMESTEPS']:,}")
    print(f"  num_envs        : {num_envs}")
    print(f"  obs_dim         : {obs_dim}")
    print(f"  act_dim         : {act_dim}")
    print(f"  msg_dim         : {msg_dim}\n")

    print("JAX devices:", jax.devices())
    print("Default backend:", jax.default_backend())

    # ------------------------------------------------------------------
    # 6. Main loop — Python for-loop owns the numpy buffer boundary
    # ------------------------------------------------------------------

    first_update_done = False
    compile_start = None

    last_eval_step = 0
    eval_interval_steps = max(1, int(config["TOTAL_TIMESTEPS"] * config["TEST_INTERVAL"]))    

    for t in range(1, total_steps_target + 1):
        global_step = t * num_envs 

        # ── Epsilon schedule ─────────────────────────────────────────
        frac    = min(1.0, global_step / max(1, eps_decay))
        epsilon = eps_start + frac * (eps_end - eps_start)

        # ── Obs ──────────────────────────────────────────────────────
        obs_all = obs_dict_to_array(obs_dict, agent_ids, num_envs, obs_dim)

        # ── Action selection ─────────────────────────────────────────
        actions_onehot, actions_idx, msgs, rng = select_actions(
            train_state, actor, obs_all, prev_msgs,
            epsilon=epsilon, rng=rng,
            num_agents=num_agents, act_dim=act_dim,
            gumbel_tau=config["GUMBEL_TAU"],
        )

        # ── Step envs ────────────────────────────────────────────────
        rng, step_rng = jax.random.split(rng)
        step_rngs = jax.random.split(step_rng, num_envs)
        action_dict = {
            f"agent_{i}": jnp.array(actions_idx[:, i])
            for i in range(num_agents)
        }
        next_obs_dict, env_states, rewards_dict, dones_dict, info = jit_step(
        step_rngs, env_states, action_dict
        )
        # check rewards used
        if t == 1:
            print("rewards_dict sample:")
            for k, v in rewards_dict.items():
                print(f"  {k}: {jnp.array(v)[:3]}")
            print("info keys:", list(info.keys()))
            if "shaped_reward" in info:
                print("shaped_rewards sample:", {k: jnp.array(v)[:3] for k, v in info["shaped_reward"].items()})

        # ── Convert ───────────────────────────────────────────────────
        next_obs_all = obs_dict_to_array(next_obs_dict, agent_ids, num_envs, obs_dim)
        dones_all = dones_dict_to_array(dones_dict, agent_ids, num_envs)

        # Use shaped rewards for training signal, sparse for logging
        if config.get("USE_SHAPED_REWARDS", True) and "shaped_reward" in info:
            shaped = info["shaped_reward"]
            # Combine: sparse delivery + shaped intermediate rewards
            combined_rewards = {
                aid: jnp.array(rewards_dict[aid]) + jnp.array(shaped[aid])
                for aid in agent_ids
            }
            rewards_all = rewards_dict_to_array(combined_rewards, agent_ids, num_envs)
            ep_deliveries += (rewards_all > 0).any(axis=1).astype(np.float32)
        else:
            rewards_all = rewards_dict_to_array(rewards_dict, agent_ids, num_envs)            

        # ── Buffer ───────────────────────────────────────────────────
        for e in range(num_envs):
            buffer_state = buffer_add(
                buffer_state,
                obs=           obs_all[e],
                prev_msgs=     prev_msgs[e],
                actions=       actions_onehot[e],
                msgs=          msgs[e],
                rewards=       rewards_all[e],
                next_obs=      next_obs_all[e],
                next_prev_msgs=msgs[e],
                done=          bool(dones_all[e]),
            )

        # ── Reward type tracking ──────────────────────────────────────
        if "shaped_reward" in info:
            shaped = info["shaped_reward"]
            for e in range(num_envs):
                for aid in agent_ids:
                    sv = float(jnp.array(shaped[aid])[e])
                    rv = float(jnp.array(rewards_dict[aid])[e])

                    # Infer reward type from value
                    if sv == 6.0:
                        ep_reward_types["soup_in_dish"][e] += 1
                    # elif sv == 4.0:
                    #     ep_reward_types["pot_start_cooking"][e] += 1
                    elif sv == 3.0:
                        ep_reward_types["placement_in_pot"][e] += 1
                    elif sv == 2.0:
                        ep_reward_types["plate_pickup"][e] += 1

                    if rv == 20.0:
                        ep_reward_types["delivery"][e] += 1
                    # elif rv == -5.0:
                    #     ep_reward_types["burn_penalty"][e] += 1            

        # ── Episode tracking ─────────────────────────────────────────
        ep_returns += rewards_all
        for e in range(num_envs):
            if dones_all[e]:
                all_returns.append(float(ep_returns[e].sum()))
                all_deliveries.append(float(ep_deliveries[e]))  # deliveries this episode
                ep_returns[e]   = 0.0
                ep_deliveries[e] = 0.0
                for k in reward_type_counts:
                    reward_type_history[k].append(float(ep_reward_types[k][e]))
                    reward_type_counts[k] += ep_reward_types[k][e]
                    ep_reward_types[k][e] = 0.0

        # ── Auto-reset done envs ──────────────────────────────────────
        # JaxMARL does NOT auto-reset — when an env is done it stays
        # in terminal state returning done=True every subsequent step
        # until manually reset. This is why dones_all stays [1,1,1,1].
        if jnp.any(dones_all):
            rng, reset_rng = jax.random.split(rng)
            reset_rngs = jax.random.split(reset_rng, num_envs)

            # Reset ALL envs that are done
            new_obs, new_states = jax.vmap(env_vec.reset)(reset_rngs)

            # Only replace done envs — keep running envs as-is
            done_mask = jnp.array(dones_all, dtype=jnp.bool_)  # (num_envs,)

            # Merge: use new state for done envs, keep old state for running envs
            env_states = jax.tree_util.tree_map(
                lambda new, old: jnp.where(
                    done_mask.reshape([-1] + [1] * (new.ndim - 1)),
                    new,
                    old,
                ),
                new_states,
                env_states,
            )

            # Update obs for done envs
            for aid in agent_ids:
                obs_dict[aid] = jnp.where(
                    done_mask.reshape([-1] + [1] * (jnp.array(obs_dict[aid]).ndim - 1)),
                    new_obs[aid],
                    obs_dict[aid],
                )   

        # ── Advance state ─────────────────────────────────────────────
        obs_dict  = next_obs_dict   # contains reset obs for done envs
        prev_msgs = msgs
        for e in range(num_envs):
            if dones_all[e]:
                prev_msgs[e] = 0.0

        # ── Gradient updates ─────────────────────────────────────────
        if (buffer_is_ready(buffer_state, batch_size)
                and global_step >= learn_start
                and t % update_every == 0):

            for _ in range(updates_per):
                # Warn once that JIT compilation is about to happen
                if not first_update_done:
                    print(
                        f"  [step={global_step:,}] Buffer ready — "
                        f"JIT compiling train_step (may take 1-3 min)..."
                    )
                    compile_start = time.time()

                batch, rng = buffer_sample(buffer_state, batch_size, rng)
                train_state, last_metrics = jit_train_step(train_state, batch)

                if not first_update_done:
                    # block_until_ready forces JAX to finish compilation
                    # before we print the compile time
                    jax.block_until_ready(train_state.actor_params)
                    compile_secs = time.time() - compile_start
                    print(f"  [JIT done] Compilation took {compile_secs:.1f}s — training now running.")
                    first_update_done = True

                total_updates += 1

                # NaN guard — stop immediately with diagnostics
                if jnp.isnan(last_metrics.critic_loss):
                    print(
                        f"\n[NaN detected at step={global_step}, update={total_updates}]"
                        f"\n  critic_loss : {last_metrics.critic_loss}"
                        f"\n  actor_loss  : {last_metrics.actor_loss}"
                        f"\n  pred_loss   : {last_metrics.pred_loss}"
                        f"\n  q_mean      : {last_metrics.q_mean}"
                        f"\n  Check: learning rates too high, grad_clip too loose,"
                        f"\n         or reward scale mismatch."
                    )
                    raise ValueError("NaN in training metrics — see above for diagnostics.")                

        # ── Progress every 100 steps before first log_every ──────────
        # Shows the loop is alive during buffer fill and compilation
        if t % 100 == 0 and total_updates == 0:
            buf_pct = 100.0 * buffer_state.size / config["BUFFER_SIZE"]
            print(
                f"  [step={global_step:>7,}] filling buffer "
                f"{buffer_state.size:>6,}/{config['BUFFER_SIZE']:,} "
                f"({buf_pct:.1f}%)  eps={epsilon:.3f}",
                flush=True,
            )

        # ── Logging every log_every steps ────────────────────────────
        if t % log_every == 0 and first_update_done:
            recent = all_returns[-100:] if all_returns else [0.0]
            recent_deliveries = all_deliveries[-100:] if all_deliveries else [0.0]
            sps    = global_step / max(1.0, time.time() - t_start)

            metrics_log = {
                "env_step":    global_step,
                "update_step": total_updates,
                "epsilon":     epsilon,
                "return_mean": float(np.mean(recent)),
                "return_std":  float(np.std(recent)) if len(recent) > 1 else 0.0,
                "critic_loss": float(last_metrics.critic_loss) if last_metrics else 0.0,
                "actor_loss":  float(last_metrics.actor_loss)  if last_metrics else 0.0,
                "pred_loss":   float(last_metrics.pred_loss)   if last_metrics else 0.0,
                "q_mean":      float(last_metrics.q_mean)      if last_metrics else 0.0,
                "steps_per_sec": sps,
            }

            if monitor is not None:
                monitor.update(total_updates, metrics_log)
            else:
                print(
                    f"  step={global_step:>8,} "
                    f"upd={total_updates:>5d} "
                    f"episodes={len(all_returns):>4d} "
                    f"ret={metrics_log['return_mean']:>7.2f}±{metrics_log['return_std']:.2f} "
                    f"deliveries={np.mean(recent_deliveries):.2f} "  # avg deliveries per episode
                    f"c_loss={metrics_log['critic_loss']:>7.4f} "
                    f"a_loss={metrics_log['actor_loss']:>7.4f} "
                    f"pred={metrics_log['pred_loss']:>7.4f} "
                    f"q={metrics_log['q_mean']:>7.4f} "
                    f"eps={epsilon:.3f} "
                    f"sps={sps:>6.0f}",
                    flush=True,
                )

            if config["WANDB_MODE"] != "disabled":
                wandb.log(metrics_log, step=global_step)

        # ── Evaluation ───────────────────────────────────────────────
        # if (total_updates > 0
        #         and global_step - last_eval_step >= eval_interval_steps):
        #     last_eval_step = global_step            
        #     rng, eval_rng = jax.random.split(rng)
        #     eval_metrics = evaluate(
        #         train_state, actor, env_eval,
        #         eval_rng, config=config, num_episodes=5,
        #     )
        #     print(
        #         f"  [EVAL] step={global_step:,} "
        #         f"test_return={eval_metrics['test_return_mean']:.2f}"
        #         f" ± {eval_metrics['test_return_std']:.2f}",
        #         flush=True,
        #     )
        #     if config["WANDB_MODE"] != "disabled":
        #         wandb.log(eval_metrics, step=global_step)

        # ── Checkpoint ───────────────────────────────────────────────
        # if (ckpt_dir is not None and total_updates > 0
        #         and total_updates % max(1, total_steps_target // 5) == 0):
        #     ckpt_path = os.path.join(
        #         ckpt_dir,
        #         f"is_maddpg_{config['LAYOUT']}_step{global_step}.pkl"
        #     )
        #     save_checkpoint(train_state, ckpt_path, config)

    
    # ── Post-training summary + plots ───────────────────────────────────────────────
    import matplotlib.pyplot as plt

    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    print(f"  Total env steps      : {config['TOTAL_TIMESTEPS']:,}")
    print(f"  Total grad updates   : {total_updates:,}")
    print(f"  Episodes completed   : {len(all_returns)}")
    if all_returns:
        print(f"  Best episode return  : {max(all_returns):.2f}")
        print(f"  Final 100-ep mean   : {np.mean(all_returns[-100:]):.2f}")
        print(f"  Final 100-ep std    : {np.std(all_returns[-100:]):.2f}")
    print("="*60)

    if all_returns:
        plot_dir = ckpt_dir if ckpt_dir else "."
        os.makedirs(plot_dir, exist_ok=True)

        fig, axes = plt.subplots(1, 2, figsize=(14, 4))

        # --- Episode returns ---
        ax = axes[0]
        ax.plot(all_returns, alpha=0.3, color="steelblue", label="raw")
        # Smooth with a rolling window
        window = min(50, len(all_returns))
        if len(all_returns) >= window:
            smoothed = np.convolve(
                all_returns,
                np.ones(window) / window,
                mode="valid",
            )
            ax.plot(
                range(window - 1, len(all_returns)),
                smoothed,
                color="steelblue",
                linewidth=2,
                label=f"{window}-ep moving avg",
            )
        ax.set_xlabel("Episode")
        ax.set_ylabel("Return")
        ax.set_title(f"IS-MADDPG — OvercookedV3 / {config['LAYOUT']}")
        ax.legend()
        ax.grid(alpha=0.3)

        # --- Steps per episode (proxy for episode length / efficiency) ---
        ax = axes[1]
        ax.hist(all_returns, bins=30, color="steelblue", alpha=0.7, edgecolor="white")
        ax.set_xlabel("Episode Return")
        ax.set_ylabel("Count")
        ax.set_title("Return Distribution (all episodes)")
        ax.grid(alpha=0.3)

        plt.tight_layout()
        plot_path = os.path.join(plot_dir, f"is_maddpg_{config['LAYOUT']}_returns.png")
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        print(f"\nPlot saved → {plot_path}")
        plt.show()
    
    # ---------------------------------------------------------------------------
    # Reward type histograms
    # ---------------------------------------------------------------------------
    if any(len(v) > 0 for v in reward_type_history.values()):
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        axes = axes.flatten()

        reward_labels = {
            "delivery":          f"Delivery (+20)",
            # "placement_in_pot":  f"Placement in Pot (+3)",
            "plate_pickup":      f"Plate Pickup (+2)",
            "pot_start_cooking": f"Pot Start Cooking (+4)",
            "soup_in_dish":      f"Soup in Dish (+6)",
            # "burn_penalty":      f"Burn Penalty (-5)",
        }
        colors = {
            "delivery":          "green",
            # "placement_in_pot":  "steelblue",
            "plate_pickup":      "yellow",
            "pot_start_cooking": "orange",
            "soup_in_dish":      "purple",
            # "burn_penalty":      "red",
        }

        for idx, (key, label) in enumerate(reward_labels.items()):
            ax = axes[idx]
            data = reward_type_history[key]
            if data:
                ax.hist(data, bins=20, color=colors[key], alpha=0.7, edgecolor="white")
                ax.set_title(label)
                ax.set_xlabel("Count per episode")
                ax.set_ylabel("Episodes")
                ax.grid(alpha=0.3)
                ax.axvline(np.mean(data), color="black", linestyle="--",
                           linewidth=1.5, label=f"mean={np.mean(data):.2f}")
                ax.legend(fontsize=8)
            else:
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=ax.transAxes)
                ax.set_title(label)

        # Summary bar chart
        summary_idx = len(reward_labels)
        ax = axes[summary_idx]

        # turn off any remaining unused axes
        for i in range(summary_idx + 1, len(axes)):
            axes[i].axis("off")
            
        keys   = list(reward_labels.keys())
        totals = [reward_type_counts[k] for k in keys]
        bars   = ax.bar(range(len(keys)), totals,
                        color=[colors[k] for k in keys], alpha=0.7, edgecolor="white")
        ax.set_xticks(range(len(keys)))
        ax.set_xticklabels([k.replace("_", "\n") for k in keys], fontsize=7)
        ax.set_title("Total reward events (all episodes)")
        ax.set_ylabel("Count")
        ax.grid(alpha=0.3, axis="y")
        for bar, total in zip(bars, totals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f"{int(total)}", ha="center", va="bottom", fontsize=8)

        plt.suptitle(f"IS-MADDPG Reward Breakdown — {config['LAYOUT']}", fontsize=13)
        plt.tight_layout()

        hist_path = os.path.join(
            ckpt_dir if ckpt_dir else ".",
            f"is_maddpg_{config['LAYOUT']}_reward_breakdown.png",
        )
        os.makedirs(os.path.dirname(hist_path) if os.path.dirname(hist_path) else ".", exist_ok=True)
        plt.savefig(hist_path, dpi=150, bbox_inches="tight")
        print(f"Reward breakdown plot saved → {hist_path}")
        plt.show()

    # Print summary table
    print("\nReward event totals across all episodes:")
    print(f"  {'Event':<25} {'Total':>8}  {'Per Episode':>12}")
    print("  " + "-" * 48)
    n_eps = max(1, len(all_returns))
    for k, label in reward_labels.items():
        total = reward_type_counts[k]
        per_ep = total / n_eps
        print(f"  {label:<25} {int(total):>8}  {per_ep:>11.2f}")

    return {
        "train_state":   train_state,
        "returns":       all_returns,
        "total_updates": total_updates,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="IS-MADDPG on OvercookedV3")
    parser.add_argument(
        "--layout", type=str, default="cramped_room", choices=LAYOUTS,
        help="Overcooked layout"
    )
    parser.add_argument(
        "--total_timesteps", type=int, default=500_000,
        help="Total environment steps"
    )
    parser.add_argument(
        "--num_envs", type=int, default=6,
        help="Number of parallel environments"
    )
    parser.add_argument(
        "--seed", type=int, default=4,
        help="Random seed"
    )
    parser.add_argument(
        "--save_path", type=str, default="results",
        help="Results directory (None to disable)"
    )
    parser.add_argument(
        "--wandb", action="store_true",
        help="Enable W&B logging"
    )
    parser.add_argument(
        "--wandb_entity", type=str, default="",
        help="W&B entity"
    )
    args = parser.parse_args()

    # ── Probe env for dims ────────────────────────────────────────────
    env_info = probe_env(args.layout)
    config   = make_overcooked_config(args.layout, args, env_info)

    # ── W&B ──────────────────────────────────────────────────────────
    import wandb
    wandb.init(
        project= "is-maddpg-overcooked-v3",       # your project name
        entity=  config["WANDB_ENTITY"],           # passed via --wandb_entity
        name=    f"is_maddpg_{args.layout}_seed{args.seed}",
        config=  {k: v for k, v in config.items() if k != "ENV_KWARGS"},
        mode=    config["WANDB_MODE"],             # "online" when --wandb is set
    )

    # ── Monitor ──────────────────────────────────────────────────────
    monitor = None
    try:
        from utils.monitor import TrainingMonitor
        num_updates = config["TOTAL_TIMESTEPS"] // config["NUM_ENVS"]
        monitor = TrainingMonitor(
            total_updates=num_updates,
            config_dict={
                "layout":    args.layout,
                "num_envs":  args.num_envs,
                "obs_dim":   config["OBS_DIM"],
                "act_dim":   config["ACT_DIM"],
                "msg_dim":   config["MSG_DIM"],
                "horizon_H": config["HORIZON_H"],
                "actor_lr":  config["ACTOR_LR"],
                "gamma":     config["GAMMA"],
            },
            title=f"IS-MADDPG — OvercookedV3 / {args.layout}",
        )
    except ImportError:
        pass

    # ── Instantiate envs ─────────────────────────────────────────────
    # env_vec is vmapped in run() — we just pass the base instance
    # env_eval is used single-threaded for greedy evaluation
    env_vec  = OvercookedV3(layout=args.layout)
    # env_eval = OvercookedV3(layout=args.layout)

    # ── Run ──────────────────────────────────────────────────────────
    class _nullctx:
        def __enter__(self): return self
        def __exit__(self, *_): pass

    ctx = monitor if monitor is not None else _nullctx()
    with ctx:
        results = run(config, env_vec, monitor=monitor)

    print(f"\n Training complete.")
    print(f"   Total gradient updates : {results['total_updates']}")
    if results["returns"]:
        print(f"   Last 100-ep mean return: {np.mean(results['returns'][-100:]):.2f}")

    wandb.finish()


if __name__ == "__main__":
    main()