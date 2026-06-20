# run_overcooked_v3.py
"""
Entry point for IS-MADDPG on OvercookedV3 (custom fork).

Usage:
    python run_overcooked_v3.py                          # cramped_room default
    python run_overcooked_v3.py --layout asymmetric_advantages
    python run_overcooked_v3.py --layout coordination_ring --num_envs 16 --wandb
    python run_overcooked_v3.py --total_timesteps 50000  # quick smoke test
    python run_overcooked_v3.py --num_envs 64 --total_timesteps 8_000_000 --max_steps 400
"""

import argparse
import os
import time
from typing import NamedTuple
import numpy as np
import jax
import jax.numpy as jnp
import csv
from functools import partial
import flashbax as fbx
from collections import namedtuple
import matplotlib.pyplot as plt

from jaxmarl.environments.overcooked_v3.overcooked import OvercookedV3, State

from networks import ISAgentNet, ISCriticNet
from buffer import buffer_init, buffer_add, buffer_add_batch, buffer_is_ready, buffer_sample, buffer_sample_prioritized
from update import TrainState, UpdateMetrics, init_train_state, train_step
from loss import received_messages
from train import save_checkpoint, save_checkpoint_zip, DEFAULT_CONFIG


# ---------------------------------------------------------------------------
# Available Layouts
# ---------------------------------------------------------------------------

LAYOUTS = [
    "cramped_room",
    "asymm_advantages",
    "coord_ring",
    "forced_coord",
    "counter_circuit",
    "cramped_room_v2",
    "conveyor_demo",
    "player_conveyor_demo",
    "player_conveyor_loop",
    "race_against_the_clock",
    "maze_conveyor_hell",
    "coordinated_temporal_conveyor",
    "general_conveyor_level_1",
    "general_conveyor_level_2",
    "general_conveyor_level_3",
    "middle_conveyor",
    "follow_the_leader",
    "around_the_island",
    "single_file",
    "moving_wall_demo",
    "moving_wall_bounce_demo",
    "barrier_demo",
    "timed_barrier_demo",
    "moving_wall_barrier_button_demo"
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
        "MSG_DIM":          5,
        "HORIZON_H":        2,
        "HIDDEN_DIM":       128,
        "ACTOR_LR":         5e-5,
        "CRITIC_LR":        1e-4,
        "GAMMA":            0.90, # lower gamma = smaller Bellman targets = more stable
        "TAU":              0.01,
        "GRAD_CLIP":        0.1, # tight clip
        "GUMBEL_TAU":       1.0,
        "GUMBEL_HARD":      True,
        "PRED_LOSS_COEF":   0.05,

        # ── Training schedule ────────────────────────────────────────────
        "TOTAL_TIMESTEPS":  args.total_timesteps,
        "NUM_ENVS":         args.num_envs,
        "MAX_STEPS":        args.max_steps,
        "BATCH_SIZE":       512,
        "BUFFER_SIZE":      200_000,
        "LEARNING_STARTS":  5_000,
        "UPDATE_EVERY":     1,
        "UPDATES_PER_STEP": 2,
        "NUM_EPOCHS":       2,

        # ── Exploration ──────────────────────────────────────────────────
        # Decay epsilon over first 30% of training — Overcooked is dense
        # reward so the policy picks up signal quickly
        "EPSILON_START":    1.0,
        "EPSILON_END":      0.1, # higher minimum epsilon — don't go fully greedy
        "EPSILON_DECAY":    int(args.total_timesteps * 0.75),

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


# vmap over agents, one JIT dispatch for all agents
@partial(jax.jit, static_argnums=(1, 4))
def select_actions_jit(train_state, actor, obs_all, prev_msgs, num_agents, gumbel_tau, rng):
    """Single JIT call for all agents via vmap.
    
    obs_all:   (num_envs, N, obs_dim)
    prev_msgs: (num_envs, N, msg_dim)
    Returns:
        logits:   (N, num_envs, act_dim)
        msgs_out: (N, num_envs, msg_dim)
    """
    received = received_messages(prev_msgs)  # (num_envs, N, N-1, msg_dim)

    def single_agent(j, key):
        return actor.apply(
            train_state,
            obs_all[:, j, :],           # (num_envs, obs_dim)
            received[:, j, :, :],       # (num_envs, N-1, msg_dim)
            rng=key,
            gumbel_tau=gumbel_tau,
            gumbel_hard=True,
        )

    keys = jax.random.split(rng, num_agents)
    # vmap over agent index
    logits_all, _, msgs_all, _ = jax.vmap(
        single_agent, in_axes=(0, 0)
    )(jnp.arange(num_agents), keys)

    return logits_all, msgs_all   # (N, num_envs, act_dim), (N, num_envs, msg_dim)


@partial(jax.jit, static_argnums=(3, 4, 5))
def apply_epsilon_greedy(logits_all, rng, epsilon, num_agents, num_envs, act_dim):
    """Apply epsilon-greedy entirely in JAX — no Python loop, no numpy.
    
    logits_all: (N, num_envs, act_dim)
    Returns:
        actions_idx:    (num_envs, N)   int32
        actions_onehot: (num_envs, N, act_dim)
    """
    greedy = jnp.argmax(logits_all, axis=-1)              # (N, num_envs)

    rng, eps_key = jax.random.split(rng)
    random_acts  = jax.random.randint(
        eps_key, (num_agents, num_envs), 0, act_dim
    )

    explore = jax.random.uniform(
        jax.random.split(rng)[1], (num_agents, num_envs)
    ) < epsilon
    final   = jnp.where(explore, random_acts, greedy)     # (N, num_envs)

    actions_idx    = final.T                               # (num_envs, N)
    actions_onehot = jax.nn.one_hot(actions_idx, act_dim) # (num_envs, N, act_dim)
    return actions_idx, actions_onehot


# ------------------------------------------------------------------
    # JIT-compiled action selection (defined once before loop)
    # ------------------------------------------------------------------

@partial(jax.jit, static_argnums=(1, 4, 5, 6))
def select_and_explore(actor_params, actor, obs_all, prev_msgs, num_agents, num_envs, act_dim, rng, gumbel_tau, epsilon):
    """Single JIT dispatch: actor forward + epsilon-greedy, all on GPU.

    Args:
        actor_params: actor parameter pytree
        obs_all:      (num_envs, N, obs_dim)  JAX array
        prev_msgs:    (num_envs, N, msg_dim)   JAX array
        rng:          PRNGKey
        epsilon:      scalar float

    Returns:
        actions_idx:    (num_envs, N)           int32
        actions_onehot: (num_envs, N, act_dim)  float32
        msgs_out:       (num_envs, N, msg_dim)  float32
        rng:            updated key
    """
    received = received_messages(prev_msgs)  # (num_envs, N, N-1, msg_dim)

    # Split one key per agent
    rng, *agent_keys = jax.random.split(rng, num_agents + 1)
    agent_keys = jnp.stack(agent_keys)  # (N, 2)

    def single_agent_forward(j):
        logits, _, msg, _ = actor.apply(
            actor_params,
            obs_all[:, j, :],        # (num_envs, obs_dim)
            received[:, j, :, :],    # (num_envs, N-1, msg_dim)
            rng=agent_keys[j],
            gumbel_tau=gumbel_tau,
            gumbel_hard=True,
        )
        return logits, msg  # (num_envs, act_dim), (num_envs, msg_dim)

    # vmap over agent index
    logits_all, msgs_all = jax.vmap(single_agent_forward)(
        jnp.arange(num_agents)
    )  # (N, num_envs, act_dim), (N, num_envs, msg_dim)

    # Transpose to (num_envs, N, ...)
    logits_all = logits_all.transpose(1, 0, 2)  # (num_envs, N, act_dim)
    msgs_out   = msgs_all.transpose(1, 0, 2)    # (num_envs, N, msg_dim)

    # Epsilon-greedy entirely in JAX
    greedy = jnp.argmax(logits_all, axis=-1)    # (num_envs, N)

    rng, eps_key, rand_key = jax.random.split(rng, 3)
    random_acts = jax.random.randint(
        rand_key, (num_envs, num_agents), 0, act_dim
    )
    explore_mask = jax.random.uniform(
        eps_key, (num_envs, num_agents)
    ) < epsilon

    actions_idx    = jnp.where(explore_mask, random_acts, greedy)  # (num_envs, N)
    actions_onehot = jax.nn.one_hot(actions_idx, act_dim)          # (num_envs, N, act_dim)

    return actions_idx, actions_onehot, msgs_out, rng


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

            rng, act_rng = jax.random.split(rng)
            logits_all, msgs = select_actions_jit(
                train_state.actor_params,
                actor,
                jnp.array(obs_all),
                jnp.array(prev_msgs),
                num_agents,
                gumbel_tau,
                act_rng,
            )

            rng, eps_rng = jax.random.split(rng)
            actions_idx, actions_onehot = apply_epsilon_greedy(
                logits_all, eps_rng, 0.0, num_agents, 1, act_dim
            )

            # Keep as JAX arrays until buffer write — no numpy conversion
            msgs = np.array(msgs.transpose(1, 0, 2))        # (num_envs, N, msg_dim)
            acts_idx = np.array(actions_idx)                # only convert for env.step
            actions_onehot = np.array(actions_onehot)             # only convert for buffer
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
    gumbel_tau  = config["GUMBEL_TAU"]
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

    # buffer = fbx.make_flat_buffer(
    #     max_length=config["BUFFER_SIZE"],
    #     min_length=config["LEARNING_STARTS"],
    #     sample_batch_size=config["BATCH_SIZE"],
    #     add_sequences=False,
    #     add_batch_size=num_envs,
    # )

    # # Init — lives entirely on GPU
    # dummy_transition = {
    #     "obs": jnp.zeros((num_agents, obs_dim), dtype=jnp.int32),
    #     "prev_msgs": jnp.zeros((num_agents, msg_dim)),
    #     "actions": jnp.zeros((num_agents, act_dim)),
    #     "msgs": jnp.zeros((num_agents, msg_dim)),
    #     "rewards": jnp.zeros((num_agents,)),
    #     # "next_obs": jnp.zeros((num_agents, obs_dim), dtype=jnp.int32),
    #     # "next_prev_msgs": jnp.zeros((num_agents, msg_dim)),
    #     "dones": jnp.zeros(()),
    # }

    # buffer_state = buffer.init(dummy_transition)

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
    ep_lengths   = np.zeros((num_envs,), dtype=np.int32)
    all_returns  = []
    all_lengths  = []
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
    "placement_in_pot": 0,   # shaped reward = 6
    "ingredient_pickup": 0,  # shaped reward = 3
    "soup_in_dish":      0,  # shaped reward = 12
    "plate_pickup":      0,  # shaped reward = 4
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


    t_step   = 0.0  # env stepping
    t_buffer = 0.0  # buffer adds
    t_train  = 0.0  # gradient updates
    t_other  = 0.0  # everything else

    CKPT_INTERVAL = 1500
    if ckpt_dir is not None:
        os.makedirs(ckpt_dir, exist_ok=True)
        print(f"  Checkpoints will be saved to: {ckpt_dir}/", flush=True)
    else:
        print("  [WARNING] No save_path set — checkpoints will NOT be saved.", flush=True)

    # # ------------------------------------------------------------------
    # # Runner state — everything that crosses scan iterations
    # # ------------------------------------------------------------------
    # RunnerState = namedtuple("RunnerState", [
    #     "train_state",
    #     "buffer_state",
    #     "obs_dict",
    #     "prev_msgs",
    #     "env_states",
    #     "rng",

    #     # episode tracking
    #     "ep_returns",
    #     "ep_lengths",
    #     "ep_deliveries",

    #     "ep_ingredient_pickup",
    #     "ep_plate_pickup",
    #     "ep_placement_in_pot",
    #     "ep_soup_in_dish",
    #     "ep_delivery",
    # ])

    # runner_state = RunnerState(
    #     train_state=train_state,
    #     buffer_state=buffer_state,
    #     obs_dict=obs_dict,
    #     prev_msgs=jnp.zeros((num_envs, num_agents, msg_dim)),
    #     env_states=env_states,
    #     rng=rng,

    #     ep_returns=jnp.zeros(num_envs),
    #     ep_lengths=jnp.zeros(num_envs, dtype=jnp.int32),
    #     ep_deliveries=jnp.zeros(num_envs),

    #     ep_ingredient_pickup=jnp.zeros(num_envs),
    #     ep_plate_pickup=jnp.zeros(num_envs),
    #     ep_placement_in_pot=jnp.zeros(num_envs),
    #     ep_soup_in_dish=jnp.zeros(num_envs),
    #     ep_delivery=jnp.zeros(num_envs),
    # )

    # from typing import NamedTuple
    # class Batch(NamedTuple):
    #     obs: jnp.ndarray
    #     prev_msgs: jnp.ndarray
    #     actions: jnp.ndarray
    #     msgs: jnp.ndarray
    #     rewards: jnp.ndarray
    #     dones: jnp.ndarray
    #     next_obs: jnp.ndarray
    #     next_prev_msgs: jnp.ndarray

    # # ------------------------------------------------------------------
    # # Single compiled update step (replaces the entire Python for-loop)
    # # ------------------------------------------------------------------
    # @jax.jit
    # def _update_step(runner_state, step_idx):
    #     train_state  = runner_state.train_state
    #     buffer_state = runner_state.buffer_state
    #     obs_dict     = runner_state.obs_dict
    #     prev_msgs    = runner_state.prev_msgs
    #     env_states   = runner_state.env_states
    #     rng          = runner_state.rng

    #     ep_returns = runner_state.ep_returns
    #     ep_lengths = runner_state.ep_lengths
    #     ep_deliveries = runner_state.ep_deliveries

    #     ep_ingredient_pickup = runner_state.ep_ingredient_pickup
    #     ep_plate_pickup = runner_state.ep_plate_pickup
    #     ep_placement_in_pot = runner_state.ep_placement_in_pot
    #     ep_soup_in_dish = runner_state.ep_soup_in_dish
    #     ep_delivery = runner_state.ep_delivery

    #     global_step = step_idx * num_envs
    #     frac    = jnp.clip(global_step / max(1, eps_decay), 0.0, 1.0)
    #     epsilon = eps_start + frac * (eps_end - eps_start)

    #     # ── Action selection ─────────────────────────────────────────
    #     rng, act_rng = jax.random.split(rng)
    #     obs_jax = jnp.stack(
    #         [obs_dict[aid].reshape(num_envs, obs_dim) for aid in agent_ids],
    #         axis=1,
    #     )
    #     actions_idx, actions_onehot, msgs, _ = select_and_explore(
    #         train_state.actor_params, actor, obs_jax, prev_msgs, num_agents, num_envs, act_dim, act_rng, gumbel_tau, epsilon
    #     )

    #     # ── Env step ─────────────────────────────────────────────────
    #     rng, step_rng = jax.random.split(rng)
    #     step_rngs = jax.random.split(step_rng, num_envs)
    #     action_dict = {
    #         f"agent_{i}": actions_idx[:, i] for i in range(num_agents)
    #     }
    #     next_obs_dict, env_states, rewards_dict, dones_dict, info = jax.vmap(
    #         env_vec.step_env
    #     )(step_rngs, env_states, action_dict)

    #     # ── Rewards ──────────────────────────────────────────────────
    #     next_obs_jax = jnp.stack(
    #         [next_obs_dict[aid].reshape(num_envs, obs_dim) for aid in agent_ids],
    #         axis=1,
    #     )
    #     raw_rewards = jnp.stack(
    #         [rewards_dict[aid] for aid in agent_ids], axis=1
    #     )
    #     shaped_rewards = jnp.stack(
    #         [info["shaped_reward"][aid] for aid in agent_ids], axis=1
    #     )
    #     rewards_all = raw_rewards + shaped_rewards

    #     step_reward = rewards_all.sum(axis=1)

    #     delivery_event = (
    #         (raw_rewards >= 20.0)
    #         .any(axis=1)
    #         .astype(jnp.float32)
    #     )

    #     ep_returns = ep_returns + step_reward
    #     ep_lengths = ep_lengths + 1
    #     ep_deliveries = ep_deliveries + delivery_event

    #     sv = shaped_rewards[:, 0]

    #     ep_ingredient_pickup = (
    #         ep_ingredient_pickup +
    #         (sv == 3.0).astype(jnp.float32)
    #     )

    #     ep_plate_pickup = (
    #         ep_plate_pickup +
    #         (sv == 4.0).astype(jnp.float32)
    #     )

    #     ep_placement_in_pot = (
    #         ep_placement_in_pot +
    #         (sv == 6.0).astype(jnp.float32)
    #     )

    #     ep_soup_in_dish = (
    #         ep_soup_in_dish +
    #         (sv == 12.0).astype(jnp.float32)
    #     )

    #     ep_delivery = ep_delivery + delivery_event           

    #     dones = jnp.stack(
    #         [dones_dict[aid] for aid in agent_ids], axis=1
    #     ).all(axis=1).astype(jnp.float32)  # (num_envs,)

    #     completed_returns = jnp.where(
    #         dones.astype(bool),
    #         ep_returns,
    #         0.0,
    #     )

    #     completed_lengths = jnp.where(
    #         dones.astype(bool),
    #         ep_lengths,
    #         0,
    #     )

    #     completed_deliveries = jnp.where(
    #         dones.astype(bool),
    #         ep_deliveries,
    #         0.0,
    #     )

    #     completed_ingredient_pickup = jnp.where(
    #         dones.astype(bool),
    #         ep_ingredient_pickup,
    #         0.0,
    #     )

    #     completed_plate_pickup = jnp.where(
    #         dones.astype(bool),
    #         ep_plate_pickup,
    #         0.0,
    #     )

    #     completed_placement_in_pot = jnp.where(
    #         dones.astype(bool),
    #         ep_placement_in_pot,
    #         0.0,
    #     )

    #     completed_soup_in_dish = jnp.where(
    #         dones.astype(bool),
    #         ep_soup_in_dish,
    #         0.0,
    #     )

    #     completed_delivery = jnp.where(
    #         dones.astype(bool),
    #         ep_delivery,
    #         0.0,
    #     )        

    #     # ── Buffer add ───────────────────────────────────────────────
    #     transition = {
    #         "obs":            obs_jax,
    #         "prev_msgs":      prev_msgs,
    #         "actions":        actions_onehot,
    #         "msgs":           msgs,
    #         "rewards":        rewards_all,
    #         # "next_obs":       next_obs_jax,
    #         # "next_prev_msgs": msgs,
    #         "dones":          dones,
    #     }
    #     buffer_state = buffer.add(buffer_state, transition)

    #     # ── Auto-reset done envs ──────────────────────────────────────
    #     rng, reset_rng = jax.random.split(rng)
    #     reset_rngs = jax.random.split(reset_rng, num_envs)
    #     new_obs, new_states = jit_reset(reset_rngs)

    #     done_mask = dones.astype(bool)
    #     env_states = jax.tree_util.tree_map(
    #         lambda new, old: jnp.where(
    #             done_mask.reshape([-1] + [1] * (new.ndim - 1)), new, old
    #         ),
    #         new_states, env_states,
    #     )

    #     next_obs_stacked = jnp.stack(
    #         [next_obs_dict[aid] for aid in agent_ids], axis=1
    #     )  # (num_envs, N, H, W, C)
    #     new_obs_stacked = jnp.stack(
    #         [new_obs[aid] for aid in agent_ids], axis=1
    #     )
    #     merged = jnp.where(
    #         done_mask[:, None, None, None, None],  # broadcast over N,H,W,C
    #         new_obs_stacked,
    #         next_obs_stacked,
    #     )
    #     # Rebuild dict
    #     next_obs_dict = {
    #         aid: merged[:, i] for i, aid in enumerate(agent_ids)
    #     }

    #     # Reset messages for done envs
    #     prev_msgs = jnp.where(
    #         done_mask[:, None, None],
    #         jnp.zeros_like(msgs),
    #         msgs,
    #     )

    #     ep_returns = jnp.where(
    #         done_mask,
    #         0.0,
    #         ep_returns,
    #     )

    #     ep_lengths = jnp.where(
    #         done_mask,
    #         0,
    #         ep_lengths,
    #     )

    #     ep_deliveries = jnp.where(
    #         done_mask,
    #         0.0,
    #         ep_deliveries,
    #     )

    #     ep_ingredient_pickup = jnp.where(
    #         done_mask,
    #         0.0,
    #         ep_ingredient_pickup,
    #     )

    #     ep_plate_pickup = jnp.where(
    #         done_mask,
    #         0.0,
    #         ep_plate_pickup,
    #     )

    #     ep_placement_in_pot = jnp.where(
    #         done_mask,
    #         0.0,
    #         ep_placement_in_pot,
    #     )

    #     ep_soup_in_dish = jnp.where(
    #         done_mask,
    #         0.0,
    #         ep_soup_in_dish,
    #     )

    #     ep_delivery = jnp.where(
    #         done_mask,
    #         0.0,
    #         ep_delivery,
    #     )

    #     # ── Train step (conditional on buffer ready) ──────────────────
    #     rng, sample_rng = jax.random.split(rng)

    #     def do_train(args):
    #         train_state, buffer_state, rng = args
    #         sample = buffer.sample(buffer_state, rng)
    #         pair = sample.experience

    #         batch = Batch(
    #             obs=pair.first["obs"],
    #             prev_msgs=pair.first["prev_msgs"],
    #             actions=pair.first["actions"],
    #             msgs=pair.first["msgs"],
    #             rewards=pair.first["rewards"],
    #             dones=pair.first["dones"],
    #             next_obs=pair.second["obs"],
    #             next_prev_msgs=pair.second["prev_msgs"],
    #         )

    #         train_state, metrics = jit_train_step(train_state, batch)
    #         return train_state, metrics

    #     def skip_train(args):
    #         train_state, buffer_state, rng = args
    #         return train_state, UpdateMetrics(
    #             critic_loss=jnp.zeros(()),
    #             actor_loss= jnp.zeros(()),
    #             pred_loss=  jnp.zeros(()),
    #             q_mean=     jnp.zeros(()),
    #         )

    #     train_state, metrics = jax.lax.cond(
    #         buffer.can_sample(buffer_state),
    #         do_train,
    #         skip_train,
    #         (train_state, buffer_state, sample_rng),
    #     )

    #     # ── Metrics for logging ───────────────────────────────────────
    #     step_metrics = {
    #         "rewards": step_reward.mean(),
    #         "deliveries": delivery_event.mean(),

    #         "critic_loss": metrics.critic_loss,
    #         "actor_loss": metrics.actor_loss,
    #         "pred_loss": metrics.pred_loss,
    #         "q_mean": metrics.q_mean,

    #         "completed_returns": completed_returns,
    #         "completed_lengths": completed_lengths,
    #         "completed_deliveries": completed_deliveries,

    #         "completed_ingredient_pickup": completed_ingredient_pickup,
    #         "completed_plate_pickup": completed_plate_pickup,
    #         "completed_placement_in_pot": completed_placement_in_pot,
    #         "completed_soup_in_dish": completed_soup_in_dish,
    #         "completed_delivery": completed_delivery,
    #     }

    #     new_runner_state = RunnerState(
    #         train_state=train_state,
    #         buffer_state=buffer_state,
    #         obs_dict=next_obs_dict,
    #         prev_msgs=prev_msgs,
    #         env_states=env_states,
    #         rng=rng,

    #         ep_returns=ep_returns,
    #         ep_lengths=ep_lengths,
    #         ep_deliveries=ep_deliveries,

    #         ep_ingredient_pickup=ep_ingredient_pickup,
    #         ep_plate_pickup=ep_plate_pickup,
    #         ep_placement_in_pot=ep_placement_in_pot,
    #         ep_soup_in_dish=ep_soup_in_dish,
    #         ep_delivery=ep_delivery,
    #     )
    #     return new_runner_state, step_metrics

    # # ── Run entire training as one compiled scan ──────────────────────
    # print("JIT compiling full training loop (first run takes several minutes)...")
    # t_compile = time.time()
    # runner_state, all_metrics = jax.lax.scan(
    #     _update_step,
    #     runner_state,
    #     jnp.arange(total_steps_target),
    #     length=total_steps_target,
    # )
    # jax.block_until_ready(all_metrics)
    # print(f"Training complete. Compile+run took {time.time()-t_compile:.1f}s")



    for t in range(1, total_steps_target + 1):
        t_loop_start = time.time()
        global_step = t * num_envs 

        # ── Epsilon schedule ─────────────────────────────────────────
        frac    = min(1.0, global_step / max(1, eps_decay))
        epsilon = eps_start + frac * (eps_end - eps_start)

        # ── Obs ──────────────────────────────────────────────────────
        obs_all = obs_dict_to_array(obs_dict, agent_ids, num_envs, obs_dim)

        # ── Action selection ─────────────────────────────────────────
        rng, act_rng = jax.random.split(rng)
        logits_all, msgs = select_actions_jit(
            train_state.actor_params,
            actor,
            jnp.array(obs_all),
            jnp.array(prev_msgs),
            num_agents = num_agents,
            gumbel_tau = config["GUMBEL_TAU"],
            rng = act_rng,
        )    

        actions_idx, actions_onehot = apply_epsilon_greedy(
            logits_all,
            act_rng,
            epsilon,
            num_agents,
            num_envs,
            act_dim,
        )

        # Keep as JAX arrays until buffer write — no numpy conversion
        msgs = np.array(msgs.transpose(1, 0, 2))        # (num_envs, N, msg_dim)
        actions_idx = np.array(actions_idx)                # only convert for env.step
        actions_onehot = np.array(actions_onehot)       # only convert for buffer

        # ── Step envs ────────────────────────────────────────────────
        rng, step_rng = jax.random.split(rng)
        step_rngs = jax.random.split(step_rng, num_envs)
        action_dict = {
            f"agent_{i}": jnp.array(actions_idx[:, i])
            for i in range(num_agents)
        }
        t0 = time.time()
        next_obs_dict, env_states, rewards_dict, dones_dict, info = jit_step(
        step_rngs, env_states, action_dict
        )
        # jax.block_until_ready(next_obs_dict)   # force sync for accurate timing
        t_step += time.time() - t0
        # print(f"First jit_step: {time.time()-t0:.1f}s (includes compilation)")

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

        # Raw rewards always needed for delivery tracking
        raw_rewards = rewards_dict_to_array(rewards_dict, agent_ids, num_envs)        

        # Use shaped rewards for training signal, sparse for logging
        if config.get("USE_SHAPED_REWARDS", True) and "shaped_reward" in info:
            shaped = info["shaped_reward"]
            # Combine: sparse delivery + shaped intermediate rewards
            combined_rewards = {
                aid: jnp.array(rewards_dict[aid]) + jnp.array(shaped[aid])
                for aid in agent_ids
            }
            rewards_all = rewards_dict_to_array(combined_rewards, agent_ids, num_envs)
        else:
            rewards_all = raw_rewards
        
        # Make sure they are np arrays before adding to the buffer
        rewards_all=np.asarray(rewards_all)
        next_obs_all=np.asarray(next_obs_all)
        obs_all=np.asarray(obs_all)

        # Track deliveries from raw rewards
        ep_deliveries += (raw_rewards >= 20.0).any(axis=1).astype(np.float32)          

        # ── Buffer ───────────────────────────────────────────────────
        t0 = time.time()
        # for e in range(num_envs):
        #     buffer_state = buffer_add(
        #         buffer_state,
        #         obs=           obs_all[e],
        #         prev_msgs=     prev_msgs[e],
        #         actions=       actions_onehot[e],
        #         msgs=          msgs[e],
        #         rewards=       rewards_all[e],
        #         next_obs=      next_obs_all[e],
        #         next_prev_msgs=msgs[e],
        #         done=          bool(dones_all[e]),
            # )

        buffer_state = buffer_add_batch(
            buffer_state,
            obs=np.asarray(obs_all),
            prev_msgs=prev_msgs,
            actions=np.asarray(actions_onehot),
            msgs=msgs,
            rewards=np.asarray(rewards_all),
            next_obs=np.asarray(next_obs_all),
            next_prev_msgs=msgs,
            dones=dones_all,
        )

        t_buffer += time.time() - t0

        # ── Reward type tracking ──────────────────────────────────────
        # Prefer explicit event flags if provided by the environment via
        # info["reward_events"]. Fall back to shaped-reward inference.
        if "reward_events" in info:
            re = info["reward_events"]
            for e in range(num_envs):
                for ev in list(reward_type_counts.keys()):
                    count = 0.0
                    # Case A: re is mapping agent_id -> {event: array}
                    try:
                        if isinstance(re, dict) and agent_ids[0] in re:
                            for aid in agent_ids:
                                sub = re.get(aid, {})
                                if isinstance(sub, dict) and ev in sub:
                                    count += float(jnp.array(sub[ev])[e])
                        # Case B: re is mapping event -> agent mapping or event -> array
                        elif isinstance(re, dict) and ev in re:
                            val = re[ev]
                            if isinstance(val, dict):
                                for aid in agent_ids:
                                    if aid in val:
                                        count += float(jnp.array(val[aid])[e])
                            else:
                                # val might be per-agent array or per-env scalar array
                                try:
                                    arr = jnp.array(val)
                                    # If arr has agent axis, try summing across agents
                                    if arr.ndim == 2:
                                        count += float(arr[:, :][e].sum())
                                    else:
                                        count += float(arr[e])
                                except Exception:
                                    pass
                    except Exception:
                        count = 0.0

                    ep_reward_types[ev][e] += count

        elif "shaped_reward" in info:
            shaped = info["shaped_reward"]
            for e in range(num_envs):
                aid0 = agent_ids[0]
                sv = float(jnp.array(shaped[aid0])[e])
                rv = float(jnp.array(rewards_dict[aid0])[e])

                # Infer reward type from shaped value
                if sv == 12.0:
                    ep_reward_types["soup_in_dish"][e] += 1
                elif sv == 3.0:
                    ep_reward_types["ingredient_pickup"][e] += 1
                elif sv == 6.0:
                    ep_reward_types["placement_in_pot"][e] += 1
                elif sv == 4.0:
                    ep_reward_types["plate_pickup"][e] += 1

                if rv >= 20.0:
                    ep_reward_types["delivery"][e] += 1

        # ── Episode tracking ─────────────────────────────────────────
        ep_returns += rewards_all
        ep_lengths += 1
        for e in range(num_envs):
            if dones_all[e]:
                all_returns.append(float(ep_returns[e].sum()))
                all_lengths.append(int(ep_lengths[e]))
                all_deliveries.append(float(ep_deliveries[e]))  # deliveries this episode
                ep_returns[e]   = 0.0
                ep_deliveries[e] = 0.0
                ep_lengths[e] = 0
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
            # new_obs, new_states = jax.vmap(env_vec.reset)(reset_rngs)
            new_obs, new_states = jit_reset(reset_rngs)


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
        # for e in range(num_envs):
        #     if dones_all[e]:
        #         prev_msgs[e] = 0.0
        done_mask_np = dones_all.astype(bool)  # (num_envs,)
        prev_msgs[done_mask_np] = 0.0          # one numpy op

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
                
                t0 = time.time()
                # batch, rng = buffer_sample_prioritized(buffer_state, batch_size, rng, priority_reward_weight=10.0)
                batch, rng = buffer_sample(buffer_state, batch_size, rng)
                train_state, last_metrics = jit_train_step(train_state, batch)
                # jax.block_until_ready(train_state.actor_params)  # force sync
                t_train += time.time() - t0

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
        
        # Print breakdown every 500 steps
        if t % 1000 == 0 and t > 0:
            total = t_step + t_buffer + t_train
            print(f"\n[t={t}] Time breakdown:")
            print(f"  env step    : {t_step:.1f}s  ({100*t_step/total:.0f}%)")
            print(f"  buffer add  : {t_buffer:.1f}s  ({100*t_buffer/total:.0f}%)")
            print(f"  train step  : {t_train:.1f}s  ({100*t_train/total:.0f}%)")
            t_step = t_buffer = t_train = 0.0

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

        # ── Checkpoint every CKPT_INTERVAL loop iterations ────────────
        if (ckpt_dir is not None
                and t > 0
                and t % CKPT_INTERVAL == 0):
            ckpt_path = os.path.join(
                ckpt_dir,
                f"is_maddpg_{config['LAYOUT']}_step{t:08d}.zip"
            )
            save_checkpoint_zip(train_state, ckpt_path, config, t)
            print(f"\nCheckpoint saved → {ckpt_path}", flush=True)        

    
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

    # --- Per-episode event counts (non-cumulative) ---
    if all_lengths:
        hist_dir = ckpt_dir if ckpt_dir else "."
        os.makedirs(hist_dir, exist_ok=True)

        # Events to include (match DEBUG_HISTOGRAM.md semantics)
        event_cols = ["ingredient_pickup", "placement_in_pot", "plate_pickup", "soup_in_dish", "delivery"]

        csv_path = os.path.join(hist_dir, f"is_maddpg_{config['LAYOUT']}_episode_events.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["episode", *event_cols])
            n_eps = len(all_lengths)
            # prepare per-event lists (pad with zeros if necessary)
            per_event = {}
            for ev in event_cols:
                vals = reward_type_history.get(ev, [])
                if len(vals) < n_eps:
                    vals = vals + [0.0] * (n_eps - len(vals))
                per_event[ev] = vals

            for i in range(n_eps):
                row = [i + 1]
                row += [int(per_event[ev][i]) for ev in event_cols]
                writer.writerow(row)

        try:
            fig, ax = plt.subplots(figsize=(10, 5))
            episodes = np.arange(1, len(all_lengths) + 1)

            # plot each event as a line (per-episode counts, non-cumulative)
            for ev in event_cols:
                vals = reward_type_history.get(ev, [])
                if len(vals) < len(episodes):
                    vals = vals + [0.0] * (len(episodes) - len(vals))
                ax.plot(episodes, vals, marker="o", linewidth=1.2, label=ev.replace("_", " "))

            ax.set_title(f"Per-episode event counts — IS-MADDPG / {config['LAYOUT']}")
            ax.set_xlabel("Episode")
            ax.set_ylabel("Count (per episode)")
            ax.grid(alpha=0.3)
            ax.legend()
            fig.tight_layout()
            png_path = os.path.join(hist_dir, f"is_maddpg_{config['LAYOUT']}_episode_events.png")
            fig.savefig(png_path, dpi=150, bbox_inches="tight")
            print(f"Saved episode-event CSV to {csv_path} and {png_path}")
            plt.show()
        except Exception as exc:
            print(f"Saved episode-event CSV to {csv_path}; plot skipped: {exc}")
    
    # ---------------------------------------------------------------------------
    # Reward type histograms
    # ---------------------------------------------------------------------------
    reward_labels = {
        "ingredient_pickup": f"Ingredient Pickup (+2)",
        "delivery":          f"Delivery (+20)",
        "placement_in_pot":  f"Placement in Pot (+6)",
        "plate_pickup":      f"Plate Pickup (+4)",
        # "pot_start_cooking": f"Pot Start Cooking (+4)",
        "ingredient_pickup":      f"Onion Pickup (+3)",
        "soup_in_dish":      f"Soup in Dish (+12)",
        # "burn_penalty":      f"Burn Penalty (-5)",
    }
    colors = {
        "ingredient_pickup": "orange",
        "delivery":          "green",
        "placement_in_pot":  "steelblue",
        "plate_pickup":      "yellow",
        # "pot_start_cooking": "orange",
        "ingredient_pickup": "pink",
        # "pot_start_cooking": "black",
        "soup_in_dish":      "purple",
        # "burn_penalty":      "red",
    }

    if any(len(v) > 0 for v in reward_type_history.values()):
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        axes = axes.flatten()

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

    # # ── Post-processing of scan outputs ───────────────────────────────
    # def extract_completed(metric_2d):
    #     """Flatten (total_steps, num_envs) and keep only completed episodes."""
    #     flat = np.asarray(metric_2d).reshape(-1)
    #     mask = np.asarray(all_metrics["completed_lengths"]).reshape(-1) > 0
    #     return flat[mask]

    # all_returns    = extract_completed(all_metrics["completed_returns"])
    # all_lengths    = extract_completed(all_metrics["completed_lengths"])
    # all_deliveries = extract_completed(all_metrics["completed_deliveries"])

    # reward_type_history = {
    #     "ingredient_pickup": extract_completed(all_metrics["completed_ingredient_pickup"]),
    #     "plate_pickup":      extract_completed(all_metrics["completed_plate_pickup"]),
    #     "placement_in_pot":  extract_completed(all_metrics["completed_placement_in_pot"]),
    #     "soup_in_dish":      extract_completed(all_metrics["completed_soup_in_dish"]),
    #     "delivery":          extract_completed(all_metrics["completed_delivery"]),
    # }
    # reward_type_counts = {k: int(v.sum()) for k, v in reward_type_history.items()}

    # # Total gradient updates = steps where buffer was ready
    # # Approximate from scan length minus learning_starts buffer fill
    # total_updates = int(np.asarray(all_metrics["critic_loss"] != 0).sum())

    # # ── Post-training summary ──────────────────────────────────────────
    # print("\n" + "="*60)
    # print("TRAINING SUMMARY")
    # print("="*60)
    # print(f"  Total env steps      : {config['TOTAL_TIMESTEPS']:,}")
    # print(f"  Approx grad updates  : {total_updates:,}")
    # print(f"  Episodes completed   : {len(all_returns)}")
    # if len(all_returns) > 0:
    #     print(f"  Best episode return  : {float(np.max(all_returns)):.2f}")
    #     print(f"  Final 100-ep mean   : {float(np.mean(all_returns[-100:])):.2f}")
    #     print(f"  Final 100-ep std    : {float(np.std(all_returns[-100:])):.2f}")
    #     print(f"  Total deliveries    : {reward_type_counts['delivery']:,}")
    # print("="*60)

    # plot_dir = ckpt_dir if ckpt_dir else "."
    # os.makedirs(plot_dir, exist_ok=True)

    # # ── Plot 1: Returns over episodes + distribution ───────────────────
    # if len(all_returns) > 0:
    #     fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    #     ax = axes[0]
    #     ax.plot(all_returns, alpha=0.3, color="steelblue", label="raw")
    #     window = min(50, len(all_returns))
    #     if len(all_returns) >= window:
    #         smoothed = np.convolve(
    #             all_returns, np.ones(window) / window, mode="valid"
    #         )
    #         ax.plot(
    #             range(window - 1, len(all_returns)),
    #             smoothed,
    #             color="steelblue", linewidth=2,
    #             label=f"{window}-ep moving avg",
    #         )
    #     ax.set_xlabel("Episode")
    #     ax.set_ylabel("Return")
    #     ax.set_title(f"IS-MADDPG — OvercookedV3 / {config['LAYOUT']}")
    #     ax.legend()
    #     ax.grid(alpha=0.3)

    #     ax = axes[1]
    #     ax.hist(all_returns, bins=30, color="steelblue", alpha=0.7, edgecolor="white")
    #     ax.axvline(float(np.mean(all_returns)), color="red", linestyle="--",
    #                linewidth=1.5, label=f"mean={np.mean(all_returns):.1f}")
    #     ax.set_xlabel("Episode Return")
    #     ax.set_ylabel("Count")
    #     ax.set_title("Return Distribution (all episodes)")
    #     ax.legend()
    #     ax.grid(alpha=0.3)

    #     plt.tight_layout()
    #     plot_path = os.path.join(plot_dir, f"is_maddpg_{config['LAYOUT']}_returns.png")
    #     plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    #     print(f"\nReturn plot saved → {plot_path}")
    #     plt.show()
    #     plt.close()

    # # ── Plot 2: Per-episode event counts over time ─────────────────────
    # event_cols = ["ingredient_pickup", "placement_in_pot", "plate_pickup",
    #               "soup_in_dish", "delivery"]
    # event_colors = {
    #     "ingredient_pickup": "orange",
    #     "placement_in_pot":  "steelblue",
    #     "plate_pickup":      "yellow",
    #     "soup_in_dish":      "purple",
    #     "delivery":          "green",
    # }
    # event_labels = {
    #     "ingredient_pickup": "Ingredient Pickup (+3)",
    #     "placement_in_pot":  "Placement in Pot (+6)",
    #     "plate_pickup":      "Plate Pickup (+4)",
    #     "soup_in_dish":      "Soup in Dish (+12)",
    #     "delivery":          "Delivery (+20)",
    # }

    # if len(all_lengths) > 0:
    #     # Save CSV
    #     csv_path = os.path.join(plot_dir, f"is_maddpg_{config['LAYOUT']}_episode_events.csv")
    #     with open(csv_path, "w", newline="", encoding="utf-8") as f:
    #         writer = csv.writer(f)
    #         writer.writerow(["episode", "return", "length", "deliveries", *event_cols])
    #         for i in range(len(all_lengths)):
    #             row = [
    #                 i + 1,
    #                 float(all_returns[i]) if i < len(all_returns) else 0.0,
    #                 int(all_lengths[i]),
    #                 int(all_deliveries[i]) if i < len(all_deliveries) else 0,
    #             ]
    #             row += [int(reward_type_history[ev][i])
    #                     if i < len(reward_type_history[ev]) else 0
    #                     for ev in event_cols]
    #             writer.writerow(row)
    #     print(f"Episode CSV saved → {csv_path}")

    #     # Per-episode line plot
    #     fig, ax = plt.subplots(figsize=(12, 5))
    #     episodes = np.arange(1, len(all_lengths) + 1)

    #     for ev in event_cols:
    #         vals = reward_type_history[ev]
    #         if len(vals) > 0:
    #             # Smooth for readability
    #             w = min(20, len(vals))
    #             smoothed = np.convolve(vals, np.ones(w) / w, mode="valid")
    #             ax.plot(
    #                 range(w - 1, len(vals)),
    #                 smoothed,
    #                 linewidth=1.5,
    #                 color=event_colors[ev],
    #                 label=event_labels[ev],
    #             )

    #     ax.set_title(f"Per-episode event counts (smoothed) — IS-MADDPG / {config['LAYOUT']}")
    #     ax.set_xlabel("Episode")
    #     ax.set_ylabel("Count per episode (smoothed)")
    #     ax.grid(alpha=0.3)
    #     ax.legend()
    #     plt.tight_layout()
    #     png_path = os.path.join(plot_dir, f"is_maddpg_{config['LAYOUT']}_episode_events.png")
    #     plt.savefig(png_path, dpi=150, bbox_inches="tight")
    #     print(f"Episode events plot saved → {png_path}")
    #     plt.show()
    #     plt.close()

    # # ── Plot 3: Reward breakdown histograms ────────────────────────────
    # has_data = any(len(v) > 0 for v in reward_type_history.values())
    # if has_data:
    #     fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    #     axes = axes.flatten()

    #     for idx, ev in enumerate(event_cols):
    #         ax   = axes[idx]
    #         data = reward_type_history[ev]
    #         lbl  = event_labels[ev]
    #         col  = event_colors[ev]

    #         if len(data) > 0 and data.max() > 0:
    #             ax.hist(data, bins=20, color=col, alpha=0.7, edgecolor="white")
    #             mean_val = float(np.mean(data))
    #             ax.axvline(mean_val, color="black", linestyle="--",
    #                        linewidth=1.5, label=f"mean={mean_val:.2f}")
    #             ax.legend(fontsize=8)
    #         else:
    #             ax.text(0.5, 0.5, "No events recorded",
    #                     ha="center", va="center",
    #                     transform=ax.transAxes, color="grey")

    #         ax.set_title(lbl)
    #         ax.set_xlabel("Count per episode")
    #         ax.set_ylabel("Episodes")
    #         ax.grid(alpha=0.3)

    #     # Summary bar chart in last panel
    #     ax     = axes[5]
    #     keys   = event_cols
    #     totals = [reward_type_counts[k] for k in keys]
    #     cols   = [event_colors[k] for k in keys]
    #     bars   = ax.bar(range(len(keys)), totals, color=cols, alpha=0.7, edgecolor="white")
    #     ax.set_xticks(range(len(keys)))
    #     ax.set_xticklabels([k.replace("_", "\n") for k in keys], fontsize=7)
    #     ax.set_title("Total reward events (all episodes)")
    #     ax.set_ylabel("Count")
    #     ax.grid(alpha=0.3, axis="y")
    #     for bar, total in zip(bars, totals):
    #         ax.text(
    #             bar.get_x() + bar.get_width() / 2,
    #             bar.get_height() + max(1, max(totals) * 0.01),
    #             f"{int(total):,}",
    #             ha="center", va="bottom", fontsize=8,
    #         )

    #     plt.suptitle(f"IS-MADDPG Reward Breakdown — {config['LAYOUT']}", fontsize=13)
    #     plt.tight_layout()
    #     hist_path = os.path.join(plot_dir, f"is_maddpg_{config['LAYOUT']}_reward_breakdown.png")
    #     plt.savefig(hist_path, dpi=150, bbox_inches="tight")
    #     print(f"Reward breakdown plot saved → {hist_path}")
    #     plt.show()
    #     plt.close()

    # # ── Summary table ─────────────────────────────────────────────────
    # print("\nReward event totals across all episodes:")
    # print(f"  {'Event':<25} {'Total':>8}  {'Per Episode':>12}")
    # print("  " + "-" * 48)
    # n_eps = max(1, len(all_returns))
    # for ev in event_cols:
    #     total  = reward_type_counts[ev]
    #     per_ep = total / n_eps
    #     print(f"  {event_labels[ev]:<25} {total:>8,}  {per_ep:>11.2f}")

    # return {
    #     "train_state":    runner_state.train_state,
    #     "returns":        all_returns,
    #     "total_updates":  total_updates,
    #     "all_deliveries": all_deliveries,
    # }    


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
        "--total_timesteps", type=int, default=1_000_000,
        help="Total environment steps"
    )
    parser.add_argument(
        "--num_envs", type=int, default=6,
        help="Number of parallel environments"
    )
    parser.add_argument(
        "--max_steps", type=int, default=400,
        help="Maximum steps per episode"
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
    env_vec  = OvercookedV3(layout=args.layout, max_steps=args.max_steps)
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