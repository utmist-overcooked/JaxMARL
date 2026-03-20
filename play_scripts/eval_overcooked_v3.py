"""Evaluate a trained IPPO-RNN model on Overcooked V3.

Usage:
    python play_scripts/eval_overcooked_v3.py \
        /scratch/zachtang/jaxmarl/ippo_overcooked_v3/models/ippo_rnn_overcooked_v3_cramped_room_seed0_vmap0.safetensors \
        --layout cramped_room \
        --save-gif $SCRATCH/jaxmarl/ippo_overcooked_v3/rollout.gif
"""

import argparse
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from jaxmarl.wrappers.baselines import load_params
from jaxmarl.environments.overcooked_v3 import OvercookedV3
from jaxmarl.viz.overcooked_v3_visualizer import OvercookedV3Visualizer
from baselines.IPPO.ippo_rnn_overcooked_v3 import ActorCriticRNN, ScannedRNN


def rollout(params, env, network, config, rng):
    """Run a single episode, returning stacked states and total reward."""
    rng, rng_reset = jax.random.split(rng)
    obs, state = env.reset(rng_reset)

    hstate = ScannedRNN.initialize_carry(
        env.num_agents, config["GRU_HIDDEN_DIM"]
    )

    state_seq = [state]
    total_reward = 0.0

    for step in range(config["max_steps"]):
        rng, rng_act = jax.random.split(rng)

        obs_batch = jnp.stack([obs[a] for a in env.agents])
        done_batch = jnp.zeros(env.num_agents)

        ac_in = (obs_batch[np.newaxis, :], done_batch[np.newaxis, :])
        hstate, pi, _ = network.apply(params, hstate, ac_in)

        if config["greedy"]:
            action = jnp.argmax(pi.logits, axis=-1)
        else:
            action = pi.sample(seed=rng_act)

        actions = {a: action.squeeze()[i] for i, a in enumerate(env.agents)}

        rng, rng_step = jax.random.split(rng)
        obs, state, reward, dones, info = env.step(rng_step, state, actions)

        state_seq.append(state)
        total_reward += float(reward[env.agents[0]])

        if dones["__all__"]:
            break

    # Stack states into a single pytree for the visualizer
    stacked_states = jax.tree.map(lambda *xs: jnp.stack(xs), *state_seq)
    return stacked_states, total_reward, step + 1


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained IPPO-RNN model on Overcooked V3"
    )
    parser.add_argument("model_path", help="Path to .safetensors model file")
    parser.add_argument("--layout", default="cramped_room")
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--greedy", action="store_true", help="Use argmax instead of sampling")
    parser.add_argument("--agent-view-size", type=int, default=None, help="Partial obs window (None for full)")
    parser.add_argument("--random-agent-positions", action="store_true", help="Randomize agent start positions")
    parser.add_argument("--save-gif", default=None, help="Save last episode as GIF")
    parser.add_argument("--gru-hidden-dim", type=int, default=128)
    parser.add_argument("--fc-dim-size", type=int, default=128)
    parser.add_argument("--activation", default="relu", choices=["relu", "tanh"])
    args = parser.parse_args()

    # Load model
    params = load_params(args.model_path)
    print(f"Loaded model from {args.model_path}")

    # Create env
    env_kwargs = {
        "layout": args.layout,
        "agent_view_size": args.agent_view_size,
        "shaped_rewards": True,
        "random_agent_positions": args.random_agent_positions,
        "max_steps": args.max_steps,
    }
    env = OvercookedV3(**env_kwargs)

    # Create network
    network_config = {
        "GRU_HIDDEN_DIM": args.gru_hidden_dim,
        "FC_DIM_SIZE": args.fc_dim_size,
        "ACTIVATION": args.activation,
    }
    network = ActorCriticRNN(env.action_space(env.agents[0]).n, config=network_config)

    config = {
        **network_config,
        "max_steps": args.max_steps,
        "greedy": args.greedy,
    }

    # Run episodes
    all_rewards = []
    last_states = None

    for ep in range(args.episodes):
        rng = jax.random.PRNGKey(args.seed + ep)
        stacked_states, total_reward, steps = rollout(params, env, network, config, rng)
        all_rewards.append(total_reward)
        last_states = stacked_states
        print(f"Episode {ep + 1}: reward={total_reward:.1f}, steps={steps}")

    if args.episodes > 1:
        print(f"\nMean: {np.mean(all_rewards):.1f} +/- {np.std(all_rewards):.1f}")

    # Save GIF
    if args.save_gif and last_states is not None:
        viz = OvercookedV3Visualizer(env)
        viz.animate(last_states, filename=args.save_gif)
        print(f"Saved GIF to {args.save_gif}")


if __name__ == "__main__":
    main()
