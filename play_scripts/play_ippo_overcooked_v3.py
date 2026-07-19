#!/usr/bin/env python3
"""Visualize a trained IPPO-RNN policy on Overcooked V3."""

import argparse
import sys
import os

import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn
import pygame
import functools
from flax.linen.initializers import constant, orthogonal
from typing import Callable, Any, Dict, Sequence
import distrax

from jaxmarl.environments.overcooked_v3 import OvercookedV3
from jaxmarl.viz.overcooked_v3_visualizer import OvercookedV3Visualizer
from jaxmarl.wrappers.baselines import load_params


# ── Model (must match training architecture) ────────────────────────────

class ScannedRNN(nn.Module):
    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        rnn_state = carry
        ins, resets = x
        new_carry = self.initialize_carry(ins.shape[0], ins.shape[1])
        rnn_state = jnp.where(resets[:, np.newaxis], new_carry, rnn_state)
        new_rnn_state, y = nn.GRUCell(features=ins.shape[1])(rnn_state, ins)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        cell = nn.GRUCell(features=hidden_size)
        return cell.initialize_carry(jax.random.PRNGKey(0), (batch_size, hidden_size))


class CNN(nn.Module):
    output_size: int = 64
    activation: Callable[..., Any] = nn.relu

    @nn.compact
    def __call__(self, x, train=False):
        x = nn.Conv(128, (1, 1), kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(x)
        x = self.activation(x)
        x = nn.Conv(128, (1, 1), kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(x)
        x = self.activation(x)
        x = nn.Conv(8, (1, 1), kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(x)
        x = self.activation(x)
        x = nn.Conv(16, (3, 3), kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(x)
        x = self.activation(x)
        x = nn.Conv(32, (3, 3), kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(x)
        x = self.activation(x)
        x = nn.Conv(32, (3, 3), kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(x)
        x = self.activation(x)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(self.output_size, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(x)
        x = self.activation(x)
        return x


class ActorCriticRNN(nn.Module):
    action_dim: Sequence[int]
    config: Dict

    @nn.compact
    def __call__(self, hidden, x):
        obs, dones = x
        activation = nn.relu if self.config["ACTIVATION"] == "relu" else nn.tanh
        embed_model = CNN(output_size=self.config["GRU_HIDDEN_DIM"], activation=activation)
        embedding = jax.vmap(embed_model)(obs)
        embedding = nn.LayerNorm()(embedding)
        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)
        actor_mean = nn.Dense(self.config["FC_DIM_SIZE"], kernel_init=orthogonal(2), bias_init=constant(0.0))(embedding)
        actor_mean = nn.relu(actor_mean)
        actor_mean = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(actor_mean)
        pi = distrax.Categorical(logits=actor_mean)
        critic = nn.Dense(self.config["FC_DIM_SIZE"], kernel_init=orthogonal(2), bias_init=constant(0.0))(embedding)
        critic = nn.relu(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(critic)
        return hidden, pi, jnp.squeeze(critic, axis=-1)


# ── Main ─────────────────────────────────────────────────────────────────

ACTION_NAMES = ["right", "down", "left", "up", "stay", "interact"]


def main():
    parser = argparse.ArgumentParser(description="Watch a trained IPPO-RNN policy play Overcooked V3")
    parser.add_argument("params_path", type=str, help="Path to .safetensors file with trained params")
    parser.add_argument("--layout", type=str, default="around_the_island")
    parser.add_argument("--agent-view-size", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--gru-hidden-dim", type=int, default=128)
    parser.add_argument("--fc-dim-size", type=int, default=128)
    parser.add_argument("--activation", type=str, default="relu", choices=["relu", "tanh"])
    parser.add_argument("--fps", type=int, default=8, help="Playback speed (frames per second)")
    parser.add_argument("--tile-size", type=int, default=48)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--deterministic", action="store_true", help="Use argmax instead of sampling")
    parser.add_argument("--gif", type=str, default=None, help="Save a GIF to this path instead of launching interactive viewer")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes to record (only used with --gif)")
    args = parser.parse_args()

    config = {
        "GRU_HIDDEN_DIM": args.gru_hidden_dim,
        "FC_DIM_SIZE": args.fc_dim_size,
        "ACTIVATION": args.activation,
    }

    # Create environment
    env = OvercookedV3(
        layout=args.layout,
        agent_view_size=args.agent_view_size,
        max_steps=args.max_steps,
        shaped_rewards=True,
        random_agent_positions=True,
    )

    # Build network and load params
    num_agents = env.num_agents
    num_actors = num_agents  # 1 env
    obs_shape = env.observation_space().shape
    action_dim = env.action_space(env.agents[0]).n

    network = ActorCriticRNN(action_dim, config=config)

    # Dummy forward pass to init
    init_x = (
        jnp.zeros((1, num_actors, *obs_shape)),
        jnp.zeros((1, num_actors)),
    )
    init_hstate = ScannedRNN.initialize_carry(num_actors, config["GRU_HIDDEN_DIM"])
    network_params = network.init(jax.random.PRNGKey(0), init_hstate, init_x)

    # Load trained params
    trained_params = load_params(args.params_path)
    network_params = {"params": trained_params["params"]}

    # JIT the forward pass
    @jax.jit
    def policy_step(params, hstate, obs_batch, done_batch, rng):
        ac_in = (obs_batch[np.newaxis, :], done_batch[np.newaxis, :])
        hstate, pi, value = network.apply(params, hstate, ac_in)
        action = pi.sample(seed=rng)
        return hstate, action.squeeze(0), value.squeeze(0), pi.logits.squeeze(0)

    @jax.jit
    def policy_step_deterministic(params, hstate, obs_batch, done_batch, rng):
        ac_in = (obs_batch[np.newaxis, :], done_batch[np.newaxis, :])
        hstate, pi, value = network.apply(params, hstate, ac_in)
        action = jnp.argmax(pi.logits, axis=-1)
        return hstate, action.squeeze(0), value.squeeze(0), pi.logits.squeeze(0)

    step_fn = policy_step_deterministic if args.deterministic else policy_step

    # Visualizer
    viz = OvercookedV3Visualizer(env, tile_size=args.tile_size)

    if args.gif:
        _run_gif(args, env, viz, network_params, step_fn, num_actors, config)
    else:
        _run_interactive(args, env, viz, network_params, step_fn, num_actors, config)


def _run_gif(args, env, viz, network_params, step_fn, num_actors, config):
    from PIL import Image

    rng = jax.random.PRNGKey(args.seed)
    frames = []

    for ep in range(args.episodes):
        rng, reset_rng = jax.random.split(rng)
        obs, state = env.reset(reset_rng)
        hstate = ScannedRNN.initialize_carry(num_actors, config["GRU_HIDDEN_DIM"])
        done_batch = jnp.zeros(num_actors, dtype=bool)
        total_reward = 0.0
        step_count = 0

        print(f"Recording episode {ep + 1}/{args.episodes}...")
        frames.append(Image.fromarray(np.array(viz.render_state(state))))

        while True:
            obs_batch = jnp.stack([obs[a] for a in env.agents])
            rng, act_rng = jax.random.split(rng)
            hstate, actions, values, logits = step_fn(
                network_params, hstate, obs_batch, done_batch, act_rng
            )
            env_actions = {a: int(actions[i]) for i, a in enumerate(env.agents)}

            rng, step_rng = jax.random.split(rng)
            obs, state, rewards, dones, info = env.step(step_rng, state, env_actions)

            done_batch = jnp.array([dones[a] for a in env.agents])
            reward = float(rewards[env.agents[0]])
            total_reward += reward
            step_count += 1

            frames.append(Image.fromarray(np.array(viz.render_state(state))))

            if reward > 0:
                print(f"  +{reward:.0f} delivery! (total: {total_reward:.0f}, step {step_count})")

            if dones["__all__"]:
                print(f"  Episode {ep + 1} done. Score: {total_reward:.0f} in {step_count} steps")
                break

    frame_duration = int(1000 / args.fps)
    frames[0].save(
        args.gif,
        save_all=True,
        append_images=frames[1:],
        duration=frame_duration,
        loop=0,
    )
    print(f"Saved {len(frames)} frames to {args.gif}")


def _run_interactive(args, env, viz, network_params, step_fn, num_actors, config):
    # Pygame setup
    pygame.init()
    width = env.width * args.tile_size
    height = env.height * args.tile_size
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption(f"IPPO Policy - {args.layout}")
    clock = pygame.time.Clock()

    # Init state
    rng = jax.random.PRNGKey(args.seed)
    rng, reset_rng = jax.random.split(rng)
    obs, state = env.reset(reset_rng)
    hstate = ScannedRNN.initialize_carry(num_actors, config["GRU_HIDDEN_DIM"])
    done_batch = jnp.zeros(num_actors, dtype=bool)

    total_reward = 0.0
    step_count = 0
    episode = 0
    running = True
    paused = False

    print("=" * 50)
    print("  IPPO Policy Viewer - Overcooked V3")
    print("=" * 50)
    print(f"  Layout:  {args.layout}")
    print(f"  Params:  {args.params_path}")
    print(f"  Mode:    {'deterministic (argmax)' if args.deterministic else 'stochastic (sampling)'}")
    print()
    print("  SPACE = Pause/Resume")
    print("  R     = Reset episode")
    print("  Q/ESC = Quit")
    print("=" * 50)

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                elif event.key == pygame.K_r:
                    rng, reset_rng = jax.random.split(rng)
                    obs, state = env.reset(reset_rng)
                    hstate = ScannedRNN.initialize_carry(num_actors, config["GRU_HIDDEN_DIM"])
                    done_batch = jnp.zeros(num_actors, dtype=bool)
                    total_reward = 0.0
                    step_count = 0
                    episode += 1
                    print(f"\n--- Episode {episode} ---")
                elif event.key == pygame.K_SPACE:
                    paused = not paused

        if not paused:
            # Get actions from policy
            obs_batch = jnp.stack([obs[a] for a in env.agents])
            rng, act_rng = jax.random.split(rng)
            hstate, actions, values, logits = step_fn(
                network_params, hstate, obs_batch, done_batch, act_rng
            )

            env_actions = {a: int(actions[i]) for i, a in enumerate(env.agents)}

            # Step environment
            rng, step_rng = jax.random.split(rng)
            obs, state, rewards, dones, info = env.step(step_rng, state, env_actions)

            done_batch = jnp.array([dones[a] for a in env.agents])
            reward = float(rewards[env.agents[0]])
            total_reward += reward
            step_count += 1

            if reward > 0:
                print(f"  +{reward:.0f} delivery! (total: {total_reward:.0f}, step {step_count})")

            if dones["__all__"]:
                print(f"  Episode done. Score: {total_reward:.0f} in {step_count} steps")
                # Auto-reset
                rng, reset_rng = jax.random.split(rng)
                obs, state = env.reset(reset_rng)
                hstate = ScannedRNN.initialize_carry(num_actors, config["GRU_HIDDEN_DIM"])
                done_batch = jnp.zeros(num_actors, dtype=bool)
                total_reward = 0.0
                step_count = 0
                episode += 1
                print(f"\n--- Episode {episode} ---")

        # Render
        img = viz.render_state(state)
        img_np = np.array(img)
        surf = pygame.surfarray.make_surface(img_np.swapaxes(0, 1))
        screen.blit(surf, (0, 0))

        # HUD
        font = pygame.font.Font(None, 24)
        hud = f"Step: {step_count}  Score: {total_reward:.0f}  Ep: {episode}"
        if paused:
            hud += "  [PAUSED]"
        text_surf = font.render(hud, True, (255, 255, 255))
        screen.blit(text_surf, (5, 5))

        pygame.display.flip()
        clock.tick(args.fps)

    pygame.quit()
    print(f"\nDone. Final score: {total_reward:.0f}")


if __name__ == "__main__":
    main()
