#!/usr/bin/env python3
"""Render a trained IPPO Overcooked-V3 checkpoint as an animated GIF.

Loads a msgpack checkpoint saved by baselines/IPPO/ippo_rnn_overcooked_v3.py,
rolls out the policy on a chosen layout, and writes the episode frames to a GIF.
Because the IPPO network definition has changed over time, this script inspects
the parameter tree to detect which of three checkpoint formats it was saved
with, and instantiates a matching network:

- ActorCriticRNN (current):    auto-named layers (Dense_0, CNN_0, ...)
- ActorCriticRNNCompat:        explicitly-named GRU weights (gru_Wi_z, ...)
- ActorCriticRNNLegacy:        named 'cnn' / 'gru_cell' submodules

Usage:
    python scripts/generate_ippo_v3_gif.py \
        --checkpoint path/to/model.msgpack \
        --layout cramped_room --output out.gif --frames 240
"""
import argparse
from pathlib import Path
import os
import sys
import numpy as np
import imageio.v2 as imageio

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import jax
import jax.numpy as jnp
import flax.linen as nn
import distrax
from flax.linen.initializers import constant, orthogonal
from flax import serialization

import jaxmarl
from baselines.IPPO.ippo_rnn_overcooked_v3 import CNN, ActorCriticRNN
from jaxmarl.viz.overcooked_v3_visualizer import OvercookedV3Visualizer


class ActorCriticRNNCompat(nn.Module):
    """IPPO actor-critic matching checkpoints with explicitly-named GRU weights.

    Re-implements the GRU cell by hand so parameter names line up with
    checkpoints that stored individual gate matrices (gru_Wi_z, gru_Wh_z, ...)
    instead of a single nn.GRUCell submodule.

    Call: (hidden, (obs, dones)) -> (final_hidden, action_dist, value)
      obs: (T, num_actors, H, W, C), dones: (T, num_actors)
    """
    action_dim: int
    config: dict

    @nn.compact
    def __call__(self, hidden, x):
        obs, dones = x
        t = obs.shape[0]
        hidden_dim = self.config.get("GRU_HIDDEN_DIM", 128)

        activation = nn.relu if self.config.get("ACTIVATION", "relu") == "relu" else nn.tanh

        embed_model = CNN(output_size=hidden_dim, activation=activation)
        embedding = jax.vmap(embed_model)(obs)
        embedding = nn.LayerNorm()(embedding)

        num_actors = obs.shape[1]
        flat_emb = embedding.reshape(-1, hidden_dim)

        wi_z = nn.Dense(hidden_dim, use_bias=False, name="gru_Wi_z")(flat_emb)
        wi_r = nn.Dense(hidden_dim, use_bias=False, name="gru_Wi_r")(flat_emb)
        wi_h = nn.Dense(hidden_dim, use_bias=False, name="gru_Wi_h")(flat_emb)

        wi_z = wi_z.reshape(t, num_actors, hidden_dim)
        wi_r = wi_r.reshape(t, num_actors, hidden_dim)
        wi_h = wi_h.reshape(t, num_actors, hidden_dim)

        wh_z = self.param("gru_Wh_z", nn.initializers.orthogonal(), (hidden_dim, hidden_dim))
        wh_r = self.param("gru_Wh_r", nn.initializers.orthogonal(), (hidden_dim, hidden_dim))
        wh_h = self.param("gru_Wh_h", nn.initializers.orthogonal(), (hidden_dim, hidden_dim))
        b_z = self.param("gru_b_z", nn.initializers.zeros_init(), (hidden_dim,))
        b_r = self.param("gru_b_r", nn.initializers.zeros_init(), (hidden_dim,))
        b_h = self.param("gru_b_h", nn.initializers.zeros_init(), (hidden_dim,))

        def _gru_step(h, inp):
            wiz_t, wir_t, wih_t, done_t = inp
            h = jnp.where(done_t[:, None], jnp.zeros_like(h), h)
            z = jax.nn.sigmoid(wiz_t + h @ wh_z + b_z)
            r = jax.nn.sigmoid(wir_t + h @ wh_r + b_r)
            h_hat = jnp.tanh(wih_t + (r * h) @ wh_h + b_h)
            new_h = (1 - z) * h + z * h_hat
            return new_h, new_h

        final_hidden, embedding = jax.lax.scan(_gru_step, hidden, (wi_z, wi_r, wi_h, dones))

        actor = nn.Dense(
            self.config.get("FC_DIM_SIZE", 128),
            kernel_init=orthogonal(2),
            bias_init=constant(0.0),
            name="actor_fc",
        )(embedding)
        actor = nn.relu(actor)
        actor = nn.Dense(
            self.action_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
            name="actor_out",
        )(actor)

        critic = nn.Dense(
            self.config.get("FC_DIM_SIZE", 128),
            kernel_init=orthogonal(2),
            bias_init=constant(0.0),
            name="critic_fc",
        )(embedding)
        critic = nn.relu(critic)
        critic = nn.Dense(
            1,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
            name="critic_out",
        )(critic)

        pi = distrax.Categorical(logits=actor)
        return final_hidden, pi, jnp.squeeze(critic, axis=-1)


class ActorCriticRNNLegacy(nn.Module):
    """IPPO actor-critic matching older checkpoints with named submodules.

    Uses a standard nn.GRUCell under the module names ('cnn', 'ln', 'gru_cell')
    that legacy training runs saved their parameters with.

    Call signature is identical to ActorCriticRNNCompat.
    """
    action_dim: int
    config: dict

    @nn.compact
    def __call__(self, hidden, x):
        obs, dones = x
        hidden_dim = self.config.get("GRU_HIDDEN_DIM", 128)
        cnn_out_dim = self.config.get("CNN_OUT_DIM", 64)
        activation = nn.relu if self.config.get("ACTIVATION", "relu") == "relu" else nn.tanh

        cnn = CNN(output_size=cnn_out_dim, activation=activation, name="cnn")
        embedding = jax.vmap(cnn)(obs)
        embedding = nn.LayerNorm(name="ln")(embedding)

        gru_cell = nn.GRUCell(features=hidden_dim, name="gru_cell")

        def _gru_step(h, inp):
            emb_t, done_t = inp
            h = jnp.where(done_t[:, None], jnp.zeros_like(h), h)
            new_h, _ = gru_cell(h, emb_t)
            return new_h, new_h

        final_hidden, embedding = jax.lax.scan(_gru_step, hidden, (embedding, dones))

        actor = nn.Dense(
            self.config.get("FC_DIM_SIZE", 128),
            kernel_init=orthogonal(2),
            bias_init=constant(0.0),
            name="actor_fc",
        )(embedding)
        actor = nn.relu(actor)
        actor = nn.Dense(
            self.action_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
            name="actor_out",
        )(actor)

        critic = nn.Dense(
            self.config.get("FC_DIM_SIZE", 128),
            kernel_init=orthogonal(2),
            bias_init=constant(0.0),
            name="critic_fc",
        )(embedding)
        critic = nn.relu(critic)
        critic = nn.Dense(
            1,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
            name="critic_out",
        )(critic)

        pi = distrax.Categorical(logits=actor)
        return final_hidden, pi, jnp.squeeze(critic, axis=-1)


def run_episode(
    checkpoint_path: Path,
    layout: str,
    output_gif: Path,
    max_steps: int = 200,
    seed: int = 0,
    target_frames: int | None = None,
    sample_actions: bool = True,
):
    """Roll out a checkpoint on `layout` and save the frames as a GIF.

    Auto-detects the checkpoint format (see module docstring), infers hidden
    sizes from the parameter shapes, and keeps stepping (resetting the env at
    episode boundaries) until `target_frames` states have been collected.

    Args:
        checkpoint_path: msgpack file with the trained parameters
        layout: overcooked_v3 layout name
        output_gif: where to write the GIF (parent dirs are created)
        max_steps: env episode cap
        seed: PRNG seed for reset/action sampling
        target_frames: total frames to render (defaults to max_steps + 1)
        sample_actions: sample from the policy if True, else greedy argmax

    Returns:
        (num_frames, output_gif)
    """
    env = jaxmarl.make("overcooked_v3", layout=layout, max_steps=max_steps)

    config = {
        "CNN_OUT_DIM": 64,
        "GRU_HIDDEN_DIM": 128,
        "FC_DIM_SIZE": 128,
        "ACTIVATION": "relu",
    }

    # Unwrap the parameter tree: checkpoints may be saved as raw params or
    # nested one or two levels deep under a "params" key (e.g. a TrainState)
    raw = checkpoint_path.read_bytes()
    restored = serialization.msgpack_restore(raw)
    params = restored["params"] if isinstance(restored, dict) and "params" in restored else restored
    if isinstance(params, dict) and "params" in params and isinstance(params["params"], dict):
        params = params["params"]
    params_for_apply = dict(params)

    # Detect the checkpoint format from its parameter names and build the
    # matching network, inferring layer sizes from the stored kernel shapes
    is_legacy = "cnn" in params_for_apply and "gru_cell" in params_for_apply
    is_dense_named = "Dense_0" in params_for_apply and "actor_fc" not in params_for_apply
    if is_legacy:
        if "cnn" in params_for_apply and "Dense_0" in params_for_apply["cnn"]:
            config["CNN_OUT_DIM"] = int(params_for_apply["cnn"]["Dense_0"]["kernel"].shape[1])
        if "actor_fc" in params_for_apply and "kernel" in params_for_apply["actor_fc"]:
            config["GRU_HIDDEN_DIM"] = int(params_for_apply["actor_fc"]["kernel"].shape[0])
            config["FC_DIM_SIZE"] = int(params_for_apply["actor_fc"]["kernel"].shape[1])
        network = ActorCriticRNNLegacy(env.action_space(env.agents[0]).n, config=config)
    elif is_dense_named:
        if "Dense_0" in params_for_apply and "kernel" in params_for_apply["Dense_0"]:
            config["FC_DIM_SIZE"] = int(params_for_apply["Dense_0"]["kernel"].shape[1])
        network = ActorCriticRNN(env.action_space(env.agents[0]).n, config=config)
    else:
        network = ActorCriticRNNCompat(env.action_space(env.agents[0]).n, config=config)
        # Some checkpoints nested the CNN params one level down; flatten them
        # so they line up with the Compat module's expected names
        if "CNN_0" in params_for_apply and "Dense_0" not in params_for_apply:
            params_for_apply.update(params_for_apply["CNN_0"])

    variables = {"params": params_for_apply}

    key = jax.random.PRNGKey(seed)
    key, reset_key = jax.random.split(key)
    obs, state = env.reset(reset_key)

    num_agents = env.num_agents
    hidden = jnp.zeros((num_agents, config["GRU_HIDDEN_DIM"]))
    done_vec = jnp.zeros((num_agents,), dtype=bool)

    state_seq = [state]
    desired_frames = target_frames if target_frames is not None else (max_steps + 1)

    # Step the policy until enough frames are collected, resetting the env
    # (and the GRU hidden state) whenever an episode ends
    while len(state_seq) < desired_frames:
        obs_batch = jnp.stack([obs[a] for a in env.agents]).reshape(
            num_agents, *env.observation_space(env.agents[0]).shape
        )

        hidden, pi, _ = network.apply(variables, hidden, (obs_batch[jnp.newaxis, :], done_vec[jnp.newaxis, :]))
        if sample_actions:
            key, action_key = jax.random.split(key)
            action = pi.sample(seed=action_key).squeeze(0)
        else:
            action = pi.mode().squeeze(0)

        actions = {agent: action[idx] for idx, agent in enumerate(env.agents)}

        key, step_key = jax.random.split(key)
        obs, state, reward, done, info = env.step(step_key, state, actions)
        state_seq.append(state)

        done_vec = jnp.array([done[a] for a in env.agents], dtype=bool)
        if bool(done["__all__"]):
            key, reset_key = jax.random.split(key)
            obs, state = env.reset(reset_key)
            hidden = jnp.zeros((num_agents, config["GRU_HIDDEN_DIM"]))
            done_vec = jnp.zeros((num_agents,), dtype=bool)
            state_seq.append(state)

    output_gif.parent.mkdir(parents=True, exist_ok=True)
    viz = OvercookedV3Visualizer(env)
    frames = [np.array(viz.render_state(s)) for s in state_seq]
    imageio.mimsave(str(output_gif), frames, format="GIF", duration=0.2)

    return len(state_seq), output_gif


def main():
    parser = argparse.ArgumentParser(
        description="Render an IPPO Overcooked-V3 checkpoint rollout as a GIF")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--layout", type=str, default="cramped_room")
    parser.add_argument("--output", type=str, default="outputs/ippo_v3_best_run_inference.gif")
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--frames", type=int, default=240)
    parser.add_argument("--greedy", action="store_true", help="Use greedy mode actions instead of sampling")
    args = parser.parse_args()

    steps, out_path = run_episode(
        checkpoint_path=Path(args.checkpoint),
        layout=args.layout,
        output_gif=Path(args.output),
        max_steps=args.max_steps,
        seed=args.seed,
        target_frames=args.frames,
        sample_actions=not args.greedy,
    )

    print(f"Saved GIF with {steps} frames: {out_path.resolve()}")


if __name__ == "__main__":
    main()
