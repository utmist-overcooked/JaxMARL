import os
import glob
import sys

sys.path.insert(0, "/workspace/JaxMARL")

import jax
import jax.numpy as jnp
from flax import serialization
import jaxmarl

from scripts.generate_ippo_v3_gif import ActorCriticRNNCompat


def main():
    env = jaxmarl.make("overcooked_v3", layout="cramped_room", max_steps=200)
    net = ActorCriticRNNCompat(
        env.action_space(env.agents[0]).n,
        config={"GRU_HIDDEN_DIM": 128, "FC_DIM_SIZE": 128, "ACTIVATION": "relu"},
    )

    ckpts = sorted(
        glob.glob(
            "/workspace/JaxMARL/checkpoints/ippo_overcooked_v3*/**/*model*.msgpack",
            recursive=True,
        )
    )
    print(f"found {len(ckpts)} checkpoints")

    best_path = None
    best_score = -1

    for ckpt in ckpts:
        try:
            obj = serialization.msgpack_restore(open(ckpt, "rb").read())
            p = obj["params"] if isinstance(obj, dict) and "params" in obj else obj
            p2 = dict(p)
            if "CNN_0" in p2 and "Dense_0" not in p2:
                p2.update(p2["CNN_0"])
            vars_ = {"params": p2}

            key = jax.random.PRNGKey(0)
            key, rk = jax.random.split(key)
            obs, state = env.reset(rk)
            hidden = jnp.zeros((env.num_agents, 128))
            done_vec = jnp.zeros((env.num_agents,), dtype=bool)
            counts = jnp.zeros((env.num_agents, 6), dtype=jnp.int32)

            for _ in range(120):
                obs_batch = jnp.stack([obs[a] for a in env.agents]).reshape(
                    env.num_agents, *env.observation_space(env.agents[0]).shape
                )
                hidden, pi, _ = net.apply(vars_, hidden, (obs_batch[jnp.newaxis, :], done_vec[jnp.newaxis, :]))
                action = pi.mode().squeeze(0)

                for i in range(env.num_agents):
                    counts = counts.at[i, action[i]].add(1)

                actions = {a: action[i] for i, a in enumerate(env.agents)}
                key, sk = jax.random.split(key)
                obs, state, reward, done, info = env.step(sk, state, actions)
                done_vec = jnp.array([done[a] for a in env.agents], dtype=bool)
                if bool(done["__all__"]):
                    break

            unique_actions = int((counts > 0).sum())
            stay_ratio = float((counts[:, 4].sum()) / jnp.maximum(1, counts.sum()))
            score = unique_actions - 10.0 * stay_ratio

            rel = os.path.relpath(ckpt, "/workspace/JaxMARL")
            print(f"{rel} | unique={unique_actions} stay={stay_ratio:.3f} score={score:.3f}")

            if score > best_score:
                best_score = score
                best_path = ckpt
        except Exception as err:
            rel = os.path.relpath(ckpt, "/workspace/JaxMARL")
            print(f"{rel} | ERROR {type(err).__name__}")

    print("BEST_CHECKPOINT", best_path)
    print("BEST_SCORE", best_score)


if __name__ == "__main__":
    main()
