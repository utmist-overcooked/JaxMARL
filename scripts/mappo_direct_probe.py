import jax
from omegaconf import OmegaConf

from baselines.MAPPO.mappo_rnn_overcooked_v3 import make_train


def main():
    cfg = OmegaConf.to_container(
        OmegaConf.load("/workspace/JaxMARL/baselines/MAPPO/config/mappo_homogenous_rnn_overcooked_v3.yaml"),
        resolve=True,
    )
    cfg.update(
        {
            "TOTAL_TIMESTEPS": 256,
            "NUM_ENVS": 1,
            "NUM_STEPS": 32,
            "NUM_MINIBATCHES": 1,
            "UPDATE_EPOCHS": 1,
            "NUM_TEST_ENVS": 1,
            "TEST_INTERVAL": 1.0,
            "SAVE_PATH": "/workspace/JaxMARL/outputs/mappo_smoke_ckpt",
            "WANDB_MODE": "disabled",
        }
    )
    cfg["ENV_KWARGS"]["layout"] = "single_file"
    train = make_train(cfg)
    out = jax.block_until_ready(jax.jit(train)(jax.random.PRNGKey(cfg["SEED"])))
    print(sorted(out["metrics"].keys()))
    print(out["metrics"]["returned_episode_returns"].shape)


if __name__ == "__main__":
    main()