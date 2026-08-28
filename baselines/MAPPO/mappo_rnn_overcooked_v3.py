"""Compatibility entrypoint for the Overcooked V3 MAPPO-RNN trainer."""

from baselines.overcooked_v3.models.mappo_rnn import (
    ActorRNN,
    CriticRNN,
    MAPPORNNPolicy,
    ScannedRNN,
    batchify,
    unbatchify,
)
from baselines.overcooked_v3.trainers.mappo import (
    OvercookedWorldStateWrapper,
    Transition,
    flatten_info_leaf,
    main,
    make_train,
)

__all__ = [
    "ActorRNN",
    "CriticRNN",
    "MAPPORNNPolicy",
    "OvercookedWorldStateWrapper",
    "ScannedRNN",
    "Transition",
    "batchify",
    "flatten_info_leaf",
    "main",
    "make_train",
    "unbatchify",
]


if __name__ == "__main__":
    main()
