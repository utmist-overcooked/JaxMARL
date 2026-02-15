"""IC3Net-family baselines (IC / IRIC / CommNet / IC3Net) for JaxMARL.

This module contains JAX/Flax implementations of:
  - Independent Controllers (IC/IRIC): no communication (feedforward & LSTM)
  - CommNet: continuous communication without gating (feedforward & LSTM)
  - IC3Net: CommNet with hard-attention gating (feedforward & LSTM)

References:
    Singh, A., Jain, T., & Sukhbaatar, S. (2018). Learning when to
    Communicate at Scale in Multiagent Cooperative and Competitive Tasks.
    arXiv:1812.09755
"""

from baselines.IC3Net.models import (
    IndependentMLP,
    IndependentLSTM,
    CommNetDiscrete,
    CommNetLSTM,
)

__all__ = [
    "IndependentMLP",
    "IndependentLSTM",
    "CommNetDiscrete",
    "CommNetLSTM",
]
