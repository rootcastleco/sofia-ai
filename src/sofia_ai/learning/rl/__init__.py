"""Experimental reinforcement learning. Training and simulation only.

Online RL must never automatically control industrial machinery. Any action
proposed by an agent here must pass through
:class:`sofia_ai.decision.policy.PolicyEngine` before it reaches equipment.

Torch is imported lazily: this package is importable without the ``torch`` extra.
"""

from __future__ import annotations

from .agent import CHECKPOINT_VERSION, RLConfig, SofiaRLAgent
from .buffers import Experience, PrioritizedReplayBuffer, ReplayBuffer
from .networks import DuelingQNetwork, QNetwork, build_mlp, torch_available

__all__ = [
    "CHECKPOINT_VERSION",
    "DuelingQNetwork",
    "Experience",
    "PrioritizedReplayBuffer",
    "QNetwork",
    "RLConfig",
    "ReplayBuffer",
    "SofiaRLAgent",
    "build_mlp",
    "torch_available",
]
