"""Backward-compatibility shims for the pre-2.0 Sofia API.

Everything here is deprecated and will be removed in 3.0.0. Names are resolved
lazily through module ``__getattr__`` so that simply importing ``sofia_ai`` is
cheap and does not emit warnings — only *using* a deprecated name warns.
"""

from __future__ import annotations

import warnings
from typing import Any

__all__ = [
    "DEPRECATED",
    "ModelConfig",
    "NLPConfig",
    "NLPProcessor",
    "ProcessedText",
    "QNetwork",
    "QuantumConfig",
    "QuantumNeuralEngine",
    "QuantumState",
    "ReplayBuffer",
    "SofiaModel",
    "SofiaRLAgent",
    "Token",
]

#: Names re-exported from other modules. Declared here so static tools can see the
#: public surface; resolution happens lazily through module ``__getattr__``.
ModelConfig: Any
NLPConfig: Any
NLPProcessor: Any
ProcessedText: Any
QNetwork: Any
QuantumConfig: Any
QuantumNeuralEngine: Any
QuantumState: Any
ReplayBuffer: Any
SofiaModel: Any
SofiaRLAgent: Any
Token: Any

#: Deprecation target version. Breaking removal happens no earlier than this.
REMOVAL_VERSION: str = "3.0.0"

DEPRECATED: dict[str, str] = {
    "SofiaModel": "sofia_ai.diagnostics / sofia_ai.inference",
    "QuantumNeuralEngine": "sofia_ai.experimental.quantum.QuantumNeuralEngine",
    "QuantumState": "sofia_ai.experimental.quantum.QuantumState",
    "NLPProcessor": "sofia_ai.copilot.legacy_nlp.NLPProcessor",
    "Token": "sofia_ai.copilot.legacy_nlp.Token",
    "ProcessedText": "sofia_ai.copilot.legacy_nlp.ProcessedText",
    "QuantumConfig": "sofia_ai.config.legacy.QuantumConfig",
    "NLPConfig": "sofia_ai.config.legacy.NLPConfig",
    "ModelConfig": "sofia_ai.config.legacy.ModelConfig",
    "SofiaRLAgent": "sofia_ai.learning.rl.SofiaRLAgent",
    "QNetwork": "sofia_ai.learning.rl.QNetwork",
    "ReplayBuffer": "sofia_ai.learning.rl.ReplayBuffer",
}

_LEGACY_MODEL: dict[str, Any] = {}


def _warn(name: str) -> None:
    replacement = DEPRECATED.get(name, "the 2.0 API")
    warnings.warn(
        f"{name} is deprecated since Sofia 2.0.0 and will be removed in "
        f"{REMOVAL_VERSION}. Use {replacement} instead.",
        DeprecationWarning,
        stacklevel=3,
    )


def _load(name: str) -> Any:
    _warn(name)
    if name == "SofiaModel":
        from .legacy_model import SofiaModel

        return SofiaModel
    if name in ("QuantumNeuralEngine", "QuantumState"):
        from .experimental.quantum import QuantumNeuralEngine, QuantumState

        return {"QuantumNeuralEngine": QuantumNeuralEngine,
                "QuantumState": QuantumState}[name]
    if name in ("NLPProcessor", "Token", "ProcessedText"):
        from .copilot.legacy_nlp import NLPProcessor, ProcessedText, Token

        return {"NLPProcessor": NLPProcessor, "Token": Token,
                "ProcessedText": ProcessedText}[name]
    if name in ("QuantumConfig", "NLPConfig", "ModelConfig"):
        from .config.legacy import ModelConfig, NLPConfig, QuantumConfig

        return {"QuantumConfig": QuantumConfig, "NLPConfig": NLPConfig,
                "ModelConfig": ModelConfig}[name]
    if name in ("SofiaRLAgent", "QNetwork", "ReplayBuffer"):
        from .learning.rl import QNetwork, ReplayBuffer, SofiaRLAgent

        return {"SofiaRLAgent": SofiaRLAgent, "QNetwork": QNetwork,
                "ReplayBuffer": ReplayBuffer}[name]
    raise AttributeError(name)


def __getattr__(name: str) -> Any:
    if name in DEPRECATED:
        return _load(name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(DEPRECATED))
