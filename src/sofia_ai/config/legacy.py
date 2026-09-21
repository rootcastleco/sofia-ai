"""Preserved pre-2.0 configuration classes. Deprecated.

``QuantumConfig``, ``NLPConfig`` and ``ModelConfig`` are preserved with their
original field names and ``to_dict``/``from_dict``/``save``/``load`` methods.

Change relative to 1.x (documented in ``migration.md``):

* ``QuantumConfig.qubits`` is capped at 12. The 1.x range allowed up to 20, but
  the simulated weight matrices are dense ``2**n x 2**n``, so ``qubits=20``
  required ~26 TB and could never be constructed. The ceiling is now enforced at
  validation time rather than as an out-of-memory crash.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from ..core.errors import ConfigurationError

#: Simulated-qubit ceiling. Mirrors
#: :data:`~experimental.quantum.engine.MAX_QUBITS`; declared locally so the
#: configuration layer does not depend on the experimental package.
MAX_QUBITS = 12

__all__ = ["LEGACY_CONFIG_VERSION", "ModelConfig", "NLPConfig", "QuantumConfig"]

LEGACY_CONFIG_VERSION: str = "1.0.0"


@dataclass
class QuantumConfig:
    """Configuration for the quantum-inspired simulation parameters."""

    qubits: int = 8
    entanglement_depth: int = 3
    optimization_steps: int = 50
    learning_rate: float = 0.01
    coherence_threshold: float = 0.5
    measurement_shots: int = 100
    use_gpu: bool = False
    precision: str = "float32"

    def __post_init__(self) -> None:
        if self.qubits < 1:
            raise ConfigurationError("Qubits must be at least 1",
                                     details={"qubits": self.qubits})
        if self.qubits > MAX_QUBITS:
            raise ConfigurationError(
                f"Qubits must be at most {MAX_QUBITS}: simulated weight matrices are "
                f"dense 2**n x 2**n and memory grows as 4**n",
                details={"qubits": self.qubits, "max": MAX_QUBITS},
            )
        if not 1 <= self.entanglement_depth <= 10:
            raise ConfigurationError("Entanglement depth must be between 1 and 10",
                                     details={"depth": self.entanglement_depth})
        if not 0 < self.learning_rate <= 1:
            raise ConfigurationError("Learning rate must be between 0 and 1",
                                     details={"lr": self.learning_rate})

    def to_dict(self) -> dict[str, Any]:
        return {
            "qubits": self.qubits,
            "entanglement_depth": self.entanglement_depth,
            "optimization_steps": self.optimization_steps,
            "learning_rate": self.learning_rate,
            "coherence_threshold": self.coherence_threshold,
            "measurement_shots": self.measurement_shots,
            "use_gpu": self.use_gpu,
            "precision": self.precision,
        }

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> QuantumConfig:
        return cls(**config_dict)

    def save(self, filepath: str) -> None:
        with open(filepath, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2)

    @classmethod
    def load(cls, filepath: str) -> QuantumConfig:
        with open(filepath, encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))


@dataclass
class NLPConfig:
    """Configuration for the preserved rule-based NLP processor."""

    vocab_size: int = 50000
    embedding_dim: int = 768
    max_sequence_length: int = 512
    num_heads: int = 12
    num_layers: int = 12
    dropout_rate: float = 0.1
    languages: list[str] = field(default_factory=lambda: ["en"])

    def __post_init__(self) -> None:
        if self.vocab_size < 1000:
            raise ConfigurationError("Vocabulary size must be at least 1000",
                                     details={"vocab_size": self.vocab_size})
        if not 64 <= self.embedding_dim <= 4096:
            raise ConfigurationError("Embedding dimension must be between 64 and 4096",
                                     details={"dim": self.embedding_dim})
        if not 0 <= self.dropout_rate <= 0.9:
            raise ConfigurationError("Dropout rate must be between 0 and 0.9",
                                     details={"dropout": self.dropout_rate})

    def to_dict(self) -> dict[str, Any]:
        return {
            "vocab_size": self.vocab_size,
            "embedding_dim": self.embedding_dim,
            "max_sequence_length": self.max_sequence_length,
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
            "dropout_rate": self.dropout_rate,
            "languages": self.languages,
        }

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> NLPConfig:
        return cls(**config_dict)


@dataclass
class ModelConfig:
    """Complete configuration for the preserved 1.x model facade.

    ``nlp`` must be an :class:`NLPConfig`. The 1.x test suite passed an
    ``NLPProcessor`` here, which only worked by accidental duck typing; the field is
    now validated.
    """

    quantum: QuantumConfig = field(default_factory=QuantumConfig)
    nlp: NLPConfig = field(default_factory=NLPConfig)
    model_name: str = "sofia-base"
    version: str = LEGACY_CONFIG_VERSION
    description: str = "Sofia legacy facade (deprecated)"

    batch_size: int = 32
    num_epochs: int = 100
    early_stopping_patience: int = 10

    temperature: float = 1.0
    top_p: float = 0.9
    top_k: int = 50

    max_memory_gb: float = 16.0
    num_workers: int = 4

    def __post_init__(self) -> None:
        if not isinstance(self.quantum, QuantumConfig):
            raise ConfigurationError(
                "ModelConfig.quantum must be a QuantumConfig",
                details={"type": type(self.quantum).__name__},
            )
        if not isinstance(self.nlp, NLPConfig):
            raise ConfigurationError(
                "ModelConfig.nlp must be an NLPConfig; passing a runtime "
                "NLPProcessor here was never valid",
                details={"type": type(self.nlp).__name__},
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "quantum": self.quantum.to_dict(),
            "nlp": self.nlp.to_dict(),
            "model_name": self.model_name,
            "version": self.version,
            "description": self.description,
            "batch_size": self.batch_size,
            "num_epochs": self.num_epochs,
            "early_stopping_patience": self.early_stopping_patience,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "max_memory_gb": self.max_memory_gb,
            "num_workers": self.num_workers,
        }

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> ModelConfig:
        return cls(
            quantum=QuantumConfig.from_dict(config_dict.get("quantum", {})),
            nlp=NLPConfig.from_dict(config_dict.get("nlp", {})),
            model_name=config_dict.get("model_name", "sofia-base"),
            version=config_dict.get("version", LEGACY_CONFIG_VERSION),
            description=config_dict.get("description", "Sofia legacy facade (deprecated)"),
            batch_size=config_dict.get("batch_size", 32),
            num_epochs=config_dict.get("num_epochs", 100),
            early_stopping_patience=config_dict.get("early_stopping_patience", 10),
            temperature=config_dict.get("temperature", 1.0),
            top_p=config_dict.get("top_p", 0.9),
            top_k=config_dict.get("top_k", 50),
            max_memory_gb=config_dict.get("max_memory_gb", 16.0),
            num_workers=config_dict.get("num_workers", 4),
        )

    def save(self, filepath: str) -> None:
        with open(filepath, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2)

    @classmethod
    def load(cls, filepath: str) -> ModelConfig:
        with open(filepath, encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

    def get_optimized_config(self, use_case: str = "general") -> ModelConfig:
        """Return a copy tuned for ``general|realtime|accuracy|low_resource``."""
        config = ModelConfig(
            quantum=QuantumConfig(**self.quantum.to_dict()),
            nlp=NLPConfig(**self.nlp.to_dict()),
            model_name=self.model_name,
            version=self.version,
        )
        if use_case == "realtime":
            config.quantum.qubits = 6
            config.quantum.entanglement_depth = 2
            config.nlp.max_sequence_length = 256
            config.batch_size = 1
        elif use_case == "accuracy":
            config.quantum.qubits = 12
            config.quantum.entanglement_depth = 5
            config.quantum.optimization_steps = 100
            config.nlp.embedding_dim = 1024
            config.nlp.num_layers = 24
        elif use_case == "low_resource":
            config.quantum.qubits = 4
            config.quantum.entanglement_depth = 1
            config.nlp.vocab_size = 10000
            config.nlp.embedding_dim = 256
            config.max_memory_gb = 4.0
        return config
