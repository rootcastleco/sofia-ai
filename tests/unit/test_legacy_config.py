"""Legacy configuration classes preserved for 1.x callers (deprecated API)."""

from __future__ import annotations

import pytest

from sofia_ai.config.legacy import (
    LEGACY_CONFIG_VERSION,
    MAX_QUBITS,
    ModelConfig,
    NLPConfig,
    QuantumConfig,
)
from sofia_ai.core.errors import ConfigurationError


class TestQuantumConfig:
    def test_defaults(self) -> None:
        cfg = QuantumConfig()
        assert cfg.qubits == 8
        assert cfg.to_dict()["qubits"] == 8

    def test_roundtrip(self, tmp_path) -> None:
        path = tmp_path / "q.json"
        cfg = QuantumConfig(qubits=6, entanglement_depth=2)
        cfg.save(str(path))
        loaded = QuantumConfig.load(str(path))
        assert loaded.to_dict() == cfg.to_dict()

    def test_from_dict(self) -> None:
        cfg = QuantumConfig.from_dict({"qubits": 4, "precision": "float64"})
        assert cfg.qubits == 4
        assert cfg.precision == "float64"

    def test_rejects_too_many_qubits(self) -> None:
        assert MAX_QUBITS == 12
        with pytest.raises(ConfigurationError, match="at most 12"):
            QuantumConfig(qubits=13)

    def test_rejects_zero_qubits(self) -> None:
        with pytest.raises(ConfigurationError, match="at least 1"):
            QuantumConfig(qubits=0)

    def test_rejects_bad_depth(self) -> None:
        with pytest.raises(ConfigurationError):
            QuantumConfig(entanglement_depth=0)

    def test_rejects_bad_learning_rate(self) -> None:
        with pytest.raises(ConfigurationError):
            QuantumConfig(learning_rate=2.0)


class TestNLPConfig:
    def test_defaults(self) -> None:
        cfg = NLPConfig()
        assert cfg.languages == ["en"]

    def test_roundtrip(self) -> None:
        cfg = NLPConfig(vocab_size=2000, embedding_dim=128, dropout_rate=0.5)
        assert NLPConfig.from_dict(cfg.to_dict()).to_dict() == cfg.to_dict()

    def test_rejects_small_vocab(self) -> None:
        with pytest.raises(ConfigurationError):
            NLPConfig(vocab_size=999)

    def test_rejects_oversized_embedding(self) -> None:
        with pytest.raises(ConfigurationError):
            NLPConfig(embedding_dim=8192)

    def test_rejects_bad_dropout(self) -> None:
        with pytest.raises(ConfigurationError):
            NLPConfig(dropout_rate=1.0)


class TestModelConfig:
    def test_defaults(self) -> None:
        cfg = ModelConfig()
        assert cfg.version == LEGACY_CONFIG_VERSION
        assert isinstance(cfg.quantum, QuantumConfig)
        assert isinstance(cfg.nlp, NLPConfig)

    def test_roundtrip(self, tmp_path) -> None:
        path = tmp_path / "m.json"
        cfg = ModelConfig(quantum=QuantumConfig(qubits=6), nlp=NLPConfig(vocab_size=2000))
        cfg.save(str(path))
        loaded = ModelConfig.load(str(path))
        assert loaded.to_dict() == cfg.to_dict()

    def test_from_dict(self) -> None:
        cfg = ModelConfig.from_dict({"quantum": {"qubits": 4}})
        assert cfg.quantum.qubits == 4

    def test_rejects_wrong_nlp_type(self) -> None:
        with pytest.raises(ConfigurationError, match="NLPConfig"):
            ModelConfig(nlp=object())  # type: ignore[arg-type]

    def test_rejects_wrong_quantum_type(self) -> None:
        with pytest.raises(ConfigurationError, match="QuantumConfig"):
            ModelConfig(quantum=object())  # type: ignore[arg-type]

    def test_optimized_realtime(self) -> None:
        tuned = ModelConfig().get_optimized_config("realtime")
        assert tuned.quantum.qubits == 6
        assert tuned.batch_size == 1

    def test_optimized_accuracy(self) -> None:
        tuned = ModelConfig().get_optimized_config("accuracy")
        assert tuned.quantum.qubits == 12
        assert tuned.nlp.num_layers == 24

    def test_optimized_low_resource(self) -> None:
        tuned = ModelConfig().get_optimized_config("low_resource")
        assert tuned.quantum.qubits == 4
        assert tuned.max_memory_gb == 4.0

    def test_optimized_general_is_change_only(self) -> None:
        base = ModelConfig()
        tuned = base.get_optimized_config("general")
        assert tuned.quantum.qubits == base.quantum.qubits
