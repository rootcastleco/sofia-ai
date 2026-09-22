"""Unit tests for Model Manifest and Safe Serialization (SOFIA-ML-005, SOFIA-SEC-002)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sofia_ai.core.errors import ValidationError
from sofia_ai.learning.asm import AssemblyNeuralNetwork
from sofia_ai.learning.manifest import (
    ModelIntegrityError,
    load_assembly_model,
    save_assembly_model,
)


def test_model_manifest_roundtrip(tmp_path: Path) -> None:
    """SOFIA-ML-005: Model saves as manifest JSON + NPZ weights and reloads with verified SHA-256."""
    model = AssemblyNeuralNetwork(input_dim=4, hidden_dim=6, output_dim=2, seed=42)
    x = np.array([0.5, -0.2, 1.1, 0.4])
    y_orig = model.forward(x)

    manifest_path = save_assembly_model(
        model=model,
        destination_dir=tmp_path,
        model_name="pump_classifier",
        feature_schema="machine-health-v3",
    )
    assert manifest_path.exists()
    assert (tmp_path / "pump_classifier.weights.npz").exists()

    loaded_model, manifest = load_assembly_model(manifest_path)
    assert manifest.input_dimension == 4
    assert manifest.hidden_dimension == 6
    assert manifest.output_dimension == 2
    assert manifest.feature_schema == "machine-health-v3"

    y_loaded = loaded_model.forward(x)
    np.testing.assert_allclose(y_loaded, y_orig)


def test_model_integrity_tamper_detection(tmp_path: Path) -> None:
    """SOFIA-SEC-002: Tampered weight file fails SHA-256 check and raises ModelIntegrityError."""
    model = AssemblyNeuralNetwork(input_dim=2, hidden_dim=4, output_dim=1, seed=42)
    manifest_path = save_assembly_model(
        model=model,
        destination_dir=tmp_path,
        model_name="tamper_test",
    )

    weights_path = tmp_path / "tamper_test.weights.npz"
    # Tamper with the weight file bytes
    data = bytearray(weights_path.read_bytes())
    data[-1] ^= 0xFF # Flip last byte
    weights_path.write_bytes(data)

    with pytest.raises(ModelIntegrityError, match="failed SHA-256 integrity check"):
        load_assembly_model(manifest_path)


def test_model_manifest_schema_rejection(tmp_path: Path) -> None:
    """Reject invalid or incompatible model schema versions."""
    model = AssemblyNeuralNetwork(input_dim=2, hidden_dim=4, output_dim=1, seed=42)
    manifest_path = save_assembly_model(model=model, destination_dir=tmp_path, model_name="bad_schema")

    # Corrupt schema in manifest
    text = manifest_path.read_text(encoding="utf-8")
    text = text.replace('"schema": "sofia.model.v1"', '"schema": "unsupported.model.v99"')
    manifest_path.write_text(text, encoding="utf-8")

    with pytest.raises(ValidationError, match="Unsupported model schema version"):
        load_assembly_model(manifest_path)
