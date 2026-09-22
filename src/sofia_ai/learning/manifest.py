"""Safe Model Manifest & Serialization for Sofia AI.

Implements SOFIA-ML-005 and SOFIA-SEC-002:
Stores neural models as a versioned JSON manifest (`sofia.model.v1`) paired with
deterministic NumPy NPZ weight archives. Verifies SHA-256 checksums before
loading to prevent model tampering. Completely avoids Python's `pickle` module.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Final

import numpy as np

from ..core.errors import ValidationError
from .asm.neural import AssemblyNeuralNetwork

__all__ = [
    "ModelIntegrityError",
    "ModelManifest",
    "load_assembly_model",
    "save_assembly_model",
]

MODEL_SCHEMA_VERSION: Final[str] = "sofia.model.v1"


class ModelIntegrityError(ValidationError):
    """Raised when a model artifact fails integrity or checksum verification."""


@dataclass(frozen=True, slots=True)
class ModelManifest:
    """Safe, versioned metadata manifest for a saved model artifact."""

    schema: str
    model_type: str
    feature_schema: str
    created_with: str
    parameters_sha256: str
    input_dimension: int
    hidden_dimension: int
    output_dimension: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelManifest:
        schema = str(data.get("schema", ""))
        if schema != MODEL_SCHEMA_VERSION:
            raise ValidationError(
                f"Unsupported model schema version: {schema!r}, expected {MODEL_SCHEMA_VERSION!r}",
                details={"schema": schema, "expected": MODEL_SCHEMA_VERSION},
            )
        return cls(
            schema=schema,
            model_type=str(data["model_type"]),
            feature_schema=str(data["feature_schema"]),
            created_with=str(data["created_with"]),
            parameters_sha256=str(data["parameters_sha256"]),
            input_dimension=int(data["input_dimension"]),
            hidden_dimension=int(data["hidden_dimension"]),
            output_dimension=int(data["output_dimension"]),
            metadata=dict(data.get("metadata", {})),
        )


def save_assembly_model(
    model: AssemblyNeuralNetwork,
    destination_dir: str | Path,
    model_name: str = "assembly_model",
    feature_schema: str = "machine-health-v3",
    extra_metadata: dict[str, Any] | None = None,
) -> Path:
    """Save an AssemblyNeuralNetwork as a safe JSON manifest and NPZ weight archive."""
    dest = Path(destination_dir)
    dest.mkdir(parents=True, exist_ok=True)

    weights = model.get_weights()
    weights_path = dest / f"{model_name}.weights.npz"
    manifest_path = dest / f"{model_name}.manifest.json"

    # 1. Save weights into deterministic NPZ
    np.savez(weights_path, **weights)  # type: ignore[arg-type]

    # 2. Compute SHA-256 of weights file
    hasher = hashlib.sha256()
    with weights_path.open("rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    weights_sha256 = hasher.hexdigest()

    # 3. Create and save manifest
    manifest = ModelManifest(
        schema=MODEL_SCHEMA_VERSION,
        model_type="assembly-neural-network",
        feature_schema=feature_schema,
        created_with="sofia-engine 3.0.0a1",
        parameters_sha256=weights_sha256,
        input_dimension=model.input_dim,
        hidden_dimension=model.hidden_dim,
        output_dimension=model.output_dim,
        metadata=extra_metadata or {},
    )

    manifest_path.write_text(json.dumps(manifest.to_dict(), indent=2), encoding="utf-8")
    return manifest_path


def load_assembly_model(
    manifest_path: str | Path,
    learning_rate: float = 0.01,
) -> tuple[AssemblyNeuralNetwork, ModelManifest]:
    """Load an AssemblyNeuralNetwork and verify its SHA-256 parameters checksum."""
    m_path = Path(manifest_path)
    if not m_path.exists():
        raise FileNotFoundError(f"Model manifest not found: {m_path}")

    manifest_data = json.loads(m_path.read_text(encoding="utf-8"))
    manifest = ModelManifest.from_dict(manifest_data)

    weights_path = m_path.parent / f"{m_path.stem.replace('.manifest', '')}.weights.npz"
    if not weights_path.exists():
        raise FileNotFoundError(f"Model weights not found: {weights_path}")

    # Verify SHA-256 checksum before reading any array
    hasher = hashlib.sha256()
    with weights_path.open("rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    actual_sha256 = hasher.hexdigest()

    if actual_sha256 != manifest.parameters_sha256:
        raise ModelIntegrityError(
            f"Model weights failed SHA-256 integrity check. Expected {manifest.parameters_sha256}, got {actual_sha256}",
            details={"expected": manifest.parameters_sha256, "actual": actual_sha256},
        )

    # Load verified NPZ archive (allow_pickle=False explicitly for security)
    with np.load(weights_path, allow_pickle=False) as npz:
        weights = {key: npz[key] for key in npz.files}

    model = AssemblyNeuralNetwork(
        input_dim=manifest.input_dimension,
        hidden_dim=manifest.hidden_dimension,
        output_dim=manifest.output_dimension,
        learning_rate=learning_rate,
    )
    model.set_weights(weights)
    return model, manifest
