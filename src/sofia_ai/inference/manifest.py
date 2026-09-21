"""Model manifests: identity, schema, provenance and integrity.

A manifest is the contract that prevents silently loading an incompatible model
(SOFIA-INF-004/005) and makes model replacement tamper-evident (threat T-14).

Manifest fields:
    model_id, version, kind, input_schema, output_schema, preprocessing_version,
    engine_compatibility, training_metadata, created_at, artifact_path,
    artifact_checksum, signature (optional)
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from ..core.errors import (
    ModelCompatibilityError,
    ModelError,
    ModelIntegrityError,
    ModelSchemaError,
    ValidationError,
)
from ..core.hashing import sha256_file, verify_checksum
from ..core.validation import validate_identifier

__all__ = [
    "ENGINE_VERSION",
    "MANIFEST_VERSION",
    "SEMVER_PATTERN",
    "ModelManifest",
    "SchemaSpec",
    "load_manifest",
    "save_manifest",
    "validate_manifest",
]

MANIFEST_VERSION: Final[str] = "2.0"
ENGINE_VERSION: Final[str] = "2.0"

#: Strict semantic version: ``major.minor.patch`` with optional pre-release.
SEMVER_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)"
    r"(?:-(?:0|[1-9]\d*|\d*[A-Za-z-][0-9A-Za-z-]*)"
    r"(?:\.(?:0|[1-9]\d*|\d*[A-Za-z-][0-9A-Za-z-]*))*)?$"
)


@dataclass(frozen=True, slots=True)
class SchemaSpec:
    """A declared input or output schema.

    ``features`` names the ordered inputs a model expects. Order matters: a model
    trained on a specific feature ordering must not silently accept a different one.
    """

    features: tuple[str, ...] = ()
    shape: tuple[int, ...] = ()
    dtype: str = "float64"

    def __post_init__(self) -> None:
        if len(self.features) != len(set(self.features)):
            raise ModelSchemaError("schema feature names must be unique", details={})
        if len(self.shape) > 2:
            raise ModelSchemaError("schema shape must be at most 2-D",
                                   details={"shape": list(self.shape)})

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> SchemaSpec:
        if not data:
            return cls()
        shape = tuple(int(v) for v in data.get("shape", ()) or ())
        return cls(features=tuple(str(f) for f in data.get("features", ()) or ()),
                   shape=shape, dtype=str(data.get("dtype", "float64")))

    def to_dict(self) -> dict[str, Any]:
        return {"features": list(self.features), "shape": list(self.shape),
                "dtype": self.dtype}


@dataclass(frozen=True, slots=True)
class ModelManifest:
    """Complete, validated description of a model artifact."""

    model_id: str
    version: str
    kind: str = "statistical"
    input_schema: SchemaSpec = field(default_factory=SchemaSpec)
    output_schema: SchemaSpec = field(default_factory=SchemaSpec)
    preprocessing_version: str = "1.0"
    engine_compatibility: str = ">=2.0,<3.0"
    training_metadata: Mapping[str, Any] = field(default_factory=dict)
    created_at: str = ""
    artifact_path: str = ""
    artifact_checksum: str = ""
    signature: str | None = None
    manifest_version: str = MANIFEST_VERSION

    def __post_init__(self) -> None:
        if not SEMVER_PATTERN.match(self.version):
            raise ModelError(
                f"version {self.version!r} is not valid semantic versioning",
                details={"version": self.version},
            )
        if not validate_identifier(self.model_id, "model_id"):
            raise ModelError("model_id is invalid", details={})

    @property
    def identity(self) -> str:
        return f"{self.model_id}@{self.version}"

    def requires_features(self) -> tuple[str, ...]:
        return self.input_schema.features

    def check_engine(self, engine_version: str = ENGINE_VERSION) -> None:
        """Raise :class:`ModelCompatibilityError` if the engine is out of range."""
        if not _version_satisfies(engine_version, self.engine_compatibility):
            raise ModelCompatibilityError(
                f"model {self.identity} requires engine {self.engine_compatibility}, "
                f"running {engine_version}",
                details={"required": self.engine_compatibility, "engine": engine_version},
            )

    def check_input(self, names: tuple[str, ...]) -> None:
        """Validate an offered feature vector against the declared input schema.

        The declared features must all be present, and must appear in the declared
        relative order. Extra features are permitted: a detector that consumes
        ``rms`` may receive a 31-feature vector. Order is enforced because a model
        is a function of an ordered vector, and silently reordering inputs is one
        of the most common sources of wrong industrial inference.

        Raises:
            ModelSchemaError: on missing features or wrong relative ordering.
        """
        required = self.requires_features()
        if not required:
            return
        offered = tuple(names)
        missing = [n for n in required if n not in offered]
        if missing:
            raise ModelSchemaError(
                f"feature mismatch for {self.identity}: missing={missing}",
                details={"missing": missing, "required": list(required)},
            )
        positions = [offered.index(n) for n in required]
        if positions != sorted(positions):
            raise ModelSchemaError(
                f"feature mismatch for {self.identity}: required features appear in a "
                f"different relative order than declared",
                details={"required": list(required), "offered": list(offered)},
            )

    def verify_artifact(self, root: str | None = None, *, strict: bool = True) -> bool:
        """Verify the artifact checksum.

        Args:
            root: Optional directory the artifact path is confined to.
            strict: Raise on mismatch instead of returning False.

        Raises:
            ModelIntegrityError: on checksum mismatch when ``strict``.
        """
        if not self.artifact_path or not self.artifact_checksum:
            if strict:
                raise ModelIntegrityError(
                    f"model {self.identity} has no artifact checksum to verify",
                    details={"model": self.identity},
                )
            return False
        path = Path(self.artifact_path)
        if root is not None:
            from ..security.safeio import resolve_within

            path = resolve_within(root, self.artifact_path)
        if not path.is_file():
            raise ModelIntegrityError(
                f"artifact {path} does not exist", details={"path": str(path)}
            )
        ok = verify_checksum(path, self.artifact_checksum)
        if not ok and strict:
            raise ModelIntegrityError(
                f"artifact checksum mismatch for {self.identity}",
                details={"model": self.identity,
                         "expected": self.artifact_checksum,
                         "actual": sha256_file(path)},
            )
        return ok

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_version": self.manifest_version,
            "model_id": self.model_id,
            "version": self.version,
            "kind": self.kind,
            "input_schema": self.input_schema.to_dict(),
            "output_schema": self.output_schema.to_dict(),
            "preprocessing_version": self.preprocessing_version,
            "engine_compatibility": self.engine_compatibility,
            "training_metadata": dict(self.training_metadata),
            "created_at": self.created_at,
            "artifact_path": self.artifact_path,
            "artifact_checksum": self.artifact_checksum,
            "signature": self.signature,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ModelManifest:
        version = str(data.get("manifest_version", MANIFEST_VERSION))
        if version.split(".", maxsplit=1)[0] != MANIFEST_VERSION.split(".")[0]:
            raise ModelCompatibilityError(
                f"manifest version {version!r} is incompatible with {MANIFEST_VERSION!r}",
                details={"manifest_version": version},
            )
        return cls(
            model_id=str(data["model_id"]),
            version=str(data["version"]),
            kind=str(data.get("kind", "statistical")),
            input_schema=SchemaSpec.from_dict(data.get("input_schema")),
            output_schema=SchemaSpec.from_dict(data.get("output_schema")),
            preprocessing_version=str(data.get("preprocessing_version", "1.0")),
            engine_compatibility=str(data.get("engine_compatibility", ">=2.0,<3.0")),
            training_metadata=dict(data.get("training_metadata", {}) or {}),
            created_at=str(data.get("created_at", "")),
            artifact_path=str(data.get("artifact_path", "")),
            artifact_checksum=str(data.get("artifact_checksum", "")),
            signature=None if data.get("signature") is None else str(data["signature"]),
            manifest_version=version,
        )


def _version_satisfies(version: str, constraint: str) -> bool:
    """Minimal version-range evaluation supporting ``>=x,<y`` and exact pins."""
    try:
        actual = tuple(int(p) for p in version.split(".")[:3])
    except ValueError:
        return False
    for clause in (c.strip() for c in constraint.split(",") if c.strip()):
        operator = clause[:2] if clause[:2] in (">=", "<=", "==", "!=") else clause[:1]
        operand = clause[len(operator):].strip()
        try:
            target = tuple(int(p) for p in operand.split(".")[:3])
        except ValueError:
            return False
        if operator == ">=" and not actual >= target:
            return False
        if operator == "<=" and not actual <= target:
            return False
        if operator == ">" and not actual > target:
            return False
        if operator == "<" and not actual < target:
            return False
        if operator == "==" and actual != target:
            return False
        if operator == "!=" and actual == target:
            return False
    return True


def validate_manifest(manifest: ModelManifest, *, engine_version: str = ENGINE_VERSION) -> None:
    """Run every structural and compatibility check on a manifest."""
    if not manifest.model_id:
        raise ValidationError("manifest model_id must not be empty", details={})
    manifest.check_engine(engine_version)
    if manifest.input_schema.features and not manifest.input_schema.shape:
        raise ModelSchemaError(
            "input_schema declares features but no shape", details={}
        )
    if len(manifest.artifact_checksum) not in (0, 64):
        raise ModelIntegrityError(
            "artifact_checksum must be a 64-character hex SHA-256 digest",
            details={"length": len(manifest.artifact_checksum)},
        )


def load_manifest(path: str, *, engine_version: str = ENGINE_VERSION) -> ModelManifest:
    """Load, parse and validate a manifest JSON file.

    Raises:
        ModelError: if the file is missing, malformed, or fails validation.
    """
    p = Path(path)
    if not p.is_file():
        raise ModelError(f"manifest not found: {path}", details={"path": path})
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except (ValueError, OSError) as exc:
        raise ModelError(
            f"manifest {path} is not readable JSON", details={"path": path}
        ) from exc
    if not isinstance(payload, dict):
        raise ModelError(f"manifest {path} must contain a JSON object",
                         details={"path": path})
    manifest = ModelManifest.from_dict(payload)
    validate_manifest(manifest, engine_version=engine_version)
    return manifest


def save_manifest(manifest: ModelManifest, path: str) -> Path:
    """Write a manifest deterministically (sorted keys, no NaN)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(manifest.to_dict(), sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return p
