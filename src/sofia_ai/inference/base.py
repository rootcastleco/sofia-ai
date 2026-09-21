"""Model backend abstraction and the inference result contract.

Inference in Sofia is backend-agnostic. A backend loads an artifact, exposes
metadata, and turns a :class:`FeatureVector` into an :class:`InferenceResult`.

Every result carries:

* ``score`` — the backend's primary decision statistic,
* ``confidence`` — how much the backend trusts the score, in ``[0, 1]``,
* ``uncertainty`` — the complement of confidence; never hidden,
* ``latency_s`` — measured with ``time.perf_counter``,
* ``evidence`` — the values the decision was based on.

Uncertainty is never converted into certainty downstream: diagnostics multiply,
they never override.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Final

import numpy as np

from ..core.contracts import DataQuality, FeatureVector
from ..core.errors import InferenceError, ModelError
from ..core.validation import validate_probability
from .manifest import ModelManifest

__all__ = [
    "DEFAULT_UNCERTAINTY",
    "BackendMetadata",
    "BackendState",
    "InferenceOutcome",
    "InferenceResult",
    "ModelBackend",
]

DEFAULT_UNCERTAINTY: Final[float] = 1.0


class BackendState(StrEnum):
    """Backend lifecycle."""

    CREATED = "CREATED"
    LOADED = "LOADED"
    FAILED = "FAILED"
    CLOSED = "CLOSED"


class InferenceOutcome(StrEnum):
    """Normalized outcome of a single inference."""

    NORMAL = "NORMAL"
    ANOMALOUS = "ANOMALOUS"
    UNKNOWN = "UNKNOWN"
    DEGRADED = "DEGRADED"


@dataclass(frozen=True, slots=True)
class InferenceResult:
    """The result of one model invocation.

    Attributes:
        model_id: Manifest model identifier.
        model_version: Semantic version of the model.
        outcome: Normalized classification.
        score: Primary decision statistic (backend-defined scale).
        confidence: Trust in the score, in ``[0, 1]``.
        uncertainty: ``1 - confidence``, always available.
        latency_s: Wall time of the inference call only.
        evidence: Values that produced the decision, for auditability.
        timestamp: Processing time (epoch seconds).
        quality: Worst data quality seen by the backend.
        metadata: Backend-specific extras.
    """

    model_id: str
    model_version: str
    outcome: InferenceOutcome
    score: float
    confidence: float
    latency_s: float
    evidence: Mapping[str, Any] = field(default_factory=dict)
    timestamp: float = 0.0
    quality: DataQuality = DataQuality.GOOD
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "confidence",
                           validate_probability(float(self.confidence), "confidence"))
        if not np.isfinite(float(self.score)):
            raise InferenceError("inference score must be finite",
                                 details={"score": repr(self.score)})
        if self.latency_s < 0:
            raise InferenceError("latency must be non-negative",
                                 details={"latency": self.latency_s})

    @property
    def uncertainty(self) -> float:
        """Always-available uncertainty: ``1 - confidence``."""
        return 1.0 - float(self.confidence)

    @property
    def is_anomalous(self) -> bool:
        return self.outcome is InferenceOutcome.ANOMALOUS

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "model_version": self.model_version,
            "outcome": self.outcome.value,
            "score": float(self.score),
            "confidence": float(self.confidence),
            "uncertainty": self.uncertainty,
            "latency_s": float(self.latency_s),
            "evidence": dict(self.evidence),
            "timestamp": self.timestamp,
            "quality": self.quality.value,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class BackendMetadata:
    """Static description of a backend."""

    backend_id: str
    kind: str
    version: str
    supports_batch: bool = False
    requires_extra: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend_id": self.backend_id,
            "kind": self.kind,
            "version": self.version,
            "supports_batch": self.supports_batch,
            "requires_extra": self.requires_extra,
        }


class ModelBackend(ABC):
    """Abstract model backend.

    Implementations must:
        * validate the manifest and refuse incompatible models,
        * raise :class:`InferenceError` subclasses on failure, never silently return,
        * measure latency with ``time.perf_counter``,
        * scale confidence by input data quality.
    """

    def __init__(self, manifest: ModelManifest | None = None) -> None:
        self.manifest = manifest
        self.state: BackendState = BackendState.CREATED

    # -- lifecycle ---------------------------------------------------------

    def load(self, manifest: ModelManifest, artifact: Any = None) -> None:
        """Validate and load a model. Sets state to LOADED or FAILED."""
        try:
            self._load(manifest, artifact)
            self.manifest = manifest
            self.state = BackendState.LOADED
        except ModelError:
            self.state = BackendState.FAILED
            raise
        except Exception as exc:
            self.state = BackendState.FAILED
            raise InferenceError(
                f"backend {type(self).__name__} failed to load "
                f"{manifest.identity if manifest else '<none>'}: {exc}",
                details={"backend": type(self).__name__},
            ) from exc

    def close(self) -> None:
        self._close()
        self.state = BackendState.CLOSED

    def infer(self, features: FeatureVector) -> InferenceResult:
        """Run one inference with latency measurement and error normalization."""
        if self.state is not BackendState.LOADED:
            raise InferenceError(
                f"backend is {self.state.value}, not LOADED",
                details={"state": self.state.value},
            )
        assert self.manifest is not None
        self.manifest.check_input(features.names)
        started = time.perf_counter()
        try:
            result = self._infer(features)
        except InferenceError:
            raise
        except Exception as exc:
            raise InferenceError(
                f"inference failed in {type(self).__name__}: {exc}",
                details={"backend": type(self).__name__,
                         "model": self.manifest.identity},
            ) from exc
        latency = time.perf_counter() - started
        object.__setattr__(result, "latency_s", latency)
        return result

    # -- hooks -------------------------------------------------------------

    @abstractmethod
    def _load(self, manifest: ModelManifest, artifact: Any) -> None: ...

    @abstractmethod
    def _infer(self, features: FeatureVector) -> InferenceResult: ...

    def _close(self) -> None:  # noqa: B027 - deliberate optional hook
        """Optional teardown."""

    @abstractmethod
    def metadata(self) -> BackendMetadata: ...

    # -- helpers -----------------------------------------------------------

    def _quality_confidence(self, features: FeatureVector, base: float) -> float:
        """Scale a base confidence by the input data quality."""
        from ..core.quality import quality_confidence_factor

        factor = quality_confidence_factor(features.quality)
        return float(np.clip(base * factor, 0.0, 1.0))

    def _result(
        self,
        outcome: InferenceOutcome,
        score: float,
        confidence: float,
        *,
        evidence: Mapping[str, Any] | None = None,
        features: FeatureVector | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> InferenceResult:
        """Build a result, applying quality-based confidence attenuation."""
        assert self.manifest is not None
        confidence = float(np.clip(confidence, 0.0, 1.0))
        if features is not None:
            confidence = self._quality_confidence(features, confidence)
        return InferenceResult(
            model_id=self.manifest.model_id,
            model_version=self.manifest.version,
            outcome=outcome,
            score=float(score),
            confidence=confidence,
            latency_s=0.0,
            evidence=dict(evidence or {}),
            timestamp=time.time(),
            quality=features.quality if features is not None else DataQuality.GOOD,
            metadata=dict(metadata or {}),
        )
