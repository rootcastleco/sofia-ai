"""Inference: the model backend abstraction.

The core never imports PyTorch, ONNX Runtime or any ML framework. Backends are
registered explicitly and loaded through a validated, checksum-protected manifest.
"""

from __future__ import annotations

from .base import (
    BackendMetadata,
    BackendState,
    InferenceOutcome,
    InferenceResult,
    ModelBackend,
)
from .detectors import (
    CusumDetector,
    DetectorConfig,
    EwmaDetector,
    IqrDetector,
    MadDetector,
    ThresholdDetector,
    ZScoreDetector,
    build_manifest_for,
    cusum,
    ewma,
    iqr_bounds,
    median_absolute_deviation,
    robust_zscore,
    zscore,
)
from .manifest import (
    ModelManifest,
    SchemaSpec,
    load_manifest,
    save_manifest,
    validate_manifest,
)
from .registry import (
    CallableBackend,
    create_backend,
    model_registry,
    register_model_backend,
)

__all__ = [
    "BackendMetadata",
    "BackendState",
    "CallableBackend",
    "CusumDetector",
    "DetectorConfig",
    "EwmaDetector",
    "InferenceOutcome",
    "InferenceResult",
    "IqrDetector",
    "MadDetector",
    "ModelBackend",
    "ModelManifest",
    "SchemaSpec",
    "ThresholdDetector",
    "ZScoreDetector",
    "build_manifest_for",
    "create_backend",
    "cusum",
    "ewma",
    "iqr_bounds",
    "load_manifest",
    "median_absolute_deviation",
    "model_registry",
    "register_model_backend",
    "robust_zscore",
    "save_manifest",
    "validate_manifest",
    "zscore",
]
