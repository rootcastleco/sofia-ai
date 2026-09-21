"""ONNX Runtime backend (optional extra: ``pip install sofia-engine[onnx]``).

ONNX Runtime is the preferred edge inference path: it is a small, self-contained
runtime with no Python training stack. The artifact is loaded from a file with a
checksum-protected manifest; there is no pickle and no arbitrary deserialization.

``onnxruntime`` is imported lazily and the session object is injectable, so unit
tests run without the extra installed.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Final

import numpy as np

from ..core.contracts import FeatureVector
from ..core.errors import InferenceError, ModelError
from .base import BackendMetadata, InferenceOutcome, InferenceResult, ModelBackend
from .manifest import ModelManifest

__all__ = ["OnnxBackend"]

_ANOMALY_THRESHOLD_KEY: Final[str] = "anomaly_threshold"


class OnnxBackend(ModelBackend):
    """Run an ONNX model over a :class:`FeatureVector`.

    The manifest must declare ``input_schema.features`` in the exact order the
    model expects, and may declare ``training_metadata["anomaly_threshold"]`` to
    map the output score onto an outcome. Without a threshold the outcome is
    ``UNKNOWN`` with confidence 0 — Sofia never guesses a threshold.
    """

    def __init__(self, *, session_factory: Callable[[str], Any] | None = None) -> None:
        super().__init__()
        self._session_factory = session_factory
        self._session: Any = None
        self._input_name: str | None = None
        self._threshold: float | None = None

    def _load(self, manifest: ModelManifest, artifact: Any) -> None:
        if not manifest.artifact_path:
            raise ModelError(
                f"model {manifest.identity} declares no artifact_path",
                details={"model": manifest.identity},
            )
        manifest.verify_artifact()
        self._session = self._make_session(manifest.artifact_path)
        self._input_name = self._first_input_name(self._session)
        metadata = dict(manifest.training_metadata or {})
        self._threshold = (
            float(metadata[_ANOMALY_THRESHOLD_KEY])
            if _ANOMALY_THRESHOLD_KEY in metadata else None
        )
        manifest.check_input(manifest.input_schema.features)

    def _make_session(self, path: str) -> Any:
        if self._session_factory is not None:
            return self._session_factory(path)
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ModelError(
                "onnxruntime is not installed. Install the 'onnx' extra: "
                "pip install sofia-engine[onnx]",
                details={"extra": "onnx"},
            ) from exc
        try:
            return ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        except Exception as exc:
            raise ModelError(
                f"failed to create an ONNX session for {path}: {exc}",
                details={"path": path},
            ) from exc

    @staticmethod
    def _first_input_name(session: Any) -> str:
        inputs = getattr(session, "get_inputs", None)
        if callable(inputs):
            entries = inputs()
            if entries:
                return str(entries[0].name)
        raise ModelError("ONNX session exposes no inputs", details={})

    def _infer(self, features: FeatureVector) -> InferenceResult:
        assert self.manifest is not None
        names = self.manifest.input_schema.features or features.names
        self.manifest.check_input(features.names)
        try:
            ordered = features.select(tuple(names)).as_array()
        except ModelError:
            raise
        except Exception as exc:
            raise InferenceError(
                f"feature selection failed: {exc}", details={}
            ) from exc

        array = np.asarray(ordered, dtype=np.float32).reshape(
            1, -1
        )
        try:
            outputs = self._session.run(None, {self._input_name: array})
        except Exception as exc:
            raise InferenceError(
                f"ONNX inference failed: {exc}",
                details={"model": self.manifest.identity},
            ) from exc
        if not outputs:
            raise InferenceError("ONNX session returned no output", details={})

        raw = np.asarray(outputs[0], dtype=np.float64).ravel()
        if raw.size == 0:
            raise InferenceError("ONNX output is empty", details={})
        score = float(raw[0])
        if not np.isfinite(score):
            raise InferenceError("ONNX produced a non-finite score",
                                 details={"score": repr(score)})

        if self._threshold is None:
            return self._result(
                InferenceOutcome.UNKNOWN,
                score=score,
                confidence=0.0,
                evidence={"raw_output": [float(v) for v in raw[:8]],
                          "reason": "no anomaly_threshold in manifest"},
                features=features,
                metadata={"backend": "onnx"},
            )

        anomalous = score >= self._threshold
        margin = abs(score - self._threshold) / max(abs(self._threshold), 1e-9)
        confidence = float(np.clip(margin / (1.0 + margin), 0.0, 1.0))
        return self._result(
            InferenceOutcome.ANOMALOUS if anomalous else InferenceOutcome.NORMAL,
            score=score,
            confidence=confidence,
            evidence={"threshold": self._threshold, "raw_output_size": int(raw.size)},
            features=features,
            metadata={"backend": "onnx"},
        )

    def metadata(self) -> BackendMetadata:
        return BackendMetadata(
            backend_id="OnnxBackend",
            kind="onnx",
            version=self._session_version(),
            supports_batch=True,
            requires_extra="onnx",
        )

    def _session_version(self) -> str:
        try:
            import onnxruntime as ort

            return str(ort.__version__)
        except ImportError:
            return "not-installed"

    def close(self) -> None:
        self._session = None
        super().close()

    def __enter__(self) -> OnnxBackend:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

