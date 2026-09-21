"""Model backend registry and a small plugin backend for user-supplied callables.

Registration is explicit (threat T-17): backends are registered by the
application. Nothing scans the filesystem and nothing imports a module named by
untrusted data.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Final

from ..core.contracts import FeatureVector
from ..core.errors import PluginError
from ..core.validation import validate_identifier
from .base import BackendMetadata, InferenceOutcome, InferenceResult, ModelBackend
from .manifest import ModelManifest

__all__ = [
    "BACKEND_PREFIX",
    "CallableBackend",
    "create_backend",
    "model_registry",
    "register_model_backend",
]

BACKEND_PREFIX: Final[str] = "sofia_"


class _BackendRegistry:
    """Explicit name → factory mapping for model backends."""

    def __init__(self) -> None:
        self._factories: dict[str, Callable[..., ModelBackend]] = {}

    def register(self, key: str, factory: Callable[..., ModelBackend],
                 *, overwrite: bool = False) -> None:
        name = validate_identifier(key, "backend.key").lower()
        if name in self._factories and not overwrite:
            raise PluginError(
                f"backend {name!r} is already registered",
                details={"backend": name},
            )
        if not callable(factory):
            raise PluginError(f"backend factory for {name!r} is not callable",
                              details={"backend": name})
        self._factories[name] = factory

    def get(self, key: str) -> Callable[..., ModelBackend]:
        name = validate_identifier(key, "backend.key").lower()
        factory = self._factories.get(name)
        if factory is None:
            raise PluginError(
                f"backend {name!r} is not registered. Known: {sorted(self._factories)}",
                details={"backend": name, "known": sorted(self._factories)},
            )
        return factory

    def create(self, key: str, **kwargs: object) -> ModelBackend:
        factory = self.get(key)
        try:
            return factory(**kwargs)
        except PluginError:
            raise
        except Exception as exc:
            raise PluginError(
                f"backend factory {key!r} failed: {exc}", details={"backend": key}
            ) from exc

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._factories))

    def as_mapping(self) -> Mapping[str, Callable[..., ModelBackend]]:
        return dict(self._factories)

    def __contains__(self, key: str) -> bool:
        return key.lower() in self._factories

    def __len__(self) -> int:
        return len(self._factories)


model_registry: Final[_BackendRegistry] = _BackendRegistry()


def register_model_backend(key: str, factory: Callable[..., ModelBackend],
                           *, overwrite: bool = False) -> None:
    """Register a model backend factory."""
    model_registry.register(key, factory, overwrite=overwrite)


def create_backend(key: str, **kwargs: object) -> ModelBackend:
    """Instantiate a registered backend."""
    return model_registry.create(key, **kwargs)


class CallableBackend(ModelBackend):
    """Wrap a user-supplied function as a backend (Demo 4: custom model adapter).

    The wrapped function receives a :class:`FeatureVector` and must return
    ``(score, confidence, evidence)``. It runs inside the standard
    ``ModelBackend`` error handling, timing and quality attenuation, so a user
    model gets the same guarantees as a built-in one.
    """

    def __init__(
        self,
        fn: Callable[[FeatureVector], tuple[float, float, dict[str, object]]] | None = None,
        *,
        backend_id: str = "callable",
        anomaly_threshold: float | None = None,
    ) -> None:
        super().__init__()
        self._fn = fn
        self.backend_id = backend_id
        self.anomaly_threshold = anomaly_threshold

    def _load(self, manifest: ModelManifest, artifact: Any = None) -> None:
        if artifact is not None and callable(artifact) and self._fn is None:
            self._fn = artifact
        if self._fn is None:
            raise PluginError(
                "CallableBackend requires a callable via constructor or artifact",
                details={},
            )
        meta = dict(manifest.training_metadata or {})
        if self.anomaly_threshold is None and "anomaly_threshold" in meta:
            self.anomaly_threshold = float(meta["anomaly_threshold"])

    def _infer(self, features: FeatureVector) -> InferenceResult:
        assert self._fn is not None
        try:
            score, confidence, evidence = self._fn(features)
        except Exception as exc:
            raise PluginError(f"custom model function failed: {exc}", details={}) from exc
        score = float(score)
        confidence = float(confidence)
        if self.anomaly_threshold is not None:
            anomalous = score >= self.anomaly_threshold
            outcome = InferenceOutcome.ANOMALOUS if anomalous else InferenceOutcome.NORMAL
        else:
            outcome = InferenceOutcome.UNKNOWN
        return self._result(
            outcome,
            score=score,
            confidence=confidence,
            evidence=dict(evidence or {}),
            features=features,
            metadata={"backend": self.backend_id},
        )

    def metadata(self) -> BackendMetadata:
        return BackendMetadata(backend_id=self.backend_id, kind="custom", version="1.0")


def register_builtin_backends() -> None:
    """Register the backends that ship with Sofia."""
    from .detectors import (
        CusumDetector,
        EwmaDetector,
        IqrDetector,
        MadDetector,
        ThresholdDetector,
        ZScoreDetector,
    )

    register_model_backend("threshold", ThresholdDetector)
    register_model_backend("zscore", ZScoreDetector)
    register_model_backend("mad", MadDetector)
    register_model_backend("ewma", EwmaDetector)
    register_model_backend("iqr", IqrDetector)
    register_model_backend("cusum", CusumDetector)
    register_model_backend("callable", CallableBackend)
    register_model_backend("onnx", _lazy_onnx)
    register_model_backend("torch", _lazy_torch)


def _lazy_onnx(**kwargs: object) -> ModelBackend:
    from .onnx_backend import OnnxBackend

    return OnnxBackend(**kwargs)  # type: ignore[arg-type]


def _lazy_torch(**kwargs: object) -> ModelBackend:
    from .torch_backend import TorchBackend

    return TorchBackend(**kwargs)  # type: ignore[arg-type]


register_builtin_backends()
