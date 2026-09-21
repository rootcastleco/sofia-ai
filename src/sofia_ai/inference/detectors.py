"""Baseline anomaly detectors.

These are deliberately simple, auditable and cheap. In industrial telemetry a
robust median/MAD detector on a well-chosen feature routinely outperforms an
unnecessary neural model, and it can be explained to a maintenance engineer —
which matters more than marginal accuracy.

Every detector is a :class:`~sofia_ai.inference.base.ModelBackend`:

* it declares the single feature it consumes via its manifest input schema,
* it maintains a **bounded** baseline window (``window_size``),
* :meth:`fit` establishes the baseline; :meth:`infer` scores one feature vector,
* confidence is attenuated by input data quality.

Pure functional forms are exported too (``zscore``, ``mad_score``, ``ewma``, ...)
so they can be unit-tested and used without the backend machinery.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

import numpy as np

from ..core.contracts import FeatureVector
from ..core.errors import ConfigurationError, InsufficientDataError, ModelError
from ..core.validation import validate_positive_float, validate_positive_int
from .base import BackendMetadata, InferenceOutcome, InferenceResult, ModelBackend
from .manifest import ModelManifest, SchemaSpec

__all__ = [
    "CusumDetector",
    "DetectorConfig",
    "EwmaDetector",
    "IqrDetector",
    "MadDetector",
    "ThresholdDetector",
    "ZScoreDetector",
    "cusum",
    "ewma",
    "iqr_bounds",
    "median_absolute_deviation",
    "robust_zscore",
    "zscore",
]

_MAD_TO_SIGMA: Final[float] = 1.4826  # MAD -> sigma for a Gaussian


# ---------------------------------------------------------------------------
# pure functions
# ---------------------------------------------------------------------------


def median_absolute_deviation(values: Sequence[float] | np.ndarray) -> float:
    """Median absolute deviation. Robust to up to 50 % contaminated samples."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return 0.0
    return float(np.median(np.abs(arr - np.median(arr))))


def zscore(value: float, mean: float, std: float) -> float:
    """Standard score, with a defined result when the spread is zero."""
    if not math.isfinite(std) or std <= 1e-12:
        return 0.0 if abs(value - mean) <= 1e-12 else math.copysign(float("inf"), value - mean)
    return float((value - mean) / std)


def robust_zscore(value: float, median: float, mad: float) -> float:
    """Median/MAD-based score. Uses ``1.4826 * MAD`` as the sigma estimate."""
    sigma = mad * _MAD_TO_SIGMA
    if not math.isfinite(sigma) or sigma <= 1e-12:
        return 0.0 if abs(value - median) <= 1e-12 else math.copysign(
            float("inf"), value - median
        )
    return float((value - median) / sigma)


def ewma(values: Sequence[float] | np.ndarray, alpha: float) -> np.ndarray:
    """Exponentially weighted moving average.

    Args:
        values: 1-D input.
        alpha: Smoothing factor in ``(0, 1]``.

    Returns:
        Smoothed series, same length.
    """
    validate_positive_float(alpha, "alpha", maximum=1.0)
    arr = np.asarray(values, dtype=np.float64)
    out = np.empty_like(arr)
    if arr.size == 0:
        return out
    acc = float(arr[0])
    out[0] = acc
    for i in range(1, arr.size):
        acc = alpha * float(arr[i]) + (1.0 - alpha) * acc
        out[i] = acc
    return out


def iqr_bounds(values: Sequence[float] | np.ndarray, k: float = 1.5
               ) -> tuple[float, float]:
    """Tukey fences: ``(q1 - k*iqr, q3 + k*iqr)``."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return 0.0, 0.0
    q1 = float(np.percentile(arr, 25))
    q3 = float(np.percentile(arr, 75))
    iqr = q3 - q1
    return q1 - float(k) * iqr, q3 + float(k) * iqr


def cusum(values: Sequence[float] | np.ndarray, *, target: float, slack: float,
          threshold: float) -> tuple[np.ndarray, int | None]:
    """One-sided cumulative sum change detection.

    Args:
        values: 1-D input.
        target: Reference (in-control) mean.
        slack: Allowance ``k``; deviations smaller than this are not accumulated.
        threshold: Alarm threshold ``h``.

    Returns:
        ``(cumulative_series, alarm_index_or_None)``.
    """
    arr = np.asarray(values, dtype=np.float64)
    out = np.zeros(arr.size, dtype=np.float64)
    if arr.size == 0:
        return out, None
    validate_positive_float(slack, "slack")
    validate_positive_float(threshold, "threshold")
    high = 0.0
    alarm: int | None = None
    for i in range(arr.size):
        high = max(0.0, high + float(arr[i]) - target - slack)
        out[i] = high
        if alarm is None and high > threshold:
            alarm = i
    return out, alarm


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DetectorConfig:
    """Shared detector settings. All bounds are validated."""

    feature: str = "rms"
    window_size: int = 256
    warmup: int = 32
    threshold: float = 3.0
    direction: str = "both"  # both | high | low
    alpha: float = 0.2
    k: float = 1.5
    slack: float = 0.5
    alarm_threshold: float = 5.0
    min_samples: int = 8

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "window_size",
            validate_positive_int(self.window_size, "window_size", maximum=1_000_000),
        )
        object.__setattr__(
            self, "warmup", validate_positive_int(self.warmup, "warmup", maximum=1_000_000),
        )
        if self.warmup > self.window_size:
            raise ConfigurationError(
                "warmup must not exceed window_size",
                details={"warmup": self.warmup, "window_size": self.window_size},
            )
        object.__setattr__(
            self, "min_samples",
            validate_positive_int(self.min_samples, "min_samples", maximum=1_000_000),
        )
        validate_positive_float(self.threshold, "threshold", maximum=1e9)
        if self.direction not in ("both", "high", "low"):
            raise ConfigurationError(
                f"direction must be both|high|low, got {self.direction!r}",
                details={"direction": self.direction},
            )
        validate_positive_float(self.alpha, "alpha", maximum=1.0)
        validate_positive_float(self.k, "k", maximum=100.0)
        validate_positive_float(self.slack, "slack")
        validate_positive_float(self.alarm_threshold, "alarm_threshold")


class _WindowedDetector(ModelBackend):
    """Shared machinery for detectors with a bounded rolling baseline."""

    def __init__(self, config: DetectorConfig | None = None) -> None:
        super().__init__()
        self.config = config or DetectorConfig()
        self._window: deque[float] = deque(maxlen=self.config.window_size)
        self._samples_seen = 0

    @property
    def feature(self) -> str:
        return self.config.feature

    @property
    def is_warm(self) -> bool:
        return len(self._window) >= max(self.config.min_samples, self.config.warmup)

    def _load(self, manifest: ModelManifest, artifact: Any) -> None:
        if manifest.input_schema.features and self.feature not in manifest.input_schema.features:
            raise ModelError(
                f"manifest input schema {manifest.input_schema.features} does not "
                f"include detector feature {self.feature!r}",
                details={"feature": self.feature},
            )
        self._window.clear()
        self._samples_seen = 0
        if artifact is not None:
            self.fit_from(artifact)

    def _close(self) -> None:
        self._window.clear()

    def fit_from(self, values: Sequence[float] | np.ndarray) -> None:
        """Prime the baseline from a recorded reference series."""
        arr = np.asarray(values, dtype=np.float64).ravel()
        for value in arr[-self.config.window_size :]:
            if math.isfinite(float(value)):
                self._window.append(float(value))
        self._samples_seen += int(arr.size)

    def _observe(self, value: float) -> None:
        """Push a value into the rolling baseline (bounded by window_size)."""
        if math.isfinite(value):
            self._window.append(float(value))
        self._samples_seen += 1

    def _baseline_array(self) -> np.ndarray:
        return np.asarray(self._window, dtype=np.float64)

    def _value_of(self, features: FeatureVector) -> float:
        try:
            index = features.names.index(self.feature)
        except ValueError as exc:
            raise ModelError(
                f"feature {self.feature!r} is not present in the vector",
                details={"feature": self.feature, "available": list(features.names)},
            ) from exc
        return float(features.values[index])

    def _require_warm(self) -> None:
        if not self.is_warm:
            raise InsufficientDataError(
                f"detector needs {max(self.config.min_samples, self.config.warmup)} "
                f"samples, has {len(self._window)}",
                details={"have": len(self._window)},
            )

    def _decide(self, score: float) -> tuple[InferenceOutcome, float]:
        """Map an absolute score onto an outcome and a confidence."""
        threshold = self.config.threshold
        magnitude = abs(score)
        if magnitude < threshold:
            return InferenceOutcome.NORMAL, float(np.clip(1.0 - magnitude / threshold, 0.0, 1.0))
        excess = (magnitude - threshold) / max(threshold, 1e-9)
        confidence = float(np.clip(0.5 + 0.5 * excess / (1.0 + excess), 0.0, 1.0))
        return InferenceOutcome.ANOMALOUS, confidence

    def _directional(self, deviation: float) -> float:
        if self.config.direction == "high":
            return deviation if deviation > 0 else 0.0
        if self.config.direction == "low":
            return -deviation if deviation < 0 else 0.0
        return deviation

    def metadata(self) -> BackendMetadata:
        return BackendMetadata(
            backend_id=type(self).__name__,
            kind="statistical",
            version="1.0",
            supports_batch=False,
        )

    def _infer(self, features: FeatureVector) -> InferenceResult:  # pragma: no cover - abstract
        raise NotImplementedError


class ThresholdDetector(ModelBackend):
    """Static threshold detector. The simplest possible baseline.

    No baseline is learned: ``high``/``low`` limits come from the manifest metadata
    or the constructor. This is the reference against which more complex detectors
    must justify themselves.
    """

    def __init__(self, *, feature: str = "rms", high: float | None = None,
                 low: float | None = None) -> None:
        super().__init__()
        self.feature = feature
        self.high = high
        self.low = low

    def _load(self, manifest: ModelManifest, artifact: Any) -> None:
        meta = dict(manifest.training_metadata or {})
        self.high = float(meta["high"]) if "high" in meta else self.high
        self.low = float(meta["low"]) if "low" in meta else self.low
        if self.high is None and self.low is None:
            raise ModelError(
                "ThresholdDetector requires 'high' or 'low' in manifest metadata "
                "or constructor arguments",
                details={"model": manifest.identity},
            )
        if self.high is not None and self.low is not None and self.low > self.high:
            raise ModelError("low threshold exceeds high threshold", details={})

    def _infer(self, features: FeatureVector) -> InferenceResult:
        try:
            value = float(features.values[features.names.index(self.feature)])
        except ValueError as exc:
            raise ModelError(
                f"feature {self.feature!r} missing from vector", details={}
            ) from exc
        breached: list[str] = []
        distance = 0.0
        if self.high is not None and value > self.high:
            breached.append("high")
            distance = max(distance, (value - self.high) / max(abs(self.high), 1e-9))
        if self.low is not None and value < self.low:
            breached.append("low")
            distance = max(distance, (self.low - value) / max(abs(self.low), 1e-9))
        outcome = InferenceOutcome.ANOMALOUS if breached else InferenceOutcome.NORMAL
        return self._result(
            outcome,
            score=value,
            confidence=float(np.clip(0.5 + 0.5 * distance / (1.0 + distance), 0.0, 1.0))
            if breached else 1.0,
            evidence={"value": value, "high": self.high, "low": self.low,
                      "breached": breached},
            features=features,
            metadata={"detector": "threshold"},
        )

    def metadata(self) -> BackendMetadata:
        return BackendMetadata(backend_id="ThresholdDetector", kind="statistical",
                               version="1.0")


class ZScoreDetector(_WindowedDetector):
    """Rolling mean/standard-deviation z-score detector.

    Sensitive to the very outliers it is meant to find (they inflate sigma), which
    is exactly why :class:`MadDetector` exists. Both are provided so the trade-off
    can be measured rather than assumed.
    """

    def _infer(self, features: FeatureVector) -> InferenceResult:
        value = self._value_of(features)
        if not self.is_warm:
            self._observe(value)
            return self._result(
                InferenceOutcome.UNKNOWN,
                score=0.0,
                confidence=0.0,
                evidence={"value": value, "warmup": True,
                          "have": len(self._window)},
                features=features,
                metadata={"detector": "zscore", "warmup": True},
            )
        baseline = self._baseline_array()
        mean = float(np.mean(baseline))
        std = float(np.std(baseline))
        score = zscore(value, mean, std)
        directed = self._directional(score)
        outcome, confidence = self._decide(directed)
        self._observe(value)
        return self._result(
            outcome,
            score=directed,
            confidence=confidence,
            evidence={"value": value, "mean": mean, "std": std,
                      "n": int(baseline.size)},
            features=features,
            metadata={"detector": "zscore"},
        )


class MadDetector(_WindowedDetector):
    """Robust median/MAD detector. Breakdown point 50 %."""

    def _infer(self, features: FeatureVector) -> InferenceResult:
        value = self._value_of(features)
        if not self.is_warm:
            self._observe(value)
            return self._result(
                InferenceOutcome.UNKNOWN,
                score=0.0,
                confidence=0.0,
                evidence={"value": value, "warmup": True, "have": len(self._window)},
                features=features,
                metadata={"detector": "mad", "warmup": True},
            )
        baseline = self._baseline_array()
        median = float(np.median(baseline))
        mad = median_absolute_deviation(baseline)
        score = robust_zscore(value, median, mad)
        directed = self._directional(score)
        outcome, confidence = self._decide(directed)
        self._observe(value)
        return self._result(
            outcome,
            score=directed,
            confidence=confidence,
            evidence={"value": value, "median": median, "mad": mad,
                      "n": int(baseline.size)},
            features=features,
            metadata={"detector": "mad"},
        )


class EwmaDetector(_WindowedDetector):
    """EWMA control chart. Detects small sustained shifts faster than a z-score."""

    def __init__(self, config: DetectorConfig | None = None) -> None:
        super().__init__(config)
        self._ewma: float | None = None

    def _load(self, manifest: ModelManifest, artifact: Any) -> None:
        super()._load(manifest, artifact)
        self._ewma = None

    def _infer(self, features: FeatureVector) -> InferenceResult:
        value = self._value_of(features)
        alpha = self.config.alpha
        self._ewma = value if self._ewma is None else alpha * value + (1 - alpha) * self._ewma
        if not self.is_warm:
            self._observe(value)
            return self._result(
                InferenceOutcome.UNKNOWN,
                score=0.0,
                confidence=0.0,
                evidence={"value": value, "ewma": self._ewma, "warmup": True},
                features=features,
                metadata={"detector": "ewma", "warmup": True},
            )
        baseline = self._baseline_array()
        centre = float(np.mean(baseline))
        sigma = float(np.std(baseline))
        # Standard EWMA control-chart statistic: the EWMA statistic is compared
        # against the in-control mean with the asymptotic EWMA standard error.
        standard_error = sigma * math.sqrt(alpha / (2.0 - alpha))
        score = zscore(self._ewma, centre, max(standard_error, 1e-12))
        directed = self._directional(score)
        outcome, confidence = self._decide(directed)
        self._observe(value)
        return self._result(
            outcome,
            score=directed,
            confidence=confidence,
            evidence={"value": value, "ewma": self._ewma, "centre": centre,
                      "sigma": sigma, "standard_error": standard_error},
            features=features,
            metadata={"detector": "ewma"},
        )


class IqrDetector(_WindowedDetector):
    """Tukey-fence detector. Non-parametric; makes no Gaussian assumption."""

    def _infer(self, features: FeatureVector) -> InferenceResult:
        value = self._value_of(features)
        if not self.is_warm:
            self._observe(value)
            return self._result(
                InferenceOutcome.UNKNOWN,
                score=0.0,
                confidence=0.0,
                evidence={"value": value, "warmup": True},
                features=features,
                metadata={"detector": "iqr", "warmup": True},
            )
        baseline = self._baseline_array()
        low, high = iqr_bounds(baseline, self.config.k)
        outside = value > high or value < low
        spread = max(high - low, 1e-12)
        distance = max(0.0, (value - high) if value > high else (low - value)) / spread
        outcome = InferenceOutcome.ANOMALOUS if outside else InferenceOutcome.NORMAL
        confidence = float(np.clip(0.5 + distance / (1.0 + distance), 0.0, 1.0)) \
            if outside else 1.0
        self._observe(value)
        return self._result(
            outcome,
            score=distance if outside else 0.0,
            confidence=confidence,
            evidence={"value": value, "low": low, "high": high},
            features=features,
            metadata={"detector": "iqr"},
        )


class CusumDetector(_WindowedDetector):
    """CUSUM detector. Designed for sustained small mean shifts."""

    def __init__(self, config: DetectorConfig | None = None) -> None:
        super().__init__(config)
        self._high = 0.0
        self._low = 0.0

    def _load(self, manifest: ModelManifest, artifact: Any) -> None:
        super()._load(manifest, artifact)
        self._high = 0.0
        self._low = 0.0

    def _infer(self, features: FeatureVector) -> InferenceResult:
        value = self._value_of(features)
        if not self.is_warm:
            self._observe(value)
            return self._result(
                InferenceOutcome.UNKNOWN,
                score=0.0,
                confidence=0.0,
                evidence={"value": value, "warmup": True},
                features=features,
                metadata={"detector": "cusum", "warmup": True},
            )
        baseline = self._baseline_array()
        target = float(np.mean(baseline))
        slack = self.config.slack * max(float(np.std(baseline)), 1e-12)
        self._high = max(0.0, self._high + value - target - slack)
        self._low = max(0.0, self._low + target - value - slack)
        statistic = max(self._high, self._low)
        alarm = statistic > self.config.alarm_threshold * max(
            float(np.std(baseline)), 1e-12
        )
        outcome = InferenceOutcome.ANOMALOUS if alarm else InferenceOutcome.NORMAL
        scale = max(float(np.std(baseline)), 1e-12)
        normalized = statistic / scale
        confidence = float(np.clip(normalized / (1.0 + normalized), 0.0, 1.0)) \
            if alarm else 1.0
        self._observe(value)
        return self._result(
            outcome,
            score=normalized,
            confidence=confidence,
            evidence={"value": value, "target": target, "cusum_high": self._high,
                      "cusum_low": self._low},
            features=features,
            metadata={"detector": "cusum"},
        )


def build_manifest_for(detector: ModelBackend, *, model_id: str, version: str = "1.0.0",
                       metadata: Mapping[str, Any] | None = None) -> ModelManifest:
    """Build a minimal valid manifest for a detector instance."""
    features: tuple[str, ...] = ()
    feature = getattr(detector, "feature", None)
    if isinstance(feature, str):
        features = (feature,)
    else:
        config = getattr(detector, "config", None)
        configured = getattr(config, "feature", None)
        if isinstance(configured, str):
            features = (configured,)
    return ModelManifest(
        model_id=model_id,
        version=version,
        kind="statistical",
        input_schema=SchemaSpec(features=features, shape=(len(features),)),
        output_schema=SchemaSpec(features=("score", "confidence"), shape=(2,)),
        preprocessing_version="1.0",
        training_metadata=dict(metadata or {}),
    )
