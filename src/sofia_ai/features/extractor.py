"""Deterministic feature extraction over signal windows.

The extractor is the boundary between the signal pipeline and inference. Its
output, :class:`~sofia_ai.core.contracts.FeatureVector`, is:

* **ordered** — names are a fixed tuple, never derived from dict iteration,
* **versioned** — ``EXTRACTOR_ID`` / ``EXTRACTOR_VERSION`` travel with every vector,
  so a model trained against one version can refuse a mismatched vector,
* **finite** — any non-finite feature raises rather than poisoning a model.

This is the same contract implemented by ``embedded/src/sofia_features.c``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np

from ..core.contracts import FeatureVector, SignalWindow
from ..core.errors import FeatureError
from ..core.validation import validate_positive_float
from .rotating import OrderSpec, harmonic_energies
from .spectral import SPECTRAL_FEATURE_NAMES, spectral_features
from .statistical import STATISTICAL_FEATURE_NAMES, statistical_features

__all__ = [
    "EXTRACTOR_ID",
    "EXTRACTOR_VERSION",
    "FEATURE_NAMES",
    "FeatureExtractor",
    "extract_from_array",
]

EXTRACTOR_ID: Final[str] = "sofia.core.vibration"
EXTRACTOR_VERSION: Final[str] = "1.0"

#: Canonical feature ordering. This is the embedded export contract.
FEATURE_NAMES: Final[tuple[str, ...]] = STATISTICAL_FEATURE_NAMES + SPECTRAL_FEATURE_NAMES


def extract_from_array(
    values: np.ndarray,
    sample_rate: float,
    *,
    channel: str = "",
    quality: object | None = None,
    shaft_hz: float | None = None,
) -> FeatureVector:
    """Extract the canonical feature vector from a raw array.

    Args:
        values: 1-D float array.
        sample_rate: Sample rate in Hz.
        channel: Source channel name (provenance only).
        quality: Optional :class:`DataQuality` to attach.
        shaft_hz: Optional shaft frequency enabling order-energy features.

    Raises:
        FeatureError: if any computed feature is non-finite.
    """
    sample_rate = validate_positive_float(sample_rate, "sample_rate")
    arr = np.asarray(values, dtype=np.float64).ravel()
    if arr.size == 0:
        raise FeatureError("cannot extract features from an empty window", details={})

    features: dict[str, float] = {}
    features.update(statistical_features(arr))
    features.update(spectral_features(arr, sample_rate))
    if shaft_hz is not None and shaft_hz > 0:
        features.update(harmonic_energies(arr, sample_rate, OrderSpec(shaft_hz=shaft_hz)))

    names = tuple(name for name in FEATURE_NAMES if name in features)
    names += tuple(sorted(n for n in features if n not in FEATURE_NAMES))
    values_tuple = tuple(float(features[name]) for name in names)

    for name, value in zip(names, values_tuple, strict=True):
        if not np.isfinite(value):
            raise FeatureError(
                f"feature {name!r} is not finite", details={"feature": name}
            )

    from ..core.quality import DataQuality

    return FeatureVector(
        names=names,
        values=values_tuple,
        extractor_id=EXTRACTOR_ID,
        extractor_version=EXTRACTOR_VERSION,
        source_channel=channel,
        sample_rate=float(sample_rate),
        quality=quality if isinstance(quality, DataQuality) else DataQuality.GOOD,
        metadata={"shaft_hz": shaft_hz} if shaft_hz else {},
    )


@dataclass(frozen=True, slots=True)
class FeatureExtractor:
    """Configured, reusable feature extractor.

    Args:
        shaft_hz: Optional shaft frequency for order-energy extension features.
        require_finite: Reject windows producing non-finite features.
    """

    shaft_hz: float | None = None
    require_finite: bool = True

    def extract(self, window: SignalWindow) -> FeatureVector:
        """Extract features from a :class:`SignalWindow`."""
        vector = extract_from_array(
            window.values,
            window.sample_rate,
            channel=window.channel,
            quality=window.quality,
            shaft_hz=self.shaft_hz,
        )
        return vector

    def extract_many(self, windows: list[SignalWindow]) -> list[FeatureVector]:
        """Extract features from several windows, preserving order."""
        return [self.extract(w) for w in windows]

    @property
    def extractor_id(self) -> str:
        return EXTRACTOR_ID

    @property
    def extractor_version(self) -> str:
        return EXTRACTOR_VERSION
