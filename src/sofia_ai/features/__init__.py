"""Feature extraction: statistical, spectral and rotating-machinery primitives.

Deterministic and independently testable. The canonical ordering in
:data:`sofia_ai.features.extractor.FEATURE_NAMES` is the contract shared with the
embedded C runtime.
"""

from __future__ import annotations

from .extractor import (
    EXTRACTOR_ID,
    EXTRACTOR_VERSION,
    FEATURE_NAMES,
    FeatureExtractor,
    extract_from_array,
)
from .rotating import (
    OrderSpec,
    bearing_band_energies,
    envelope_analysis,
    harmonic_energies,
    orders_from_rpm,
    rpm_to_hz,
    sideband_energy,
    speed_normalized_resample,
)
from .spectral import SPECTRAL_FEATURE_NAMES, spectral_features
from .statistical import (
    STATISTICAL_FEATURE_NAMES,
    crest_factor,
    energy,
    kurtosis,
    peak,
    peak_to_peak,
    rms,
    skewness,
    statistical_features,
    std_dev,
    zero_crossing_rate,
)

__all__ = [
    "EXTRACTOR_ID",
    "EXTRACTOR_VERSION",
    "FEATURE_NAMES",
    "SPECTRAL_FEATURE_NAMES",
    "STATISTICAL_FEATURE_NAMES",
    "FeatureExtractor",
    "OrderSpec",
    "bearing_band_energies",
    "crest_factor",
    "energy",
    "envelope_analysis",
    "extract_from_array",
    "harmonic_energies",
    "kurtosis",
    "orders_from_rpm",
    "peak",
    "peak_to_peak",
    "rms",
    "rpm_to_hz",
    "sideband_energy",
    "skewness",
    "spectral_features",
    "speed_normalized_resample",
    "statistical_features",
    "std_dev",
    "zero_crossing_rate",
]
