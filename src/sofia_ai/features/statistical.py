"""Time-domain statistical features.

Every function is pure and validated against closed-form expectations in
``tests/property/test_signal_properties.py``. Conventions:

* kurtosis is **excess-free** (Fisher is available separately) — the Gaussian
  reference value is 3.0, which is the convention used in vibration monitoring;
* population variance/standard deviation (``ddof=0``) is used so results are
  independent of sample count for a stationary signal;
* division-by-zero cases return a defined value rather than NaN.
"""

from __future__ import annotations

import math

import numpy as np

from ..core.errors import InsufficientDataError

__all__ = [
    "STATISTICAL_FEATURE_NAMES",
    "crest_factor",
    "energy",
    "excess_kurtosis",
    "impulse_factor",
    "kurtosis",
    "margin_factor",
    "mean",
    "peak",
    "peak_to_peak",
    "rms",
    "shape_factor",
    "skewness",
    "statistical_features",
    "std_dev",
    "variance",
    "zero_crossing_rate",
]


def _as_1d(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).ravel()
    if arr.size == 0:
        raise InsufficientDataError("feature input must not be empty", details={})
    return arr


def mean(values: np.ndarray) -> float:
    """Arithmetic mean."""
    return float(np.mean(_as_1d(values)))


def rms(values: np.ndarray) -> float:
    """Root mean square. For a constant signal ``c`` this equals ``|c|``."""
    x = _as_1d(values)
    return float(np.sqrt(np.mean(x * x)))


def peak(values: np.ndarray) -> float:
    """Maximum absolute value."""
    return float(np.max(np.abs(_as_1d(values))))


def peak_to_peak(values: np.ndarray) -> float:
    """Maximum minus minimum."""
    x = _as_1d(values)
    return float(np.max(x) - np.min(x))


def variance(values: np.ndarray) -> float:
    """Population variance (ddof=0)."""
    return float(np.var(_as_1d(values)))


def std_dev(values: np.ndarray) -> float:
    """Population standard deviation (ddof=0)."""
    return float(np.std(_as_1d(values)))


def _safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    if not math.isfinite(denominator) or abs(denominator) < 1e-15:
        return default
    return float(numerator / denominator)


def crest_factor(values: np.ndarray) -> float:
    """peak / RMS. Always >= 1 for a non-zero signal."""
    return _safe_divide(peak(values), rms(values), default=0.0)


def shape_factor(values: np.ndarray) -> float:
    """RMS / mean-absolute."""
    x = _as_1d(values)
    return _safe_divide(rms(x), float(np.mean(np.abs(x))), default=0.0)


def impulse_factor(values: np.ndarray) -> float:
    """peak / mean-absolute."""
    x = _as_1d(values)
    return _safe_divide(peak(x), float(np.mean(np.abs(x))), default=0.0)


def margin_factor(values: np.ndarray) -> float:
    """peak / (mean of sqrt(|x|))^2 — a sensitive spikiness indicator."""
    x = _as_1d(values)
    root_mean_square_amplitude = float(np.mean(np.sqrt(np.abs(x))))
    return _safe_divide(peak(x), root_mean_square_amplitude**2, default=0.0)


def skewness(values: np.ndarray) -> float:
    """Population skewness. Zero for a symmetric signal."""
    x = _as_1d(values)
    sigma = float(np.std(x))
    if sigma < 1e-15:
        return 0.0
    return float(np.mean(((x - np.mean(x)) / sigma) ** 3))


def kurtosis(values: np.ndarray) -> float:
    """Population kurtosis. Equals 3.0 for a Gaussian signal."""
    x = _as_1d(values)
    sigma = float(np.std(x))
    if sigma < 1e-15:
        return 0.0
    return float(np.mean(((x - np.mean(x)) / sigma) ** 4))


def excess_kurtosis(values: np.ndarray) -> float:
    """Kurtosis minus 3. Zero for a Gaussian signal."""
    return kurtosis(values) - 3.0


def zero_crossing_rate(values: np.ndarray) -> float:
    """Fraction of adjacent sample pairs whose product is negative, in ``[0, 1]``.

    The signal is mean-removed first, which is the convention used for rotating
    machinery: a zero-mean sinusoid at frequency ``f`` and rate ``fs`` has a rate
    of ``2f/fs``.
    """
    x = _as_1d(values)
    if x.size < 2:
        return 0.0
    centred = x - float(np.mean(x))
    signs = np.sign(centred)
    nonzero = signs[signs != 0]
    if nonzero.size < 2:
        return 0.0
    crossings = int(np.count_nonzero(np.diff(nonzero) != 0))
    return float(crossings / (x.size - 1))


def energy(values: np.ndarray) -> float:
    """Sum of squares."""
    x = _as_1d(values)
    return float(np.sum(x * x))


STATISTICAL_FEATURE_NAMES: tuple[str, ...] = (
    "mean",
    "rms",
    "peak",
    "peak_to_peak",
    "variance",
    "std",
    "crest_factor",
    "shape_factor",
    "impulse_factor",
    "margin_factor",
    "skewness",
    "kurtosis",
    "zero_crossing_rate",
    "energy",
)


def statistical_features(values: np.ndarray) -> dict[str, float]:
    """Compute all time-domain features in a fixed order.

    Returns:
        Mapping with exactly :data:`STATISTICAL_FEATURE_NAMES` as keys.
    """
    x = _as_1d(values)
    return {
        "mean": mean(x),
        "rms": rms(x),
        "peak": peak(x),
        "peak_to_peak": peak_to_peak(x),
        "variance": variance(x),
        "std": std_dev(x),
        "crest_factor": crest_factor(x),
        "shape_factor": shape_factor(x),
        "impulse_factor": impulse_factor(x),
        "margin_factor": margin_factor(x),
        "skewness": skewness(x),
        "kurtosis": kurtosis(x),
        "zero_crossing_rate": zero_crossing_rate(x),
        "energy": energy(x),
    }
