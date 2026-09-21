"""Sample-rate conversion: decimation and interpolation.

Resampling declares its input and output rates explicitly. Decimation applies an
anti-aliasing low-pass first; naive decimation is never performed.
"""

from __future__ import annotations

import numpy as np

from ..core.errors import SignalError
from ..core.validation import validate_positive_float, validate_positive_int
from .filters import butterworth_lowpass

__all__ = ["decimate", "interpolate_linear", "resample_linear", "resample_ratio"]


def decimate(values: np.ndarray, factor: int, sample_rate: float) -> tuple[np.ndarray, float]:
    """Decimate by an integer factor with anti-alias filtering.

    Returns:
        ``(decimated, new_sample_rate)``.
    """
    factor = validate_positive_int(factor, "factor", maximum=1024)
    sample_rate = validate_positive_float(sample_rate, "sample_rate")
    x = np.asarray(values, dtype=np.float64)
    if x.size < factor:
        raise SignalError(
            f"signal of {x.size} samples is too short to decimate by {factor}",
            details={"size": int(x.size), "factor": factor},
        )
    if factor == 1:
        return x.copy(), sample_rate
    cutoff = 0.4 * (sample_rate / factor)
    sections = butterworth_lowpass(sample_rate, cutoff, order=4)
    filtered = x
    for section in sections:
        filtered = _apply(filtered, section)
    return np.ascontiguousarray(filtered[::factor]), sample_rate / factor


def _apply(values: np.ndarray, section: object) -> np.ndarray:
    from .filters import apply_biquad

    return apply_biquad(values, section)  # type: ignore[arg-type]


def interpolate_linear(
    values: np.ndarray, source_rate: float, target_rate: float
) -> tuple[np.ndarray, float]:
    """Linear interpolation to a higher sample rate.

    Returns:
        ``(interpolated, target_rate)``.
    """
    source_rate = validate_positive_float(source_rate, "source_rate")
    target_rate = validate_positive_float(target_rate, "target_rate")
    if target_rate < source_rate:
        raise SignalError(
            "target_rate must be >= source_rate for interpolation; use decimate()",
            details={"source": source_rate, "target": target_rate},
        )
    x = np.asarray(values, dtype=np.float64)
    if x.size < 2:
        raise SignalError("interpolation needs at least 2 samples",
                          details={"size": int(x.size)})
    ratio = target_rate / source_rate
    n_out = round((x.size - 1) * ratio) + 1
    source_index = np.arange(x.size, dtype=np.float64)
    target_index = np.linspace(0.0, float(x.size - 1), n_out)
    return np.interp(target_index, source_index, x), target_rate


def resample_linear(
    values: np.ndarray, source_rate: float, target_rate: float
) -> tuple[np.ndarray, float]:
    """Resample to ``target_rate`` by decimation and/or interpolation."""
    if target_rate == source_rate:
        return np.asarray(values, dtype=np.float64).copy(), float(target_rate)
    if target_rate < source_rate:
        ratio = source_rate / target_rate
        if abs(ratio - round(ratio)) < 1e-9:
            return decimate(values, round(ratio), source_rate)
        signal = values
        rate = source_rate
    else:
        signal, rate = values, source_rate
    signal, rate = interpolate_linear(signal, rate, max(target_rate, rate))
    if rate > target_rate:
        ratio = rate / target_rate
        if abs(ratio - round(ratio)) < 1e-9 and round(ratio) > 1:
            return decimate(signal, round(ratio), rate)
    return signal, rate


def resample_ratio(values: np.ndarray, up: int, down: int) -> np.ndarray:
    """Rational resampling by integer ``up``/``down`` with linear interpolation."""
    up = validate_positive_int(up, "up", maximum=1024)
    down = validate_positive_int(down, "down", maximum=1024)
    x = np.asarray(values, dtype=np.float64)
    if x.size < 2:
        raise SignalError("resample_ratio needs at least 2 samples",
                          details={"size": int(x.size)})
    source_index = np.arange(x.size, dtype=np.float64)
    n_out = round((x.size - 1) * up / down) + 1
    target_index = np.linspace(0.0, float(x.size - 1), n_out)
    return np.interp(target_index, source_index, x)
