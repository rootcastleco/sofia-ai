"""Time-domain filtering.

All filters are causal, deterministic and operate on float64 arrays. There is no
state carried between calls, so every function is pure and directly testable.

The IIR filters are implemented as second-order sections (biquads) realized via a
direct-form transposed loop. Coefficients are computed analytically for
Butterworth low-pass/high-pass/band-pass responses.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..core.errors import SignalError
from ..core.validation import validate_positive_float, validate_positive_int

__all__ = [
    "Biquad",
    "apply_biquad",
    "apply_cascade",
    "butterworth_bandpass",
    "butterworth_highpass",
    "butterworth_lowpass",
    "detrend_constant",
    "envelope_remove_dc",
    "median_filter",
    "moving_average",
    "remove_dc",
]


@dataclass(frozen=True, slots=True)
class Biquad:
    """Second-order IIR section: ``b0,b1,b2 / a0,a1,a2`` with ``a0 = 1``."""

    b0: float
    b1: float
    b2: float
    a1: float
    a2: float

    def __post_init__(self) -> None:
        for name in ("b0", "b1", "b2", "a1", "a2"):
            if not math.isfinite(float(getattr(self, name))):
                raise SignalError(f"biquad coefficient {name} is not finite",
                                  details={"coefficient": name})
        if abs(self.a2) >= 1.0:
            raise SignalError("unstable biquad: |a2| must be < 1", details={"a2": self.a2})


def _butterworth_q(order: int, index: int) -> float:
    """Pole Q for Butterworth section ``index`` of ``order``."""
    k = index + 1
    theta = math.pi * (2 * k - 1) / (2 * order)
    return 1.0 / (2.0 * math.cos(theta)) if abs(math.cos(theta)) > 1e-12 else 1e12


def _lowpass_biquad(sample_rate: float, cutoff: float, q: float) -> Biquad:
    w0 = 2.0 * math.pi * cutoff / sample_rate
    if not 0.0 < w0 < math.pi:
        raise SignalError(
            f"cutoff {cutoff} Hz must be below the Nyquist frequency {sample_rate / 2}",
            details={"cutoff": cutoff, "nyquist": sample_rate / 2},
        )
    alpha = math.sin(w0) / (2.0 * q)
    cos_w0 = math.cos(w0)
    a0 = 1.0 + alpha
    return Biquad(
        b0=((1.0 - cos_w0) / 2.0) / a0,
        b1=(1.0 - cos_w0) / a0,
        b2=((1.0 - cos_w0) / 2.0) / a0,
        a1=(-2.0 * cos_w0) / a0,
        a2=(1.0 - alpha) / a0,
    )


def _highpass_biquad(sample_rate: float, cutoff: float, q: float) -> Biquad:
    w0 = 2.0 * math.pi * cutoff / sample_rate
    if not 0.0 < w0 < math.pi:
        raise SignalError(
            f"cutoff {cutoff} Hz must be below the Nyquist frequency {sample_rate / 2}",
            details={"cutoff": cutoff, "nyquist": sample_rate / 2},
        )
    alpha = math.sin(w0) / (2.0 * q)
    cos_w0 = math.cos(w0)
    a0 = 1.0 + alpha
    return Biquad(
        b0=((1.0 + cos_w0) / 2.0) / a0,
        b1=(-(1.0 + cos_w0)) / a0,
        b2=((1.0 + cos_w0) / 2.0) / a0,
        a1=(-2.0 * cos_w0) / a0,
        a2=(1.0 - alpha) / a0,
    )


def _bandpass_biquad(sample_rate: float, low: float, high: float) -> Biquad:
    if low >= high:
        raise SignalError("low cutoff must be below high cutoff",
                          details={"low": low, "high": high})
    f0 = math.sqrt(low * high)
    q = f0 / (high - low)
    w0 = 2.0 * math.pi * f0 / sample_rate
    if not 0.0 < w0 < math.pi:
        raise SignalError(
            f"band centre {f0} Hz must be below the Nyquist frequency {sample_rate / 2}",
            details={"centre": f0, "nyquist": sample_rate / 2},
        )
    alpha = math.sin(w0) / (2.0 * q)
    cos_w0 = math.cos(w0)
    a0 = 1.0 + alpha
    return Biquad(
        b0=alpha / a0,
        b1=0.0,
        b2=-alpha / a0,
        a1=(-2.0 * cos_w0) / a0,
        a2=(1.0 - alpha) / a0,
    )


def butterworth_lowpass(sample_rate: float, cutoff: float, order: int = 4) -> list[Biquad]:
    """Butterworth low-pass as a cascade of biquads."""
    return _cascade(sample_rate, cutoff, order, _lowpass_biquad)


def butterworth_highpass(sample_rate: float, cutoff: float, order: int = 4) -> list[Biquad]:
    """Butterworth high-pass as a cascade of biquads."""
    return _cascade(sample_rate, cutoff, order, _highpass_biquad)


def butterworth_bandpass(sample_rate: float, low: float, high: float,
                         order: int = 2) -> list[Biquad]:
    """Butterworth band-pass as a cascade of biquads."""
    validate_positive_float(sample_rate, "sample_rate")
    sections = max(1, int(order) // 2)
    return [_bandpass_biquad(sample_rate, low, high) for _ in range(sections)]


def _cascade(sample_rate: float, cutoff: float, order: int,
             factory: object) -> list[Biquad]:
    validate_positive_float(sample_rate, "sample_rate")
    validate_positive_int(order, "order", maximum=20)
    even_order = int(order) + (int(order) % 2)
    return [
        factory(sample_rate, cutoff, _butterworth_q(even_order, i))  # type: ignore[operator]
        for i in range(even_order // 2)
    ]


def apply_biquad(values: np.ndarray, section: Biquad) -> np.ndarray:
    """Apply one biquad using the direct-form-II transposed structure.

    Bounded, allocation-free after the output array, and numerically stable for
    the coefficient ranges produced here.
    """
    x = np.asarray(values, dtype=np.float64)
    out = np.empty_like(x)
    z1 = 0.0
    z2 = 0.0
    for i in range(x.size):
        xi = float(x[i])
        yi = section.b0 * xi + z1
        z1 = section.b1 * xi - section.a1 * yi + z2
        z2 = section.b2 * xi - section.a2 * yi
        out[i] = yi
    return out


def apply_cascade(values: np.ndarray, sections: list[Biquad]) -> np.ndarray:
    """Apply a biquad cascade in order."""
    signal = np.asarray(values, dtype=np.float64)
    for section in sections:
        signal = apply_biquad(signal, section)
    return signal


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    """Causal moving average with edge padding of the partial window."""
    validate_positive_int(window, "window", maximum=1 << 20)
    x = np.asarray(values, dtype=np.float64)
    if x.size == 0:
        return x.copy()
    kernel = np.ones(int(window), dtype=np.float64) / float(window)
    padded = np.concatenate([np.full(min(window - 1, x.size), x[0]), x])
    convolved = np.convolve(padded, kernel, mode="full")
    return convolved[min(window - 1, x.size) :][: x.size]


def median_filter(values: np.ndarray, window: int) -> np.ndarray:
    """Causal sliding median. Robust against isolated outliers (threat T-01)."""
    validate_positive_int(window, "window", maximum=4096)
    if window % 2 == 0:
        raise SignalError("median_filter window must be odd", details={"window": window})
    x = np.asarray(values, dtype=np.float64)
    if x.size == 0:
        return x.copy()
    half = window // 2
    padded = np.concatenate([np.full(half, x[0]), x, np.full(half, x[-1])])
    out = np.empty(x.size, dtype=np.float64)
    for i in range(x.size):
        out[i] = float(np.median(padded[i : i + window]))
    return out


def detrend_constant(values: np.ndarray) -> np.ndarray:
    """Remove the mean (DC) component."""
    x = np.asarray(values, dtype=np.float64)
    if x.size == 0:
        return x.copy()
    return x - float(np.mean(x))


def remove_dc(values: np.ndarray) -> np.ndarray:
    """Alias of :func:`detrend_constant`."""
    return detrend_constant(values)


def envelope_remove_dc(values: np.ndarray, sample_rate: float, cutoff: float = 5.0) -> np.ndarray:
    """High-pass by Butterworth, removing slow drift before envelope analysis."""
    sections = butterworth_highpass(sample_rate, cutoff, order=4)
    if not sections:
        return detrend_constant(values)
    return apply_cascade(values, sections)
