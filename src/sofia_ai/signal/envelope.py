"""Envelope extraction via the Hilbert transform.

The analytic-signal envelope is the standard amplitude-demodulation primitive for
bearing and gear condition indicators. The Hilbert transform is implemented with
the FFT, so it is deterministic and O(n log n).
"""

from __future__ import annotations

import numpy as np

from ..core.errors import InsufficientDataError

__all__ = ["amplitude_envelope", "envelope_features", "envelope_spectrum", "hilbert_analytic"]


def hilbert_analytic(values: np.ndarray) -> np.ndarray:
    """Analytic signal of a real input (Hilbert transform via FFT).

    Args:
        values: Real 1-D input.

    Returns:
        Complex analytic signal whose magnitude is the envelope.
    """
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 1:
        raise InsufficientDataError("hilbert_analytic expects a 1-D array",
                                    details={"ndim": x.ndim})
    n = x.size
    if n < 2:
        raise InsufficientDataError("hilbert_analytic needs at least 2 samples",
                                    details={"size": int(n)})
    spectrum = np.fft.fft(x)
    h = np.zeros(n, dtype=np.float64)
    if n % 2 == 0:
        h[0] = 1.0
        h[n // 2] = 1.0
        h[1 : n // 2] = 2.0
    else:
        h[0] = 1.0
        h[1 : (n + 1) // 2] = 2.0
    return np.fft.ifft(spectrum * h)


def amplitude_envelope(values: np.ndarray) -> np.ndarray:
    """Magnitude of the analytic signal."""
    return np.abs(hilbert_analytic(np.asarray(values, dtype=np.float64)))


def envelope_spectrum(
    values: np.ndarray, sample_rate: float, *, remove_dc: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """Envelope (Hilbert) spectrum of a band-limited signal.

    Returns:
        ``(frequencies, magnitudes)`` for the one-sided envelope spectrum, with the
        DC bin removed when ``remove_dc`` is set.
    """
    from .filters import detrend_constant
    from .spectral import rfft_magnitude

    x = np.asarray(values, dtype=np.float64)
    if remove_dc and x.size:
        x = detrend_constant(x)
    envelope = amplitude_envelope(x)
    envelope = envelope - float(np.mean(envelope))
    spectrum = rfft_magnitude(envelope, sample_rate, window="hann")
    return spectrum.frequencies, spectrum.values


def envelope_features(values: np.ndarray) -> dict[str, float]:
    """Descriptive statistics of the amplitude envelope."""
    env = amplitude_envelope(np.asarray(values, dtype=np.float64))
    if env.size == 0:
        return {}
    mean = float(np.mean(env))
    std = float(np.std(env))
    return {
        "envelope_mean": mean,
        "envelope_std": std,
        "envelope_peak": float(np.max(env)),
        "envelope_min": float(np.min(env)),
        "envelope_peak_to_peak": float(np.max(env) - np.min(env)),
    }
