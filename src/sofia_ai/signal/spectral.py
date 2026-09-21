"""Spectral analysis: FFT, PSD, Welch PSD, spectral descriptors.

All functions are pure and deterministic. The frequency axis is derived from the
sample rate, never from wall-clock timestamps.

Conventions:
    * ``rfft`` is used for real input; outputs are one-sided.
    * PSD is normalized so that ``sum(psd) * df`` approximates the signal mean
      square (Parseval's relation), which is asserted in the property tests.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.errors import InsufficientDataError, SignalError
from ..core.validation import validate_positive_int
from .windowing import WindowSpec, hann

__all__ = [
    "Spectrum",
    "band_energy",
    "fft_frequencies",
    "peak_frequency",
    "power_spectral_density",
    "rfft_magnitude",
    "spectral_centroid",
    "spectral_entropy",
    "spectral_flatness",
    "spectral_peaks",
    "welch_psd",
]


@dataclass(frozen=True, slots=True)
class Spectrum:
    """One-sided spectral estimate."""

    frequencies: np.ndarray
    values: np.ndarray
    sample_rate: float
    method: str
    unit: str = "g^2/Hz"

    @property
    def df(self) -> float:
        """Frequency resolution in Hz."""
        if self.frequencies.size < 2:
            return self.sample_rate
        return float(self.frequencies[1] - self.frequencies[0])

    @property
    def total_power(self) -> float:
        return float(np.sum(self.values) * self.df)

    def to_dict(self) -> dict[str, object]:
        return {
            "frequencies": self.frequencies.tolist(),
            "values": self.values.tolist(),
            "sample_rate": self.sample_rate,
            "method": self.method,
            "unit": self.unit,
        }


def fft_frequencies(n: int, sample_rate: float) -> np.ndarray:
    """One-sided FFT frequency axis: ``0 .. sample_rate/2``."""
    validate_positive_int(n, "n", maximum=1 << 22)
    return np.fft.rfftfreq(n, d=1.0 / float(sample_rate))


def rfft_magnitude(values: np.ndarray, sample_rate: float, *,
                   window: str = "hann", detrend_mean: bool = True) -> Spectrum:
    """One-sided FFT magnitude spectrum.

    Args:
        values: Real 1-D input.
        sample_rate: Sample rate in Hz.
        window: Window function name. ``"rectangular"`` disables windowing.
        detrend_mean: Remove the mean before transforming.
    """
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1 or arr.size < 2:
        raise InsufficientDataError("rfft_magnitude needs at least 2 samples",
                                    details={"size": int(arr.size)})
    x = arr - np.mean(arr) if detrend_mean else arr.copy()
    if window != "rectangular":
        x = x * _window(arr.size, window)
    spectrum = np.abs(np.fft.rfft(x))
    return Spectrum(frequencies=fft_frequencies(arr.size, sample_rate),
                    values=spectrum, sample_rate=float(sample_rate), method="rfft",
                    unit="amplitude")


def _window(n: int, name: str) -> np.ndarray:
    if name == "hann":
        return hann(n)
    if name == "hamming":
        from .windowing import hamming

        return hamming(n)
    if name == "blackman":
        from .windowing import blackman

        return blackman(n)
    if name == "rectangular":
        return np.ones(n, dtype=np.float64)
    raise SignalError(f"unknown window {name!r}", details={"window": name})


def power_spectral_density(values: np.ndarray, sample_rate: float, *,
                           window: str = "hann") -> Spectrum:
    """One-sided periodogram-style PSD satisfying Parseval's relation.

    ``sum(psd) * df`` ≈ mean square of the (detrended) signal.
    """
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1 or arr.size < 2:
        raise InsufficientDataError("power_spectral_density needs at least 2 samples",
                                    details={"size": int(arr.size)})
    n = arr.size
    x = arr - np.mean(arr)
    w = _window(n, window)
    xw = x * w
    spectrum = np.fft.rfft(xw)
    scale = 1.0 / (float(sample_rate) * float(np.sum(w**2)))
    psd = (np.abs(spectrum) ** 2) * scale
    _fold_negative_frequencies(psd, n)
    return Spectrum(frequencies=fft_frequencies(n, sample_rate), values=psd,
                    sample_rate=float(sample_rate), method="periodogram")


def _fold_negative_frequencies(psd_one_sided: np.ndarray, n_samples: int) -> None:
    """Fold negative-frequency energy into the one-sided spectrum.

    For an even transform length the last bin is Nyquist and is shared, so it must
    not be doubled. For an odd length there is no Nyquist bin and every bin after
    DC is doubled. Getting this wrong breaks Parseval's relation for odd lengths.
    """
    if psd_one_sided.size < 2:
        return
    if n_samples % 2 == 0:
        psd_one_sided[1:-1] *= 2.0
    else:
        psd_one_sided[1:] *= 2.0


def welch_psd(
    values: np.ndarray,
    sample_rate: float,
    *,
    segment_length: int | None = None,
    overlap: float = 0.5,
    window: str = "hann",
) -> Spectrum:
    """Welch's averaged periodogram.

    Args:
        values: Real 1-D input.
        sample_rate: Sample rate in Hz.
        segment_length: Samples per segment. Defaults to ``min(n, 256)``.
        overlap: Fraction of overlap in ``[0, 1)``.
        window: Window applied to each segment.

    Raises:
        SignalError: if ``overlap`` is outside ``[0, 1)`` or the signal is too short.
    """
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1 or arr.size < 2:
        raise InsufficientDataError("welch_psd needs at least 2 samples",
                                    details={"size": int(arr.size)})
    if not 0.0 <= overlap < 1.0:
        raise SignalError("overlap must be within [0, 1)", details={"overlap": overlap})

    n = arr.size
    seg = int(segment_length) if segment_length else min(n, 256)
    seg = int(np.clip(seg, 2, n))
    step = max(1, round(seg * (1.0 - overlap)))
    n_segments = 1 + (n - seg) // step
    if n_segments < 1:
        raise InsufficientDataError("signal too short for the requested segment length",
                                    details={"n": n, "segment_length": seg})

    from .windowing import frame_signal

    frames = frame_signal(arr, WindowSpec(length=seg, hop=step))[:n_segments]
    w = _window(seg, window)
    scale = 1.0 / (float(sample_rate) * float(np.sum(w**2)))
    spectrum = np.fft.rfft(frames * w, axis=1)
    psd = (np.abs(spectrum) ** 2) * scale
    for row in range(psd.shape[0]):
        _fold_negative_frequencies(psd[row], seg)
    averaged = np.mean(psd, axis=0)
    return Spectrum(frequencies=fft_frequencies(seg, sample_rate), values=averaged,
                    sample_rate=float(sample_rate), method="welch")


def spectral_centroid(frequencies: np.ndarray, magnitudes: np.ndarray) -> float:
    """Amplitude-weighted mean frequency (Hz)."""
    f = np.asarray(frequencies, dtype=np.float64)
    m = np.asarray(magnitudes, dtype=np.float64)
    if f.size != m.size:
        raise SignalError("frequencies and magnitudes must have equal length",
                          details={"f": f.size, "m": m.size})
    total = float(np.sum(m))
    if total <= 0.0:
        return 0.0
    return float(np.sum(f * m) / total)


def peak_frequency(frequencies: np.ndarray, magnitudes: np.ndarray) -> float:
    """Frequency of the largest magnitude bin."""
    f = np.asarray(frequencies, dtype=np.float64)
    m = np.asarray(magnitudes, dtype=np.float64)
    if m.size == 0:
        return 0.0
    return float(f[int(np.argmax(m))])


def spectral_peaks(
    frequencies: np.ndarray,
    magnitudes: np.ndarray,
    *,
    max_peaks: int = 8,
    prominence_factor: float = 0.05,
) -> list[tuple[float, float]]:
    """Find local maxima above a fraction of the global maximum.

    Returns:
        List of ``(frequency, magnitude)`` sorted by magnitude, descending, capped
        at ``max_peaks``. Deterministic: ties break by index.
    """
    f = np.asarray(frequencies, dtype=np.float64)
    m = np.asarray(magnitudes, dtype=np.float64)
    if m.size < 3:
        return []
    threshold = float(np.max(m)) * float(prominence_factor)
    found: list[tuple[float, float]] = []
    for i in range(1, m.size - 1):
        if m[i] >= m[i - 1] and m[i] >= m[i + 1] and m[i] >= threshold:
            found.append((float(f[i]), float(m[i])))
    found.sort(key=lambda item: (-item[1], item[0]))
    return found[: max(1, int(max_peaks))]


def band_energy(
    frequencies: np.ndarray,
    psd: np.ndarray,
    low_hz: float,
    high_hz: float,
) -> float:
    """Integrate PSD over ``[low_hz, high_hz)`` using the trapezoid rule.

    Raises:
        SignalError: if the band is empty or ``low_hz >= high_hz``.
    """
    f = np.asarray(frequencies, dtype=np.float64)
    p = np.asarray(psd, dtype=np.float64)
    if low_hz >= high_hz:
        raise SignalError("low_hz must be below high_hz",
                          details={"low": low_hz, "high": high_hz})
    mask = (f >= low_hz) & (f < high_hz)
    if not np.any(mask):
        return 0.0
    # Rectangle rule over bins. This is the band-power convention that matches the
    # PSD normalization (sum(psd) * df = total power), and it is monotone in band
    # width — the trapezoid rule is not, for bands narrower than a few bins.
    return float(np.sum(p[mask]) * _infer_df(f))


def _infer_df(frequencies: np.ndarray) -> float:
    """Infer bin spacing from a frequency axis."""
    if frequencies.size < 2:
        return 1.0
    span = float(frequencies[-1] - frequencies[0])
    return span / (frequencies.size - 1) if span > 0 else 1.0


def spectral_entropy(magnitudes: np.ndarray) -> float:
    """Shannon entropy of the normalized magnitude distribution, in bits."""
    m = np.asarray(magnitudes, dtype=np.float64)
    total = float(np.sum(m))
    if total <= 0.0 or m.size == 0:
        return 0.0
    p = m / total
    p = p[p > 0.0]
    return float(-np.sum(p * np.log2(p)))


def spectral_flatness(magnitudes: np.ndarray) -> float:
    """Geometric mean / arithmetic mean of the magnitude spectrum in ``[0, 1]``.

    Near 1 for noise-like signals, near 0 for tonal signals.
    """
    m = np.asarray(magnitudes, dtype=np.float64)
    positive = m[m > 0.0]
    if positive.size == 0:
        return 0.0
    log_mean = float(np.mean(np.log(positive)))
    geometric = float(np.exp(log_mean))
    arithmetic = float(np.mean(positive))
    if arithmetic <= 0.0:
        return 0.0
    return float(np.clip(geometric / arithmetic, 0.0, 1.0))
