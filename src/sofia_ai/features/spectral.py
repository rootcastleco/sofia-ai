"""Frequency-domain features.

All features are derived from a Welch PSD by default, because the averaged
periodogram has far lower variance than a single periodogram and is the standard
choice for condition monitoring.
"""

from __future__ import annotations

import numpy as np

from ..core.errors import InsufficientDataError, SignalError
from ..core.validation import validate_positive_float
from ..signal.spectral import (
    band_energy,
    peak_frequency,
    spectral_centroid,
    spectral_entropy,
    spectral_flatness,
    spectral_peaks,
    welch_psd,
)

__all__ = [
    "SPECTRAL_BANDS",
    "SPECTRAL_FEATURE_NAMES",
    "band_energy_features",
    "spectral_features",
]

#: Default fractional bands of the Nyquist range used for band-energy features.
SPECTRAL_BANDS: tuple[tuple[str, float, float], ...] = (
    ("band_0_25", 0.0, 0.25),
    ("band_25_50", 0.25, 0.50),
    ("band_50_75", 0.50, 0.75),
    ("band_75_100", 0.75, 1.00),
)

SPECTRAL_FEATURE_NAMES: tuple[str, ...] = (
    "spectral_centroid_hz",
    "peak_frequency_hz",
    "peak_magnitude",
    "spectral_entropy_bits",
    "spectral_flatness",
    "total_power",
    "dominant_harmonic_ratio",
    *[name for name, _, _ in SPECTRAL_BANDS],
)


def spectral_features(
    values: np.ndarray,
    sample_rate: float,
    *,
    segment_length: int | None = None,
) -> dict[str, float]:
    """Compute frequency-domain features from a Welch PSD.

    Returns:
        Mapping with exactly :data:`SPECTRAL_FEATURE_NAMES` as keys.
    """
    sample_rate = validate_positive_float(sample_rate, "sample_rate")
    psd = welch_psd(values, sample_rate, segment_length=segment_length)
    freq = psd.frequencies
    power = psd.values

    if freq.size == 0 or power.size == 0:
        raise InsufficientDataError("empty spectrum", details={})

    peaks = spectral_peaks(freq, power, max_peaks=4)
    top = float(peaks[0][1]) if peaks else 0.0
    total = float(np.sum(power)) or 1.0

    features: dict[str, float] = {
        "spectral_centroid_hz": spectral_centroid(freq, power),
        "peak_frequency_hz": peak_frequency(freq, power),
        "peak_magnitude": top,
        "spectral_entropy_bits": spectral_entropy(power),
        "spectral_flatness": spectral_flatness(power),
        "total_power": psd.total_power,
        "dominant_harmonic_ratio": float(top / total),
    }
    features.update(band_energy_features(freq, power, sample_rate))
    return features


def band_energy_features(
    frequencies: np.ndarray,
    psd: np.ndarray,
    sample_rate: float,
) -> dict[str, float]:
    """Fractional band energies across the Nyquist range.

    Each value is the fraction of total energy inside the band, so the features
    are scale-invariant.
    """
    sample_rate = validate_positive_float(sample_rate, "sample_rate")
    nyquist = sample_rate / 2.0
    frequencies = np.asarray(frequencies, dtype=np.float64)
    psd_array = np.asarray(psd, dtype=np.float64)
    total = float(np.sum(psd_array) * _df(frequencies))
    if total <= 0.0:
        return {name: 0.0 for name, _, _ in SPECTRAL_BANDS}
    out: dict[str, float] = {}
    for name, low_fraction, high_fraction in SPECTRAL_BANDS:
        energy = band_energy(frequencies, psd, low_fraction * nyquist,
                             high_fraction * nyquist)
        if low_fraction == 0.0:
            # Exclude DC so a sensor offset cannot dominate the lowest band.
            energy = band_energy(frequencies, psd, 1e-9, high_fraction * nyquist)
        out[name] = float(np.clip(energy / total, 0.0, 1.0))
    return out


def order_features(
    frequencies: np.ndarray,
    psd: np.ndarray,
    shaft_hz: float,
    *,
    max_order: int = 6,
) -> dict[str, float]:
    """Energy fraction near each shaft harmonic order (extension point)."""
    if shaft_hz <= 0:
        raise SignalError("shaft_hz must be positive", details={"shaft_hz": shaft_hz})
    axis = np.asarray(frequencies, dtype=np.float64)
    total = float(np.sum(np.asarray(psd, dtype=np.float64)) * _df(axis))
    out: dict[str, float] = {}
    for order in range(1, int(max_order) + 1):
        centre = shaft_hz * order
        width = max(shaft_hz * 0.1, 1e-6)
        energy = band_energy(frequencies, psd, centre - width, centre + width)
        out[f"order_{order}_energy"] = 0.0 if total <= 0 else float(np.clip(energy / total, 0.0, 1.0))
    return out


def _df(frequencies: np.ndarray) -> float:
    """Infer bin spacing from a frequency axis."""
    if frequencies.size < 2:
        return 1.0
    span = float(frequencies[-1] - frequencies[0])
    return span / (frequencies.size - 1) if span > 0 else 1.0
