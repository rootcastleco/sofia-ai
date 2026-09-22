"""Acoustic emission, ultrasound, and fluid cavitation signal processing.

Implements parameter extraction for high-frequency acoustic emission (AE) sensors,
ultrasound micro-crack detection, partial electrical discharge, and cavitation
monitoring in industrial hydraulic and rotating assets.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "AcousticEmissionFeatures",
    "compute_acoustic_emission_features",
    "compute_cavitation_index",
]


@dataclass(frozen=True, slots=True)
class AcousticEmissionFeatures:
    """Standard acoustic emission parameters (ASTM E1316)."""

    peak_amplitude: float
    energy: float
    counts: int
    duration_s: float
    rise_time_s: float
    average_frequency_hz: float
    rms: float
    crest_factor: float

    def to_dict(self) -> dict[str, float | int]:
        return {
            "peak_amplitude": self.peak_amplitude,
            "energy": self.energy,
            "counts": self.counts,
            "duration_s": self.duration_s,
            "rise_time_s": self.rise_time_s,
            "average_frequency_hz": self.average_frequency_hz,
            "rms": self.rms,
            "crest_factor": self.crest_factor,
        }


def compute_acoustic_emission_features(
    signal: np.ndarray,
    fs: float,
    threshold: float | None = None,
) -> AcousticEmissionFeatures:
    """Extract standard Acoustic Emission (AE) transient parameters.

    Args:
        signal: 1D array of acoustic/ultrasound telemetry samples.
        fs: Sampling frequency in Hz.
        threshold: Detection threshold. If None, defaults to 3.0 * RMS.
    """
    sig = np.asarray(signal, dtype=np.float64)
    n = len(sig)
    if n == 0:
        return AcousticEmissionFeatures(
            peak_amplitude=0.0,
            energy=0.0,
            counts=0,
            duration_s=0.0,
            rise_time_s=0.0,
            average_frequency_hz=0.0,
            rms=0.0,
            crest_factor=0.0,
        )

    sig_rms = float(np.sqrt(np.mean(sig**2)))
    th = threshold if threshold is not None else max(1e-6, 3.0 * sig_rms)

    abs_sig = np.abs(sig)
    peak_idx = int(np.argmax(abs_sig))
    peak_amp = float(abs_sig[peak_idx])

    # True energy: integral of squared voltage (dt = 1/fs)
    dt = 1.0 / fs
    energy = float(np.sum(sig**2) * dt)

    # Threshold crossings (rising edges)
    above = abs_sig > th
    crossings = np.where(~above[:-1] & above[1:])[0]
    counts = len(crossings)

    if counts > 0:
        first_idx = int(crossings[0])
        # Find last sample above threshold
        last_indices = np.where(above)[0]
        last_idx = int(last_indices[-1]) if len(last_indices) > 0 else first_idx

        duration_s = max(0.0, (last_idx - first_idx) * dt)
        rise_time_s = max(0.0, (peak_idx - first_idx) * dt)
        avg_freq = counts / duration_s if duration_s > 0 else 0.0
    else:
        duration_s = 0.0
        rise_time_s = 0.0
        avg_freq = 0.0

    cf = peak_amp / sig_rms if sig_rms > 1e-12 else 0.0

    return AcousticEmissionFeatures(
        peak_amplitude=round(peak_amp, 4),
        energy=round(energy, 6),
        counts=counts,
        duration_s=round(duration_s, 6),
        rise_time_s=round(rise_time_s, 6),
        average_frequency_hz=round(avg_freq, 1),
        rms=round(sig_rms, 4),
        crest_factor=round(cf, 2),
    )


def compute_cavitation_index(
    vibration_or_acoustic: np.ndarray,
    fs: float,
    cavitation_band: tuple[float, float] = (5000.0, 20000.0),
) -> float:
    """Compute normalized cavitation intensity index in high-frequency acoustic band.

    Cavitation produces broadband high-frequency acoustic bursts typically in the
    5 kHz - 20+ kHz range.
    """
    sig = np.asarray(vibration_or_acoustic, dtype=np.float64)
    n = len(sig)
    if n < 32:
        return 0.0

    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    fft_mag = np.abs(np.fft.rfft(sig))

    low_f, high_f = cavitation_band
    band_mask = (freqs >= low_f) & (freqs <= high_f)

    total_power = np.sum(fft_mag**2)
    band_power = np.sum(fft_mag[band_mask] ** 2) if np.any(band_mask) else 0.0

    if total_power < 1e-12:
        return 0.0

    return float(np.clip(band_power / total_power, 0.0, 1.0))
