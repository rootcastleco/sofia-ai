"""Signal processing: windowing, filtering, spectral analysis, resampling, envelopes.

Every function here is pure: ``f(ndarray, config) -> ndarray``. No clocks, no RNG,
no global state. The frequency axis always comes from the sample rate.
"""

from __future__ import annotations

from .detrend import detrend, detrend_linear, polynomial_trend
from .envelope import amplitude_envelope, envelope_spectrum, hilbert_analytic
from .filters import (
    Biquad,
    apply_biquad,
    apply_cascade,
    butterworth_bandpass,
    butterworth_highpass,
    butterworth_lowpass,
    median_filter,
    moving_average,
)
from .resample import decimate, interpolate_linear, resample_linear, resample_ratio
from .spectral import (
    Spectrum,
    band_energy,
    peak_frequency,
    power_spectral_density,
    rfft_magnitude,
    spectral_centroid,
    spectral_entropy,
    spectral_flatness,
    spectral_peaks,
    welch_psd,
)
from .windowing import (
    WindowSpec,
    apply_window_function,
    blackman,
    frame_signal,
    hamming,
    hann,
    rectangular,
    sliding_windows,
    windows_from_samples,
)

__all__ = [
    "Biquad",
    "Spectrum",
    "WindowSpec",
    "amplitude_envelope",
    "apply_biquad",
    "apply_cascade",
    "apply_window_function",
    "band_energy",
    "blackman",
    "butterworth_bandpass",
    "butterworth_highpass",
    "butterworth_lowpass",
    "decimate",
    "detrend",
    "detrend_linear",
    "envelope_spectrum",
    "frame_signal",
    "hamming",
    "hann",
    "hilbert_analytic",
    "interpolate_linear",
    "median_filter",
    "moving_average",
    "peak_frequency",
    "polynomial_trend",
    "power_spectral_density",
    "rectangular",
    "resample_linear",
    "resample_ratio",
    "rfft_magnitude",
    "sliding_windows",
    "spectral_centroid",
    "spectral_entropy",
    "spectral_flatness",
    "spectral_peaks",
    "welch_psd",
    "windows_from_samples",
]
