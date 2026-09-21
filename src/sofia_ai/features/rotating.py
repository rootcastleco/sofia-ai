"""Rotating-machinery extension points.

These are *foundations*, not diagnoses. Sofia does not claim to identify a
specific bearing or gear fault from a spectrum; it exposes the primitives an
engineer needs — order tracking, harmonic extraction, sideband measurement and
speed-aware resampling — and leaves the interpretation to a validated diagnostic
rule that the user supplies.

Nothing here hard-codes a fault conclusion.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.errors import InsufficientDataError, SignalError
from ..core.validation import validate_positive_float, validate_positive_int
from ..signal.envelope import amplitude_envelope
from ..signal.filters import apply_cascade as apply_filter
from ..signal.filters import butterworth_bandpass
from ..signal.spectral import band_energy, rfft_magnitude, welch_psd

__all__ = [
    "OrderSpec",
    "bearing_band_energies",
    "envelope_analysis",
    "harmonic_energies",
    "orders_from_rpm",
    "rpm_to_hz",
    "sideband_energy",
    "speed_normalized_resample",
]


@dataclass(frozen=True, slots=True)
class OrderSpec:
    """Shaft-speed reference for order analysis."""

    shaft_hz: float
    max_order: int = 6
    tolerance_fraction: float = 0.05

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "shaft_hz", validate_positive_float(self.shaft_hz, "shaft_hz")
        )
        object.__setattr__(
            self, "max_order", validate_positive_int(self.max_order, "max_order", maximum=64)
        )
        if not 0.0 < self.tolerance_fraction < 0.5:
            raise SignalError(
                "tolerance_fraction must be within (0, 0.5)",
                details={"value": self.tolerance_fraction},
            )


def rpm_to_hz(rpm: float) -> float:
    """Convert revolutions per minute to shaft frequency in Hz."""
    return validate_positive_float(rpm, "rpm") / 60.0


def orders_from_rpm(rpm: float, max_order: int = 6) -> OrderSpec:
    """Build an :class:`OrderSpec` from a tachometer reading in RPM."""
    return OrderSpec(shaft_hz=rpm_to_hz(rpm), max_order=max_order)


def harmonic_energies(
    values: np.ndarray, sample_rate: float, spec: OrderSpec
) -> dict[str, float]:
    """Energy fraction near each shaft harmonic.

    Returns:
        ``{"order_1_energy": ..., "order_2_energy": ...}`` plus a ``total`` key.
    """
    psd = welch_psd(values, sample_rate)
    total = float(psd.total_power)
    if total <= 0.0:
        return {f"order_{o}_energy": 0.0 for o in range(1, spec.max_order + 1)}

    out: dict[str, float] = {}
    # Never ask for a band narrower than one spectral bin: the estimate would
    # always be zero regardless of the signal.
    bin_width = psd.df
    for order in range(1, spec.max_order + 1):
        centre = spec.shaft_hz * order
        half_width = max(centre * spec.tolerance_fraction, bin_width)
        out[f"order_{order}_energy"] = float(
            np.clip(band_energy(psd.frequencies, psd.values,
                                centre - half_width, centre + half_width) / total, 0.0, 1.0)
        )
    return out


def sideband_energy(
    values: np.ndarray,
    sample_rate: float,
    spec: OrderSpec,
    *,
    modulation_hz: float,
    order: int = 1,
) -> tuple[float, float]:
    """Measure sidebands around a shaft harmonic.

    Returns:
        ``(lower_sideband_fraction, upper_sideband_fraction)``.
    """
    modulate = validate_positive_float(modulation_hz, "modulation_hz")
    psd = welch_psd(values, sample_rate)
    total = float(psd.total_power)
    if total <= 0.0:
        return 0.0, 0.0
    centre = spec.shaft_hz * int(order)
    # Never request a band narrower than one spectral bin.
    half_width = max(centre * spec.tolerance_fraction, psd.df)
    lower = band_energy(psd.frequencies, psd.values,
                        centre - modulate - half_width, centre - modulate + half_width)
    upper = band_energy(psd.frequencies, psd.values,
                        centre + modulate - half_width, centre + modulate + half_width)
    return float(np.clip(lower / total, 0.0, 1.0)), float(np.clip(upper / total, 0.0, 1.0))


def speed_normalized_resample(
    values: np.ndarray,
    sample_rate: float,
    rpm_profile: np.ndarray,
    *,
    samples_per_revolution: int = 256,
) -> np.ndarray:
    """Resample from the time domain to the angular (order) domain.

    Args:
        values: Time-domain signal.
        sample_rate: Original sample rate.
        rpm_profile: Instantaneous shaft speed per input sample, in RPM.
        samples_per_revolution: Output resolution.

    Returns:
        Signal resampled onto a uniform angle grid.

    Raises:
        InsufficientDataError: if lengths disagree or the profile is degenerate.
    """
    x = np.asarray(values, dtype=np.float64).ravel()
    rpm = np.asarray(rpm_profile, dtype=np.float64).ravel()
    if x.size != rpm.size:
        raise InsufficientDataError(
            "rpm_profile must have the same length as the signal",
            details={"signal": int(x.size), "rpm": int(rpm.size)},
        )
    if x.size < 2:
        raise InsufficientDataError("speed_normalized_resample needs at least 2 samples",
                                    details={})
    samples_per_revolution = validate_positive_int(
        samples_per_revolution, "samples_per_revolution", maximum=1 << 16
    )

    shaft_hz = np.clip(rpm, 0.0, None) / 60.0
    shaft_hz[shaft_hz <= 0] = float(np.mean(shaft_hz[shaft_hz > 0]) or 1.0)
    phase = np.cumsum(shaft_hz) / float(sample_rate)  # revolutions
    target_phase = np.linspace(phase[0], phase[-1], samples_per_revolution)
    if phase[-1] <= phase[0]:
        raise InsufficientDataError("cumulative shaft phase did not advance", details={})
    return np.interp(target_phase, phase, x)


def bearing_band_energies(
    values: np.ndarray, sample_rate: float, *, bands_hz: list[tuple[float, float]] | None = None
) -> dict[str, float]:
    """Energy fraction inside user-supplied bearing-related frequency bands.

    No default band set is built in: bearing frequencies depend on geometry, race
    counts and contact angle, and hard-coding them would be an unvalidated
    diagnosis. The caller supplies the bands from the machine datasheet.
    """
    if bands_hz is None:
        return {}
    psd = welch_psd(values, sample_rate)
    total = float(psd.total_power)
    out: dict[str, float] = {}
    for index, (low, high) in enumerate(bands_hz):
        key = f"bearing_band_{index}"
        if total <= 0.0:
            out[key] = 0.0
            continue
        out[key] = float(
            np.clip(band_energy(psd.frequencies, psd.values, low, high) / total, 0.0, 1.0)
        )
    return out


def envelope_analysis(
    values: np.ndarray,
    sample_rate: float,
    *,
    band_low_hz: float,
    band_high_hz: float,
) -> dict[str, float]:
    """Band-pass, then Hilbert envelope, then envelope-spectrum descriptors.

    Returns:
        Envelope statistics plus the dominant envelope-spectrum frequency.
    """
    validate_positive_float(sample_rate, "sample_rate")
    if band_low_hz >= band_high_hz:
        raise SignalError("band_low_hz must be below band_high_hz",
                          details={"low": band_low_hz, "high": band_high_hz})
    if band_high_hz >= sample_rate / 2:
        raise SignalError(
            f"band_high_hz must be below Nyquist ({sample_rate / 2} Hz)",
            details={"high": band_high_hz, "nyquist": sample_rate / 2},
        )
    x = np.asarray(values, dtype=np.float64).ravel()
    if x.size < 8:
        raise InsufficientDataError("envelope_analysis needs at least 8 samples", details={})

    bandpassed = apply_filter(x, butterworth_bandpass(sample_rate, band_low_hz, band_high_hz))
    envelope = amplitude_envelope(bandpassed)
    envelope = envelope - float(np.mean(envelope))

    spectrum = rfft_magnitude(envelope, sample_rate, window="hann")
    freqs = spectrum.frequencies
    mags = spectrum.values
    if mags.size > 1:
        mags = mags.copy()
        mags[0] = 0.0
    dominant = float(freqs[int(np.argmax(mags))]) if mags.size else 0.0

    return {
        "envelope_rms": float(np.sqrt(np.mean(envelope * envelope))),
        "envelope_peak": float(np.max(np.abs(envelope))),
        "envelope_kurtosis": float(
            np.mean(((envelope - np.mean(envelope)) / (np.std(envelope) or 1.0)) ** 4)
        ),
        "envelope_dominant_hz": dominant,
        "envelope_spectrum_centroid_hz": (
            float(np.sum(freqs * mags) / np.sum(mags)) if np.sum(mags) > 0 else 0.0
        ),
    }

