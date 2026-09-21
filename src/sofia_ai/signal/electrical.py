"""Industrial electrical and power quality signal processing.

Implements IEEE 519 and IEC 61000-4-30 compliant power metrics, total harmonic
distortion (THD), Fortescue 3-phase symmetrical components, and power quality
events (sags, swells, interruptions, inrush).

Ported and enhanced from Rootcastle REI SignalLab (rei-signallab).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Final

import numpy as np

__all__ = [
    "ElectricalPowerMetrics",
    "HarmonicSpectrum",
    "PowerQualityEvent",
    "SymmetricalComponentsResult",
    "analyze_harmonics_50",
    "compute_electrical_power_metrics",
    "compute_symmetrical_components",
    "compute_thd",
    "detect_power_quality_events",
]

SPEED_OF_LIGHT: Final[float] = 299792458.0


@dataclass(frozen=True, slots=True)
class ElectricalPowerMetrics:
    """Core electrical power metrics for single-phase or per-phase measurements."""

    v_rms: float
    i_rms: float
    active_power_w: float
    reactive_power_var: float
    apparent_power_va: float
    power_factor: float
    thd_v_percent: float
    thd_i_percent: float
    fundamental_freq_hz: float

    def to_dict(self) -> dict[str, float]:
        return {
            "v_rms": self.v_rms,
            "i_rms": self.i_rms,
            "active_power_w": self.active_power_w,
            "reactive_power_var": self.reactive_power_var,
            "apparent_power_va": self.apparent_power_va,
            "power_factor": self.power_factor,
            "thd_v_percent": self.thd_v_percent,
            "thd_i_percent": self.thd_i_percent,
            "fundamental_freq_hz": self.fundamental_freq_hz,
        }


@dataclass(frozen=True, slots=True)
class SymmetricalComponentsResult:
    """Fortescue 3-Phase symmetrical components representation."""

    v0_zero_seq_v: float
    v1_pos_seq_v: float
    v2_neg_seq_v: float
    vuf_percent: float

    def to_dict(self) -> dict[str, float]:
        return {
            "v0_zero_seq_v": self.v0_zero_seq_v,
            "v1_pos_seq_v": self.v1_pos_seq_v,
            "v2_neg_seq_v": self.v2_neg_seq_v,
            "vuf_percent": self.vuf_percent,
        }


@dataclass(frozen=True, slots=True)
class HarmonicSpectrum:
    """Individual harmonic component according to IEEE 519."""

    order: int
    frequency_hz: float
    v_magnitude_percent: float
    i_magnitude_percent: float
    ieee_519_limit_percent: float
    status: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "order": self.order,
            "frequency_hz": self.frequency_hz,
            "v_magnitude_percent": self.v_magnitude_percent,
            "i_magnitude_percent": self.i_magnitude_percent,
            "ieee_519_limit_percent": self.ieee_519_limit_percent,
            "status": self.status,
        }


@dataclass(frozen=True, slots=True)
class PowerQualityEvent:
    """Detected power disturbance event."""

    event_type: str
    severity: str
    description: str

    def to_dict(self) -> dict[str, str]:
        return {
            "event_type": self.event_type,
            "severity": self.severity,
            "description": self.description,
        }


def compute_thd(signal: np.ndarray, fs: float, fundamental_hz: float = 50.0) -> float:
    """Compute Total Harmonic Distortion (THD) as a percentage up to 50th harmonic.

    Follows IEEE 519:
        THD = (sqrt(sum_{h=2}^{50} V_h^2) / V_1) * 100%
    """
    sig = np.asarray(signal, dtype=np.float64)
    n = len(sig)
    if n < 16:
        return 0.0

    fft_vals = np.abs(np.fft.rfft(sig))
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)

    fund_idx = int(np.argmin(np.abs(freqs - fundamental_hz)))
    h1_mag = float(fft_vals[fund_idx])
    if h1_mag < 1e-6:
        return 0.0

    harmonic_sum_sq = 0.0
    for h in range(2, 51):
        h_freq = h * fundamental_hz
        if h_freq > freqs[-1]:
            break
        h_idx = int(np.argmin(np.abs(freqs - h_freq)))
        mag = float(fft_vals[h_idx])
        harmonic_sum_sq += mag * mag

    thd = (math.sqrt(harmonic_sum_sq) / h1_mag) * 100.0
    return float(np.clip(thd, 0.0, 500.0))


def compute_electrical_power_metrics(
    v_sig: np.ndarray,
    i_sig: np.ndarray,
    fs: float,
    nom_freq: float = 50.0,
) -> ElectricalPowerMetrics:
    """Calculate comprehensive power metrics from instantaneous voltage and current.

    Formulas:
        V_rms = sqrt(mean(v^2))
        I_rms = sqrt(mean(i^2))
        Active Power P = mean(v * i)
        Apparent Power S = V_rms * I_rms
        Reactive Power Q = sqrt(S^2 - P^2)
        Power Factor PF = P / S
    """
    v = np.asarray(v_sig, dtype=np.float64)
    i = np.asarray(i_sig, dtype=np.float64)
    if len(v) == 0 or len(i) == 0:
        raise ValueError("Voltage and current signals must not be empty.")
    if len(v) != len(i):
        min_len = min(len(v), len(i))
        v = v[:min_len]
        i = i[:min_len]

    v_rms = float(np.sqrt(np.mean(v**2)))
    i_rms = float(np.sqrt(np.mean(i**2)))

    active_power = float(np.mean(v * i))
    apparent_power = float(v_rms * i_rms)

    if apparent_power >= abs(active_power):
        reactive_power = float(np.sqrt(max(0.0, apparent_power**2 - active_power**2)))
    else:
        reactive_power = 0.0

    power_factor = (
        float(active_power / apparent_power) if apparent_power > 1e-6 else 1.0
    )
    power_factor = float(np.clip(power_factor, -1.0, 1.0))

    thd_v = compute_thd(v, fs, nom_freq)
    thd_i = compute_thd(i, fs, nom_freq)

    return ElectricalPowerMetrics(
        v_rms=round(v_rms, 3),
        i_rms=round(i_rms, 3),
        active_power_w=round(active_power, 2),
        reactive_power_var=round(reactive_power, 2),
        apparent_power_va=round(apparent_power, 2),
        power_factor=round(power_factor, 4),
        thd_v_percent=round(thd_v, 2),
        thd_i_percent=round(thd_i, 2),
        fundamental_freq_hz=round(nom_freq, 2),
    )


def compute_symmetrical_components(
    va_amp: float,
    va_phase_deg: float,
    vb_amp: float,
    vb_phase_deg: float,
    vc_amp: float,
    vc_phase_deg: float,
) -> SymmetricalComponentsResult:
    """Fortescue Transformation for 3-Phase Systems.

    Decomposes an unbalanced 3-phase system into zero, positive, and negative
    sequence components:
        a = exp(j * 120°) = -0.5 + j * sqrt(3)/2
        V0 = (Va + Vb + Vc) / 3
        V1 = (Va + a * Vb + a^2 * Vc) / 3   (Positive sequence)
        V2 = (Va + a^2 * Vb + a * Vc) / 3   (Negative sequence)
        VUF = (|V2| / |V1|) * 100%          (Voltage Unbalance Factor)
    """
    a = np.exp(1j * np.radians(120.0))

    va = va_amp * np.exp(1j * np.radians(va_phase_deg))
    vb = vb_amp * np.exp(1j * np.radians(vb_phase_deg))
    vc = vc_amp * np.exp(1j * np.radians(vc_phase_deg))

    v0 = (va + vb + vc) / 3.0
    v1 = (va + a * vb + (a**2) * vc) / 3.0
    v2 = (va + (a**2) * vb + a * vc) / 3.0

    v0_mag = float(np.abs(v0))
    v1_mag = float(np.abs(v1))
    v2_mag = float(np.abs(v2))

    vuf = (v2_mag / v1_mag * 100.0) if v1_mag > 1e-6 else 0.0

    return SymmetricalComponentsResult(
        v0_zero_seq_v=round(v0_mag, 2),
        v1_pos_seq_v=round(v1_mag, 2),
        v2_neg_seq_v=round(v2_mag, 2),
        vuf_percent=round(vuf, 2),
    )


def analyze_harmonics_50(
    v_sig: np.ndarray,
    i_sig: np.ndarray,
    fs: float,
    fund_freq: float = 50.0,
) -> list[HarmonicSpectrum]:
    """Harmonic analysis up to 50th order with IEEE 519 threshold verification."""
    v = np.asarray(v_sig, dtype=np.float64)
    i = np.asarray(i_sig, dtype=np.float64)
    n = len(v)

    v_fft = np.abs(np.fft.rfft(v))
    i_fft = np.abs(np.fft.rfft(i))
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)

    fund_idx = int(np.argmin(np.abs(freqs - fund_freq)))
    v_fund = float(v_fft[fund_idx]) if float(v_fft[fund_idx]) > 1e-6 else 1.0
    i_fund = float(i_fft[fund_idx]) if float(i_fft[fund_idx]) > 1e-6 else 1.0

    harmonics: list[HarmonicSpectrum] = []
    for h in range(1, 51):
        target_f = h * fund_freq
        if target_f > freqs[-1]:
            v_mag_pct = 0.0
            i_mag_pct = 0.0
        else:
            idx = int(np.argmin(np.abs(freqs - target_f)))
            v_mag_pct = float((v_fft[idx] / v_fund) * 100.0)
            i_mag_pct = float((i_fft[idx] / i_fund) * 100.0)

        ieee_limit = 100.0 if h == 1 else (3.0 if h % 2 == 1 else 1.5)
        status = "PASS" if v_mag_pct <= ieee_limit else "IEEE_519_EXCEEDED"

        harmonics.append(
            HarmonicSpectrum(
                order=h,
                frequency_hz=round(target_f, 1),
                v_magnitude_percent=round(v_mag_pct, 2),
                i_magnitude_percent=round(i_mag_pct, 2),
                ieee_519_limit_percent=ieee_limit,
                status=status,
            )
        )
    return harmonics


def detect_power_quality_events(
    v_sig: np.ndarray,
    i_sig: np.ndarray,
    v_nom_rms: float = 230.0,
) -> list[PowerQualityEvent]:
    """Detect voltage sags, swells, interruptions, and inrush currents."""
    v = np.asarray(v_sig, dtype=np.float64)
    i = np.asarray(i_sig, dtype=np.float64)

    events: list[PowerQualityEvent] = []
    v_rms_actual = float(np.sqrt(np.mean(v**2)))
    i_peak = float(np.max(np.abs(i))) if len(i) > 0 else 0.0
    i_rms = float(np.sqrt(np.mean(i**2))) if len(i) > 0 else 0.0

    ratio = v_rms_actual / v_nom_rms if v_nom_rms > 0 else 1.0

    if ratio < 0.1:
        events.append(
            PowerQualityEvent(
                event_type="INTERRUPTION",
                severity="alarm",
                description=f"Voltage Interruption ({v_rms_actual:.1f}V RMS, <10% nominal)",
            )
        )
    elif ratio < 0.9:
        events.append(
            PowerQualityEvent(
                event_type="VOLTAGE_SAG",
                severity="warning",
                description=f"Voltage Sag ({v_rms_actual:.1f}V RMS, {ratio*100:.1f}% nominal)",
            )
        )
    elif ratio > 1.1:
        events.append(
            PowerQualityEvent(
                event_type="VOLTAGE_SWELL",
                severity="warning",
                description=f"Voltage Swell ({v_rms_actual:.1f}V RMS, {ratio*100:.1f}% nominal)",
            )
        )

    if i_rms > 0 and (i_peak / i_rms) > 3.0:
        events.append(
            PowerQualityEvent(
                event_type="INRUSH_CURRENT",
                severity="warning",
                description=f"Inrush Current / Peak-to-RMS ratio: {(i_peak/i_rms):.2f}",
            )
        )

    if not events:
        events.append(
            PowerQualityEvent(
                event_type="NORMAL",
                severity="normal",
                description=f"Normal Power Quality ({v_rms_actual:.1f}V RMS, nominal)",
            )
        )

    return events
