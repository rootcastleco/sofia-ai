"""Multi-domain industrial feature extractors.

Extends Sofia Engine beyond vibration to:
- Electrical power & power quality (IEEE 519 / IEC 61000-4-30)
- Acoustic emissions & ultrasound (ASTM E1316)
- Thermal & process telemetry
- Fluid pressure & cavitation
- Multi-axis inertial motion & IMU dynamics
"""

from __future__ import annotations

import math
from typing import Final

import numpy as np

from ..core.contracts import FeatureVector
from ..signal.acoustic import compute_acoustic_emission_features, compute_cavitation_index
from ..signal.electrical import compute_electrical_power_metrics

MULTIDOMAIN_EXTRACTOR_ID: Final[str] = "sofia.multidomain"
MULTIDOMAIN_EXTRACTOR_VERSION: Final[str] = "2.0.0"

__all__ = [
    "MULTIDOMAIN_EXTRACTOR_ID",
    "MULTIDOMAIN_EXTRACTOR_VERSION",
    "extract_acoustic_features",
    "extract_electrical_features",
    "extract_motion_features",
    "extract_process_features",
]


def extract_electrical_features(
    v_sig: np.ndarray,
    i_sig: np.ndarray,
    fs: float,
    nom_freq: float = 50.0,
    device_id: str = "electrical-bus",
) -> FeatureVector:
    """Extract industrial electrical power quality features."""
    metrics = compute_electrical_power_metrics(v_sig, i_sig, fs, nom_freq=nom_freq)
    d = metrics.to_dict()
    names = tuple(sorted(d.keys()))
    values = np.asarray([d[k] for k in names], dtype=np.float64)

    return FeatureVector(
        names=names,
        values=values,
        extractor_id=f"{MULTIDOMAIN_EXTRACTOR_ID}.electrical",
        extractor_version=MULTIDOMAIN_EXTRACTOR_VERSION,
    )


def extract_acoustic_features(
    signal: np.ndarray,
    fs: float,
    threshold: float | None = None,
) -> FeatureVector:
    """Extract acoustic emission and cavitation features."""
    ae = compute_acoustic_emission_features(signal, fs, threshold=threshold)
    cavitation = compute_cavitation_index(signal, fs)

    d = ae.to_dict()
    d["cavitation_index"] = cavitation

    names = tuple(sorted(d.keys()))
    values = np.asarray([float(d[k]) for k in names], dtype=np.float64)

    return FeatureVector(
        names=names,
        values=values,
        extractor_id=f"{MULTIDOMAIN_EXTRACTOR_ID}.acoustic",
        extractor_version=MULTIDOMAIN_EXTRACTOR_VERSION,
    )


def extract_process_features(
    temperature_sig: np.ndarray,
    pressure_sig: np.ndarray | None = None,
    dt_seconds: float = 1.0,
) -> FeatureVector:
    """Extract thermal and fluid pressure dynamics.

    Computes:
    - Temperature mean, min, max, delta, rate of change (dT/dt)
    - Thermal acceleration (d2T/dt2)
    - Pressure mean, pulsation peak-to-peak, pressure variance
    """
    temp = np.asarray(temperature_sig, dtype=np.float64)
    d: dict[str, float] = {
        "temp_mean": float(np.mean(temp)) if len(temp) > 0 else 0.0,
        "temp_min": float(np.min(temp)) if len(temp) > 0 else 0.0,
        "temp_max": float(np.max(temp)) if len(temp) > 0 else 0.0,
        "temp_delta": float(np.max(temp) - np.min(temp)) if len(temp) > 0 else 0.0,
    }

    if len(temp) > 1 and dt_seconds > 0:
        dt_dt = np.gradient(temp, dt_seconds)
        d["temp_rate_mean"] = float(np.mean(dt_dt))
        d["temp_rate_max"] = float(np.max(np.abs(dt_dt)))
    else:
        d["temp_rate_mean"] = 0.0
        d["temp_rate_max"] = 0.0

    if pressure_sig is not None and len(pressure_sig) > 0:
        p = np.asarray(pressure_sig, dtype=np.float64)
        d["pressure_mean"] = float(np.mean(p))
        d["pressure_p2p"] = float(np.max(p) - np.min(p))
        d["pressure_std"] = float(np.std(p))
        d["pressure_crest_factor"] = (
            float(np.max(np.abs(p)) / np.sqrt(np.mean(p**2)))
            if np.mean(p**2) > 1e-12
            else 0.0
        )
    else:
        d["pressure_mean"] = 0.0
        d["pressure_p2p"] = 0.0
        d["pressure_std"] = 0.0
        d["pressure_crest_factor"] = 0.0

    names = tuple(sorted(d.keys()))
    values = np.asarray([d[k] for k in names], dtype=np.float64)

    return FeatureVector(
        names=names,
        values=values,
        extractor_id=f"{MULTIDOMAIN_EXTRACTOR_ID}.process",
        extractor_version=MULTIDOMAIN_EXTRACTOR_VERSION,
    )


def extract_motion_features(
    ax: np.ndarray,
    ay: np.ndarray,
    az: np.ndarray,
    fs: float,
) -> FeatureVector:
    """Extract 3-axis IMU accelerometer dynamics: magnitude, tilt, and jerk."""
    x = np.asarray(ax, dtype=np.float64)
    y = np.asarray(ay, dtype=np.float64)
    z = np.asarray(az, dtype=np.float64)

    min_len = min(len(x), len(y), len(z))
    x, y, z = x[:min_len], y[:min_len], z[:min_len]

    # Total acceleration vector magnitude
    mag = np.sqrt(x**2 + y**2 + z**2)
    mag_mean = float(np.mean(mag)) if min_len > 0 else 0.0
    mag_max = float(np.max(mag)) if min_len > 0 else 0.0

    # Dynamic pitch and roll angles in degrees
    pitch = np.degrees(np.arctan2(-x, np.sqrt(y**2 + z**2))) if min_len > 0 else np.zeros(0)
    roll = np.degrees(np.arctan2(y, z)) if min_len > 0 else np.zeros(0)

    # Dynamic Jerk (da/dt in g/s)
    if min_len > 1 and fs > 0:
        dt = 1.0 / fs
        jerk = np.gradient(mag, dt)
        jerk_max = float(np.max(np.abs(jerk)))
        jerk_rms = float(np.sqrt(np.mean(jerk**2)))
    else:
        jerk_max = 0.0
        jerk_rms = 0.0

    d = {
        "accel_mag_mean": round(mag_mean, 4),
        "accel_mag_max": round(mag_max, 4),
        "pitch_mean_deg": round(float(np.mean(pitch)), 2) if len(pitch) > 0 else 0.0,
        "roll_mean_deg": round(float(np.mean(roll)), 2) if len(roll) > 0 else 0.0,
        "jerk_max": round(jerk_max, 2),
        "jerk_rms": round(jerk_rms, 2),
    }

    names = tuple(sorted(d.keys()))
    values = np.asarray([d[k] for k in names], dtype=np.float64)

    return FeatureVector(
        names=names,
        values=values,
        extractor_id=f"{MULTIDOMAIN_EXTRACTOR_ID}.motion",
        extractor_version=MULTIDOMAIN_EXTRACTOR_VERSION,
    )
