"""Unit tests for multi-domain feature extraction."""

from __future__ import annotations

import numpy as np

from sofia_ai.features.multidomain import (
    extract_acoustic_features,
    extract_electrical_features,
    extract_motion_features,
    extract_process_features,
)


def test_extract_electrical_features() -> None:
    fs = 2000.0
    t = np.arange(2000) / fs
    v = 230.0 * np.sqrt(2) * np.sin(2 * np.pi * 50.0 * t)
    i = 10.0 * np.sqrt(2) * np.sin(2 * np.pi * 50.0 * t)

    fv = extract_electrical_features(v, i, fs)
    assert fv.size == 9
    d = fv.as_dict()
    assert "v_rms" in d
    assert "power_factor" in d
    assert "thd_v_percent" in d
    assert abs(d["v_rms"] - 230.0) < 1.0


def test_extract_acoustic_features() -> None:
    fs = 50_000.0
    t = np.arange(5000) / fs
    sig = np.sin(2 * np.pi * 8000.0 * t)

    fv = extract_acoustic_features(sig, fs)
    d = fv.as_dict()
    assert "energy" in d
    assert "cavitation_index" in d
    assert "peak_amplitude" in d


def test_extract_process_features() -> None:
    # Temperature heating ramp from 20C to 80C over 60 seconds
    temp = np.linspace(20.0, 80.0, 60)
    pressure = np.full(60, 5.5)

    fv = extract_process_features(temp, pressure, dt_seconds=1.0)
    d = fv.as_dict()
    assert abs(d["temp_min"] - 20.0) < 0.1
    assert abs(d["temp_max"] - 80.0) < 0.1
    assert abs(d["temp_rate_mean"] - 1.0) < 0.1
    assert abs(d["pressure_mean"] - 5.5) < 0.1


def test_extract_motion_features() -> None:
    fs = 100.0
    n = 200
    # Gravity on Z axis
    ax = np.zeros(n)
    ay = np.zeros(n)
    az = np.ones(n) * 1.0

    fv = extract_motion_features(ax, ay, az, fs)
    d = fv.as_dict()
    assert abs(d["accel_mag_mean"] - 1.0) < 0.01
    assert abs(d["roll_mean_deg"]) < 1.0
