"""Unit tests for electrical and power quality signal processing."""

from __future__ import annotations

import math

import numpy as np
import pytest

from sofia_ai.signal.electrical import (
    analyze_harmonics_50,
    compute_electrical_power_metrics,
    compute_symmetrical_components,
    compute_thd,
    detect_power_quality_events,
)


def test_pure_sinusoid_thd() -> None:
    fs = 4000.0
    f0 = 50.0
    t = np.arange(4000) / fs
    # Pure sine wave has 0% THD
    v = 230.0 * np.sqrt(2) * np.sin(2 * np.pi * f0 * t)
    thd = compute_thd(v, fs, f0)
    assert thd < 0.5


def test_distorted_signal_thd() -> None:
    fs = 4000.0
    f0 = 50.0
    t = np.arange(4000) / fs
    # Fundamental + 10% 3rd harmonic + 5% 5th harmonic
    v = (
        230.0 * np.sqrt(2) * np.sin(2 * np.pi * f0 * t)
        + 23.0 * np.sqrt(2) * np.sin(2 * np.pi * 3 * f0 * t)
        + 11.5 * np.sqrt(2) * np.sin(2 * np.pi * 5 * f0 * t)
    )
    thd = compute_thd(v, fs, f0)
    # Expected: sqrt(10^2 + 5^2) = sqrt(125) ≈ 11.18%
    assert abs(thd - 11.18) < 1.0


def test_electrical_power_metrics() -> None:
    fs = 4000.0
    f0 = 50.0
    t = np.arange(4000) / fs
    v = 230.0 * np.sqrt(2) * np.sin(2 * np.pi * f0 * t)
    # Current in phase with voltage (purely resistive)
    i = 10.0 * np.sqrt(2) * np.sin(2 * np.pi * f0 * t)

    metrics = compute_electrical_power_metrics(v, i, fs, nom_freq=f0)
    assert abs(metrics.v_rms - 230.0) < 1.0
    assert abs(metrics.i_rms - 10.0) < 0.1
    # P = V * I = 2300 W
    assert abs(metrics.active_power_w - 2300.0) < 10.0
    assert abs(metrics.power_factor - 1.0) < 0.05
    assert metrics.reactive_power_var < 50.0


def test_symmetrical_components() -> None:
    # Balanced 3-phase voltages: 230V separated by 120 degrees
    res = compute_symmetrical_components(
        va_amp=230.0, va_phase_deg=0.0,
        vb_amp=230.0, vb_phase_deg=-120.0,
        vc_amp=230.0, vc_phase_deg=120.0,
    )
    # In balanced system: V1 = 230V, V0 = 0V, V2 = 0V, VUF = 0%
    assert abs(res.v1_pos_seq_v - 230.0) < 2.0
    assert res.v0_zero_seq_v < 2.0
    assert res.v2_neg_seq_v < 2.0
    assert res.vuf_percent < 1.0


def test_power_quality_events() -> None:
    fs = 1000.0
    t = np.arange(1000) / fs
    # Voltage sag: 150V instead of nominal 230V
    v_sag = 150.0 * np.sqrt(2) * np.sin(2 * np.pi * 50.0 * t)
    i = 5.0 * np.sin(2 * np.pi * 50.0 * t)

    events = detect_power_quality_events(v_sag, i, v_nom_rms=230.0)
    types = [e.event_type for e in events]
    assert "VOLTAGE_SAG" in types


def test_harmonics_50() -> None:
    fs = 4000.0
    t = np.arange(4000) / fs
    v = 230.0 * np.sqrt(2) * np.sin(2 * np.pi * 50.0 * t)
    i = 10.0 * np.sqrt(2) * np.sin(2 * np.pi * 50.0 * t)
    harmonics = analyze_harmonics_50(v, i, fs, 50.0)
    assert len(harmonics) == 50
    assert harmonics[0].order == 1
    assert harmonics[0].status == "PASS"
