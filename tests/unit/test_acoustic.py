"""Unit tests for acoustic emission and cavitation analysis."""

from __future__ import annotations

import numpy as np

from sofia_ai.signal.acoustic import (
    compute_acoustic_emission_features,
    compute_cavitation_index,
)


def test_acoustic_emission_features() -> None:
    fs = 100_000.0
    t = np.arange(10_000) / fs
    # Decaying transient burst (typical AE event)
    transient = np.exp(-t * 2000.0) * np.sin(2 * np.pi * 15000.0 * t)

    features = compute_acoustic_emission_features(transient, fs, threshold=0.1)
    assert features.peak_amplitude > 0.8
    assert features.energy > 0.0
    assert features.counts > 0
    assert features.duration_s > 0.0
    assert features.crest_factor > 3.0


def test_cavitation_index() -> None:
    fs = 50_000.0
    t = np.arange(5000) / fs
    # Low frequency machine vibration (50 Hz)
    low_f = np.sin(2 * np.pi * 50.0 * t)
    idx_low = compute_cavitation_index(low_f, fs, cavitation_band=(5000.0, 20000.0))
    assert idx_low < 0.05

    # Cavitation noise (broadband high frequency 10 kHz)
    cav_sig = np.sin(2 * np.pi * 10000.0 * t)
    idx_high = compute_cavitation_index(cav_sig, fs, cavitation_band=(5000.0, 20000.0))
    assert idx_high > 0.9
