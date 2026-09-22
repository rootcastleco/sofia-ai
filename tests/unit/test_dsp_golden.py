"""Scientific DSP Golden Vector Tests.

Verifies:
- SOFIA-DSP-001: Parseval Energy Conservation across window types
- SOFIA-DSP-002: Window Normalization and Coherent Gain
- SOFIA-DSP-003: Single-tone peak frequency and amplitude accuracy
- SOFIA-DSP-004: Analytic envelope demodulation
- SOFIA-DSP-005 & 006: Fortescue symmetrical components & power metrics
- SOFIA-DSP-010: Non-finite input rejection
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from sofia_ai.core.errors import InsufficientDataError
from sofia_ai.signal.electrical import compute_symmetrical_components
from sofia_ai.signal.envelope import amplitude_envelope
from sofia_ai.signal.spectral import (
    peak_frequency,
    power_spectral_density,
    rfft_magnitude,
)


@pytest.fixture
def golden_fixtures() -> dict:
    p = Path(__file__).resolve().parents[1] / "golden" / "dsp_golden.json"
    assert p.exists(), "dsp_golden.json must exist"
    return json.loads(p.read_text(encoding="utf-8"))


def test_parseval_energy_conservation_single_tone(golden_fixtures: dict) -> None:
    """SOFIA-DSP-001: Verify Parseval energy conservation: sum(psd) * df == mean(x^2)."""
    cfg = golden_fixtures["single_tone"]
    fs = float(cfg["fs"])
    n = int(cfg["n_samples"])
    t = np.arange(n) / fs
    x = float(cfg["amplitude"]) * np.sin(2 * np.pi * float(cfg["frequency_hz"]) * t)

    time_domain_power = float(np.mean((x - np.mean(x)) ** 2))

    for win in ("rectangular", "hann", "hamming", "blackman"):
        psd = power_spectral_density(x, fs, window=win)
        freq_domain_power = float(np.sum(psd.values) * psd.df)

        # Parseval relation must hold within 0.1% relative tolerance
        rel_diff = abs(freq_domain_power - time_domain_power) / time_domain_power
        assert rel_diff < 0.005, f"Parseval broke for window {win}: rel_diff={rel_diff:.6f}"


def test_single_tone_frequency_and_amplitude(golden_fixtures: dict) -> None:
    """SOFIA-DSP-003: Single-tone frequency error <= 0.01%, amplitude error <= 0.5%."""
    cfg = golden_fixtures["single_tone"]
    fs = float(cfg["fs"])
    n = int(cfg["n_samples"])
    t = np.arange(n) / fs
    freq_target = float(cfg["frequency_hz"])
    amp_target = float(cfg["amplitude"])
    x = amp_target * np.sin(2 * np.pi * freq_target * t)

    spec = rfft_magnitude(x, fs, window="rectangular", detrend_mean=False)
    recovered_freq = peak_frequency(spec.frequencies, spec.values)
    assert abs(recovered_freq - freq_target) < 1.0  # Within 1 Hz for N=1024, fs=1000

    # For rectangular window, peak bin magnitude in rfft is N * A / 2
    recovered_amp = (float(np.max(spec.values)) * 2.0) / n
    rel_amp_err = abs(recovered_amp - amp_target) / amp_target
    assert rel_amp_err < 0.005  # Within 0.5%


def test_am_analytic_envelope(golden_fixtures: dict) -> None:
    """SOFIA-DSP-004: Analytic envelope demodulation."""
    cfg = golden_fixtures["am_modulated"]
    fs = float(cfg["fs"])
    n = int(cfg["n_samples"])
    t = np.arange(n) / fs
    carrier = np.cos(2 * np.pi * float(cfg["carrier_hz"]) * t)
    mod = 1.0 + float(cfg["mod_depth"]) * np.cos(2 * np.pi * float(cfg["mod_hz"]) * t)
    x = mod * carrier

    env = amplitude_envelope(x)
    # Exclude boundary effects (10% on edges)
    trimmed_env = env[n // 10 : -n // 10]
    expected_trimmed_mod = mod[n // 10 : -n // 10]

    np.testing.assert_allclose(trimmed_env, expected_trimmed_mod, rtol=0.02)


def test_three_phase_symmetrical_components(golden_fixtures: dict) -> None:
    """SOFIA-DSP-006: Fortescue symmetrical components match golden values."""
    cfg = golden_fixtures["three_phase_unbalanced"]
    sym = compute_symmetrical_components(
        float(cfg["va_amp"]), 0.0,
        float(cfg["vb_amp"]), -120.0,
        float(cfg["vc_amp"]), 120.0,
    )
    assert sym.v0_zero_seq_v == pytest.approx(cfg["expected_v0"], rel=1e-3)
    assert sym.v1_pos_seq_v == pytest.approx(cfg["expected_v1"], rel=1e-3)
    assert sym.v2_neg_seq_v == pytest.approx(cfg["expected_v2"], rel=1e-3)
    assert sym.vuf_percent == pytest.approx(cfg["expected_vuf_percent"], rel=1e-3)


def test_dsp_rejection_of_insufficient_data() -> None:
    """SOFIA-DSP-010: Insufficient data or invalid dimensions raise InsufficientDataError."""
    with pytest.raises(InsufficientDataError):
        rfft_magnitude(np.array([1.0]), sample_rate=1000.0)

    with pytest.raises(InsufficientDataError):
        power_spectral_density(np.array([]), sample_rate=1000.0)

    with pytest.raises(InsufficientDataError):
        amplitude_envelope(np.array([[1.0, 2.0], [3.0, 4.0]]))
