"""Unit tests for feature extraction (SOFIA-SIG-001/002/008/009)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from sofia_ai.core.errors import FeatureError, InsufficientDataError, SignalError
from sofia_ai.features.extractor import (
    EXTRACTOR_ID,
    EXTRACTOR_VERSION,
    FEATURE_NAMES,
    FeatureExtractor,
    extract_from_array,
)
from sofia_ai.features.rotating import (
    OrderSpec,
    bearing_band_energies,
    envelope_analysis,
    harmonic_energies,
    orders_from_rpm,
    rpm_to_hz,
    sideband_energy,
    speed_normalized_resample,
)
from sofia_ai.features.statistical import (
    STATISTICAL_FEATURE_NAMES,
    crest_factor,
    energy,
    kurtosis,
    mean,
    peak,
    peak_to_peak,
    rms,
    skewness,
    statistical_features,
    std_dev,
    variance,
    zero_crossing_rate,
)


class TestStatistical:
    def test_constant_signal(self) -> None:
        x = np.full(128, 3.0)
        assert math.isclose(mean(x), 3.0)
        assert math.isclose(rms(x), 3.0)
        assert math.isclose(std_dev(x), 0.0)
        assert math.isclose(variance(x), 0.0)
        assert math.isclose(peak(x), 3.0)
        assert math.isclose(peak_to_peak(x), 0.0)
        assert math.isclose(crest_factor(x), 1.0)
        assert math.isclose(zero_crossing_rate(x), 0.0)
        assert math.isclose(energy(x), 128 * 9.0)

    def test_negative_constant(self) -> None:
        x = np.full(64, -2.0)
        assert math.isclose(rms(x), 2.0)
        assert math.isclose(crest_factor(x), 1.0)

    def test_unit_sine(self) -> None:
        # 1000 samples at 1 kHz is exactly 25 cycles, so the discrete RMS equals
        # the continuous value without leakage.
        n, fs = 1000, 1000.0
        t = np.arange(n) / fs
        x = np.sin(2 * np.pi * 25 * t)
        assert math.isclose(rms(x), 1 / math.sqrt(2), rel_tol=1e-6)
        assert math.isclose(peak(x), 1.0, rel_tol=1e-3)
        assert math.isclose(crest_factor(x), math.sqrt(2), rel_tol=1e-3)
        assert math.isclose(peak_to_peak(x), 2.0, rel_tol=1e-3)
        assert math.isclose(kurtosis(x), 1.5, abs_tol=0.05)
        assert math.isclose(skewness(x), 0.0, abs_tol=0.05)

    def test_zero_crossing_rate_matches_frequency(self) -> None:
        n, fs = 4096, 1000.0
        t = np.arange(n) / fs
        x = np.sin(2 * np.pi * 25 * t)
        expected = 2 * 25 / fs
        assert math.isclose(zero_crossing_rate(x), expected, rel_tol=0.05)

    def test_skewness_sign(self) -> None:
        positive_skew = np.array([0.0, 0.0, 0.0, 0.0, 10.0])
        assert skewness(positive_skew) > 0

    def test_empty_raises(self) -> None:
        with pytest.raises(InsufficientDataError):
            rms(np.array([]))

    def test_feature_set_completeness(self) -> None:
        table = statistical_features(np.random.default_rng(0).normal(size=64))
        assert tuple(table) == STATISTICAL_FEATURE_NAMES
        assert len(table) == 14

    def test_guarded_division(self) -> None:
        """Zero-spread signals must not produce NaN or inf."""
        table = statistical_features(np.full(16, 4.0))
        for value in table.values():
            assert math.isfinite(value)


class TestExtractor:
    def test_names_are_canonical(self) -> None:
        vector = extract_from_array(np.random.default_rng(1).normal(size=256), 1000.0)
        assert vector.names[: len(STATISTICAL_FEATURE_NAMES)] == STATISTICAL_FEATURE_NAMES
        assert vector.extractor_id == EXTRACTOR_ID
        assert vector.extractor_version == EXTRACTOR_VERSION
        assert vector.size == len(FEATURE_NAMES)

    def test_deterministic(self) -> None:
        signal = np.random.default_rng(2).normal(size=256)
        a = extract_from_array(signal, 1000.0)
        b = extract_from_array(signal, 1000.0)
        assert a == b

    def test_empty_raises(self) -> None:
        with pytest.raises(FeatureError):
            extract_from_array(np.array([]), 1000.0)

    def test_shaft_hz_adds_order_features(self) -> None:
        signal = np.sin(2 * np.pi * 25 * np.arange(512) / 1000.0)
        plain = extract_from_array(signal, 1000.0)
        with_orders = extract_from_array(signal, 1000.0, shaft_hz=25.0)
        assert with_orders.size > plain.size
        assert "order_1_energy" in with_orders.names

    def test_order_1_energy_detects_shaft_tone(self) -> None:
        signal = np.sin(2 * np.pi * 25 * np.arange(1024) / 1000.0)
        vector = extract_from_array(signal, 1000.0, shaft_hz=25.0)
        table = vector.as_dict()
        assert table["order_1_energy"] > table["order_5_energy"]

    def test_extractor_class(self) -> None:
        from sofia_ai.core.contracts import SignalWindow

        window = SignalWindow(values=np.random.default_rng(3).normal(size=256),
                              sample_rate=1000.0, device_id="d", channel="c", unit="g")
        extractor = FeatureExtractor()
        vector = extractor.extract(window)
        assert vector.source_channel == "c"
        assert extractor.extract_many([window])[0] == vector


class TestRotating:
    def test_rpm_to_hz(self) -> None:
        assert math.isclose(rpm_to_hz(1500.0), 25.0)

    def test_orders_from_rpm(self) -> None:
        spec = orders_from_rpm(1500.0, max_order=4)
        assert math.isclose(spec.shaft_hz, 25.0)
        assert spec.max_order == 4

    def test_order_spec_validates(self) -> None:
        with pytest.raises(SignalError):
            OrderSpec(shaft_hz=25.0, tolerance_fraction=0.9)

    def test_harmonic_energies_sum_at_most_one(self) -> None:
        signal = np.sin(2 * np.pi * 25 * np.arange(1024) / 1000.0)
        energies = harmonic_energies(signal, 1000.0, OrderSpec(shaft_hz=25.0))
        # Bands are one bin wide at minimum, so adjacent orders can double-count
        # a fraction of a bin. Allow a small numerical overlap.
        assert 0.0 <= sum(energies.values()) <= 1.0 + 1e-3
        assert energies["order_1_energy"] > 0.0

    def test_sidebands(self) -> None:
        fs = 1000.0
        t = np.arange(2048) / fs
        carrier = np.sin(2 * np.pi * 50 * t)
        modulation = 1.0 + 0.5 * np.sin(2 * np.pi * 5 * t)
        lower, upper = sideband_energy(carrier * modulation, fs,
                                       OrderSpec(shaft_hz=50.0), modulation_hz=5.0)
        assert lower > 0.0 and upper > 0.0

    def test_speed_normalized_resample(self) -> None:
        fs = 1000.0
        t = np.arange(1024) / fs
        signal = np.sin(2 * np.pi * 25 * t)
        rpm = np.full(1024, 25.0 * 60.0)
        angular = speed_normalized_resample(signal, fs, rpm, samples_per_revolution=128)
        assert angular.size == 128

    def test_speed_normalized_length_mismatch(self) -> None:
        with pytest.raises(InsufficientDataError):
            speed_normalized_resample(np.ones(16), 1000.0, np.ones(8))

    def test_bearing_bands_require_explicit_bands(self) -> None:
        assert bearing_band_energies(np.ones(64), 1000.0) == {}

    def test_bearing_bands_computed(self) -> None:
        signal = np.sin(2 * np.pi * 120 * np.arange(1024) / 1000.0)
        bands = bearing_band_energies(signal, 1000.0, bands_hz=[(100.0, 140.0)])
        assert bands["bearing_band_0"] > 0.0

    def test_envelope_analysis(self) -> None:
        fs = 4000.0
        t = np.arange(2048) / fs
        signal = (1.0 + 0.6 * np.sin(2 * np.pi * 20 * t)) * np.sin(2 * np.pi * 500 * t)
        result = envelope_analysis(signal, fs, band_low_hz=300.0, band_high_hz=800.0)
        assert math.isclose(result["envelope_dominant_hz"], 20.0, abs_tol=8.0)

    def test_envelope_analysis_validates_band(self) -> None:
        with pytest.raises(SignalError):
            envelope_analysis(np.ones(64), 1000.0, band_low_hz=900.0, band_high_hz=100.0)

    def test_envelope_analysis_rejects_above_nyquist(self) -> None:
        with pytest.raises(SignalError):
            envelope_analysis(np.ones(64), 1000.0, band_low_hz=100.0, band_high_hz=900.0)


class TestSpectralFeatures:
    def test_band_energy_features_zero_power(self) -> None:
        from sofia_ai.features.spectral import band_energy_features

        bands = band_energy_features(np.array([0.0, 1.0]), np.zeros(2), 1000.0)
        assert set(bands) == {"band_0_25", "band_25_50", "band_50_75", "band_75_100"}
        assert all(value == 0.0 for value in bands.values())

    def test_band_energy_features_normalised(self) -> None:
        from sofia_ai.features.spectral import band_energy_features

        freqs = np.linspace(0.0, 500.0, 128)
        psd = np.full(128, 1.0)
        bands = band_energy_features(freqs, psd, 1000.0)
        total = sum(bands.values())
        assert math.isclose(total, 1.0, rel_tol=0.05)

    def test_order_features_rejects_non_positive_shaft(self) -> None:
        from sofia_ai.features.spectral import order_features

        with pytest.raises(SignalError):
            order_features(np.array([0.0, 1.0, 2.0]), np.ones(3), 0.0)

    def test_order_features_zero_power(self) -> None:
        from sofia_ai.features.spectral import order_features

        out = order_features(np.array([0.0, 1.0]), np.zeros(2), shaft_hz=10.0)
        assert all(value == 0.0 for value in out.values())
