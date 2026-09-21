"""Unit tests for signal processing against closed-form expectations."""

from __future__ import annotations

import math

import numpy as np
import pytest

from sofia_ai.core.errors import InsufficientDataError, SignalError
from sofia_ai.signal.detrend import detrend, detrend_linear, polynomial_trend
from sofia_ai.signal.envelope import amplitude_envelope, hilbert_analytic
from sofia_ai.signal.filters import (
    butterworth_bandpass,
    butterworth_highpass,
    butterworth_lowpass,
    median_filter,
    moving_average,
)
from sofia_ai.signal.resample import (
    decimate,
    interpolate_linear,
    resample_linear,
    resample_ratio,
)
from sofia_ai.signal.spectral import (
    band_energy,
    power_spectral_density,
    rfft_magnitude,
    spectral_centroid,
    spectral_entropy,
    spectral_flatness,
    spectral_peaks,
    welch_psd,
)
from sofia_ai.signal.windowing import (
    WindowSpec,
    apply_window_function,
    blackman,
    frame_signal,
    hamming,
    hann,
    rectangular,
    sliding_windows,
)


class TestWindowing:
    def test_spec_rejects_hop_gt_length(self) -> None:
        with pytest.raises(SignalError):
            WindowSpec(length=8, hop=16)

    def test_count_for(self) -> None:
        spec = WindowSpec(length=4, hop=2)
        assert spec.count_for(10) == 4
        assert spec.count_for(3) == 0

    def test_frame_shape(self) -> None:
        signal = np.arange(10, dtype=np.float64)
        frames = frame_signal(signal, WindowSpec(length=4, hop=2))
        assert frames.shape == (4, 4)
        assert frames[0].tolist() == [0.0, 1.0, 2.0, 3.0]
        assert frames[1].tolist() == [2.0, 3.0, 4.0, 5.0]

    def test_frame_insufficient(self) -> None:
        with pytest.raises(InsufficientDataError):
            frame_signal(np.arange(3, dtype=np.float64), WindowSpec(length=8, hop=4))

    def test_frame_rejects_2d(self) -> None:
        with pytest.raises(SignalError):
            frame_signal(np.zeros((2, 2)), WindowSpec(length=2, hop=1))

    def test_sliding_windows_metadata(self) -> None:
        windows = list(sliding_windows(
            np.arange(8, dtype=np.float64), length=4, hop=2, sample_rate=10.0,
            device_id="d", channel="c", unit="g", start_index=4, start_time=1.7e9,
        ))
        assert len(windows) == 3
        assert windows[0].start_index == 4
        assert windows[1].start_index == 6
        assert windows[0].start_time == pytest.approx(1.7e9 + 0.4)

    def test_window_functions_are_non_negative(self) -> None:
        for fn in (rectangular, hann, hamming, blackman):
            w = fn(64)
            assert w.shape == (64,)
            assert np.all(w >= -1e-12)

    def test_hann_symmetric_variant_is_symmetric(self) -> None:
        w = hann(64, periodic=False)
        assert np.allclose(w, w[::-1], atol=1e-12)

    def test_hann_periodic_is_for_spectral_use(self) -> None:
        w = hann(64)
        # Periodic Hann is deliberately not symmetric: it is the correct window
        # for FFT analysis because it removes the duplicate endpoint.
        assert not np.allclose(w, w[::-1], atol=1e-12)
        assert math.isclose(w[0], 0.0, abs_tol=1e-12)

    def test_hann_endpoints(self) -> None:
        w = hann(64)
        assert math.isclose(w[0], 0.0, abs_tol=1e-12)

    def test_apply_window_unknown(self) -> None:
        with pytest.raises(SignalError):
            apply_window_function(np.ones(4), "kaiser")

    def test_apply_window_rectangular_is_identity(self) -> None:
        x = np.arange(8, dtype=np.float64)
        assert np.allclose(apply_window_function(x, "rectangular"), x)


class TestSpectral:
    def test_frequency_axis(self) -> None:
        freqs = rfft_magnitude(np.zeros(8), 8.0).frequencies
        assert math.isclose(freqs[-1], 4.0)  # Nyquist

    def test_pure_tone_peak(self) -> None:
        n, fs = 256, 1000.0
        t = np.arange(n) / fs
        signal = np.sin(2 * np.pi * 100.0 * t)
        spectrum = rfft_magnitude(signal, fs)
        peak_index = int(np.argmax(spectrum.values))
        assert math.isclose(spectrum.frequencies[peak_index], 100.0, abs_tol=5.0)

    def test_psd_parseval(self) -> None:
        n, fs = 512, 1000.0
        signal = np.sin(2 * np.pi * 50 * np.arange(n) / fs)
        psd = power_spectral_density(signal, fs)
        integral = float(np.sum(psd.values) * psd.df)
        mean_square = float(np.mean((signal - signal.mean()) ** 2))
        assert math.isclose(integral, mean_square, rel_tol=0.05)

    def test_psd_non_negative(self) -> None:
        signal = np.random.default_rng(0).normal(size=128)
        assert np.all(power_spectral_density(signal, 1000.0).values >= 0.0)

    def test_welch_shape(self) -> None:
        psd = welch_psd(np.random.default_rng(1).normal(size=512), 1000.0,
                        segment_length=128)
        assert psd.method == "welch"
        assert psd.frequencies.size == 65

    def test_welch_rejects_bad_overlap(self) -> None:
        with pytest.raises(SignalError):
            welch_psd(np.zeros(64), 1000.0, overlap=1.5)

    def test_welch_averages_more_than_one_segment(self) -> None:
        signal = np.random.default_rng(2).normal(size=1024)
        single = power_spectral_density(signal, 1000.0)
        averaged = welch_psd(signal, 1000.0, segment_length=256)
        # Welch reduces variance: total power should be in the same ballpark.
        assert math.isclose(single.total_power, averaged.total_power, rel_tol=0.5)

    def test_spectral_centroid_of_tone(self) -> None:
        n, fs = 256, 1000.0
        signal = np.sin(2 * np.pi * 100 * np.arange(n) / fs)
        psd = power_spectral_density(signal, fs)
        centroid = spectral_centroid(psd.frequencies, psd.values)
        assert math.isclose(centroid, 100.0, abs_tol=15.0)

    def test_centroid_zero_for_empty_spectrum(self) -> None:
        assert spectral_centroid(np.zeros(4), np.zeros(4)) == 0.0

    def test_centroid_length_mismatch(self) -> None:
        with pytest.raises(SignalError):
            spectral_centroid(np.zeros(4), np.zeros(3))

    def test_peaks_finds_single_tone(self) -> None:
        n, fs = 256, 1000.0
        signal = np.sin(2 * np.pi * 100 * np.arange(n) / fs)
        psd = power_spectral_density(signal, fs)
        peaks = spectral_peaks(psd.frequencies, psd.values)
        assert peaks
        assert math.isclose(peaks[0][0], 100.0, abs_tol=10.0)

    def test_peaks_respects_max(self) -> None:
        signal = np.random.default_rng(3).normal(size=256)
        psd = power_spectral_density(signal, 1000.0)
        assert len(spectral_peaks(psd.frequencies, psd.values, max_peaks=2)) <= 2

    def test_band_energy(self) -> None:
        freqs = np.array([0.0, 10.0, 20.0, 30.0])
        psd = np.array([0.0, 1.0, 1.0, 0.0])
        # Band power = sum(psd inside the band) * df = (1 + 1) * 10.
        assert math.isclose(band_energy(freqs, psd, 5.0, 25.0), 20.0)

    def test_band_energy_rejects_inverted_band(self) -> None:
        with pytest.raises(SignalError):
            band_energy(np.zeros(4), np.zeros(4), 10.0, 5.0)

    def test_band_energy_empty_band(self) -> None:
        assert band_energy(np.array([0.0, 1.0]), np.ones(2), 100.0, 200.0) == 0.0

    def test_entropy_of_delta_is_zero(self) -> None:
        mags = np.zeros(8)
        mags[3] = 1.0
        assert math.isclose(spectral_entropy(mags), 0.0, abs_tol=1e-12)

    def test_entropy_of_uniform_is_log2(self) -> None:
        mags = np.full(8, 1.0)
        assert math.isclose(spectral_entropy(mags), 3.0, rel_tol=1e-9)

    def test_flatness_bounds(self) -> None:
        flat = spectral_flatness(np.full(16, 2.0))
        tonal = spectral_flatness(np.eye(16)[0] * 5.0)
        assert math.isclose(flat, 1.0, abs_tol=1e-9)
        assert 0.0 <= tonal < flat


class TestFilters:
    def test_lowpass_attenuates_high_frequency(self) -> None:
        fs = 1000.0
        t = np.arange(2048) / fs
        signal = np.sin(2 * np.pi * 10 * t) + np.sin(2 * np.pi * 300 * t)
        filtered = signal
        for section in butterworth_lowpass(fs, 50.0, order=4):
            filtered = _apply(filtered, section)
        low_before = _band_power(signal, fs, 5.0, 15.0)
        low_after = _band_power(filtered, fs, 5.0, 15.0)
        high_before = _band_power(signal, fs, 290.0, 310.0)
        high_after = _band_power(filtered, fs, 290.0, 310.0)
        assert high_after < high_before * 0.05
        assert low_after > low_before * 0.5

    def test_highpass_removes_dc(self) -> None:
        fs = 1000.0
        signal = np.full(512, 5.0) + np.sin(2 * np.pi * 50 * np.arange(512) / fs)
        filtered = signal
        for section in butterworth_highpass(fs, 20.0, order=4):
            filtered = _apply(filtered, section)
        assert abs(float(np.mean(filtered[-64:]))) < 0.5

    def test_bandpass_keeps_middle(self) -> None:
        fs = 1000.0
        t = np.arange(2048) / fs
        signal = np.sin(2 * np.pi * 20 * t) + np.sin(2 * np.pi * 200 * t)
        filtered = signal
        for section in butterworth_bandpass(fs, 150.0, 250.0, order=2):
            filtered = _apply(filtered, section)
        assert _band_power(filtered, fs, 190.0, 210.0) > _band_power(filtered, fs, 10.0, 30.0)

    def test_cutoff_above_nyquist_rejected(self) -> None:
        with pytest.raises(SignalError):
            butterworth_lowpass(1000.0, 600.0)

    def test_moving_average_of_constant(self) -> None:
        assert np.allclose(moving_average(np.full(16, 3.0), 4), 3.0)

    def test_moving_average_smooths(self) -> None:
        noisy = np.array([0.0, 10.0, 0.0, 10.0], dtype=np.float64)
        smoothed = moving_average(noisy, 3)
        assert smoothed[-1] < 10.0

    def test_median_filter_removes_single_outlier(self) -> None:
        signal = np.array([1.0, 1.0, 100.0, 1.0, 1.0])
        assert median_filter(signal, 3)[2] == 1.0

    def test_median_filter_rejects_even_window(self) -> None:
        with pytest.raises(SignalError):
            median_filter(np.ones(8), 4)


class TestDetrend:
    def test_linear_removes_ramp(self) -> None:
        ramp = np.arange(100, dtype=np.float64) * 2.0 + 5.0
        assert abs(float(np.mean(detrend_linear(ramp)))) < 1e-9

    def test_polynomial_trend_matches_input_for_line(self) -> None:
        ramp = np.arange(50, dtype=np.float64) * 3.0
        assert np.allclose(polynomial_trend(ramp, 1), ramp, atol=1e-9)

    def test_insufficient_for_fit(self) -> None:
        with pytest.raises(InsufficientDataError):
            polynomial_trend(np.array([1.0]), 5)

    def test_detrend_modes(self) -> None:
        x = np.arange(20, dtype=np.float64) + 3.0
        assert np.allclose(detrend(x, "none"), x)
        assert abs(float(np.mean(detrend(x, "constant")))) < 1e-9
        assert abs(float(np.mean(detrend(x, "linear")))) < 1e-9

    def test_detrend_unknown_mode(self) -> None:
        with pytest.raises(SignalError):
            detrend(np.ones(4), "quadratic")


class TestResample:
    def test_decimate_reduces_length(self) -> None:
        signal = np.random.default_rng(4).normal(size=1024)
        out, rate = decimate(signal, 4, 1000.0)
        assert out.size == 256
        assert math.isclose(rate, 250.0)

    def test_decimate_identity(self) -> None:
        signal = np.arange(16, dtype=np.float64)
        out, rate = decimate(signal, 1, 100.0)
        assert np.allclose(out, signal)
        assert rate == 100.0

    def test_decimate_too_short(self) -> None:
        with pytest.raises(SignalError):
            decimate(np.ones(2), 8, 1000.0)

    def test_interpolate_increases_length(self) -> None:
        signal = np.arange(64, dtype=np.float64)
        out, rate = interpolate_linear(signal, 100.0, 200.0)
        assert out.size == 127
        assert rate == 200.0

    def test_interpolate_rejects_downsample(self) -> None:
        with pytest.raises(SignalError):
            interpolate_linear(np.arange(16, dtype=np.float64), 100.0, 50.0)

    def test_interpolate_too_short(self) -> None:
        with pytest.raises(SignalError):
            interpolate_linear(np.ones(1), 100.0, 200.0)

    def test_resample_linear_same_rate(self) -> None:
        signal = np.arange(16, dtype=np.float64)
        out, rate = resample_linear(signal, 100.0, 100.0)
        assert rate == 100.0
        assert np.allclose(out, signal)

    def test_resample_linear_downsample_integer(self) -> None:
        signal = np.arange(1024, dtype=np.float64)
        _, rate = resample_linear(signal, 1000.0, 500.0)
        assert math.isclose(rate, 500.0)

    def test_resample_linear_upsample(self) -> None:
        signal = np.arange(64, dtype=np.float64)
        out, rate = resample_linear(signal, 100.0, 250.0)
        assert rate == 250.0
        assert out.size > signal.size

    def test_resample_ratio_downsample(self) -> None:
        out = resample_ratio(np.arange(256, dtype=np.float64), up=1, down=2)
        assert out.size == 129


class TestEnvelope:
    def test_envelope_of_tone_is_constant(self) -> None:
        fs = 1000.0
        t = np.arange(1024) / fs
        signal = np.sin(2 * np.pi * 50 * t)
        env = amplitude_envelope(signal)
        assert math.isclose(float(np.mean(env)), 1.0, rel_tol=0.05)

    def test_analytic_signal_is_complex(self) -> None:
        analytic = hilbert_analytic(np.sin(np.arange(64) * 0.1))
        assert np.iscomplexobj(analytic)

    def test_envelope_rejects_short(self) -> None:
        with pytest.raises(InsufficientDataError):
            hilbert_analytic(np.array([1.0]))

    def test_envelope_odd_length(self) -> None:
        analytic = hilbert_analytic(np.sin(np.arange(63) * 0.1))
        assert analytic.size == 63
        assert np.iscomplexobj(analytic)

    def test_envelope_rejects_2d(self) -> None:
        with pytest.raises(InsufficientDataError):
            hilbert_analytic(np.zeros((2, 2)))

    def test_envelope_features_stats(self) -> None:
        from sofia_ai.signal.envelope import envelope_features

        fs = 1000.0
        t = np.arange(1024) / fs
        signal = np.sin(2 * np.pi * 50 * t)
        stats = envelope_features(signal)
        assert stats["envelope_peak"] >= stats["envelope_mean"]
        assert stats["envelope_min"] <= stats["envelope_mean"]
        assert stats["envelope_peak_to_peak"] >= 0.0

    def test_envelope_spectrum_shape(self) -> None:
        from sofia_ai.signal.envelope import envelope_spectrum

        fs = 1000.0
        t = np.arange(1024) / fs
        signal = np.sin(2 * np.pi * 50 * t) + 0.3 * np.sin(2 * np.pi * 25 * t)
        freqs, mags = envelope_spectrum(signal, fs)
        assert freqs.size == mags.size
        assert float(np.max(mags)) > 0.0
        assert float(freqs[0]) >= 0.0


def _apply(values: np.ndarray, section) -> np.ndarray:
    from sofia_ai.signal.filters import apply_biquad

    return apply_biquad(values, section)


def _band_power(signal: np.ndarray, fs: float, low: float, high: float) -> float:
    psd = power_spectral_density(signal, fs)
    return band_energy(psd.frequencies, psd.values, low, high)
