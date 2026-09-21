"""Property tests for mathematical and signal invariants.

Hypothesis is used where it adds real value: closed-form mathematical properties
that must hold for *all* inputs, not just the hand-picked examples in the unit
tests.
"""

from __future__ import annotations

import math

import numpy as np
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from sofia_ai.core.contracts import DataQuality
from sofia_ai.core.units import Quantity, convert, dimension_of, is_known_unit, unit_alias
from sofia_ai.edge import BoundedRingBuffer
from sofia_ai.features.statistical import (
    crest_factor,
    energy,
    kurtosis,
    mean,
    peak,
    peak_to_peak,
    rms,
    skewness,
    std_dev,
    variance,
    zero_crossing_rate,
)
from sofia_ai.signal.spectral import (
    band_energy,
    power_spectral_density,
    spectral_centroid,
    spectral_entropy,
    spectral_flatness,
)

finite_floats = st.floats(allow_nan=False, allow_infinity=False,
                          min_value=-1e6, max_value=1e6)
signals = st.lists(finite_floats, min_size=1, max_size=256).map(
    lambda xs: np.asarray(xs, dtype=np.float64)
)


def _nonzero(signal: np.ndarray) -> np.ndarray:
    """Nudge a signal away from the degenerate all-zero case."""
    if np.allclose(signal, 0.0):
        return signal + 1e-9
    return signal


class TestStatisticalProperties:
    @given(signals)
    @settings(max_examples=100, deadline=None)
    def test_rms_is_at_least_abs_mean(self, signal: np.ndarray) -> None:
        assert rms(signal) >= abs(mean(signal)) - 1e-9

    @given(signals)
    @settings(max_examples=100, deadline=None)
    def test_variance_is_square_of_std(self, signal: np.ndarray) -> None:
        assert math.isclose(variance(signal), std_dev(signal) ** 2, rel_tol=1e-9,
                            abs_tol=1e-12)

    @given(signals)
    @settings(max_examples=100, deadline=None)
    def test_peak_is_non_negative(self, signal: np.ndarray) -> None:
        assert peak(signal) >= 0.0

    @given(signals)
    @settings(max_examples=100, deadline=None)
    def test_peak_to_peak_is_non_negative(self, signal: np.ndarray) -> None:
        assert peak_to_peak(signal) >= 0.0

    @given(signals)
    @settings(max_examples=100, deadline=None)
    def test_energy_is_sum_of_squares(self, signal: np.ndarray) -> None:
        assert math.isclose(energy(signal), float(np.sum(signal * signal)),
                            rel_tol=1e-9, abs_tol=1e-6)

    @given(signals.map(_nonzero))
    @settings(max_examples=100, deadline=None)
    def test_crest_factor_is_at_least_one(self, signal: np.ndarray) -> None:
        assert crest_factor(signal) >= 1.0 - 1e-9

    @given(signals)
    @settings(max_examples=100, deadline=None)
    def test_zero_crossing_rate_is_a_fraction(self, signal: np.ndarray) -> None:
        rate = zero_crossing_rate(signal)
        assert 0.0 <= rate <= 1.0

    @given(signals)
    @settings(max_examples=100, deadline=None)
    def test_kurtosis_is_non_negative(self, signal: np.ndarray) -> None:
        assert kurtosis(signal) >= 0.0

    @given(signals)
    @settings(max_examples=100, deadline=None)
    def test_skewness_is_bounded_by_kurtosis(self, signal: np.ndarray) -> None:
        # |skew| <= sqrt(kurtosis) is a standard moment inequality.
        assert abs(skewness(signal)) <= math.sqrt(kurtosis(signal)) + 1e-6

    @given(st.floats(allow_nan=False, allow_infinity=False, min_value=-1000,
                     max_value=1000),
           st.integers(min_value=2, max_value=128))
    @settings(max_examples=100, deadline=None)
    def test_constant_signal_rms_equals_abs_value(self, value: float, n: int) -> None:
        signal = np.full(n, value)
        assert math.isclose(rms(signal), abs(value), rel_tol=1e-12, abs_tol=1e-12)

    @given(signals)
    @settings(max_examples=60, deadline=None)
    def test_features_are_always_finite(self, signal: np.ndarray) -> None:
        for value in (mean(signal), rms(signal), peak(signal), peak_to_peak(signal),
                      variance(signal), std_dev(signal), crest_factor(signal),
                      skewness(signal), kurtosis(signal), zero_crossing_rate(signal),
                      energy(signal)):
            assert math.isfinite(value)


class TestSpectralProperties:
    @given(st.integers(min_value=8, max_value=128))
    @settings(max_examples=40, deadline=None,
              suppress_health_check=[HealthCheck.too_slow])
    def test_psd_is_non_negative(self, n: int) -> None:
        rng = np.random.default_rng(n)
        signal = rng.normal(size=n)
        psd = power_spectral_density(signal, 1000.0)
        assert np.all(psd.values >= 0.0)

    @given(st.integers(min_value=8, max_value=128))
    @settings(max_examples=40, deadline=None,
              suppress_health_check=[HealthCheck.too_slow])
    def test_spectral_centroid_within_nyquist(self, n: int) -> None:
        rng = np.random.default_rng(n + 1000)
        signal = rng.normal(size=n)
        psd = power_spectral_density(signal, 1000.0)
        centroid = spectral_centroid(psd.frequencies, psd.values)
        assert 0.0 <= centroid <= 500.0 + 1e-6

    @given(st.integers(min_value=8, max_value=128))
    @settings(max_examples=40, deadline=None,
              suppress_health_check=[HealthCheck.too_slow])
    def test_parseval_relation(self, n: int) -> None:
        """Exact identity: the PSD integral equals the windowed signal energy.

        ``sum(psd) * df == sum((x * w)^2) / sum(w^2)`` holds algebraically for the
        normalization used here, so it can be asserted tightly for any input.
        """
        from sofia_ai.signal.windowing import hann

        rng = np.random.default_rng(n + 2000)
        signal = rng.normal(size=n)
        centred = signal - float(np.mean(signal))
        window = hann(n)
        expected = float(np.sum((centred * window) ** 2) / np.sum(window**2))
        psd = power_spectral_density(signal, 1000.0)
        assert math.isclose(psd.total_power, expected, rel_tol=1e-9, abs_tol=1e-9)

    @given(st.integers(min_value=4, max_value=64))
    @settings(max_examples=40, deadline=None)
    def test_flatness_is_a_fraction(self, n: int) -> None:
        rng = np.random.default_rng(n + 3000)
        magnitudes = np.abs(rng.normal(size=n)) + 1e-9
        assert 0.0 <= spectral_flatness(magnitudes) <= 1.0 + 1e-9

    @given(st.integers(min_value=2, max_value=64))
    @settings(max_examples=40, deadline=None)
    def test_entropy_is_bounded_by_log2(self, n: int) -> None:
        rng = np.random.default_rng(n + 4000)
        magnitudes = np.abs(rng.normal(size=n)) + 1e-9
        assert 0.0 <= spectral_entropy(magnitudes) <= math.log2(n) + 1e-9

    @given(st.integers(min_value=8, max_value=64))
    @settings(max_examples=30, deadline=None)
    def test_band_energy_is_monotonic_in_band_width(self, n: int) -> None:
        rng = np.random.default_rng(n + 5000)
        freqs = np.linspace(0.0, 500.0, n)
        psd = np.abs(rng.normal(size=n)) + 1e-9
        narrow = band_energy(freqs, psd, 100.0, 150.0)
        wide = band_energy(freqs, psd, 100.0, 300.0)
        assert wide >= narrow - 1e-9


class TestUnitProperties:
    @given(st.sampled_from(["g", "m/s2", "mm/s", "mm/s2", "C", "K", "F", "RPM", "Hz",
                            "V", "mV", "A", "mA", "W", "kW", "Pa", "kPa", "bar",
                            "psi", "mm", "um", "mil", "in"]),
           finite_floats)
    @settings(max_examples=200, deadline=None)
    def test_si_roundtrip_is_identity(self, unit: str, value: float) -> None:
        from sofia_ai.core.units import from_si, to_si

        restored = from_si(to_si(value, unit), unit)
        assert math.isclose(restored, value, rel_tol=1e-9, abs_tol=1e-9)

    @given(st.sampled_from(["g", "m/s2"]), st.sampled_from(["g", "m/s2"]),
           finite_floats)
    @settings(max_examples=100, deadline=None)
    def test_conversion_roundtrip_is_identity(self, source: str, target: str,
                                              value: float) -> None:
        restored = convert(convert(value, source, target), target, source)
        assert math.isclose(restored, value, rel_tol=1e-8, abs_tol=1e-9)

    @given(st.text(min_size=1, max_size=20))
    @settings(max_examples=100, deadline=None)
    def test_unknown_units_are_never_silently_accepted(self, text: str) -> None:
        if is_known_unit(text):
            assert unit_alias(text)
        else:
            from sofia_ai.core.errors import UnitError

            try:
                unit_alias(text)
            except UnitError:
                pass
            else:  # pragma: no cover
                raise AssertionError(f"unknown unit {text!r} was accepted")

    @given(st.sampled_from(["g", "mm/s", "C", "RPM"]), finite_floats)
    @settings(max_examples=100, deadline=None)
    def test_quantity_scaling_is_linear(self, unit: str, value: float) -> None:
        quantity = Quantity(value, unit)
        assert math.isclose((quantity * 2.0).value, value * 2.0, rel_tol=1e-9,
                            abs_tol=1e-9)

    @given(st.sampled_from(["g", "mm/s"]))
    @settings(max_examples=20, deadline=None)
    def test_dimension_is_stable(self, unit: str) -> None:
        assert dimension_of(unit) == dimension_of(unit_alias(unit))


class TestBufferProperties:
    @given(st.integers(min_value=1, max_value=32),
           st.lists(st.integers(), min_size=0, max_size=64))
    @settings(max_examples=100, deadline=None)
    def test_size_never_exceeds_capacity(self, capacity: int, items: list[int]) -> None:
        buffer: BoundedRingBuffer[int] = BoundedRingBuffer(capacity=capacity)
        for item in items:
            buffer.push(item)
        assert len(buffer) <= capacity

    @given(st.integers(min_value=1, max_value=16),
           st.lists(st.integers(min_value=0, max_value=1000), min_size=0, max_size=64))
    @settings(max_examples=100, deadline=None)
    def test_drain_preserves_fifo_order(self, capacity: int, items: list[int]) -> None:
        buffer: BoundedRingBuffer[int] = BoundedRingBuffer(capacity=capacity)
        for item in items:
            buffer.push(item)
        retained = buffer.drain()
        expected = items[-capacity:] if len(items) > capacity else items
        assert retained == expected

    @given(st.integers(min_value=1, max_value=16),
           st.lists(st.integers(), min_size=0, max_size=32))
    @settings(max_examples=100, deadline=None)
    def test_accounting_is_exact(self, capacity: int, items: list[int]) -> None:
        buffer: BoundedRingBuffer[int] = BoundedRingBuffer(capacity=capacity)
        for item in items:
            buffer.push(item)
        assert buffer.written == len(items)
        assert buffer.dropped == max(0, len(items) - capacity)


class TestSerializationProperties:
    @given(st.dictionaries(st.text(min_size=1, max_size=8).filter(
        lambda s: "\x00" not in s),
        st.one_of(st.integers(), st.floats(allow_nan=False, allow_infinity=False),
                  st.text(max_size=16)),
        max_size=6))
    @settings(max_examples=80, deadline=None)
    def test_json_roundtrip_preserves_equality(self, payload: dict) -> None:
        from sofia_ai.core.serialization import decode, encode

        assert decode(encode(payload)) == payload

    @given(st.lists(st.floats(allow_nan=False, allow_infinity=False,
                              min_value=-1e6, max_value=1e6),
                    min_size=0, max_size=32))
    @settings(max_examples=60, deadline=None)
    def test_feature_vector_roundtrip(self, values: list[float]) -> None:
        from sofia_ai.core.contracts import FeatureVector

        names = tuple(f"f{i}" for i in range(len(values)))
        import numpy as np

        finite = [float(v) for v in values if np.isfinite(v)]
        vector = FeatureVector(names=names[: len(finite)],
                               values=tuple(finite),
                               extractor_id="p", extractor_version="1",
                               quality=DataQuality.GOOD)
        assert FeatureVector.from_dict(vector.to_dict()) == vector
