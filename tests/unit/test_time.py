"""Unit tests for time discipline and clock anomaly detection (SOFIA-TIME-*)."""

from __future__ import annotations

import math

import pytest

from sofia_ai.core.errors import TimestampError
from sofia_ai.core.time import (
    ClockAnomaly,
    ClockMonitor,
    FixedTimeSource,
    SystemTimeSource,
    iso8601_to_seconds,
    seconds_to_iso8601,
    validate_timestamp,
)


class TestValidation:
    def test_accepts_plausible(self) -> None:
        assert validate_timestamp(1_700_000_000.0) == 1_700_000_000.0

    def test_rejects_nan(self) -> None:
        with pytest.raises(TimestampError):
            validate_timestamp(float("nan"))

    def test_rejects_zero(self) -> None:
        with pytest.raises(TimestampError):
            validate_timestamp(0.0)

    def test_rejects_far_future(self) -> None:
        with pytest.raises(TimestampError):
            validate_timestamp(1e12)


class TestClockMonitor:
    def test_monotonic_stream_is_clean(self) -> None:
        monitor = ClockMonitor()
        for i in range(10):
            assert monitor.observe(1_700_000_000.0 + i * 0.1) is None
        assert monitor.anomaly_count == 0

    def test_detects_backwards(self) -> None:
        monitor = ClockMonitor()
        monitor.observe(1_700_000_000.0)
        anomaly = monitor.observe(1_699_999_999.0)
        assert anomaly is not None
        assert anomaly.kind == "backwards"

    def test_detects_duplicate(self) -> None:
        monitor = ClockMonitor()
        monitor.observe(1_700_000_000.0)
        anomaly = monitor.observe(1_700_000_000.0)
        assert anomaly is not None
        assert anomaly.kind == "duplicate"

    def test_detects_forward_jump(self) -> None:
        monitor = ClockMonitor(max_forward_jump_s=10.0)
        monitor.observe(1_700_000_000.0)
        anomaly = monitor.observe(1_700_001_000.0)
        assert anomaly is not None
        assert anomaly.kind == "jump"

    def test_reset_clears_state(self) -> None:
        monitor = ClockMonitor()
        monitor.observe(1_700_000_000.0)
        monitor.reset()
        assert monitor.observe(1_600_000_000.0) is None

    def test_anomaly_serializes(self) -> None:
        anomaly = ClockAnomaly(kind="jump", observed=1.0, reference=0.0, detail="x")
        assert anomaly.to_dict()["kind"] == "jump"


class TestTimeSources:
    def test_fixed_clock_is_deterministic(self) -> None:
        clock = FixedTimeSource()
        assert clock.wall() == clock.wall()
        first = clock.monotonic()
        second = clock.monotonic()
        assert second > first

    def test_fixed_clock_advance(self) -> None:
        clock = FixedTimeSource(wall_time=1000.0, monotonic_time=0.0)
        clock.advance(5.0)
        assert clock.wall() == 1005.0

    def test_system_clock_advances(self) -> None:
        clock = SystemTimeSource()
        assert clock.wall() > 0
        assert clock.monotonic() >= 0


class TestIso8601:
    def test_roundtrip(self) -> None:
        text = seconds_to_iso8601(1_700_000_000.0)
        assert math.isclose(iso8601_to_seconds(text), 1_700_000_000.0, abs_tol=1e-3)

    def test_format_is_utc(self) -> None:
        assert seconds_to_iso8601(1_700_000_000.0).endswith("Z")

    def test_rejects_garbage(self) -> None:
        with pytest.raises(TimestampError):
            iso8601_to_seconds("not a timestamp")


def test_dsp_axis_is_not_wall_clock() -> None:
    """The window time axis must derive from sample rate and index."""
    import numpy as np

    from sofia_ai.core.contracts import SignalWindow

    window = SignalWindow(values=np.zeros(10), sample_rate=100.0, device_id="d",
                          channel="c", unit="g", start_time=1_700_000_000.0)
    axis = window.time_axis
    assert math.isclose(axis[0], 0.0)
    assert math.isclose(axis[-1], 0.09)
