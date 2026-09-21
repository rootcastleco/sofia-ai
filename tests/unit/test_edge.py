"""Unit tests for edge runtime, buffers, store-and-forward and reconnect."""

from __future__ import annotations

import json
import math

import pytest

from sofia_ai.core.contracts import TelemetrySample
from sofia_ai.core.errors import BufferOverflowError, SofiaError, ValidationError
from sofia_ai.edge import (
    BackoffCalculator,
    BoundedRingBuffer,
    ChannelBuffer,
    ConnectionState,
    HealthMonitor,
    HealthStatus,
    OverflowPolicy,
    Reconnector,
    ReconnectPolicy,
    RuntimeConfig,
    StoreForward,
)
from sofia_ai.edge.health import DEGRADED_THRESHOLDS


def _sample(index: int, value: float = 1.0) -> TelemetrySample:
    return TelemetrySample(
        timestamp=1_700_000_000.0 + index * 0.001,
        device_id="d", channel="c", value=value, unit="g",
        source="test", sequence_number=index,
    )


class TestBoundedRingBuffer:
    def test_fifo_order(self) -> None:
        buffer: BoundedRingBuffer[int] = BoundedRingBuffer(capacity=4)
        for i in range(4):
            buffer.push(i)
        assert buffer.drain() == [0, 1, 2, 3]

    def test_drop_oldest(self) -> None:
        buffer: BoundedRingBuffer[int] = BoundedRingBuffer(capacity=3)
        for i in range(5):
            buffer.push(i)
        assert len(buffer) == 3
        assert buffer.dropped == 2
        assert buffer.drain() == [2, 3, 4]

    def test_drop_newest(self) -> None:
        buffer: BoundedRingBuffer[int] = BoundedRingBuffer(
            capacity=3, policy=OverflowPolicy.DROP_NEWEST)
        for i in range(5):
            buffer.push(i)
        assert buffer.drain() == [0, 1, 2]
        assert buffer.dropped == 2

    def test_reject_policy_raises(self) -> None:
        buffer: BoundedRingBuffer[int] = BoundedRingBuffer(
            capacity=2, policy=OverflowPolicy.REJECT)
        buffer.push(1)
        buffer.push(2)
        with pytest.raises(BufferOverflowError):
            buffer.push(3)

    def test_saturation(self) -> None:
        buffer: BoundedRingBuffer[int] = BoundedRingBuffer(capacity=4)
        buffer.push(1)
        assert math.isclose(buffer.saturation, 0.25)

    def test_pop_empty_raises(self) -> None:
        buffer: BoundedRingBuffer[int] = BoundedRingBuffer(capacity=2)
        with pytest.raises(BufferOverflowError):
            buffer.pop()

    def test_extend_counts_retained(self) -> None:
        buffer: BoundedRingBuffer[int] = BoundedRingBuffer(
            capacity=3, policy=OverflowPolicy.DROP_NEWEST)
        assert buffer.extend([1, 2, 3, 4, 5]) == 3

    def test_extend_retains_all_with_drop_oldest(self) -> None:
        buffer: BoundedRingBuffer[int] = BoundedRingBuffer(capacity=3)
        assert buffer.extend([1, 2, 3, 4, 5]) == 5
        assert buffer.dropped == 2

    def test_peek_does_not_consume(self) -> None:
        buffer: BoundedRingBuffer[int] = BoundedRingBuffer(capacity=3)
        buffer.push(7)
        assert buffer.peek() == [7]
        assert len(buffer) == 1

    def test_stats(self) -> None:
        buffer: BoundedRingBuffer[int] = BoundedRingBuffer(capacity=2)
        buffer.push(1)
        stats = buffer.stats()
        assert stats["capacity"] == 2
        assert stats["written"] == 1


class TestChannelBuffer:
    def test_per_channel_isolation(self) -> None:
        channels: ChannelBuffer[int] = ChannelBuffer(capacity=4)
        channels.push("a", 1)
        channels.push("b", 2)
        assert channels.drain("a") == [1]
        assert channels.drain("b") == [2]

    def test_channel_count_is_bounded(self) -> None:
        channels: ChannelBuffer[int] = ChannelBuffer(capacity=4, max_channels=2)
        assert channels.push("a", 1)
        assert channels.push("b", 1)
        assert not channels.push("c", 1)
        assert channels.rejected_channels == 1

    def test_totals(self) -> None:
        channels: ChannelBuffer[int] = ChannelBuffer(capacity=2)
        for i in range(4):
            channels.push("a", i)
        assert channels.total_size() == 2
        assert channels.total_dropped() == 2
        assert channels.saturation() == 1.0


class TestStoreForward:
    def test_append_and_flush(self, tmp_path) -> None:
        store = StoreForward(directory=str(tmp_path), flush_every=2)
        store.append(_sample(0))
        assert store.pending == 1
        store.append(_sample(1))
        assert store.pending == 0
        assert store.file_count == 1

    def test_rotation_enforces_byte_ceiling(self, tmp_path) -> None:
        store = StoreForward(directory=str(tmp_path), max_bytes=500, max_files=2,
                             flush_every=1)
        for i in range(30):
            store.append(_sample(i))
            store.flush()
        assert store.file_count <= 2
        assert store.bytes_used <= 500
        assert store.stats.rotations > 0

    def test_read_all_roundtrip(self, tmp_path) -> None:
        store = StoreForward(directory=str(tmp_path), flush_every=4)
        for i in range(8):
            store.append(_sample(i, float(i)))
        store.flush()
        recovered = store.read_all()
        assert len(recovered) == 8
        assert recovered[0].value == 0.0
        assert store.file_count == 0

    def test_corrupt_file_is_discarded(self, tmp_path) -> None:
        store = StoreForward(directory=str(tmp_path), flush_every=2)
        store.append(_sample(0))
        store.append(_sample(1))
        store.flush()
        for path in store._files():
            path.write_text("{not json}\n", encoding="utf-8")
        recovered = store.read_all()
        assert recovered == []
        assert store.stats.corrupt_files == 1

    def test_utilization(self, tmp_path) -> None:
        store = StoreForward(directory=str(tmp_path), max_bytes=1000, flush_every=1)
        store.append(_sample(0))
        store.flush()
        assert 0.0 < store.utilization <= 1.0

    def test_describe(self, tmp_path) -> None:
        payload = StoreForward(directory=str(tmp_path)).describe()
        assert "bytes_used" in payload and "stats" in payload

    def test_clear(self, tmp_path) -> None:
        store = StoreForward(directory=str(tmp_path), flush_every=1)
        store.append(_sample(0))
        store.flush()
        store.clear()
        assert store.file_count == 0

    def test_json_records_are_valid(self, tmp_path) -> None:
        store = StoreForward(directory=str(tmp_path), flush_every=1)
        store.append(_sample(3, 2.5))
        store.flush()
        for path in store._files():
            line = path.read_text(encoding="utf-8").strip()
            assert json.loads(line)["value"] == 2.5


class TestReconnect:
    def test_backoff_schedule_is_bounded(self) -> None:
        policy = ReconnectPolicy(max_attempts=8, initial_backoff_s=0.5,
                                 max_backoff_s=5.0, multiplier=2.0)
        schedule = BackoffCalculator(policy).schedule()
        assert len(schedule) == 8
        assert all(d <= 5.0 for d in schedule)
        assert schedule == sorted(schedule)

    def test_reconnect_succeeds_on_third_attempt(self) -> None:
        attempts = {"n": 0}

        def connect() -> bool:
            attempts["n"] += 1
            return attempts["n"] >= 3

        delays: list[float] = []
        reconnector = Reconnector(policy=ReconnectPolicy(max_attempts=8),
                                  connect=connect, sleep=delays.append)
        assert reconnector.run()
        assert reconnector.state is ConnectionState.CONNECTED
        assert reconnector.attempts == 3
        assert len(delays) == 3

    def test_exhaustion_is_an_explicit_error_state(self) -> None:
        reconnector = Reconnector(policy=ReconnectPolicy(max_attempts=3),
                                  connect=lambda: False, sleep=lambda _s: None)
        assert not reconnector.run()
        assert reconnector.state is ConnectionState.FAILED
        assert reconnector.attempts == 3

    def test_transport_exception_is_contained(self) -> None:
        def connect() -> bool:
            raise RuntimeError("socket closed")

        reconnector = Reconnector(policy=ReconnectPolicy(max_attempts=2),
                                  connect=connect, sleep=lambda _s: None)
        assert not reconnector.run()
        assert reconnector.state is ConnectionState.FAILED

    def test_mark_disconnected_resets(self) -> None:
        reconnector = Reconnector(connect=lambda: False, sleep=lambda _s: None)
        reconnector.run()
        reconnector.mark_disconnected()
        assert reconnector.state is ConnectionState.DISCONNECTED
        assert reconnector.attempts == 0

    def test_policy_validation(self) -> None:
        with pytest.raises(ValueError):
            ReconnectPolicy(max_attempts=0)
        with pytest.raises(ValueError):
            ReconnectPolicy(initial_backoff_s=10.0, max_backoff_s=5.0)

    def test_describe(self) -> None:
        payload = Reconnector(connect=lambda: True).describe()
        assert payload["state"] == "DISCONNECTED"


class TestHealthMonitor:
    def test_ready(self) -> None:
        report = HealthMonitor(lambda: {
            "stats": {"ticks": 10, "inferences": 10, "inference_failures": 0,
                      "source_errors": 0},
            "buffer": {"saturation": 0.1},
            "source": {"state": "OPEN"},
        }).check()
        assert report.status is HealthStatus.READY

    def test_degraded_on_saturation(self) -> None:
        report = HealthMonitor(lambda: {
            "stats": {"ticks": 10, "inferences": 10, "inference_failures": 0,
                      "source_errors": 0},
            "buffer": {"saturation": 0.99},
            "source": {"state": "OPEN"},
        }).check()
        assert report.status is HealthStatus.DEGRADED
        assert report.ready
        assert any("saturation" in r for r in report.reasons)

    def test_not_ready_on_failed_source(self) -> None:
        report = HealthMonitor(lambda: {
            "stats": {}, "buffer": {}, "source": {"state": "FAILED"},
        }).check()
        assert report.status is HealthStatus.NOT_READY
        assert not report.ready

    def test_probe_never_raises(self) -> None:
        def broken() -> dict:
            raise RuntimeError("boom")

        report = HealthMonitor(broken).check()
        assert report.status is HealthStatus.NOT_READY

    def test_thresholds_are_configurable(self) -> None:
        assert DEGRADED_THRESHOLDS["buffer_saturation"] == 0.9


class TestRuntimeConfig:
    def test_defaults(self) -> None:
        config = RuntimeConfig()
        assert config.window_length == 1024
        assert config.window_hop <= config.window_length

    def test_rejects_hop_gt_length(self) -> None:
        with pytest.raises(ValidationError):
            RuntimeConfig(window_length=8, window_hop=16)

    def test_rejects_bad_sample_rate(self) -> None:
        with pytest.raises(SofiaError):
            RuntimeConfig(sample_rate=0.0)
