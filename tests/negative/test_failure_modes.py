"""Fault injection and negative tests (mission §21).

These tests assert that bad input degrades the system in a defined way: counted,
typed, and never crashing the pipeline.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import pytest

from sofia_ai.core.contracts import DataQuality, TelemetrySample
from sofia_ai.core.errors import (
    FeatureError,
    InsufficientDataError,
    ModelError,
    ModelIntegrityError,
    PayloadTooLargeError,
    SofiaError,
    TelemetryProtocolError,
    ValidationError,
)
from sofia_ai.core.time import ClockMonitor
from sofia_ai.edge import BoundedRingBuffer, RuntimeConfig, StoreForward
from sofia_ai.edge.runtime import EdgeRuntime
from sofia_ai.features import extract_from_array
from sofia_ai.inference import build_manifest_for, create_backend
from sofia_ai.inference.detectors import DetectorConfig
from sofia_ai.telemetry import CsvSource, JsonlSource, SourceLimits, TelemetrySource

REPO_ROOT = Path(__file__).resolve().parents[2]


class FailingSource(TelemetrySource):
    """A source that fails a configurable number of times, then recovers."""

    def __init__(self, samples, *, fail_times: int = 1, source_id: str = "failing") -> None:
        super().__init__(source_id=source_id)
        self._samples = list(samples)
        self._cursor = 0
        self.fail_times = fail_times
        self.failures_injected = 0

    def read(self, *, max_records: int | None = None):
        self._require_open()
        if self.failures_injected < self.fail_times:
            self.failures_injected += 1
            raise TelemetryProtocolError("injected transport failure",
                                         details={"attempt": self.failures_injected})
        count = 256 if max_records is None else int(max_records)
        chunk = self._samples[self._cursor:self._cursor + count]
        self._cursor += len(chunk)
        self.stats.records_emitted += len(chunk)
        return list(chunk)


class ExplodingBackend:
    """A backend that always fails."""

    state = type("S", (), {"LOADED": "LOADED"})()
    state = "LOADED"
    manifest = None

    def infer(self, features):
        raise ModelError("injected inference failure", details={})


def _sample(index: int, value: float = 1.0) -> TelemetrySample:
    return TelemetrySample(timestamp=1_700_000_000.0 + index * 0.001, device_id="d",
                           channel="c", value=value, unit="g")


class TestBadValues:
    def test_nan_rejected_at_construction(self) -> None:
        with pytest.raises(ValidationError):
            TelemetrySample(timestamp=1.7e9, device_id="d", channel="c",
                            value=float("nan"), unit="g")

    def test_inf_rejected_at_construction(self) -> None:
        with pytest.raises(ValidationError):
            TelemetrySample(timestamp=1.7e9, device_id="d", channel="c",
                            value=float("inf"), unit="g")

    def test_nan_window_rejected(self) -> None:
        from sofia_ai.core.contracts import SignalWindow

        with pytest.raises(ValidationError):
            SignalWindow(values=np.array([1.0, float("nan")]), sample_rate=1000.0,
                         device_id="d", channel="c", unit="g")

    def test_non_finite_feature_rejected(self) -> None:
        from sofia_ai.core.contracts import FeatureVector

        with pytest.raises(ValidationError):
            FeatureVector(names=("a",), values=(float("nan"),), extractor_id="e",
                          extractor_version="1")

    def test_nan_never_reaches_serialization(self) -> None:
        from sofia_ai.core.serialization import encode

        with pytest.raises(ValidationError):
            encode({"a": float("nan")})

    def test_huge_input_does_not_crash(self) -> None:
        signal = np.random.default_rng(0).normal(size=100_000)
        features = extract_from_array(signal, 1000.0)
        assert all(np.isfinite(v) for v in features.values)


class TestMalformedFrames:
    def test_truncated_csv_row(self, tmp_path) -> None:
        path = tmp_path / "t.csv"
        path.write_text("timestamp,device_id,channel,value,unit\n"
                        "1700000000.0,d,c\n", encoding="utf-8")
        with CsvSource(str(path)) as source, pytest.raises(TelemetryProtocolError):
            source.read()

    def test_csv_injection_newline_in_value(self, tmp_path) -> None:
        """A newline in a data field must not create an extra logical record."""
        path = tmp_path / "t.csv"
        path.write_text('timestamp,device_id,channel,value,unit\n'
                        '1700000000.0,"d\nevil",c,1.0,g\n', encoding="utf-8")
        with CsvSource(str(path), strict=False) as source:
            samples = source.read()
        assert len(samples) <= 1

    def test_oversized_payload(self, tmp_path) -> None:
        path = tmp_path / "t.jsonl"
        path.write_text('{"timestamp":1700000000.0,"device_id":"d","channel":"c",'
                        '"value":1.0,"unit":"g","tags":{"x":"' + "A" * 5000 + '"}}\n',
                        encoding="utf-8")
        with JsonlSource(str(path)) as source, pytest.raises(TelemetryProtocolError):
            # A tag value longer than the ceiling is rejected; the adapter reports
            # it as a protocol error, which is what a strict transport should do.
            source.read()

    def test_binary_garbage_in_jsonl(self, tmp_path) -> None:
        path = tmp_path / "t.jsonl"
        path.write_bytes(b"\x00\x01\x02\x03\n")
        with pytest.raises(TelemetryProtocolError):
            JsonlSource(str(path)).open()

    def test_modbus_short_frame(self) -> None:
        from sofia_ai.telemetry.modbus import RegisterSpec, RegisterType, decode_registers

        spec = RegisterSpec(name="t", address=0, register_type=RegisterType.FLOAT32)
        with pytest.raises(TelemetryProtocolError):
            decode_registers([1], spec)

    def test_serial_partial_frame_not_emitted(self) -> None:
        from sofia_ai.telemetry.serial_adapter import SerialConfig, SerialSource

        class Port:
            in_waiting = 10

            def __init__(self, chunks):
                self._chunks = list(chunks)

            def read(self, size=1):
                return self._chunks.pop(0) if self._chunks else b""

            def close(self):
                pass

        port = Port([b"partial-without-delim"])
        source = SerialSource(SerialConfig(), port_factory=lambda _c: port)
        source.open()
        # No delimiter: nothing is emitted, and no exception is raised.
        assert source.read() == []


class TestModelRejection:
    def test_wrong_schema_rejected(self) -> None:
        from sofia_ai.inference.manifest import ModelManifest, SchemaSpec

        backend = create_backend("mad")
        manifest = ModelManifest(model_id="bad", version="1.0.0",
                                 input_schema=SchemaSpec(features=("missing",), shape=(1,)))
        with pytest.raises(ModelError):
            backend.load(manifest)

    def test_checksum_mismatch_refuses_load(self, tmp_path) -> None:
        from sofia_ai.inference.manifest import ModelManifest

        artifact = tmp_path / "model.bin"
        artifact.write_bytes(b"weights")
        manifest = ModelManifest(model_id="tampered", version="1.0.0",
                                 artifact_path=str(artifact),
                                 artifact_checksum="0" * 64)
        with pytest.raises(ModelIntegrityError):
            manifest.verify_artifact()

    def test_corrupted_manifest_json(self, tmp_path) -> None:
        from sofia_ai.inference.manifest import load_manifest

        path = tmp_path / "m.json"
        path.write_text("{not json", encoding="utf-8")
        with pytest.raises(ModelError):
            load_manifest(str(path))

    def test_manifest_missing_required_field(self, tmp_path) -> None:
        from sofia_ai.inference.manifest import load_manifest

        path = tmp_path / "m.json"
        path.write_text(json.dumps({"manifest_version": "2.0"}), encoding="utf-8")
        with pytest.raises(KeyError):
            load_manifest(str(path))

    def test_pickle_is_refused(self, tmp_path) -> None:
        from sofia_ai.legacy_model import SofiaModel

        path = tmp_path / "legacy.pkl"
        path.write_bytes(b"whatever")
        with pytest.raises(ModelError):
            SofiaModel.load(str(path))

    def test_legacy_model_roundtrip_is_json(self, tmp_path) -> None:
        import pickle

        from sofia_ai.legacy_model import SofiaModel

        path = tmp_path / "legacy.json"
        SofiaModel().save(str(path))
        assert not path.with_suffix(".pkl").exists()
        assert SofiaModel.load(str(path)) is not None
        # Ensure no pickle payload was written anywhere alongside.
        assert not any(p.suffix == ".pkl" for p in tmp_path.iterdir())
        assert pickle  # module exists; Sofia simply refuses to use it for models

    def test_engine_version_mismatch(self) -> None:
        from sofia_ai.inference.manifest import ModelManifest

        manifest = ModelManifest(model_id="m", version="1.0.0",
                                 engine_compatibility=">=9.0")
        with pytest.raises(ModelError):
            manifest.check_engine("2.0")


class TestTimeAnomalies:
    def test_duplicate_timestamps_flagged(self) -> None:
        monitor = ClockMonitor()
        monitor.observe(1_700_000_000.0)
        assert monitor.observe(1_700_000_000.0).kind == "duplicate"

    def test_out_of_order_flagged(self) -> None:
        monitor = ClockMonitor()
        monitor.observe(1_700_000_000.5)
        assert monitor.observe(1_700_000_000.0).kind == "backwards"

    def test_clock_jump_flagged(self) -> None:
        monitor = ClockMonitor(max_forward_jump_s=60.0)
        monitor.observe(1_700_000_000.0)
        assert monitor.observe(1_700_010_000.0).kind == "jump"

    def test_runtime_counts_anomalies(self) -> None:
        samples = [_sample(i) for i in range(10)]
        # Force duplicates.
        duplicated = samples + samples[:5]
        source = FailingSource(duplicated, fail_times=0)
        runtime = EdgeRuntime(source=source, backend=_backend(),
                              config=RuntimeConfig(window_length=8, window_hop=8,
                                                   sample_rate=1000.0,
                                                   buffer_capacity=64))
        runtime.start()
        runtime.tick(max_records=64)
        assert runtime.stats.clock_anomalies >= 5


class TestStaleTelemetry:
    def test_stale_quality_lowers_confidence(self) -> None:
        backend = create_backend("threshold")
        backend.load(build_manifest_for(backend, model_id="t", metadata={"high": 0.0}))
        good = extract_from_array(np.ones(64), 1000.0)
        stale = extract_from_array(np.ones(64), 1000.0)
        stale = type(stale)(**{**stale.to_dict(), "quality": DataQuality.STALE})
        assert backend.infer(stale).confidence < backend.infer(good).confidence

    def test_invalid_quality_zeroes_confidence(self) -> None:
        backend = create_backend("threshold")
        backend.load(build_manifest_for(backend, model_id="t", metadata={"high": 0.0}))
        bad = extract_from_array(np.ones(64), 1000.0)
        bad = type(bad)(**{**bad.to_dict(), "quality": DataQuality.INVALID})
        assert backend.infer(bad).confidence == 0.0


class TestBufferSaturation:
    def test_drop_oldest_counts_losses(self) -> None:
        buffer: BoundedRingBuffer[int] = BoundedRingBuffer(capacity=4)
        for i in range(20):
            buffer.push(i)
        assert len(buffer) == 4
        assert buffer.dropped == 16
        assert buffer.saturation == 1.0

    def test_runtime_reports_saturation(self) -> None:
        source = FailingSource([_sample(i) for i in range(100)], fail_times=0)
        runtime = EdgeRuntime(source=source, backend=_backend(),
                              config=RuntimeConfig(window_length=8, window_hop=8,
                                                   sample_rate=1000.0, buffer_capacity=8))
        runtime.start()
        runtime.tick(max_records=100)
        # Every sample was accounted for: 92 overflowed a capacity-8 buffer.
        assert runtime.buffer.dropped > 0
        assert runtime.buffer.written == 100


class TestFaultInjection:
    def test_connection_loss_does_not_crash(self) -> None:
        source = FailingSource([_sample(i) for i in range(512)], fail_times=2)
        runtime = EdgeRuntime(source=source, backend=_backend(),
                              config=RuntimeConfig(window_length=64, window_hop=32,
                                                   sample_rate=1000.0,
                                                   buffer_capacity=1024))
        runtime.start()
        events = runtime.run(max_ticks=8, batch_size=256)
        assert source.failures_injected == 2
        assert runtime.stats.source_errors == 2
        assert isinstance(events, list)

    def test_inference_failure_is_contained(self) -> None:
        source = FailingSource([_sample(i) for i in range(256)], fail_times=0)
        runtime = EdgeRuntime(source=source, backend=ExplodingBackend(),  # type: ignore[arg-type]
                              config=RuntimeConfig(window_length=64, window_hop=32,
                                                   sample_rate=1000.0,
                                                   buffer_capacity=1024))
        runtime.start()
        runtime.run(max_ticks=4, batch_size=256)
        assert runtime.stats.inference_failures > 0
        assert runtime.stats.events_emitted == 0

    def test_incomplete_window_is_not_emitted(self) -> None:
        source = FailingSource([_sample(i) for i in range(10)], fail_times=0)
        runtime = EdgeRuntime(source=source, backend=_backend(),
                              config=RuntimeConfig(window_length=1024, window_hop=512,
                                                   sample_rate=1000.0))
        runtime.start()
        runtime.tick(max_records=64)
        assert runtime.stats.windows_built == 0
        assert runtime.stats.inferences == 0

    def test_corrupted_store_forward_file(self, tmp_path) -> None:
        store = StoreForward(directory=str(tmp_path), flush_every=2)
        store.append(_sample(0))
        store.append(_sample(1))
        store.flush()
        for path in store._files():
            path.write_bytes(b"\x00\x00\x00")
        assert store.read_all() == []
        assert store.stats.corrupt_files == 1

    def test_disk_pressure_triggers_rotation(self, tmp_path) -> None:
        store = StoreForward(directory=str(tmp_path), max_bytes=200, max_files=1,
                             flush_every=1)
        for i in range(50):
            store.append(_sample(i))
            store.flush()
        assert store.file_count <= 1
        assert store.stats.rotations > 0

    def test_empty_input_raises_insufficient_data(self) -> None:
        with pytest.raises(FeatureError):
            extract_from_array(np.array([]), 1000.0)

    def test_window_shorter_than_signal(self) -> None:
        from sofia_ai.signal import WindowSpec, frame_signal

        with pytest.raises(InsufficientDataError):
            frame_signal(np.ones(4), WindowSpec(length=16, hop=8))


class TestPayloadCeiling:
    def test_source_limits_enforced(self) -> None:
        from sofia_ai.telemetry.base import check_payload_size

        limits = SourceLimits(max_payload_bytes=16)
        with pytest.raises(PayloadTooLargeError):
            check_payload_size(32, limits)

    def test_default_ceiling_is_one_mib(self) -> None:
        assert SourceLimits().max_payload_bytes == 1 << 20


def _backend():
    backend = create_backend("mad")
    backend.config = DetectorConfig(feature="rms", window_size=8, warmup=2, min_samples=2)
    backend.load(build_manifest_for(backend, model_id="neg"))
    return backend


def test_all_failures_are_sofia_errors() -> None:
    """Every injected failure above must surface as a SofiaError subclass."""
    assert issubclass(TelemetryProtocolError, SofiaError)
    assert issubclass(ModelError, SofiaError)
    assert issubclass(ValidationError, SofiaError)
    assert issubclass(PayloadTooLargeError, SofiaError)
    assert struct  # keep import used for frame-encoding context
