"""Unit tests for telemetry adapters and the registry (SOFIA-TLM-*)."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from sofia_ai.core.contracts import TelemetrySample
from sofia_ai.core.errors import (
    PayloadTooLargeError,
    PluginError,
    TelemetryError,
    TelemetryProtocolError,
    TransportClosedError,
    ValidationError,
)
from sofia_ai.core.quality import DataQuality
from sofia_ai.telemetry import (
    CsvSource,
    InMemorySource,
    JsonlSource,
    MultiChannelSyntheticSource,
    ReplayMode,
    ReplaySource,
    SourceLimits,
    SyntheticMachineProfile,
    SyntheticVibrationSource,
    TelemetrySource,
    inject_faults,
    register_telemetry_source,
    rotating_machine_signal,
    telemetry_registry,
)


def _sample(index: int, value: float = 1.0, channel: str = "c") -> TelemetrySample:
    return TelemetrySample(
        timestamp=1_700_000_000.0 + index * 0.001,
        device_id="d", channel=channel, value=value, unit="g",
        source="test", sequence_number=index,
    )


class TestCsvSource:
    def test_reads_long_format(self, tmp_path) -> None:
        path = tmp_path / "t.csv"
        path.write_text(
            "timestamp,device_id,channel,value,unit\n"
            "1700000000.0,d,c,1.5,g\n"
            "1700000000.1,d,c,2.5,g\n",
            encoding="utf-8",
        )
        with CsvSource(str(path)) as source:
            samples = source.read()
        assert len(samples) == 2
        assert samples[0].value == 1.5
        assert samples[0].unit == "g"

    def test_iso8601_timestamps(self, tmp_path) -> None:
        path = tmp_path / "t.csv"
        path.write_text("timestamp,device_id,channel,value,unit\n"
                        "2023-11-14T22:13:20.000Z,d,c,1.0,g\n", encoding="utf-8")
        with CsvSource(str(path)) as source:
            samples = source.read()
        assert samples[0].timestamp == pytest.approx(1_700_000_000.0, abs=1.0)

    def test_default_device_and_unit(self, tmp_path) -> None:
        """A blank device_id cell falls back to the configured default."""
        path = tmp_path / "t.csv"
        path.write_text("timestamp,device_id,channel,value\n1700000000.0,,c,1.0\n",
                        encoding="utf-8")
        with CsvSource(str(path), device_id="def", unit="mm/s") as source:
            samples = source.read()
        assert samples[0].device_id == "def"
        assert samples[0].unit == "mm/s"

    def test_device_id_column_is_required(self, tmp_path) -> None:
        path = tmp_path / "t.csv"
        path.write_text("timestamp,channel,value\n1700000000.0,c,1.0\n", encoding="utf-8")
        with pytest.raises(TelemetryProtocolError):
            CsvSource(str(path), device_id="def").open()

    def test_missing_column_is_protocol_error(self, tmp_path) -> None:
        path = tmp_path / "t.csv"
        path.write_text("timestamp,value\n1700000000.0,1.0\n", encoding="utf-8")
        with pytest.raises(TelemetryProtocolError):
            CsvSource(str(path)).open()

    def test_unknown_unit_rejected(self, tmp_path) -> None:
        path = tmp_path / "t.csv"
        path.write_text("timestamp,device_id,channel,value,unit\n"
                        "1700000000.0,d,c,1.0,furlongs\n", encoding="utf-8")
        with CsvSource(str(path)) as source, pytest.raises(TelemetryProtocolError):
            source.read()

    def test_non_numeric_value(self, tmp_path) -> None:
        path = tmp_path / "t.csv"
        path.write_text("timestamp,device_id,channel,value,unit\n"
                        "1700000000.0,d,c,notanumber,g\n", encoding="utf-8")
        with CsvSource(str(path)) as source, pytest.raises(TelemetryProtocolError):
            source.read()

    def test_lenient_mode_skips_bad_rows(self, tmp_path) -> None:
        path = tmp_path / "t.csv"
        path.write_text(
            "timestamp,device_id,channel,value,unit\n"
            "1700000000.0,d,c,1.0,g\n"
            "1700000000.1,d,c,bad,g\n",
            encoding="utf-8",
        )
        with CsvSource(str(path), strict=False) as source:
            samples = source.read()
        assert len(samples) == 1
        assert source.stats.records_rejected == 1

    def test_quality_column(self, tmp_path) -> None:
        path = tmp_path / "t.csv"
        path.write_text("timestamp,device_id,channel,value,unit,quality\n"
                        "1700000000.0,d,c,1.0,g,STALE\n", encoding="utf-8")
        with CsvSource(str(path)) as source:
            assert source.read()[0].quality is DataQuality.STALE

    def test_overlong_line_rejected(self, tmp_path) -> None:
        path = tmp_path / "t.csv"
        path.write_text("timestamp,device_id,channel,value,unit\n"
                        f"1700000000.0,d,c,1.0,{'a' * 3000000}\n", encoding="utf-8")
        with pytest.raises(TelemetryProtocolError):
            CsvSource(str(path)).open()

    def test_read_before_open(self, tmp_path) -> None:
        path = tmp_path / "t.csv"
        path.write_text("timestamp,device_id,channel,value,unit\n", encoding="utf-8")
        with pytest.raises(TransportClosedError):
            CsvSource(str(path)).read()

    def test_missing_file(self, tmp_path) -> None:
        with pytest.raises(TelemetryProtocolError):
            CsvSource(str(tmp_path / "nope.csv")).open()


class TestJsonlSource:
    def test_roundtrip(self, tmp_path) -> None:
        path = tmp_path / "t.jsonl"
        lines = [json.dumps(_sample(i, float(i)).to_dict(), sort_keys=True)
                 for i in range(4)]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        with JsonlSource(str(path)) as source:
            samples = source.read()
        assert len(samples) == 4
        assert samples[0].value == 0.0

    def test_malformed_json(self, tmp_path) -> None:
        path = tmp_path / "t.jsonl"
        path.write_text("{not json}\n", encoding="utf-8")
        with pytest.raises(TelemetryProtocolError):
            JsonlSource(str(path)).open()

    def test_non_object_line(self, tmp_path) -> None:
        path = tmp_path / "t.jsonl"
        path.write_text("[1,2]\n", encoding="utf-8")
        with pytest.raises(TelemetryProtocolError):
            JsonlSource(str(path)).open()

    def test_blank_lines_skipped(self, tmp_path) -> None:
        path = tmp_path / "t.jsonl"
        line = json.dumps(_sample(0).to_dict(), sort_keys=True)
        path.write_text(f"{line}\n\n{line}\n", encoding="utf-8")
        with JsonlSource(str(path)) as source:
            assert len(source.read()) == 2


class TestInMemorySource:
    def test_batches(self) -> None:
        source = InMemorySource([_sample(i) for i in range(5)], batch_size=2)
        source.open()
        assert len(source.read()) == 2
        assert source.remaining == 3
        assert len(source.read()) == 2
        assert len(source.read()) == 1
        assert source.read() == []

    def test_stream_is_bounded(self) -> None:
        source = InMemorySource([_sample(i) for i in range(10)], batch_size=3)
        source.open()
        assert len(list(source.stream(batch_size=3, max_batches=2))) == 6


class TestReplaySource:
    def test_requires_samples(self) -> None:
        with pytest.raises(ValidationError):
            ReplaySource(())

    def test_single_pass(self) -> None:
        samples = [_sample(i) for i in range(5)]
        source = ReplaySource.from_samples(samples, batch_size=2)
        source.open()
        assert len(source.read()) == 2
        assert len(source.read()) == 2
        assert len(source.read()) == 1
        assert source.exhausted

    def test_loops(self) -> None:
        samples = [_sample(i) for i in range(4)]
        source = ReplaySource.from_samples(samples, loops=2, batch_size=4)
        source.open()
        assert len(source.read()) == 4
        assert len(source.read()) == 4
        assert math.isclose(source.progress, 1.0)

    def test_max_total_records_is_a_ceiling(self) -> None:
        samples = [_sample(i) for i in range(10)]
        source = ReplaySource.from_samples(samples, loops=100, batch_size=100,
                                           max_total_records=15)
        source.open()
        assert len(source.read()) == 15
        assert source.read() == []

    def test_mode_coercion(self) -> None:
        source = ReplaySource.from_samples([_sample(0)], mode="real_time")
        assert source.mode is ReplayMode.REAL_TIME


class TestSynthetic:
    def test_signal_shape(self) -> None:
        signal = rotating_machine_signal(128, 1000.0, shaft_hz=25.0)
        assert signal.shape == (128,)

    def test_signal_is_deterministic_without_rng(self) -> None:
        a = rotating_machine_signal(64, 1000.0)
        b = rotating_machine_signal(64, 1000.0)
        assert a.tolist() == b.tolist()

    def test_noise_uses_injected_rng(self) -> None:
        signal = rotating_machine_signal(64, 1000.0, noise_std=0.5,
                                         rng=np.random.default_rng(0))
        assert signal.std() > 0.0

    def test_rejects_shaft_above_nyquist(self) -> None:
        with pytest.raises(ValidationError):
            rotating_machine_signal(64, 100.0, shaft_hz=80.0)

    def test_inject_faults(self) -> None:
        signal = rotating_machine_signal(100, 1000.0)
        degraded = inject_faults(signal, start_fraction=0.5, amplitude_gain=4.0)
        assert degraded[:50].tolist() == signal[:50].tolist()
        assert abs(degraded[-1]) > abs(signal[-1])

    def test_source_emits_samples(self) -> None:
        source = SyntheticVibrationSource(SyntheticMachineProfile(seed=1), n_samples=64,
                                          batch_size=32)
        source.open()
        total = 0
        while True:
            batch = source.read()
            if not batch:
                break
            total += len(batch)
        assert total == 64

    def test_values_property(self) -> None:
        source = SyntheticVibrationSource(n_samples=32)
        assert source.values().size == 32

    def test_rejects_wrong_harmonic_lengths(self) -> None:
        with pytest.raises(ValidationError):
            rotating_machine_signal(64, 1000.0, harmonics=(1.0, 2.0), amplitudes=(1.0,))

    def test_rejects_negative_bearing_start(self) -> None:
        with pytest.raises(ValidationError):
            inject_faults(np.zeros(16), start_fraction=-0.1)

    def test_rejects_start_fraction_above_one(self) -> None:
        with pytest.raises(ValidationError):
            inject_faults(np.zeros(16), start_fraction=1.5)

    def test_inject_faults_with_offset(self) -> None:
        signal = rotating_machine_signal(100, 1000.0, noise_std=0.0)
        degraded = inject_faults(signal, start_fraction=0.5, offset=2.0)
        assert degraded[0].item() == signal[0].item()
        assert degraded[-1].item() == pytest.approx(
            signal[-1].item() * 3.0 + 2.0)

    def test_bearing_component(self) -> None:
        signal = rotating_machine_signal(512, 1000.0, shaft_hz=20.0,
                                         bearing_hz=100.0, bearing_amplitude=0.4)
        assert signal.ndim == 1
        assert np.all(np.isfinite(signal))

    def test_multi_channel_source(self) -> None:
        source = MultiChannelSyntheticSource(n_samples=256, degrade_fraction=0.5)
        source.open()
        samples = source.read()
        channels = {s.channel for s in samples}
        assert len(channels) == 3
        assert all(s.timestamp >= source.start_time for s in samples)

    def test_multi_channel_frames(self) -> None:
        source = MultiChannelSyntheticSource(n_samples=128)
        source.open()
        frames = list(source.frames())
        assert frames
        for group in frames:
            assert len({s.timestamp for s in group}) == 1

    def test_multi_channel_custom_thermal_profile(self) -> None:
        profile = SyntheticMachineProfile(channel="temperature", unit="C")
        source = MultiChannelSyntheticSource(
            profiles=(profile,), n_samples=64, degrade_fraction=0.5)
        source.open()
        samples = source.read()
        assert len(samples) == 64
        assert all(s.unit == "C" for s in samples)


class TestLimits:
    def test_rejects_zero_payload(self) -> None:
        with pytest.raises(TelemetryError):
            SourceLimits(max_payload_bytes=0)

    def test_payload_check(self) -> None:
        limits = SourceLimits(max_payload_bytes=10)
        from sofia_ai.telemetry.base import check_payload_size

        assert check_payload_size(5, limits) == 5
        with pytest.raises(PayloadTooLargeError):
            check_payload_size(11, limits)
        with pytest.raises(TelemetryProtocolError):
            check_payload_size(-1, limits)


class TestRegistry:
    def test_builtin_sources(self) -> None:
        for name in ("csv", "jsonl", "memory", "replay", "synthetic", "mqtt",
                     "modbus", "serial"):
            assert name in telemetry_registry

    def test_create_csv(self, tmp_path) -> None:
        path = tmp_path / "t.csv"
        path.write_text("timestamp,device_id,channel,value,unit\n", encoding="utf-8")
        source = telemetry_registry.create("csv", path=str(path))
        assert isinstance(source, TelemetrySource)
        source.open()
        assert source.read() == []

    def test_unknown_source(self) -> None:
        with pytest.raises(PluginError):
            telemetry_registry.create("carrier-pigeon")

    def test_custom_registration(self) -> None:
        register_telemetry_source("unit_test_source",
                                  InMemorySource, overwrite=True)
        assert "unit_test_source" in telemetry_registry

    def test_registry_helpers(self) -> None:
        assert len(telemetry_registry) > 0
        assert "memory" in telemetry_registry.names()
        assert "memory" in telemetry_registry.as_mapping()

    def test_registry_factory_failure_wraps(self) -> None:
        def broken(**kwargs: object) -> object:
            raise RuntimeError("boom")

        register_telemetry_source("unit_broken", broken, overwrite=True)
        with pytest.raises(PluginError, match="failed"):
            telemetry_registry.create("unit_broken")

    def test_registry_unregister(self) -> None:
        register_telemetry_source("unit_ephemeral", InMemorySource, overwrite=True)
        assert "unit_ephemeral" in telemetry_registry
        telemetry_registry.unregister("unit_ephemeral")
        assert "unit_ephemeral" not in telemetry_registry
        with pytest.raises(PluginError):
            telemetry_registry.create("unit_ephemeral")

    def test_registry_prefix_enforced(self) -> None:
        from sofia_ai.telemetry.registry import Registry

        strict = Registry("strict", allowed_prefix="x_")
        with pytest.raises(PluginError, match="must start with"):
            strict.register("nope", InMemorySource)

    def test_describe(self, tmp_path) -> None:
        path = tmp_path / "t.csv"
        path.write_text("timestamp,device_id,channel,value,unit\n", encoding="utf-8")
        source = CsvSource(str(path))
        source.open()
        payload = source.describe()
        assert payload["type"] == "CsvSource"
        assert "limits" in payload


