"""Integration tests: complete pipelines running offline."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from sofia_ai.core.contracts import MachineState
from sofia_ai.diagnostics import DiagnosticEngine, RuleContext, compute_health_score
from sofia_ai.edge import EdgeRuntime, HealthMonitor, HealthStatus, RuntimeConfig
from sofia_ai.features import extract_from_array
from sofia_ai.inference import build_manifest_for, create_backend
from sofia_ai.inference.detectors import DetectorConfig
from sofia_ai.observability import MetricsRegistry, get_logger
from sofia_ai.signal import welch_psd, windows_from_samples
from sofia_ai.telemetry import (
    CsvSource,
    JsonlSource,
    ReplayMode,
    ReplaySource,
    SyntheticMachineProfile,
    SyntheticVibrationSource,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
VIBRATION_CSV = REPO_ROOT / "examples" / "data" / "vibration.csv"
TELEMETRY_JSONL = REPO_ROOT / "examples" / "data" / "telemetry.jsonl"
EDGE_JSONL = REPO_ROOT / "examples" / "data" / "edge_offline.jsonl"


def _detector(name: str = "mad", **config) -> object:
    backend = create_backend(name)
    if hasattr(backend, "config"):
        backend.config = DetectorConfig(feature="rms", window_size=32, warmup=8,
                                        min_samples=8, **config)
    backend.load(build_manifest_for(backend, model_id=f"it-{name}"))
    return backend


class TestVibrationPipeline:
    def test_signal_to_health_event(self) -> None:
        with CsvSource(str(VIBRATION_CSV)) as source:
            samples = []
            while True:
                batch = source.read(max_records=4096)
                if not batch:
                    break
                samples.extend(batch)

        windows = windows_from_samples(
            samples, length=512, hop=256, sample_rate=1000.0, unit="g",
            channel="vibration_x", device_id="sim-pump-01")
        assert windows, "expected at least one window"

        detector = _detector()
        engine = DiagnosticEngine()
        events = []
        for window in windows:
            features = extract_from_array(window.values, window.sample_rate,
                                          channel=window.channel, shaft_hz=25.0)
            result = detector.infer(features)
            event = engine.evaluate(RuleContext(
                device_id=window.device_id, channel=window.channel, score=result.score,
                confidence=result.confidence, machine_state=MachineState.RUNNING,
                feature_values=features.as_dict(), unit=window.unit,
            ), model_version=result.model_version)
            if event is not None:
                events.append(event)

        assert events, "expected at least one health event from degraded data"
        assert all(e.confidence > 0.0 for e in events)
        score = compute_health_score(events)
        assert score.score < 100.0

    def test_spectrum_identifies_shaft_frequency(self) -> None:
        with CsvSource(str(VIBRATION_CSV)) as source:
            samples = source.read(max_records=2048)
        values = np.asarray([s.value for s in samples], dtype=np.float64)
        spectrum = welch_psd(values, 1000.0)
        dominant = float(spectrum.frequencies[int(np.argmax(spectrum.values))])
        assert 20.0 <= dominant <= 30.0, dominant


class TestTelemetryPipeline:
    def test_jsonl_to_events(self) -> None:
        with JsonlSource(str(TELEMETRY_JSONL)) as source:
            recorded = []
            while True:
                batch = source.read(max_records=20_000)
                if not batch:
                    break
                recorded.extend(batch)
        channels = {s.channel for s in recorded}
        assert channels == {"vibration_x", "temperature", "rpm"}

        vibration = [s for s in recorded if s.channel == "vibration_x"]
        replay = ReplaySource.from_samples(vibration, mode=ReplayMode.AS_FAST_AS_POSSIBLE,
                                           batch_size=4096)
        runtime = EdgeRuntime(
            source=replay, backend=_detector(),
            config=RuntimeConfig(window_length=512, window_hop=256, sample_rate=1000.0,
                                 buffer_capacity=32768, max_windows_per_tick=32),
            metrics=MetricsRegistry(), logger=get_logger("it"),
        )
        runtime.start()
        events = runtime.run(max_ticks=32, batch_size=4096)
        snapshot = runtime.health_snapshot()
        runtime.stop()

        assert events
        assert snapshot["stats"]["inference_failures"] == 0
        payload = json.dumps(events[0].to_dict(), sort_keys=True, allow_nan=False)
        assert "event_id" in payload

    def test_rpm_channel_is_constant(self) -> None:
        with JsonlSource(str(TELEMETRY_JSONL)) as source:
            recorded = source.read(max_records=20_000)
        rpm = [s.value for s in recorded if s.channel == "rpm"]
        assert rpm
        assert all(abs(v - 1500.0) < 1e-9 for v in rpm)


class TestOfflineOperation:
    def test_no_network_is_required(self, monkeypatch) -> None:
        """The pipeline must complete with socket creation forbidden."""
        import socket

        def forbidden(*_args, **_kwargs):
            raise AssertionError("network access attempted")

        monkeypatch.setattr(socket, "socket", forbidden)
        monkeypatch.setattr(socket, "create_connection", forbidden)

        source = SyntheticVibrationSource(SyntheticMachineProfile(seed=21),
                                          n_samples=4096, degrade=True, batch_size=4096)
        runtime = EdgeRuntime(
            source=source, backend=_detector(),
            config=RuntimeConfig(window_length=512, window_hop=256, sample_rate=1000.0,
                                 buffer_capacity=16384, max_windows_per_tick=16),
        )
        runtime.start()
        events = runtime.run(max_ticks=30, batch_size=4096)
        runtime.stop()
        assert events, "offline pipeline produced no events"

    def test_edge_file_pipeline_runs_offline(self) -> None:
        with JsonlSource(str(EDGE_JSONL)) as source:
            runtime = EdgeRuntime(
                source=source, backend=_detector(),
                config=RuntimeConfig(window_length=512, window_hop=256,
                                     sample_rate=1000.0, buffer_capacity=8192),
            )
            runtime.start()
            events = runtime.run(max_ticks=12, batch_size=4096)
            snapshot = runtime.health_snapshot()
            runtime.stop()
        assert snapshot["stats"]["samples_ingested"] > 0
        assert isinstance(events, list)

    def test_health_monitor_reports_ready(self) -> None:
        with JsonlSource(str(EDGE_JSONL)) as source:
            runtime = EdgeRuntime(
                source=source, backend=_detector(),
                config=RuntimeConfig(window_length=512, window_hop=256,
                                     sample_rate=1000.0, buffer_capacity=8192),
            )
            runtime.start()
            runtime.run(max_ticks=4, batch_size=4096)
            report = HealthMonitor(runtime.health_snapshot).check()
            runtime.stop()
        assert report.status in (HealthStatus.READY, HealthStatus.DEGRADED)


class TestDiagnosticsToCopilot:
    def test_evidence_flows_into_copilot(self) -> None:
        from sofia_ai.copilot import CopilotRequest, EvidenceCopilot

        engine = DiagnosticEngine()
        event = engine.evaluate(RuleContext(device_id="pump-01", channel="vibration_x",
                                            score=12.0, confidence=0.9))
        assert event is not None
        copilot = EvidenceCopilot()
        response = copilot.ask(CopilotRequest(
            question="Explain this event", device_id=event.device_id,
            evidence=tuple(event.evidence.to_list()),
            events=(event.to_dict(),),
        ))
        assert response.grounded
        assert "anomaly_score" in response.text
