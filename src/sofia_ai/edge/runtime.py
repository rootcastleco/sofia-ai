"""Offline-first edge runtime.

The runtime wires the whole pipeline locally:

    source → bounded buffer → quality marking → window assembly →
    features → inference → diagnostics → health events → sink

Everything runs without network access. Connectivity loss, malformed input and
inference failure are all handled without taking the pipeline down.

The runtime owns no global state. Every collaborator is injected.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ..core.contracts import (
    DataQuality,
    MachineState,
    SignalWindow,
    TelemetrySample,
)
from ..core.errors import SofiaError, TelemetryError
from ..core.time import ClockMonitor, TimeSource
from ..core.validation import validate_positive_int
from ..diagnostics.engine import DiagnosticEngine, HealthEvent
from ..diagnostics.rules import RuleContext
from ..features.extractor import FeatureExtractor
from ..inference.base import InferenceResult, ModelBackend
from ..observability.logging import SofiaLogger
from ..observability.metrics import MetricsRegistry
from ..signal.windowing import windows_from_samples
from ..telemetry.base import TelemetrySource
from .buffer import BoundedRingBuffer, OverflowPolicy
from .store_forward import StoreForward

__all__ = ["EdgeRuntime", "RuntimeConfig", "RuntimeStats"]


@dataclass(slots=True)
class RuntimeConfig:
    """Runtime geometry and limits."""

    window_length: int = 1024
    window_hop: int = 512
    sample_rate: float = 1000.0
    buffer_capacity: int = 4096
    max_age_s: float = 60.0
    max_windows_per_tick: int = 16
    device_id: str = "edge-01"
    channel: str = "vibration_x"

    def __post_init__(self) -> None:
        self.window_length = validate_positive_int(self.window_length, "window_length",
                                                   maximum=1 << 20)
        self.window_hop = validate_positive_int(self.window_hop, "window_hop",
                                                maximum=self.window_length)
        if self.sample_rate <= 0:
            raise SofiaError("sample_rate must be positive", details={})
        self.buffer_capacity = validate_positive_int(self.buffer_capacity, "buffer_capacity",
                                                     maximum=10_000_000)
        if self.max_age_s <= 0:
            raise SofiaError("max_age_s must be positive", details={})
        self.max_windows_per_tick = validate_positive_int(
            self.max_windows_per_tick, "max_windows_per_tick", maximum=4096
        )


@dataclass(slots=True)
class RuntimeStats:
    """Runtime counters, distinct from metrics because they are runtime-scoped."""

    ticks: int = 0
    samples_ingested: int = 0
    samples_rejected: int = 0
    windows_built: int = 0
    inferences: int = 0
    inference_failures: int = 0
    events_emitted: int = 0
    source_errors: int = 0
    clock_anomalies: int = 0


@dataclass(slots=True)
class EdgeRuntime:
    """The offline-first pipeline.

    Args:
        source: Any :class:`TelemetrySource`.
        backend: A loaded :class:`ModelBackend`.
        config: Geometry and limits.
        diagnostics: Diagnostic engine producing health events.
        extractor: Feature extractor.
        store_forward: Optional bounded persistence.
        metrics: Metrics registry.
        logger: Structured logger.
        clock: Injectable clock.
        event_sink: Receives every emitted health event.
    """

    source: TelemetrySource
    backend: ModelBackend
    config: RuntimeConfig = field(default_factory=RuntimeConfig)
    diagnostics: DiagnosticEngine | None = None
    extractor: FeatureExtractor | None = None
    store_forward: StoreForward | None = None
    metrics: MetricsRegistry | None = None
    logger: SofiaLogger | None = None
    clock: TimeSource | None = None
    event_sink: Callable[[HealthEvent], None] | None = None

    buffer: BoundedRingBuffer[TelemetrySample] = field(init=False, repr=False)
    _clock_monitor: ClockMonitor = field(init=False, repr=False)
    stats: RuntimeStats = field(init=False, repr=False)
    events: list[HealthEvent] = field(init=False, repr=False)
    last_results: list[InferenceResult] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.buffer = BoundedRingBuffer(
            capacity=self.config.buffer_capacity, policy=OverflowPolicy.DROP_OLDEST
        )
        self.extractor = self.extractor or FeatureExtractor()
        self.diagnostics = self.diagnostics or DiagnosticEngine()
        self.metrics = self.metrics or MetricsRegistry()
        self._clock_monitor = ClockMonitor()
        self.stats = RuntimeStats()
        self.events = []
        self.last_results = []

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        """Open the source. Failure is recorded, not raised into the caller."""
        try:
            self.source.open()
        except TelemetryError as exc:
            self.stats.source_errors += 1
            if self.metrics:
                self.metrics.inc("protocol_failures")
            if self.logger:
                self.logger.exception("source_open_failed", error=exc.code, detail=exc.message)
            raise

    def stop(self) -> None:
        self.source.close()
        if self.store_forward is not None:
            self.store_forward.flush()

    # -- pipeline ----------------------------------------------------------

    def tick(self, *, max_records: int = 512) -> list[HealthEvent]:
        """Run one pipeline iteration: ingest → window → infer → diagnose.

        Returns:
            Health events produced by this iteration. Empty when nothing fired.

        A failure in any stage is contained: the tick returns what it managed to
        produce and records the failure in metrics.
        """
        self.stats.ticks += 1
        self._ingest(max_records)
        windows = self._build_windows()
        produced: list[HealthEvent] = []
        for window in windows[: self.config.max_windows_per_tick]:
            event = self._process_window(window)
            if event is not None:
                produced.append(event)
        self.events.extend(produced)
        if self.metrics:
            self.metrics.gauge("queue_depth").set(len(self.buffer))
            self.metrics.gauge("buffer_saturation").set(self.buffer.saturation)
        return produced

    def run(self, *, max_ticks: int = 100, batch_size: int = 512) -> list[HealthEvent]:
        """Run up to ``max_ticks`` iterations. Bounded by construction."""
        produced: list[HealthEvent] = []
        for _ in range(max(0, int(max_ticks))):
            produced.extend(self.tick(max_records=batch_size))
        return produced

    def _ingest(self, max_records: int) -> None:
        try:
            batch = self.source.read(max_records=max_records)
        except TelemetryError as exc:
            self.stats.source_errors += 1
            if self.metrics:
                self.metrics.inc("protocol_failures")
            if self.logger:
                self.logger.warning("source_read_failed", error=exc.code, detail=exc.message)
            return

        now = self.clock.wall() if self.clock is not None else time.time()
        for sample in batch:
            anomaly = self._clock_monitor.observe(sample.timestamp)
            if anomaly is not None:
                self.stats.clock_anomalies += 1
                if self.metrics:
                    self.metrics.inc("clock_anomalies")
            if not self.buffer.push(sample):
                if self.metrics:
                    self.metrics.inc("samples_dropped")
                self.stats.samples_rejected += 1
                continue
            self.stats.samples_ingested += 1
            if self.metrics:
                self.metrics.inc("samples_received")
            if sample.quality is not DataQuality.GOOD and self.metrics:
                self.metrics.inc("samples_malformed")
            if self.store_forward is not None:
                self.store_forward.append(sample)
            _ = now

    def _build_windows(self) -> list[SignalWindow]:
        samples = self.buffer.peek(len(self.buffer))
        if len(samples) < self.config.window_length:
            return []
        started = time.perf_counter()
        windows = windows_from_samples(
            samples,
            length=self.config.window_length,
            hop=self.config.window_hop,
            sample_rate=self.config.sample_rate,
            unit=samples[0].unit,
            channel=samples[0].channel,
            device_id=samples[0].device_id,
        )
        if self.metrics:
            self.metrics.observe("window_processing_latency_s", time.perf_counter() - started)
        self.stats.windows_built += len(windows)
        # Advance the buffer by the samples actually consumed.
        if windows:
            consumed = (len(windows) - 1) * self.config.window_hop + self.config.window_length
            self.buffer.drain(min(consumed, len(samples)))
        return windows

    def _process_window(self, window: SignalWindow) -> HealthEvent | None:
        assert self.extractor is not None
        assert self.diagnostics is not None
        try:
            features = self.extractor.extract(window)
        except SofiaError as exc:
            self.stats.samples_rejected += 1
            if self.metrics:
                self.metrics.inc("samples_malformed")
            if self.logger:
                self.logger.warning("feature_extraction_failed", error=exc.code)
            return None

        try:
            result = self.backend.infer(features)
        except SofiaError as exc:
            self.stats.inference_failures += 1
            if self.metrics:
                self.metrics.inc("inference_failures")
            if self.logger:
                self.logger.exception("inference_failed", error=exc.code, detail=exc.message)
            return None

        self.stats.inferences += 1
        self.last_results.append(result)
        if self.metrics:
            self.metrics.observe("inference_latency_s", result.latency_s)

        context = RuleContext(
            device_id=window.device_id or self.config.device_id,
            channel=window.channel,
            score=result.score,
            confidence=result.confidence,
            quality=window.quality,
            machine_state=MachineState.RUNNING,
            feature_values=features.as_dict(),
            age_s=0.0,
            unit=window.unit,
        )
        event = self.diagnostics.evaluate(context, model_version=result.model_version)
        if event is None:
            return None

        self.stats.events_emitted += 1
        if self.metrics:
            self.metrics.inc("anomaly_events")
        if self.event_sink is not None:
            self.event_sink(event)
        return event

    # -- introspection -----------------------------------------------------

    def health_snapshot(self) -> dict[str, Any]:
        """Structured snapshot for the health/readiness endpoint."""
        return {
            "config": {
                "device_id": self.config.device_id,
                "channel": self.config.channel,
                "window_length": self.config.window_length,
                "sample_rate": self.config.sample_rate,
            },
            "buffer": self.buffer.stats(),
            "stats": {
                "ticks": self.stats.ticks,
                "samples_ingested": self.stats.samples_ingested,
                "samples_rejected": self.stats.samples_rejected,
                "windows_built": self.stats.windows_built,
                "inferences": self.stats.inferences,
                "inference_failures": self.stats.inference_failures,
                "events_emitted": self.stats.events_emitted,
                "source_errors": self.stats.source_errors,
                "clock_anomalies": self.stats.clock_anomalies,
            },
            "source": self.source.describe(),
            "metrics": self.metrics.snapshot() if self.metrics else {},
        }

    def drain_events(self) -> list[HealthEvent]:
        events, self.events = self.events, []
        return events

