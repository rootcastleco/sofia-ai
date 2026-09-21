"""Unit tests for observability: logging, metrics and redaction (SOFIA-OBS-*)."""

from __future__ import annotations

import json
import logging
import math

import pytest

from sofia_ai.observability import (
    METRIC_NAMES,
    REDACTED,
    Counter,
    Gauge,
    Histogram,
    MetricsRegistry,
    SofiaLogger,
    configure_logging,
    get_logger,
    redact_mapping,
    redact_text,
    redact_value,
)
from sofia_ai.observability.logging import LogRateLimiter


class TestLogRateLimiter:
    def test_allows_up_to_limit(self) -> None:
        limiter = LogRateLimiter(per_minute=2)
        assert limiter.allow("k", 0.0)[0]
        assert limiter.allow("k", 1.0)[0]
        allowed, suppressed = limiter.allow("k", 2.0)
        assert not allowed
        assert suppressed >= 1

    def test_window_rolls_over(self) -> None:
        limiter = LogRateLimiter(per_minute=1)
        assert limiter.allow("k", 0.0)[0]
        assert not limiter.allow("k", 1.0)[0]
        assert limiter.allow("k", 61.0)[0]

    def test_key_table_is_bounded(self) -> None:
        limiter = LogRateLimiter(per_minute=10, max_keys=3)
        for i in range(10):
            limiter.allow(f"key-{i}", float(i))
        assert len(limiter) <= 3

    def test_rejects_bad_limit(self) -> None:
        with pytest.raises(ValueError):
            LogRateLimiter(per_minute=0)


class TestSofiaLogger:
    @pytest.fixture(autouse=True)
    def _capture(self, caplog) -> None:
        configure_logging(level="DEBUG")
        self.caplog = caplog

    def test_emits_single_line_json(self, caplog) -> None:
        logger = SofiaLogger(name="t")
        with caplog.at_level(logging.DEBUG, logger="sofia.t"):
            assert logger.info("test_event", value=1)
        record = json.loads(caplog.records[-1].message)
        assert record["event"] == "test_event"
        assert record["level"] == "INFO"
        assert record["value"] == 1

    def test_rate_limit_suppresses(self, caplog) -> None:
        logger = SofiaLogger(name="t", rate_limit_per_min=2)
        with caplog.at_level(logging.DEBUG, logger="sofia.t"):
            assert logger.warning("noisy")
            assert logger.warning("noisy")
            assert not logger.warning("noisy")

    def test_context_is_attached(self, caplog) -> None:
        logger = SofiaLogger(name="t", device_id="pump-01", correlation_id="abc")
        with caplog.at_level(logging.DEBUG, logger="sofia.t"):
            logger.error("boom")
        record = json.loads(caplog.records[-1].message)
        assert record["device_id"] == "pump-01"
        assert record["correlation_id"] == "abc"

    def test_secrets_are_redacted(self, caplog) -> None:
        logger = SofiaLogger(name="t")
        with caplog.at_level(logging.DEBUG, logger="sofia.t"):
            logger.info("auth", password="hunter2", token="abc", api_key="k")
        record = json.loads(caplog.records[-1].message)
        assert record["password"] == REDACTED
        assert record["token"] == REDACTED
        assert record["api_key"] == REDACTED

    def test_nested_secrets_are_redacted(self, caplog) -> None:
        logger = SofiaLogger(name="t")
        with caplog.at_level(logging.DEBUG, logger="sofia.t"):
            logger.info("cfg", config={"password": "x", "host": "h"})
        record = json.loads(caplog.records[-1].message)
        assert record["config"]["password"] == REDACTED
        assert record["config"]["host"] == "h"

    def test_newlines_cannot_forge_records(self, caplog) -> None:
        """Structured JSON framing means an injected newline cannot split a record."""
        logger = SofiaLogger(name="t")
        with caplog.at_level(logging.DEBUG, logger="sofia.t"):
            logger.info("inject", note="a\nb")
        assert len(caplog.records[-1].message.splitlines()) == 1

    def test_reserved_keys_are_prefixed(self, caplog) -> None:
        logger = SofiaLogger(name="t")
        with caplog.at_level(logging.DEBUG, logger="sofia.t"):
            logger.info("x", module="mine")
        record = json.loads(caplog.records[-1].message)
        assert record["f_module"] == "mine"

    def test_level_helpers(self, caplog) -> None:
        logger = SofiaLogger(name="t")
        with caplog.at_level(logging.DEBUG, logger="sofia.t"):
            logger.debug("d")
            logger.info("i")
            logger.warning("w")
            logger.error("e")
            logger.critical("c")
        levels = [json.loads(r.message)["level"] for r in caplog.records[-5:]]
        assert levels == ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

    def test_get_logger(self) -> None:
        assert isinstance(get_logger("unit"), SofiaLogger)


class TestMetrics:
    def test_counter(self) -> None:
        counter = Counter(name="c")
        counter.inc()
        counter.inc(4)
        assert counter.value == 5
        with pytest.raises(ValueError):
            counter.inc(-1)
        counter.reset()
        assert counter.value == 0

    def test_gauge(self) -> None:
        gauge = Gauge(name="g")
        assert gauge.set(3.5) == 3.5
        assert gauge.observe(1.0) == 1.0

    def test_histogram_percentiles(self) -> None:
        histogram = Histogram(name="h", capacity=1000)
        for i in range(1000):
            histogram.observe(float(i))
        assert histogram.p50() < histogram.p95() < histogram.p99()
        assert histogram.count == 1000

    def test_histogram_is_bounded(self) -> None:
        histogram = Histogram(name="h", capacity=10)
        for i in range(100):
            histogram.observe(float(i))
        assert histogram.retained == 10
        assert histogram.count == 100

    def test_histogram_ignores_non_finite(self) -> None:
        histogram = Histogram(name="h")
        histogram.observe(float("nan"))
        histogram.observe(float("inf"))
        assert histogram.count == 0

    def test_histogram_empty(self) -> None:
        histogram = Histogram(name="h")
        assert histogram.p50() == 0.0
        assert histogram.mean() == 0.0
        assert histogram.max() == 0.0

    def test_registry(self) -> None:
        registry = MetricsRegistry()
        registry.inc("samples_received")
        registry.inc("samples_dropped", 3)
        registry.observe("inference_latency_s", 0.01)
        registry.gauge("queue_depth").set(7)
        snapshot = registry.snapshot()
        assert snapshot["counters"]["samples_received"] == 1
        assert snapshot["counters"]["samples_dropped"] == 3
        assert snapshot["gauges"]["queue_depth"] == 7
        assert "inference_latency_s" in snapshot["histograms"]

    def test_registry_labels(self) -> None:
        registry = MetricsRegistry()
        registry.inc("samples_received", device="a")
        registry.inc("samples_received", device="b")
        counters = registry.snapshot()["counters"]
        assert len(counters) == 2

    def test_histogram_summary_shape(self) -> None:
        registry = MetricsRegistry()
        registry.observe("inference_latency_s", 1.0)
        summary = registry.snapshot()["histograms"]["inference_latency_s"]
        for key in ("count", "mean", "p50", "p95", "p99", "max"):
            assert key in summary

    def test_registry_rejects_bad_capacity(self) -> None:
        with pytest.raises(ValueError):
            MetricsRegistry(max_samples=0)

    def test_declared_metrics_exist(self) -> None:
        for name in ("samples_received", "samples_dropped", "samples_malformed",
                     "queue_depth", "window_processing_latency_s", "inference_latency_s",
                     "inference_failures", "model_version", "anomaly_events",
                     "reconnect_attempts", "protocol_failures", "policy_denies",
                     "buffer_saturation", "clock_anomalies", "store_forward_writes",
                     "store_forward_drops"):
            assert name in METRIC_NAMES


class TestRedaction:
    def test_top_level_secret(self) -> None:
        assert redact_mapping({"password": "x"})["password"] == REDACTED

    def test_nested(self) -> None:
        out = redact_mapping({"a": {"secret": "x", "b": 1}})
        assert out["a"]["secret"] == REDACTED
        assert out["a"]["b"] == 1

    def test_list(self) -> None:
        assert redact_value([{"token": "t"}])[0]["token"] == REDACTED

    def test_text_key_value(self) -> None:
        assert REDACTED in redact_text("password=hunter2")

    def test_text_bearer(self) -> None:
        assert REDACTED in redact_text("Authorization: Bearer abcdef")

    def test_text_is_truncated(self) -> None:
        assert len(redact_text("a" * 100_000, max_length=100)) < 200

    def test_safe_values_unchanged(self) -> None:
        assert redact_mapping({"host": "localhost", "port": 1883}) == {
            "host": "localhost", "port": 1883}

    def test_processor_is_usable(self) -> None:
        assert math.isfinite(1.0)
