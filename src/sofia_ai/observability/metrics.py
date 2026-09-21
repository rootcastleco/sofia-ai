"""Metrics: counters, gauges and bounded percentile histograms.

Every metric Sofia defines is listed in :data:`METRIC_NAMES` so the observability
contract is explicit and testable. Histograms retain a bounded reservoir so
p50/p95/p99 can be reported without unbounded memory.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Final

import numpy as np

__all__ = [
    "METRIC_NAMES",
    "Counter",
    "Gauge",
    "Histogram",
    "MetricsRegistry",
]

#: The metrics Sofia guarantees (SOFIA-OBS-004).
METRIC_NAMES: Final[tuple[str, ...]] = (
    "samples_received",
    "samples_dropped",
    "samples_malformed",
    "queue_depth",
    "window_processing_latency_s",
    "inference_latency_s",
    "inference_failures",
    "model_version",
    "anomaly_events",
    "reconnect_attempts",
    "protocol_failures",
    "policy_denies",
    "buffer_saturation",
    "clock_anomalies",
    "store_forward_writes",
    "store_forward_drops",
)


@dataclass(slots=True)
class Counter:
    """Monotonic counter."""

    name: str
    value: int = 0
    labels: dict[str, str] = field(default_factory=dict)

    def inc(self, amount: int = 1) -> int:
        if amount < 0:
            raise ValueError("Counter.inc requires a non-negative amount")
        self.value += int(amount)
        return self.value

    def reset(self) -> None:
        self.value = 0


@dataclass(slots=True)
class Gauge:
    """Point-in-time value."""

    name: str
    value: float = 0.0
    labels: dict[str, str] = field(default_factory=dict)

    def set(self, value: float) -> float:
        self.value = float(value)
        return self.value

    def observe(self, value: float) -> float:
        return self.set(value)


@dataclass(slots=True)
class Histogram:
    """Bounded reservoir histogram with exact percentiles over retained samples.

    Args:
        capacity: Maximum retained samples. Oldest are evicted first, so memory is
            bounded regardless of event rate.
    """

    name: str
    capacity: int = 10_000
    labels: dict[str, str] = field(default_factory=dict)
    _samples: list[float] = field(default_factory=list, init=False, repr=False)
    _count: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if self.capacity <= 0:
            raise ValueError("Histogram capacity must be positive")

    def observe(self, value: float) -> None:
        """Record one sample."""
        value = float(value)
        if not math.isfinite(value):
            return
        self._count += 1
        self._samples.append(value)
        if len(self._samples) > self.capacity:
            del self._samples[: len(self._samples) - self.capacity]

    @property
    def count(self) -> int:
        return self._count

    @property
    def retained(self) -> int:
        return len(self._samples)

    def percentile(self, q: float) -> float:
        """Percentile over retained samples. ``q`` in ``[0, 100]``."""
        if not self._samples:
            return 0.0
        return float(np.percentile(np.asarray(self._samples, dtype=np.float64), q))

    def p50(self) -> float:
        return self.percentile(50)

    def p95(self) -> float:
        return self.percentile(95)

    def p99(self) -> float:
        return self.percentile(99)

    def mean(self) -> float:
        return float(np.mean(self._samples)) if self._samples else 0.0

    def max(self) -> float:
        return float(np.max(self._samples)) if self._samples else 0.0

    def summary(self) -> dict[str, float]:
        return {
            "count": float(self._count),
            "retained": float(self.retained),
            "mean": self.mean(),
            "p50": self.p50(),
            "p95": self.p95(),
            "p99": self.p99(),
            "max": self.max(),
        }

    def reset(self) -> None:
        self._samples.clear()
        self._count = 0


@dataclass(slots=True)
class MetricsRegistry:
    """Named metric collection with a bounded history."""

    prefix: str = "sofia"
    max_samples: int = 10_000

    _counters: dict[str, Counter] = field(default_factory=dict, init=False, repr=False)
    _gauges: dict[str, Gauge] = field(default_factory=dict, init=False, repr=False)
    _histograms: dict[str, Histogram] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.max_samples <= 0:
            raise ValueError("max_samples must be positive")

    def counter(self, name: str, **labels: str) -> Counter:
        key = self._key(name, labels)
        metric = self._counters.get(key)
        if metric is None:
            metric = Counter(name=name, labels=dict(labels))
            self._counters[key] = metric
        return metric

    def gauge(self, name: str, **labels: str) -> Gauge:
        key = self._key(name, labels)
        metric = self._gauges.get(key)
        if metric is None:
            metric = Gauge(name=name, labels=dict(labels))
            self._gauges[key] = metric
        return metric

    def histogram(self, name: str, **labels: str) -> Histogram:
        key = self._key(name, labels)
        metric = self._histograms.get(key)
        if metric is None:
            metric = Histogram(name=name, capacity=self.max_samples, labels=dict(labels))
            self._histograms[key] = metric
        return metric

    def _key(self, name: str, labels: dict[str, str]) -> str:
        suffix = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}|{suffix}" if suffix else name

    def inc(self, name: str, amount: int = 1, **labels: str) -> int:
        return self.counter(name, **labels).inc(amount)

    def observe(self, name: str, value: float, **labels: str) -> None:
        self.histogram(name, **labels).observe(value)

    def snapshot(self) -> dict[str, Any]:
        """Full metrics snapshot, suitable for a health endpoint."""
        return {
            "counters": {k: v.value for k, v in self._counters.items()},
            "gauges": {k: v.value for k, v in self._gauges.items()},
            "histograms": {k: v.summary() for k, v in self._histograms.items()},
        }

    def reset(self) -> None:
        self._counters.clear()
        self._gauges.clear()
        self._histograms.clear()
