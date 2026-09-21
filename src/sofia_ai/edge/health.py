"""Health and readiness reporting for gateway deployment.

Three states, deliberately simple so an orchestrator can act on them:

* ``READY``  — configured, source open, pipeline healthy
* ``DEGRADED`` — running, but with recorded failures or saturation
* ``NOT_READY`` — not started, or source failed

There is no ``UNKNOWN``: an unmonitored gateway is a broken gateway.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

__all__ = ["DEGRADED_THRESHOLDS", "HealthMonitor", "HealthReport", "HealthStatus"]

#: Fraction-based thresholds above which the gateway reports DEGRADED.
DEGRADED_THRESHOLDS: dict[str, float] = {
    "buffer_saturation": 0.9,
    "inference_failure_ratio": 0.05,
    "source_error_ratio": 0.10,
}


class HealthStatus(StrEnum):
    """Readiness state."""

    READY = "READY"
    DEGRADED = "DEGRADED"
    NOT_READY = "NOT_READY"


@dataclass(frozen=True, slots=True)
class HealthReport:
    """A health snapshot suitable for an HTTP endpoint or a systemd probe."""

    status: HealthStatus
    ready: bool
    live: bool
    reasons: tuple[str, ...]
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "ready": self.ready,
            "live": self.live,
            "reasons": list(self.reasons),
            "details": self.details,
        }


@dataclass(slots=True)
class HealthMonitor:
    """Derives :class:`HealthReport` from a runtime snapshot provider.

    Args:
        snapshot_fn: Callable returning the runtime's ``health_snapshot()`` dict.
        thresholds: Overrides for :data:`DEGRADED_THRESHOLDS`.
    """

    snapshot_fn: Callable[[], dict[str, Any]]
    thresholds: dict[str, float] = field(
        default_factory=lambda: dict(DEGRADED_THRESHOLDS)
    )

    def check(self) -> HealthReport:
        """Produce a health report. Never raises: a failing probe is a report."""
        try:
            snapshot = self.snapshot_fn()
        except Exception as exc:
            return HealthReport(
                status=HealthStatus.NOT_READY,
                ready=False,
                live=False,
                reasons=(f"snapshot failed: {type(exc).__name__}",),
            )
        if not isinstance(snapshot, dict):
            return HealthReport(
                status=HealthStatus.NOT_READY, ready=False, live=False,
                reasons=("snapshot is not a mapping",),
            )

        stats = dict(snapshot.get("stats", {}) or {})
        buffer = dict(snapshot.get("buffer", {}) or {})
        source = dict(snapshot.get("source", {}) or {})

        reasons: list[str] = []
        not_ready = False

        if source.get("state") not in (None, "OPEN", "DEGRADED"):
            not_ready = True
            reasons.append(f"source state is {source.get('state')}")

        saturation = float(buffer.get("saturation", 0.0))
        if saturation >= self.thresholds["buffer_saturation"]:
            reasons.append(f"buffer saturation {saturation:.2f} at or above threshold")

        inferences = int(stats.get("inferences", 0))
        inference_failures = int(stats.get("inference_failures", 0))
        if inferences > 0:
            ratio = inference_failures / inferences
            if ratio >= self.thresholds["inference_failure_ratio"]:
                reasons.append(
                    f"inference failure ratio {ratio:.3f} at or above threshold"
                )

        ticks = int(stats.get("ticks", 0))
        source_errors = int(stats.get("source_errors", 0))
        if ticks > 0:
            error_ratio = source_errors / ticks
            if error_ratio >= self.thresholds["source_error_ratio"]:
                reasons.append(
                    f"source error ratio {error_ratio:.3f} at or above threshold"
                )

        if not_ready:
            status = HealthStatus.NOT_READY
        elif reasons:
            status = HealthStatus.DEGRADED
        else:
            status = HealthStatus.READY

        return HealthReport(
            status=status,
            ready=status is not HealthStatus.NOT_READY,
            live=True,
            reasons=tuple(reasons),
            details=snapshot,
        )
