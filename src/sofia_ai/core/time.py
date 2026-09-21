"""Time discipline for telemetry.

Telemetry code must distinguish:

* **event time** — when the sample was produced by the device (wall clock, UTC epoch seconds),
* **ingestion time** — when Sofia received it,
* **processing time** — monotonic seconds, used for latency measurement only,
* **sample index** and **sample rate** — the authoritative axis for DSP.

DSP never derives frequency content from wall-clock deltas when a sample rate and index
are available. Clocks drift, jump, and duplicate; sample indices do not.

All clocks are injectable (:class:`TimeSource`) so tests are deterministic.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Final, Protocol

from .errors import TimestampError

__all__ = [
    "MAX_REASONABLE_EPOCH_S",
    "MIN_REASONABLE_EPOCH_S",
    "ClockAnomaly",
    "ClockMonitor",
    "FixedTimeSource",
    "SystemTimeSource",
    "TimeSource",
    "iso8601_to_seconds",
    "seconds_to_iso8601",
    "validate_timestamp",
]

# Guard rails for obviously-wrong timestamps (year ~1970-01-02 .. year ~2100).
MIN_REASONABLE_EPOCH_S: Final[float] = 86_400.0
MAX_REASONABLE_EPOCH_S: Final[float] = 4_102_444_800.0


class TimeSource(Protocol):
    """Injectable clock."""

    def wall(self) -> float:
        """Wall-clock time in seconds since the Unix epoch (UTC)."""
        ...

    def monotonic(self) -> float:
        """Monotonic time in seconds, arbitrary origin, never decreases."""
        ...


class SystemTimeSource:
    """Real system clock."""

    __slots__ = ()

    def wall(self) -> float:
        return time.time()

    def monotonic(self) -> float:
        return time.monotonic()


@dataclass(slots=True)
class FixedTimeSource:
    """Deterministic clock for tests.

    ``monotonic`` advances by ``step`` on every call so latency measurements remain
    non-zero and deterministic.
    """

    wall_time: float = 1_700_000_000.0
    monotonic_time: float = 0.0
    step: float = 0.001

    def wall(self) -> float:
        return self.wall_time

    def monotonic(self) -> float:
        self.monotonic_time += self.step
        return self.monotonic_time

    def advance(self, seconds: float) -> None:
        """Move wall and monotonic time forward together."""
        self.wall_time += float(seconds)
        self.monotonic_time += float(seconds)


def validate_timestamp(value: float, *, name: str = "timestamp") -> float:
    """Validate a wall-clock timestamp.

    Raises:
        TimestampError: if the value is non-finite or absurdly out of range.
    """
    ts = float(value)
    if not math.isfinite(ts):
        raise TimestampError(
            f"{name} must be finite, got {value!r}", details={"name": name, "value": value}
        )
    if not (MIN_REASONABLE_EPOCH_S <= ts <= MAX_REASONABLE_EPOCH_S):
        raise TimestampError(
            f"{name} {ts!r} is outside the plausible epoch range "
            f"[{MIN_REASONABLE_EPOCH_S:.0f}, {MAX_REASONABLE_EPOCH_S:.0f}]",
            details={"name": name, "value": ts},
        )
    return ts


@dataclass(frozen=True, slots=True)
class ClockAnomaly:
    """A detected clock irregularity."""

    kind: str
    observed: float
    reference: float
    detail: str = ""

    def to_dict(self) -> dict[str, float | str]:
        return {
            "kind": self.kind,
            "observed": self.observed,
            "reference": self.reference,
            "detail": self.detail,
        }


@dataclass(slots=True)
class ClockMonitor:
    """Detects duplicated, backwards, and jumping timestamps per stream.

    Args:
        max_forward_jump_s: A forward jump larger than this is reported as ``jump``.
            This distinguishes "gap in data" from "clock was reset".
        duplicate_tolerance_s: Timestamps within this distance are treated as duplicates.
    """

    max_forward_jump_s: float = 300.0
    duplicate_tolerance_s: float = 0.0
    _last: float | None = field(default=None, init=False, repr=False)
    anomalies: list[ClockAnomaly] = field(default_factory=list, init=False)

    def observe(self, timestamp: float) -> ClockAnomaly | None:
        """Record a timestamp and return an anomaly, or None if consistent."""
        ts = validate_timestamp(timestamp)
        anomaly: ClockAnomaly | None = None

        if self._last is not None:
            if ts < self._last:
                anomaly = ClockAnomaly(
                    kind="backwards",
                    observed=ts,
                    reference=self._last,
                    detail="Timestamp moved backwards; possible clock reset or replay.",
                )
            elif ts == self._last or (ts - self._last) <= self.duplicate_tolerance_s:
                anomaly = ClockAnomaly(
                    kind="duplicate",
                    observed=ts,
                    reference=self._last,
                    detail="Timestamp equals or nearly equals the previous sample.",
                )
            elif (ts - self._last) > self.max_forward_jump_s:
                anomaly = ClockAnomaly(
                    kind="jump",
                    observed=ts,
                    reference=self._last,
                    detail=f"Forward jump of {ts - self._last:.3f}s exceeds tolerance.",
                )

        if ts > (self._last if self._last is not None else -math.inf):
            self._last = ts
        if anomaly is not None:
            self.anomalies.append(anomaly)
        return anomaly

    def reset(self) -> None:
        """Forget stream state (used after a transport reconnect)."""
        self._last = None

    @property
    def anomaly_count(self) -> int:
        return len(self.anomalies)


def seconds_to_iso8601(value: float) -> str:
    """Format epoch seconds as an ISO-8601 UTC string (no external dependency)."""
    import datetime as _dt

    ts = validate_timestamp(value)
    dt = _dt.datetime.fromtimestamp(ts, tz=_dt.UTC)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}Z"


def iso8601_to_seconds(text: str) -> float:
    """Parse an ISO-8601 UTC string to epoch seconds."""
    import datetime as _dt

    cleaned = text.strip()
    if cleaned.endswith("Z"):
        cleaned = cleaned[:-1] + "+00:00"
    try:
        dt = _dt.datetime.fromisoformat(cleaned)
    except ValueError as exc:
        raise TimestampError(
            f"Could not parse ISO-8601 timestamp {text!r}", details={"text": text}
        ) from exc
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=_dt.UTC)
    return validate_timestamp(dt.timestamp())
