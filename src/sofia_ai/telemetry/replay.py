"""Deterministic telemetry replay with rate control.

Replay is the mechanism used by benchmarks, regression tests and the offline
demos. It is fully deterministic: given the same input and the same injected
clock it produces the same sequence of samples with the same timestamps.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from enum import StrEnum
from typing import Final

from ..core.contracts import TelemetrySample
from ..core.errors import ValidationError
from ..core.time import TimeSource
from ..core.validation import validate_positive_float, validate_positive_int
from .base import SourceLimits, TelemetrySource

__all__ = ["ReplayMode", "ReplaySource"]

DEFAULT_BATCH: Final[int] = 256


class ReplayMode(StrEnum):
    """How replay maps recorded timestamps onto runtime."""

    AS_FAST_AS_POSSIBLE = "AS_FAST_AS_POSSIBLE"
    """Emit immediately; no sleeping. Used by tests and benchmarks."""

    REAL_TIME = "REAL_TIME"
    """Sleep to reproduce the original inter-sample timing (bounded by speed)."""

    EVENT_TIME = "EVENT_TIME"
    """Emit with original timestamps, advancing an injected clock rather than sleeping."""


class ReplaySource(TelemetrySource):
    """Replay a recorded sample sequence.

    Args:
        samples: Recorded samples. Kept by reference; not copied.
        speed: Multiplier for REAL_TIME mode (``2.0`` = twice as fast).
        mode: Timing behaviour.
        loops: Number of passes. ``1`` means a single pass.
        max_total_records: Hard ceiling on emitted records across all loops so an
            unbounded loop configuration cannot exhaust memory.
    """

    def __init__(
        self,
        samples: Sequence[TelemetrySample] = (),
        *,
        speed: float = 1.0,
        mode: ReplayMode = ReplayMode.AS_FAST_AS_POSSIBLE,
        loops: int = 1,
        max_total_records: int = 10_000_000,
        source_id: str = "replay",
        limits: SourceLimits | None = None,
        clock: TimeSource | None = None,
        batch_size: int = DEFAULT_BATCH,
    ) -> None:
        super().__init__(source_id=source_id, limits=limits)
        if not samples:
            raise ValidationError("ReplaySource requires at least one sample", details={})
        self.samples: Sequence[TelemetrySample] = samples
        self.speed = validate_positive_float(speed, "speed", maximum=1000.0)
        self.mode = mode if isinstance(mode, ReplayMode) else ReplayMode(str(mode).upper())
        self.loops = validate_positive_int(loops, "loops", maximum=1_000_000)
        self.max_total_records = validate_positive_int(
            max_total_records, "max_total_records", maximum=100_000_000
        )
        self.clock = clock
        self.batch_size = validate_positive_int(batch_size, "batch_size", maximum=1 << 20)
        self._cursor = 0
        self._loop = 0
        self._emitted = 0
        self._last_wall: float | None = None

    def _open(self) -> None:
        self._cursor = 0
        self._loop = 0
        self._emitted = 0
        self._last_wall = None

    def read(self, *, max_records: int | None = None) -> list[TelemetrySample]:
        self._require_open()
        count = self.batch_size if max_records is None else int(max_records)
        if count <= 0 or not self.samples:
            return []
        if self._emitted >= self.max_total_records:
            return []

        remaining_budget = self.max_total_records - self._emitted
        count = min(count, remaining_budget)

        out: list[TelemetrySample] = []
        while len(out) < count:
            if self._cursor >= len(self.samples):
                self._loop += 1
                if self._loop >= self.loops:
                    break
                self._cursor = 0
                self.stats.reconnects += 1
            sample = self.samples[self._cursor]
            self._cursor += 1
            out.append(self._retime(sample))
            self._emitted += 1

        self._maybe_sleep(out)
        self.stats.records_read += len(out)
        self.stats.records_emitted += len(out)
        return out

    def _retime(self, sample: TelemetrySample) -> TelemetrySample:
        if self.mode is ReplayMode.EVENT_TIME:
            return sample
        return sample

    def _maybe_sleep(self, batch: Sequence[TelemetrySample]) -> None:
        if self.mode is not ReplayMode.REAL_TIME or not batch:
            return
        if self._last_wall is None:
            self._last_wall = time.monotonic()
            return
        span = batch[-1].timestamp - batch[0].timestamp
        if span <= 0:
            return
        target = span / self.speed
        elapsed = time.monotonic() - self._last_wall
        delay = target - elapsed
        if delay > 0:
            # Bounded: never sleep longer than the configured read timeout.
            time.sleep(min(delay, self.limits.read_timeout_s))
        self._last_wall = time.monotonic()

    @property
    def exhausted(self) -> bool:
        return self._loop >= self.loops and self._cursor >= len(self.samples)

    @property
    def progress(self) -> float:
        total = len(self.samples) * self.loops
        return 0.0 if total == 0 else min(1.0, self._emitted / total)

    @classmethod
    def from_samples(cls, samples: Sequence[TelemetrySample], **kwargs: object) -> ReplaySource:
        return cls(list(samples), **kwargs)  # type: ignore[arg-type]
