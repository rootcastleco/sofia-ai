"""Telemetry source abstraction.

The core must never depend on a specific protocol. Every transport is an
implementation of :class:`TelemetrySource`; the rest of Sofia only sees
:class:`~sofia_ai.core.contracts.TelemetrySample` objects.

Every adapter inherits these guarantees from :class:`SourceLimits`:

* payload size ceiling,
* read/connect timeouts,
* bounded retry count,
* bounded record rate,
* typed failure (:class:`TelemetryError` subclasses) instead of raw exceptions.
"""

from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Final

from ..core.contracts import TelemetrySample
from ..core.errors import (
    PayloadTooLargeError,
    TelemetryError,
    TelemetryProtocolError,
    TransportClosedError,
)
from ..core.validation import MAX_PAYLOAD_BYTES, validate_identifier

__all__ = [
    "SourceLimits",
    "SourceState",
    "SourceStats",
    "TelemetrySink",
    "TelemetrySource",
    "check_payload_size",
]

MAX_FIELD_BYTES: Final[int] = 4096
MAX_RECORD_FIELDS: Final[int] = 128


class SourceState(StrEnum):
    """Lifecycle state of a telemetry source."""

    CREATED = "CREATED"
    OPEN = "OPEN"
    DEGRADED = "DEGRADED"
    CLOSED = "CLOSED"
    FAILED = "FAILED"


@dataclass(frozen=True, slots=True)
class SourceLimits:
    """Transport-level resource ceilings applied to every adapter."""

    max_payload_bytes: int = MAX_PAYLOAD_BYTES
    read_timeout_s: float = 5.0
    connect_timeout_s: float = 10.0
    max_retries: int = 3
    rate_limit_per_s: float = 0.0

    def __post_init__(self) -> None:
        if self.max_payload_bytes <= 0:
            raise TelemetryError("max_payload_bytes must be positive",
                                 details={"value": self.max_payload_bytes})
        if self.read_timeout_s <= 0 or self.connect_timeout_s <= 0:
            raise TelemetryError("timeouts must be positive", details={})
        if self.max_retries < 0:
            raise TelemetryError("max_retries must be non-negative", details={})
        if self.rate_limit_per_s < 0:
            raise TelemetryError("rate_limit_per_s must be non-negative", details={})


@dataclass(slots=True)
class SourceStats:
    """Counters every source maintains for observability."""

    records_read: int = 0
    records_emitted: int = 0
    records_rejected: int = 0
    bytes_read: int = 0
    protocol_errors: int = 0
    reconnects: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "records_read": self.records_read,
            "records_emitted": self.records_emitted,
            "records_rejected": self.records_rejected,
            "bytes_read": self.bytes_read,
            "protocol_errors": self.protocol_errors,
            "reconnects": self.reconnects,
        }


def check_payload_size(size_bytes: int, limits: SourceLimits) -> int:
    """Raise :class:`PayloadTooLargeError` if the payload exceeds the ceiling."""
    size = int(size_bytes)
    if size < 0:
        raise TelemetryProtocolError(
            f"negative payload size {size}", details={"size": size}
        )
    if size > limits.max_payload_bytes:
        raise PayloadTooLargeError(
            f"payload of {size} bytes exceeds limit of {limits.max_payload_bytes} bytes",
            details={"size": size, "limit": limits.max_payload_bytes},
        )
    return size


class TelemetrySource(ABC):
    """Abstract telemetry source.

    Implementations are context managers. ``read()`` yields validated samples and
    must raise a :class:`TelemetryError` subclass on transport failure — never a raw
    exception, and never ``None`` in place of an error.
    """

    def __init__(self, *, source_id: str = "unknown", limits: SourceLimits | None = None) -> None:
        self.source_id = validate_identifier(source_id, "source_id")
        self.limits = limits or SourceLimits()
        self.stats = SourceStats()
        self.state: SourceState = SourceState.CREATED

    # -- lifecycle ---------------------------------------------------------

    def open(self) -> None:
        """Open the transport. Idempotent."""
        if self.state in (SourceState.OPEN, SourceState.DEGRADED):
            return
        self._open()
        self.state = SourceState.OPEN

    def close(self) -> None:
        """Close the transport. Idempotent."""
        if self.state is SourceState.CLOSED:
            return
        self._close()
        self.state = SourceState.CLOSED

    def __enter__(self) -> TelemetrySource:
        self.open()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- data --------------------------------------------------------------

    @abstractmethod
    def read(self, *, max_records: int | None = None) -> list[TelemetrySample]:
        """Read up to ``max_records`` samples. Returns an empty list at end of stream."""

    def stream(self, *, batch_size: int = 256, max_batches: int | None = None
               ) -> Iterator[TelemetrySample]:
        """Iterate samples in bounded batches.

        Args:
            batch_size: Records per underlying read.
            max_batches: Hard ceiling on iterations so a live source cannot loop
                forever in a test.
        """
        if batch_size <= 0:
            raise TelemetryError("batch_size must be positive", details={})
        batches = 0
        while max_batches is None or batches < max_batches:
            if self.state is SourceState.CLOSED:
                return
            batch = self.read(max_records=batch_size)
            if not batch:
                return
            yield from batch
            batches += 1

    # -- hooks -------------------------------------------------------------

    def _open(self) -> None:  # noqa: B027 - deliberate optional hook
        """Transport-specific open. Default is a no-op for file sources."""

    def _close(self) -> None:  # noqa: B027 - deliberate optional hook
        """Transport-specific close. Default is a no-op."""

    def _require_open(self) -> None:
        if self.state is not SourceState.OPEN:
            raise TransportClosedError(
                f"source {self.source_id!r} is {self.state.value}, not OPEN",
                details={"source_id": self.source_id, "state": self.state.value},
            )

    def _reject(self, reason: str) -> None:
        """Record a protocol violation and raise.

        Only ``protocol_errors`` is incremented here; ``records_rejected`` is
        counted once by the caller that decides whether to skip or abort, so a
        single bad record is never double-counted.
        """
        self.stats.protocol_errors += 1
        raise TelemetryProtocolError(reason, details={"source_id": self.source_id})

    # -- metadata ----------------------------------------------------------

    def describe(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "type": type(self).__name__,
            "state": self.state.value,
            "limits": {
                "max_payload_bytes": self.limits.max_payload_bytes,
                "read_timeout_s": self.limits.read_timeout_s,
                "max_retries": self.limits.max_retries,
            },
            "stats": self.stats.as_dict(),
        }


@dataclass(slots=True)
class TelemetrySink:
    """Bounded in-memory collector used by tests, demos and store-and-forward.

    Args:
        capacity: Maximum retained samples. Overflow drops the oldest.
    """

    capacity: int = 4096
    samples: list[TelemetrySample] = field(default_factory=list, init=False)
    dropped: int = field(default=0, init=False)

    def write(self, sample: TelemetrySample) -> None:
        """Append a sample, dropping the oldest if at capacity."""
        if len(self.samples) >= self.capacity:
            self.samples.pop(0)
            self.dropped += 1
        self.samples.append(sample)

    def extend(self, samples: Sequence[TelemetrySample]) -> None:
        """Append a batch."""
        for sample in samples:
            self.write(sample)

    def drain(self) -> list[TelemetrySample]:
        """Remove and return all retained samples."""
        out, self.samples = self.samples, []
        return out

    def __len__(self) -> int:
        return len(self.samples)


def _rate_wait(rate_limit_per_s: float, last_emit: float) -> float:
    """Seconds to sleep to respect a rate ceiling. Returns 0 when unlimited."""
    if rate_limit_per_s <= 0:
        return 0.0
    minimum_interval = 1.0 / rate_limit_per_s
    elapsed = time.monotonic() - last_emit
    return max(0.0, minimum_interval - elapsed) if math.isfinite(elapsed) else 0.0


def _coerce_mapping(record: Any, origin: str) -> Mapping[str, Any]:
    if not isinstance(record, Mapping):
        raise TelemetryProtocolError(
            f"{origin}: record must be a mapping, got {type(record).__name__}",
            details={"origin": origin},
        )
    if len(record) > MAX_RECORD_FIELDS:
        raise TelemetryProtocolError(
            f"{origin}: record has {len(record)} fields, limit is {MAX_RECORD_FIELDS}",
            details={"origin": origin, "fields": len(record)},
        )
    return record
