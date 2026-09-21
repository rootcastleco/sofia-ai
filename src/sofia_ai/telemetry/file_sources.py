"""File-based telemetry sources: CSV and JSONL.

These are the reference adapters and the ones used by every offline demo. They
implement the full defensive contract: bounded line length, bounded field count,
strict protocol errors, and no silent skipping of malformed records.
"""

from __future__ import annotations

import csv
import io
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final

from ..core.contracts import TelemetrySample
from ..core.errors import TelemetryProtocolError, ValidationError
from ..core.quality import DataQuality
from ..core.serialization import decode
from ..core.time import iso8601_to_seconds, validate_timestamp
from ..core.units import is_known_unit
from ..core.validation import MAX_PAYLOAD_BYTES
from .base import (
    MAX_RECORD_FIELDS,
    SourceLimits,
    TelemetrySource,
)

__all__ = ["DEFAULT_TIMESTAMP_COLUMN", "CsvSource", "InMemorySource", "JsonlSource"]

DEFAULT_TIMESTAMP_COLUMN: Final[str] = "timestamp"
_MAX_LINE_BYTES: Final[int] = MAX_PAYLOAD_BYTES


def _read_lines(path: str, max_lines: int = 1_000_000) -> list[str]:
    """Read a text file with a hard per-line byte ceiling and a line-count ceiling.

    The per-line ceiling is the transport payload limit: a line longer than
    ``MAX_PAYLOAD_BYTES`` is a protocol violation, not data.
    """
    p = Path(path)
    if not p.is_file():
        raise TelemetryProtocolError(f"not a regular file: {path}", details={"path": path})
    lines: list[str] = []
    with p.open("r", encoding="utf-8", newline="") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line_number > max_lines:
                break
            raw = line.encode("utf-8")
            if len(raw) > _MAX_LINE_BYTES:
                raise TelemetryProtocolError(
                    f"{path}:{line_number} line exceeds {_MAX_LINE_BYTES} bytes",
                    details={"path": path, "line": line_number},
                )
            lines.append(line.rstrip("\r\n"))
    return lines


class CsvSource(TelemetrySource):
    """CSV telemetry source.

    Expected columns: ``timestamp, device_id, channel, value, unit`` plus optional
    ``source``, ``quality``, ``sequence_number``.

    Long-format only: one row per sample. A row with the wrong field count is a
    protocol error, not a silent skip.
    """

    REQUIRED_COLUMNS: Final[tuple[str, ...]] = ("timestamp", "device_id", "channel", "value")

    def __init__(
        self,
        path: str,
        *,
        device_id: str | None = None,
        unit: str | None = None,
        source_id: str = "csv",
        limits: SourceLimits | None = None,
        strict: bool = True,
        max_lines: int = 1_000_000,
    ) -> None:
        super().__init__(source_id=source_id, limits=limits)
        self.path = str(path)
        self.default_device_id = device_id
        self.default_unit = unit
        self.strict = strict
        self.max_lines = max_lines
        self._rows: list[dict[str, str]] = []
        self._cursor = 0

    def _open(self) -> None:
        lines = _read_lines(self.path, self.max_lines)
        if not lines:
            self._rows = []
            return
        reader = csv.DictReader(io.StringIO("\n".join(lines)))
        if reader.fieldnames is None:
            raise TelemetryProtocolError("CSV has no header row",
                                         details={"path": self.path})
        missing = [c for c in self.REQUIRED_COLUMNS if c not in reader.fieldnames]
        if missing:
            raise TelemetryProtocolError(
                f"CSV missing required column(s): {missing}",
                details={"path": self.path, "missing": missing},
            )
        self._rows = [dict(row) for row in reader]
        self._cursor = 0
        self.stats.bytes_read = sum(len(x.encode("utf-8")) for x in lines)

    def read(self, *, max_records: int | None = None) -> list[TelemetrySample]:
        self._require_open()
        count = len(self._rows) if max_records is None else int(max_records)
        if count <= 0:
            return []
        chunk = self._rows[self._cursor : self._cursor + count]
        self._cursor += len(chunk)

        out: list[TelemetrySample] = []
        for index, row in enumerate(chunk, start=self._cursor - len(chunk) + 1):
            self.stats.records_read += 1
            try:
                out.append(self._to_sample(row))
            except TelemetryProtocolError:
                if self.strict:
                    raise
                self.stats.records_rejected += 1
                self.stats.protocol_errors += 1
                continue
            except (ValidationError, ValueError, TypeError, KeyError) as exc:
                if self.strict:
                    raise TelemetryProtocolError(
                        f"{self.path}:{index} malformed row: {exc}",
                        details={"path": self.path, "line": index},
                    ) from exc
                self.stats.records_rejected += 1
                continue
        self.stats.records_emitted += len(out)
        return out

    def _to_sample(self, row: dict[str, str]) -> TelemetrySample:
        ts_raw = (row.get("timestamp") or "").strip()
        if not ts_raw:
            self._reject("row has no timestamp")
        try:
            timestamp = float(ts_raw)
        except ValueError:
            timestamp = iso8601_to_seconds(ts_raw)
        validate_timestamp(timestamp)

        device_id = (row.get("device_id") or self.default_device_id or "").strip()
        if not device_id:
            self._reject("row has no device_id and no default was configured")

        channel = (row.get("channel") or "").strip()
        if not channel:
            self._reject("row has no channel")

        try:
            value = float((row.get("value") or "").strip())
        except ValueError:
            self._reject(f"value is not numeric: {row.get('value')!r}")
            raise  # pragma: no cover - _reject always raises

        unit = (row.get("unit") or self.default_unit or "dimensionless").strip()
        if not is_known_unit(unit):
            self._reject(f"unknown unit {unit!r}")

        quality_raw = (row.get("quality") or "").strip().upper()
        quality = DataQuality(quality_raw) if quality_raw else DataQuality.GOOD

        seq_raw = (row.get("sequence_number") or "").strip()
        sequence_number = int(seq_raw) if seq_raw else None

        return TelemetrySample(
            timestamp=timestamp,
            device_id=device_id,
            channel=channel,
            value=value,
            unit=unit,
            source=self.source_id,
            quality=quality,
            sequence_number=sequence_number,
        )


class JsonlSource(TelemetrySource):
    """JSON Lines telemetry source. One ``TelemetrySample`` object per line."""

    def __init__(
        self,
        path: str,
        *,
        source_id: str = "jsonl",
        limits: SourceLimits | None = None,
        strict: bool = True,
        max_lines: int = 1_000_000,
    ) -> None:
        super().__init__(source_id=source_id, limits=limits)
        self.path = str(path)
        self.strict = strict
        self.max_lines = max_lines
        self._records: list[dict[str, Any]] = []
        self._cursor = 0

    def _open(self) -> None:
        lines = _read_lines(self.path, self.max_lines)
        self._records = []
        for line_number, line in enumerate(lines, start=1):
            if len(self._records) >= self.max_lines:
                break
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = decode(stripped)
            except ValueError as exc:
                raise TelemetryProtocolError(
                    f"{self.path}:{line_number} is not valid JSON",
                    details={"path": self.path, "line": line_number},
                ) from exc
            if not isinstance(record, dict):
                raise TelemetryProtocolError(
                    f"{self.path}:{line_number} is not a JSON object",
                    details={"path": self.path, "line": line_number},
                )
            if len(record) > MAX_RECORD_FIELDS:
                raise TelemetryProtocolError(
                    f"{self.path}:{line_number} has too many fields",
                    details={"path": self.path, "line": line_number},
                )
            self._records.append(record)
        self._cursor = 0
        self.stats.bytes_read = sum(len(x.encode("utf-8")) for x in lines)

    def read(self, *, max_records: int | None = None) -> list[TelemetrySample]:
        self._require_open()
        count = len(self._records) if max_records is None else int(max_records)
        if count <= 0:
            return []
        chunk = self._records[self._cursor : self._cursor + count]
        self._cursor += len(chunk)

        out: list[TelemetrySample] = []
        for index, record in enumerate(chunk, start=self._cursor - len(chunk) + 1):
            self.stats.records_read += 1
            try:
                sample = TelemetrySample.from_dict(record)
                sample = TelemetrySample(
                    **{**sample.to_dict(), "source": record.get("source", self.source_id)}
                )
                out.append(sample)
            except (ValidationError, ValueError, KeyError, TypeError) as exc:
                if self.strict:
                    raise TelemetryProtocolError(
                        f"{self.path}:{index} malformed record: {exc}",
                        details={"path": self.path, "line": index},
                    ) from exc
                self.stats.records_rejected += 1
                self.stats.protocol_errors += 1
        self.stats.records_emitted += len(out)
        return out


class InMemorySource(TelemetrySource):
    """Deterministic source over an in-memory sequence. Used by tests and demos."""

    def __init__(
        self,
        samples: Sequence[TelemetrySample],
        *,
        source_id: str = "memory",
        limits: SourceLimits | None = None,
        batch_size: int = 256,
    ) -> None:
        super().__init__(source_id=source_id, limits=limits)
        self._samples = list(samples)
        self.batch_size = max(1, int(batch_size))
        self._cursor = 0

    def _open(self) -> None:
        self._cursor = 0

    def read(self, *, max_records: int | None = None) -> list[TelemetrySample]:
        self._require_open()
        count = self.batch_size if max_records is None else int(max_records)
        if count <= 0:
            return []
        chunk = self._samples[self._cursor : self._cursor + count]
        self._cursor += len(chunk)
        self.stats.records_read += len(chunk)
        self.stats.records_emitted += len(chunk)
        return list(chunk)

    @property
    def remaining(self) -> int:
        return max(0, len(self._samples) - self._cursor)



