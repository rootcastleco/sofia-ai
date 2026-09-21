"""Serial line telemetry adapter (optional extra: ``sofia-engine[serial]``).

Reads framed, delimited records from a serial port. Framing is strict:

* a record is terminated by ``delimiter`` (default ``\\n``),
* a record longer than ``max_frame_bytes`` is a protocol error, the buffer is
  flushed, and the condition is counted — an attacker cannot grow the buffer
  without bound (threat T-07),
* no partial frame is ever emitted.

``pyserial`` is imported lazily and the port object is injectable, so unit tests
need no hardware. Integration against real serial hardware is **NOT VERIFIED**.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Final

from ..core.contracts import TelemetrySample
from ..core.errors import (
    ConfigurationError,
    PayloadTooLargeError,
    TelemetryError,
    TelemetryProtocolError,
)
from ..core.units import is_known_unit
from ..core.validation import validate_positive_float
from .base import SourceLimits, TelemetrySource

__all__ = ["SerialConfig", "SerialSource", "decode_delimited_record"]

_DEFAULT_MAX_FRAME: Final[int] = 4096


@dataclass(frozen=True, slots=True)
class SerialConfig:
    """Serial port settings."""

    port: str = "/dev/ttyUSB0"
    baudrate: int = 115200
    timeout_s: float = 1.0
    delimiter: bytes = b"\n"
    max_frame_bytes: int = _DEFAULT_MAX_FRAME

    def __post_init__(self) -> None:
        if not self.port:
            raise ConfigurationError("serial.port must not be empty", details={})
        if self.baudrate <= 0:
            raise ConfigurationError(f"serial.baudrate {self.baudrate} must be positive",
                                     details={"baudrate": self.baudrate})
        validate_positive_float(self.timeout_s, "serial.timeout_s", maximum=3600.0)
        if not self.delimiter:
            raise ConfigurationError("serial.delimiter must not be empty", details={})
        if self.max_frame_bytes <= 0:
            raise ConfigurationError("serial.max_frame_bytes must be positive", details={})


def decode_delimited_record(frame: bytes, *, source: str = "serial") -> TelemetrySample:
    """Decode one delimited serial frame into a sample.

    Accepts the canonical Sofia JSON envelope. CSV frames are supported when they
    contain exactly ``device_id,channel,value,unit``.

    Raises:
        TelemetryProtocolError: if the frame cannot be decoded safely.
    """
    text = frame.decode("utf-8", errors="strict").strip()
    if not text:
        raise TelemetryProtocolError("empty serial frame", details={})
    if text.startswith("{"):
        try:
            record = json.loads(text)
        except ValueError as exc:
            raise TelemetryProtocolError("serial frame is not valid JSON", details={}) from exc
        if not isinstance(record, dict) or "value" not in record:
            raise TelemetryProtocolError("serial frame is not a Sofia JSON sample",
                                        details={})
        unit = str(record.get("unit", "dimensionless"))
        if not is_known_unit(unit):
            raise TelemetryProtocolError(f"unknown unit {unit!r}", details={"unit": unit})
        return TelemetrySample(
            timestamp=float(record.get("timestamp", time.time())),
            device_id=str(record.get("device_id", "serial-device")),
            channel=str(record.get("channel", "value")),
            value=float(record["value"]),
            unit=unit,
            source=source,
        )
    parts = text.split(",")
    if len(parts) != 4:
        raise TelemetryProtocolError(
            f"CSV serial frame must have 4 fields, got {len(parts)}", details={}
        )
    device_id, channel, value_raw, unit = (p.strip() for p in parts)
    if not is_known_unit(unit):
        raise TelemetryProtocolError(f"unknown unit {unit!r}", details={"unit": unit})
    try:
        value = float(value_raw)
    except ValueError as exc:
        raise TelemetryProtocolError(f"non-numeric value {value_raw!r}", details={}) from exc
    return TelemetrySample(
        timestamp=time.time(),
        device_id=device_id,
        channel=channel,
        value=value,
        unit=unit,
        source=source,
    )


class SerialSource(TelemetrySource):
    """Framed serial telemetry reader with a bounded frame buffer."""

    def __init__(
        self,
        config: SerialConfig | None = None,
        *,
        port_factory: Callable[[SerialConfig], Any] | None = None,
        decoder: Callable[[bytes], TelemetrySample] | None = None,
        source_id: str = "serial",
        limits: SourceLimits | None = None,
    ) -> None:
        super().__init__(source_id=source_id, limits=limits)
        self.config = config or SerialConfig()
        self._port_factory = port_factory
        self._decoder = decoder or (
            lambda frame: decode_delimited_record(frame, source=source_id)
        )
        self._port: Any = None
        self._buffer = bytearray()
        self.frames_dropped = 0
        self.read_timeout_count = 0

    def _open(self) -> None:
        self._port = self._make_port()
        self._buffer = bytearray()

    def _make_port(self) -> Any:
        if self._port_factory is not None:
            return self._port_factory(self.config)
        try:
            import serial
        except ImportError as exc:
            raise TelemetryError(
                "pyserial is not installed. Install the 'serial' extra: "
                "pip install sofia-engine[serial]",
                details={"extra": "serial"},
            ) from exc
        return serial.Serial(self.config.port, baudrate=self.config.baudrate,
                             timeout=self.config.timeout_s)

    def read(self, *, max_records: int | None = None) -> list[TelemetrySample]:
        self._require_open()
        limit = 64 if max_records is None else int(max_records)
        if limit <= 0:
            return []
        out: list[TelemetrySample] = []

        while len(out) < limit:
            chunk = self._read_chunk()
            if not chunk:
                break
            self._buffer.extend(chunk)
            self.stats.bytes_read += len(chunk)
            while True:
                index = self._buffer.find(self.config.delimiter)
                if index < 0:
                    break
                if len(self._buffer) > self.config.max_frame_bytes:
                    self._buffer.clear()
                    self.frames_dropped += 1
                    self.stats.protocol_errors += 1
                    raise PayloadTooLargeError(
                        f"serial frame exceeds {self.config.max_frame_bytes} bytes",
                        details={"limit": self.config.max_frame_bytes},
                    )
                frame = bytes(self._buffer[:index])
                del self._buffer[: index + len(self.config.delimiter)]
                self.stats.records_read += 1
                try:
                    out.append(self._decoder(frame))
                except (TelemetryProtocolError, ValueError, KeyError, TypeError):
                    self.stats.protocol_errors += 1
                    self.stats.records_rejected += 1
                if len(out) >= limit:
                    break

            # A frame with no delimiter that keeps growing is an unbounded buffer.
            # Flush and report rather than let the buffer grow.
            if len(self._buffer) > self.config.max_frame_bytes:
                self._buffer.clear()
                self.frames_dropped += 1
                self.stats.protocol_errors += 1
                raise PayloadTooLargeError(
                    f"serial frame exceeds {self.config.max_frame_bytes} bytes",
                    details={"limit": self.config.max_frame_bytes},
                )

        self.stats.records_emitted += len(out)
        return out

    def _read_chunk(self) -> bytes:
        try:
            waiting = self._port.in_waiting if hasattr(self._port, "in_waiting") else 1
            size = max(1, min(int(waiting or 1), 4096))
            data = self._port.read(size)
        except Exception as exc:
            self.state = type(self.state).FAILED
            raise TelemetryError(
                f"serial read failure on {self.config.port}: {exc}",
                details={"port": self.config.port},
            ) from exc
        if not data:
            # No data available. This is a timeout, not a transport failure: the
            # pipeline must survive it, so it is counted and reported as end-of-batch.
            self.read_timeout_count += 1
            return b""
        return bytes(data)

    def _close(self) -> None:
        if self._port is not None and hasattr(self._port, "close"):
            self._port.close()
        self._port = None
        self._buffer.clear()
