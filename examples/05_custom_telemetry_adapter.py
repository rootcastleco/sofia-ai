#!/usr/bin/env python3
"""Demo 5 — Custom telemetry adapter.

Shows how to add a new industrial protocol without editing the Sofia core.

A new transport is a subclass of :class:`TelemetrySource` that implements
``read()`` and, optionally, ``_open``/``_close``. It inherits:

* payload size limits,
* timeouts and retry ceilings,
* bounded record counts,
* typed protocol errors,
* statistics for observability.

This example implements a minimal line-protocol adapter for a fixed-width binary
frame format, then registers it so it can be created by name.

Usage::

    python examples/05_custom_telemetry_adapter.py
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path
from typing import Any

from sofia_ai.core.contracts import TelemetrySample
from sofia_ai.core.errors import PayloadTooLargeError, TelemetryProtocolError
from sofia_ai.diagnostics import DiagnosticEngine, RuleContext
from sofia_ai.features import extract_from_array
from sofia_ai.inference import build_manifest_for, create_backend
from sofia_ai.inference.detectors import DetectorConfig
from sofia_ai.signal import windows_from_samples
from sofia_ai.telemetry import SourceLimits, TelemetrySource, register_telemetry_source

DATA = Path(__file__).resolve().parent / "data" / "edge_offline.jsonl"

# Frame grammar: magic(2) | seq(4, big-endian uint32) | timestamp(8, float64)
#                | value(8, float64) | crc16 placeholder(2)  => 24 bytes
FRAME_MAGIC = b"SF"
FRAME_SIZE = 24


def encode_frame(sequence: int, timestamp: float, value: float) -> bytes:
    """Encode one frame using the demo grammar."""
    return (
        FRAME_MAGIC
        + struct.pack(">I", int(sequence))
        + struct.pack(">d", float(timestamp))
        + struct.pack(">d", float(value))
        + b"\x00\x00"
    )


class BinaryFrameSource(TelemetrySource):
    """A custom adapter for a fixed-width binary frame protocol.

    Security-relevant behaviour inherited and implemented here:

    * ``read`` never returns a partial frame,
    * a buffer longer than ``max_payload_bytes`` is a protocol error, not growth,
    * a frame whose magic or length is wrong is counted and rejected,
    * the frame length is checked *before* any indexing (threat T-02).
    """

    def __init__(
        self,
        transport: Any = None,
        *,
        device_id: str = "binary-device",
        channel: str = "vibration_x",
        unit: str = "g",
        source_id: str = "binary-frame",
        limits: SourceLimits | None = None,
    ) -> None:
        super().__init__(source_id=source_id, limits=limits)
        self.transport = transport
        self.device_id = device_id
        self.channel = channel
        self.unit = unit
        self._buffer = bytearray()
        self.frames_rejected = 0

    def _open(self) -> None:
        self._buffer = bytearray()

    def read(self, *, max_records: int | None = None) -> list[TelemetrySample]:
        self._require_open()
        limit = 64 if max_records is None else int(max_records)
        if limit <= 0:
            return []

        chunk = self.transport.read() if self.transport is not None else b""
        if chunk:
            self.stats.bytes_read += len(chunk)
            self._buffer.extend(chunk)
            if len(self._buffer) > self.limits.max_payload_bytes:
                self._buffer.clear()
                self.frames_rejected += 1
                raise PayloadTooLargeError(
                    f"frame buffer exceeded {self.limits.max_payload_bytes} bytes",
                    details={"limit": self.limits.max_payload_bytes},
                )

        out: list[TelemetrySample] = []
        while len(out) < limit and len(self._buffer) >= FRAME_SIZE:
            frame = bytes(self._buffer[:FRAME_SIZE])
            del self._buffer[:FRAME_SIZE]
            sample = self._decode(frame)
            if sample is not None:
                out.append(sample)
        self.stats.records_emitted += len(out)
        return out

    def _decode(self, frame: bytes) -> TelemetrySample | None:
        """Decode one frame, returning None (counted) on a protocol violation."""
        if len(frame) != FRAME_SIZE:
            self.stats.protocol_errors += 1
            self.frames_rejected += 1
            raise TelemetryProtocolError(
                f"frame length {len(frame)} != {FRAME_SIZE}",
                details={"length": len(frame)},
            )
        if frame[:2] != FRAME_MAGIC:
            self.stats.protocol_errors += 1
            self.frames_rejected += 1
            return None
        try:
            sequence = struct.unpack(">I", frame[2:6])[0]
            timestamp = struct.unpack(">d", frame[6:14])[0]
            value = struct.unpack(">d", frame[14:22])[0]
        except struct.error as exc:
            self.stats.protocol_errors += 1
            self.frames_rejected += 1
            raise TelemetryProtocolError(f"frame unpack failed: {exc}",
                                         details={}) from exc
        self.stats.records_read += 1
        return TelemetrySample(
            timestamp=timestamp,
            device_id=self.device_id,
            channel=self.channel,
            value=value,
            unit=self.unit,
            source=self.source_id,
            sequence_number=int(sequence),
        )


class FakeTransport:
    """In-memory byte stream standing in for a serial port or socket."""

    def __init__(self, payload: bytes) -> None:
        self._payload = payload
        self._cursor = 0

    def read(self, size: int = 4096) -> bytes:
        chunk = self._payload[self._cursor : self._cursor + size]
        self._cursor += len(chunk)
        return chunk


def main() -> int:
    print("=" * 72)
    print("SOFIA ENGINE — Demo 5: custom telemetry adapter")
    print("=" * 72)

    import json

    samples = []
    with open(DATA, encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                samples.append(json.loads(stripped))
    payload = b"".join(
        encode_frame(i, s["timestamp"], s["value"]) for i, s in enumerate(samples)
    )
    print(f"\n[1] protocol      custom binary frame grammar, {FRAME_SIZE} bytes/frame")
    print(f"    payload       {len(payload)} bytes for {len(samples)} samples")

    register_telemetry_source("binary_frame", BinaryFrameSource,
                              overwrite=True)
    source = BinaryFrameSource(FakeTransport(payload), device_id="sim-pump-01",
                               channel="vibration_x", unit="g")
    with source:
        decoded = []
        while True:
            batch = source.read(max_records=64)
            if not batch:
                break
            decoded.extend(batch)

    print(f"[2] adapter       decoded {len(decoded)} samples, "
          f"rejected={source.frames_rejected}, bytes={source.stats.bytes_read}")

    # Corrupted-frame handling ---------------------------------------------
    corrupt = bytearray(payload[: FRAME_SIZE * 4])
    corrupt[0:2] = b"XX"  # bad magic in the first frame
    corrupt_source = BinaryFrameSource(FakeTransport(bytes(corrupt)),
                                       device_id="sim-pump-01",
                                       limits=SourceLimits(max_payload_bytes=4096))
    with corrupt_source:
        recovered = []
        while True:
            batch = corrupt_source.read(max_records=8)
            if not batch:
                break
            recovered.extend(batch)
    print(f"[3] bad frame     1 of 4 rejected, {len(recovered)} recovered — "
          f"no exception, no partial frame emitted")

    # Feed the custom adapter into the standard pipeline --------------------
    windows = windows_from_samples(
        decoded, length=512, hop=256, sample_rate=1000.0, unit="g",
        channel="vibration_x", device_id="sim-pump-01",
    )
    backend = create_backend("mad")
    backend.config = DetectorConfig(feature="rms", window_size=32, warmup=8,
                                    min_samples=8, threshold=3.0)
    backend.load(build_manifest_for(backend, model_id="demo-custom-adapter"))

    engine = DiagnosticEngine()
    events = 0
    for window in windows:
        features = extract_from_array(window.values, window.sample_rate)
        result = backend.infer(features)
        if engine.evaluate(RuleContext(device_id=window.device_id, channel=window.channel,
                                       score=result.score,
                                       confidence=result.confidence)) is not None:
            events += 1
    print(f"[4] pipeline      {len(windows)} windows -> {events} health events "
          f"(core untouched)")

    print("\nThe adapter was added by registration. No core module was modified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
