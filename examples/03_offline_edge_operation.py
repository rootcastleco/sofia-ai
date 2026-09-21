#!/usr/bin/env python3
"""Demo 3 — Offline edge operation.

Demonstrates that the full pipeline runs with no network access, and that a
transport failure does not stop it:

* bounded ingest buffer with drop-oldest overflow,
* bounded store-and-forward persistence with a byte ceiling and rotation,
* connection loss simulated mid-run (the source raises; the pipeline continues),
* bounded reconnect with a backoff ceiling,
* readiness reporting.

Usage::

    python examples/03_offline_edge_operation.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from sofia_ai.core.errors import TelemetryError
from sofia_ai.edge import (
    BoundedRingBuffer,
    HealthMonitor,
    Reconnector,
    ReconnectPolicy,
    RuntimeConfig,
    StoreForward,
)
from sofia_ai.edge.runtime import EdgeRuntime
from sofia_ai.inference import build_manifest_for, create_backend
from sofia_ai.inference.detectors import DetectorConfig
from sofia_ai.telemetry import SourceLimits, TelemetrySource
from sofia_ai.telemetry.file_sources import JsonlSource

DATA = Path(__file__).resolve().parent / "data" / "edge_offline.jsonl"


class FlakySource(TelemetrySource):
    """A source that drops its connection once, partway through the stream.

    This is the fault-injection harness: it raises a typed
    :class:`TelemetryError` exactly once so the runtime's containment can be
    observed.
    """

    FAIL_AT_READ = 2

    def __init__(self, samples: list, *, source_id: str = "flaky",
                 limits: SourceLimits | None = None) -> None:
        super().__init__(source_id=source_id, limits=limits)
        self._samples = list(samples)
        self._cursor = 0
        self._reads = 0
        self.failures_injected = 0

    def read(self, *, max_records: int | None = None) -> list:
        self._require_open()
        self._reads += 1
        if self._reads == self.FAIL_AT_READ:
            self.failures_injected += 1
            # State stays OPEN: a single failed read is a transient error, not a
            # dead transport. This is what lets the pipeline continue.
            raise TelemetryError("simulated connection loss",
                                 details={"source_id": self.source_id})
        count = 512 if max_records is None else int(max_records)
        chunk = self._samples[self._cursor : self._cursor + count]
        self._cursor += len(chunk)
        self.stats.records_read += len(chunk)
        self.stats.records_emitted += len(chunk)
        return list(chunk)


def main() -> int:
    print("=" * 72)
    print("SOFIA ENGINE — Demo 3: offline edge operation")
    print("=" * 72)

    # 1. Bounded buffer -----------------------------------------------------
    buffer: BoundedRingBuffer[int] = BoundedRingBuffer(capacity=8, policy="drop_oldest")
    for i in range(12):
        buffer.push(i)
    print(f"\n[1] buffer        capacity=8, pushed 12 -> size={len(buffer)}, "
          f"dropped={buffer.dropped}, saturation={buffer.saturation:.2f}")

    # 2. Store and forward --------------------------------------------------
    with tempfile.TemporaryDirectory() as tmp:
        store = StoreForward(directory=tmp, max_bytes=20_000, max_files=3,
                             flush_every=32)
        reader = JsonlSource(str(DATA), strict=False)
        reader.open()
        recorded = []
        while True:
            batch = reader.read(max_records=20_000)
            if not batch:
                break
            recorded.extend(batch)
        reader.close()
        store.extend(recorded)
        store.flush()
        print(f"[2] store/fwd     {store.file_count} files, {store.bytes_used} bytes, "
              f"utilization={store.utilization:.2f}")

        # 3. Connection loss ------------------------------------------------
        flaky = FlakySource(recorded, source_id="flaky-edge")
        backend = create_backend("mad")
        backend.config = DetectorConfig(feature="rms", window_size=32, warmup=8,
                                        min_samples=8, threshold=3.0)
        backend.load(build_manifest_for(backend, model_id="demo-offline"))

        runtime = EdgeRuntime(
            source=flaky,
            backend=backend,
            config=RuntimeConfig(window_length=512, window_hop=256, sample_rate=1000.0,
                                 buffer_capacity=8192, max_windows_per_tick=16),
            store_forward=store,
        )
        runtime.start()
        events = runtime.run(max_ticks=20, batch_size=512)
        snapshot = runtime.health_snapshot()
        report = HealthMonitor(runtime.health_snapshot).check()
        runtime.stop()

        print(f"[3] conn loss     source raised {flaky.failures_injected} time(s); "
              f"pipeline completed {snapshot['stats']['ticks']} ticks")
        print(f"    contained     source_errors={snapshot['stats']['source_errors']}, "
              f"samples_ingested={snapshot['stats']['samples_ingested']}, "
              f"windows={snapshot['stats']['windows_built']}")
        print(f"    events        {len(events)}")

        # 4. Reconnect ------------------------------------------------------
        attempts = {"n": 0}

        def connect() -> bool:
            attempts["n"] += 1
            return attempts["n"] >= 3

        delays: list[float] = []
        reconnector = Reconnector(
            policy=ReconnectPolicy(max_attempts=8, initial_backoff_s=0.1,
                                   max_backoff_s=1.0, multiplier=2.0),
            connect=connect,
            sleep=delays.append,
        )
        reconnected = reconnector.run()
        print(f"[4] reconnect     success={reconnected}, attempts={reconnector.attempts}, "
              f"delays={[round(d, 3) for d in delays]}")

        # 5. Readiness ------------------------------------------------------
        print(f"[5] readiness     {report.status.value} "
              f"(ready={report.ready}, live={report.live})")

        # 6. Replay persisted data -----------------------------------------
        replayed = store.read_all()
        print(f"[6] drain store   {len(replayed)} samples recovered from local store, "
              f"files now {store.file_count}")

    print("\nNo network access was used at any point in this demo.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
