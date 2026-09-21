#!/usr/bin/env python3
"""Demo 2 — Telemetry pipeline.

Pipeline::

    telemetry replay -> adapter -> window -> inference -> event -> JSON output

Reads the committed multi-channel JSONL file, replays it, and prints the resulting
health events as JSON (the shape an upstream system would consume).

Usage::

    python examples/02_telemetry_pipeline.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from sofia_ai.edge import EdgeRuntime, RuntimeConfig
from sofia_ai.inference import build_manifest_for, create_backend
from sofia_ai.inference.detectors import DetectorConfig
from sofia_ai.observability import MetricsRegistry, configure_logging, get_logger
from sofia_ai.telemetry import JsonlSource, ReplayMode, ReplaySource

DATA = Path(__file__).resolve().parent / "data" / "telemetry.jsonl"
SAMPLE_RATE = 1000.0


def main() -> int:
    configure_logging(level="WARNING")
    logger = get_logger("demo2", device_id="sim-pump-01")

    print("=" * 72)
    print("SOFIA ENGINE — Demo 2: telemetry pipeline (replay -> JSON events)")
    print("=" * 72)

    # 1. Adapter ------------------------------------------------------------
    reader = JsonlSource(str(DATA), strict=False)
    reader.open()
    recorded = []
    while True:
        batch = reader.read(max_records=20_000)
        if not batch:
            break
        recorded.extend(batch)
    reader.close()
    channels = sorted({s.channel for s in recorded})
    print(f"\n[1] adapter       JsonlSource: {len(recorded)} samples, channels={channels}")

    # 2. Replay -------------------------------------------------------------
    vibration = [s for s in recorded if s.channel == "vibration_x"]
    replay = ReplaySource.from_samples(
        vibration, mode=ReplayMode.AS_FAST_AS_POSSIBLE, batch_size=4096
    )
    print(f"[2] replay        {len(vibration)} vibration samples "
          f"(mode={replay.mode.value})")

    # 3-5. Runtime ----------------------------------------------------------
    backend = create_backend("mad")
    backend.config = DetectorConfig(feature="rms", window_size=32, warmup=8,
                                    min_samples=8, threshold=3.0)
    backend.load(build_manifest_for(backend, model_id="demo-mad"))

    metrics = MetricsRegistry()
    runtime = EdgeRuntime(
        source=replay,
        backend=backend,
        config=RuntimeConfig(window_length=512, window_hop=256,
                             sample_rate=SAMPLE_RATE, buffer_capacity=32768,
                             max_windows_per_tick=32),
        metrics=metrics,
        logger=logger,
    )
    runtime.start()
    events = runtime.run(max_ticks=32, batch_size=4096)
    snapshot = runtime.health_snapshot()
    runtime.stop()

    print(f"[3] runtime       {snapshot['stats']['ticks']} ticks, "
          f"{snapshot['stats']['windows_built']} windows, "
          f"{snapshot['stats']['inferences']} inferences")
    print(f"[4] inference     {snapshot['stats']['inference_failures']} failures")
    print(f"[5] events        {len(events)} health events")

    # 6. JSON output --------------------------------------------------------
    payload = {
        "source": str(DATA.name),
        "event_count": len(events),
        "events": [e.to_dict() for e in events[:3]],
        "metrics": metrics.snapshot()["counters"],
    }
    print("\n[6] JSON output (first event):")
    print(json.dumps(payload, sort_keys=True, indent=2, default=str)[:1600])

    print("\nNote: 'anomaly_score' evidence records a deviation, not a fault cause.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
