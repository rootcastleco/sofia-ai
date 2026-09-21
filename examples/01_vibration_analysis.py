#!/usr/bin/env python3
"""Demo 1 — Vibration analysis.

Pipeline::

    signal -> validation -> window -> signal features -> spectrum
           -> anomaly detector -> HealthEvent

Runs offline against ``examples/data/vibration.csv`` (deterministic synthetic
data committed to this repository). No network access.

Usage::

    python examples/01_vibration_analysis.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from sofia_ai.core.contracts import DataQuality
from sofia_ai.diagnostics import DiagnosticEngine, RuleContext, compute_health_score
from sofia_ai.features import EXTRACTOR_ID, EXTRACTOR_VERSION, extract_from_array
from sofia_ai.inference import build_manifest_for, create_backend
from sofia_ai.inference.detectors import DetectorConfig
from sofia_ai.signal import welch_psd, windows_from_samples
from sofia_ai.telemetry import CsvSource

DATA = Path(__file__).resolve().parent / "data" / "vibration.csv"
SAMPLE_RATE = 1000.0
WINDOW = 512
HOP = 256
SHAFT_HZ = 25.0


def main() -> int:
    print("=" * 72)
    print("SOFIA ENGINE — Demo 1: vibration analysis")
    print("=" * 72)

    # 1. Ingest -------------------------------------------------------------
    source = CsvSource(str(DATA), device_id="sim-pump-01", unit="g", source_id="csv")
    with source:
        samples = []
        while True:
            batch = source.read(max_records=4096)
            if not batch:
                break
            samples.extend(batch)
    print(f"\n[1] ingest        {len(samples)} samples from {DATA.name}")
    print(f"    channel       {samples[0].channel}  unit={samples[0].unit}")
    print(f"    time span     {samples[-1].timestamp - samples[0].timestamp:.3f} s")

    # 2. Validate + window --------------------------------------------------
    windows = windows_from_samples(
        samples, length=WINDOW, hop=HOP, sample_rate=SAMPLE_RATE,
        unit=samples[0].unit, channel=samples[0].channel, device_id=samples[0].device_id,
    )
    print(f"\n[2] windowing     {len(windows)} windows "
          f"(length={WINDOW}, hop={HOP}, fs={SAMPLE_RATE:.0f} Hz)")

    # 3. Features + spectrum -------------------------------------------------
    features = [
        extract_from_array(w.values, w.sample_rate, channel=w.channel, shaft_hz=SHAFT_HZ)
        for w in windows
    ]
    print(f"\n[3] features      {features[0].size} per window "
          f"({EXTRACTOR_ID}@{EXTRACTOR_VERSION})")

    spectrum = welch_psd(windows[-1].values, SAMPLE_RATE)
    dominant = spectrum.frequencies[int(np.argmax(spectrum.values))]
    print(f"    spectrum      Welch PSD, {spectrum.frequencies.size} bins, "
          f"df={spectrum.df:.2f} Hz")
    print(f"    dominant      {dominant:.2f} Hz (shaft = {SHAFT_HZ:.2f} Hz)")

    # 4. Detector ------------------------------------------------------------
    detector = create_backend("mad")
    detector.config = DetectorConfig(feature="rms", window_size=32, warmup=8,
                                     min_samples=8, threshold=3.0)
    detector.load(build_manifest_for(detector, model_id="demo-mad"))
    print("\n[4] detector      robust median/MAD on feature 'rms' (threshold 3.0)")

    # 5. Diagnostics -> HealthEvent -----------------------------------------
    engine = DiagnosticEngine()
    events = []
    for window, vector in zip(windows, features, strict=True):
        result = detector.infer(vector)
        context = RuleContext(
            device_id=window.device_id,
            channel=window.channel,
            score=result.score,
            confidence=result.confidence,
            quality=DataQuality.GOOD,
            feature_values=vector.as_dict(),
            unit=window.unit,
        )
        event = engine.evaluate(context, model_version=result.model_version)
        if event is not None:
            events.append(event)

    print(f"\n[5] diagnostics   {len(events)} health events from {len(windows)} windows")
    for event in events[:3]:
        print(f"    - {event.severity.value:<8} conf={event.confidence:.3f} "
              f"unc={event.uncertainty:.3f}  {event.recommendation[:60]}...")

    score = compute_health_score(events)
    print(f"\n[6] health score  {score.score:.1f} / 100 "
          f"(band {score.lower:.1f}-{score.upper:.1f}, conf {score.confidence:.2f})")

    print("\nNote: this demo detects a deviation from baseline. It does not diagnose a")
    print("      specific mechanical fault — no validated fault evidence is bundled.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
