#!/usr/bin/env python3
"""Demo 4 — Custom model adapter.

Shows how to integrate your own model without editing the Sofia core:

1. as a plain callable wrapped by ``CallableBackend``, and
2. as a registered backend in the model registry.

The custom model receives a :class:`FeatureVector` and returns
``(score, confidence, evidence)``. It inherits the standard guarantees: manifest
validation, input schema checking, latency measurement, typed failure handling and
data-quality confidence attenuation.

Usage::

    python examples/04_custom_model_adapter.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from sofia_ai.core.contracts import FeatureVector
from sofia_ai.features import extract_from_array
from sofia_ai.inference import (
    CallableBackend,
    build_manifest_for,
    create_backend,
    model_registry,
    register_model_backend,
)
from sofia_ai.telemetry import CsvSource

DATA = Path(__file__).resolve().parent / "data" / "vibration.csv"
SAMPLE_RATE = 1000.0


def energy_and_impulsiveness_model(
    features: FeatureVector,
) -> tuple[float, float, dict[str, Any]]:
    """A user-supplied detector combining overall level and impulsiveness.

    Two features are used because they fail differently:

    * ``rms`` rises with overall vibration energy — it catches the 3x amplitude
      increase in the degraded section of the demo data;
    * ``crest_factor`` and ``kurtosis`` are **scale-invariant**, so a uniform
      amplitude gain does *not* change them. They catch impacting, which a level
      detector cannot see. Including both makes the detector sensitive to either
      failure mode.

    Returns:
        ``(score, confidence, evidence)``.
    """
    table = features.as_dict()
    rms = table.get("rms", 0.0)
    crest = table.get("crest_factor", 0.0)
    kurt = table.get("kurtosis", 0.0)
    level_term = max(0.0, rms - 1.0)
    impulse_term = max(0.0, crest - 1.8) + 0.5 * max(0.0, kurt - 3.0)
    score = level_term + impulse_term
    confidence = min(1.0, 0.3 + 0.35 * score)
    evidence = {
        "rms": round(rms, 4),
        "crest_factor": round(crest, 4),
        "kurtosis": round(kurt, 4),
        "level_term": round(level_term, 4),
        "impulse_term": round(impulse_term, 4),
        "rule": "rms>1.0 or crest>1.8 or kurtosis>3",
    }
    return score, confidence, evidence


def main() -> int:
    print("=" * 72)
    print("SOFIA ENGINE — Demo 4: custom model adapter")
    print("=" * 72)

    source = CsvSource(str(DATA), device_id="sim-pump-01", unit="g")
    with source:
        samples = []
        while True:
            batch = source.read(max_records=4096)
            if not batch:
                break
            samples.extend(batch)
    signal = [s.value for s in samples]
    print(f"\n[1] data          {len(signal)} samples from {DATA.name}")

    # 1. Custom model as a callable ----------------------------------------
    custom = CallableBackend(
        energy_and_impulsiveness_model,
        backend_id="demo-energy-impulse",
        anomaly_threshold=0.5,
    )
    manifest = build_manifest_for(
        custom, model_id="crest-kurtosis", version="0.1.0",
        metadata={"anomaly_threshold": 0.5},
    )
    custom.load(manifest)
    print(f"[2] callable      loaded {manifest.identity} "
          f"(backend={custom.metadata().backend_id})")

    # 2. Registered backend -------------------------------------------------
    register_model_backend(
        "energy_impulse",
        lambda **kw: CallableBackend(energy_and_impulsiveness_model,
                                     backend_id="demo-energy-impulse", **kw),
        overwrite=True,
    )
    from_registry = create_backend("energy_impulse", anomaly_threshold=0.5)
    from_registry.load(manifest)
    print(f"[3] registry      backends now: {', '.join(model_registry.names())}")

    # 3. Run both -----------------------------------------------------------
    baseline = extract_from_array(signal[:512], SAMPLE_RATE)
    degraded = extract_from_array(signal[-512:], SAMPLE_RATE)
    for label, vector in (("baseline", baseline), ("degraded", degraded)):
        result = from_registry.infer(vector)
        print(f"[4] {label:<9} outcome={result.outcome.value:<9} "
              f"score={result.score:.3f} conf={result.confidence:.3f} "
              f"unc={result.uncertainty:.3f} "
              f"latency={result.latency_s * 1e6:.1f}us")
        print(f"    evidence      {result.evidence}")

    print("\nThe custom model received the standard FeatureVector contract and inherited")
    print("manifest validation, schema checking, timing and quality attenuation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
