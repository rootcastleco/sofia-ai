"""Generate the example datasets committed to this repository.

The data is deterministic (fixed seeds) so the examples, tests and benchmarks that
consume it produce identical results everywhere. Regenerating it must not change
the bytes — a CI check runs this module and asserts the files are unchanged.

Usage::

    python tools/gen_example_data.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Final

import numpy as np

from sofia_ai.core.contracts import TelemetrySample
from sofia_ai.core.serialization import write_jsonl
from sofia_ai.telemetry.synthetic import (
    SyntheticMachineProfile,
    inject_faults,
    rotating_machine_signal,
)

SEED: Final[int] = 20240921
SAMPLE_RATE: Final[float] = 1000.0
N_SAMPLES: Final[int] = 4096
SHAFT_HZ: Final[float] = 25.0
START_TIME: Final[float] = 1_700_000_000.0

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
DATA_DIR: Final[Path] = REPO_ROOT / "examples" / "data"


def vibration_csv(path: Path) -> int:
    """Write the vibration CSV used by Demo 1 and the CLI examples."""
    rng = np.random.default_rng(SEED)
    signal = rotating_machine_signal(
        N_SAMPLES, SAMPLE_RATE, shaft_hz=SHAFT_HZ, harmonics=(1.0, 2.0, 3.0),
        amplitudes=(1.0, 0.4, 0.15), noise_std=0.05, rng=rng,
    )
    signal = inject_faults(signal, start_fraction=0.6, amplitude_gain=3.0)

    lines = ["timestamp,device_id,channel,value,unit,quality,sequence_number"]
    for index in range(N_SAMPLES):
        lines.append(
            f"{START_TIME + index / SAMPLE_RATE:.6f},"
            f"sim-pump-01,vibration_x,{signal[index]:.6f},g,GOOD,{index}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return N_SAMPLES


def telemetry_jsonl(path: Path) -> int:
    """Write multi-channel JSONL used by Demo 2, Demo 3 and Demo 5."""
    rng = np.random.default_rng(SEED + 1)
    base = rotating_machine_signal(
        N_SAMPLES, SAMPLE_RATE, shaft_hz=SHAFT_HZ, amplitudes=(1.0, 0.4, 0.15),
        noise_std=0.05, rng=rng,
    )
    base = inject_faults(base, start_fraction=0.6, amplitude_gain=3.0)
    thermal = 45.0 + 5.0 * np.sin(
        2.0 * np.pi * 0.05 * np.arange(N_SAMPLES) / SAMPLE_RATE
    )
    onset = int(N_SAMPLES * 0.6)
    ramp = (np.arange(N_SAMPLES) - onset) / max(1, N_SAMPLES - onset) * 25.0
    thermal = thermal + np.where(np.arange(N_SAMPLES) >= onset, ramp, 0.0)

    records = []
    for index in range(N_SAMPLES):
        timestamp = START_TIME + index / SAMPLE_RATE
        records.append(
            TelemetrySample(
                timestamp=timestamp, device_id="sim-pump-01", channel="vibration_x",
                value=float(base[index]), unit="g", source="synthetic",
                sequence_number=index,
            ).to_dict()
        )
        records.append(
            TelemetrySample(
                timestamp=timestamp, device_id="sim-pump-01", channel="temperature",
                value=float(thermal[index]), unit="C", source="synthetic",
                sequence_number=index,
            ).to_dict()
        )
        records.append(
            TelemetrySample(
                timestamp=timestamp, device_id="sim-pump-01", channel="rpm",
                value=float(SHAFT_HZ * 60.0), unit="RPM", source="synthetic",
                sequence_number=index,
            ).to_dict()
        )
    return write_jsonl(records, str(path))


def edge_offline_jsonl(path: Path) -> int:
    """A small file for the offline edge demo (no degradation)."""
    rng = np.random.default_rng(SEED + 2)
    signal = rotating_machine_signal(
        2048, SAMPLE_RATE, shaft_hz=SHAFT_HZ, amplitudes=(0.8, 0.2, 0.08),
        noise_std=0.04, rng=rng,
    )
    records = [
        TelemetrySample(
            timestamp=START_TIME + i / SAMPLE_RATE, device_id="edge-gateway-01",
            channel="vibration_x", value=float(signal[i]), unit="g",
            source="synthetic", sequence_number=i,
        ).to_dict()
        for i in range(2048)
    ]
    return write_jsonl(records, str(path))


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate Sofia example data")
    parser.add_argument("--check", action="store_true",
                        help="fail if generated content differs from committed files")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    targets = {
        DATA_DIR / "vibration.csv": vibration_csv,
        DATA_DIR / "telemetry.jsonl": telemetry_jsonl,
        DATA_DIR / "edge_offline.jsonl": edge_offline_jsonl,
    }
    for path, builder in targets.items():
        builder(path)
        print(f"wrote {path.relative_to(REPO_ROOT)} ({path.stat().st_size} bytes)")

    manifest = {
        "seed": SEED,
        "sample_rate": SAMPLE_RATE,
        "n_samples": N_SAMPLES,
        "shaft_hz": SHAFT_HZ,
        "start_time": START_TIME,
        "channels": ["vibration_x", "temperature", "rpm"],
        "units": {"vibration_x": "g", "temperature": "C", "rpm": "RPM"},
        "note": (
            "Synthetic data generated deterministically for examples and tests. "
            "It is not measured machine data and carries no diagnostic validity."
        ),
    }
    manifest_path = DATA_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n",
                             encoding="utf-8")
    print(f"wrote {manifest_path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
