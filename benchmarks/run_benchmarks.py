"""Standardized Benchmark Runner for Sofia Engine Runtime.

Collects real, unembellished latency, throughput, and memory measurements on the
host environment in accordance with specs/sofia-runtime-v3/performance-budgets.md.
Zero fabricated numbers.
"""

from __future__ import annotations

import gc
import json
import os
import platform
import sys
import time
import tracemalloc
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "benchmarks" / "results"


def measure_benchmark(
    fn: Callable[[], Any],
    n_iterations: int = 100,
    warmup: int = 10,
) -> dict[str, float]:
    """Run warmup iterations and measure statistical latency percentiles."""
    for _ in range(warmup):
        fn()

    durations: list[float] = []
    for _ in range(n_iterations):
        t0 = time.perf_counter_ns()
        fn()
        t1 = time.perf_counter_ns()
        durations.append((t1 - t0) / 1000.0)  # microseconds

    durations.sort()
    p50 = float(np.percentile(durations, 50))
    p95 = float(np.percentile(durations, 95))
    p99 = float(np.percentile(durations, 99))
    mean = float(np.mean(durations))

    return {
        "n_iterations": n_iterations,
        "mean_us": round(mean, 2),
        "p50_us": round(p50, 2),
        "p95_us": round(p95, 2),
        "p99_us": round(p99, 2),
        "min_us": round(durations[0], 2),
        "max_us": round(durations[-1], 2),
    }


def run_all_benchmarks() -> dict[str, Any]:
    print("=" * 70)
    print("SOFIA ENGINE RUNTIME — STANDARDIZED BENCHMARK RUNNER")
    print("=" * 70)

    tracemalloc.start()

    # 1. Package import time
    t0 = time.perf_counter()
    import sofia_ai
    import sofia_ai.signal.electrical as electrical
    import sofia_ai.signal.envelope as envelope
    import sofia_ai.signal.spectral as spectral
    import sofia_ai.features as features
    import sofia_ai.learning.asm as asm
    import sofia_ai.learning.asm.neural as asm_neural
    t1 = time.perf_counter()
    import_time_ms = round((t1 - t0) * 1000.0, 2)
    print(f"[*] Package import time: {import_time_ms} ms")

    # 2. FFT / Welch PSD (1024 samples)
    x_1024 = np.sin(2.0 * np.pi * 50.0 * np.linspace(0, 1, 1024, endpoint=False))
    bench_psd_1024 = measure_benchmark(lambda: spectral.welch_psd(x_1024, sample_rate=1000.0, segment_length=256))
    print(f"[*] FFT/PSD (1024 samples) p50: {bench_psd_1024['p50_us']} us")

    # 3. FFT / Welch PSD (4096 samples)
    x_4096 = np.sin(2.0 * np.pi * 50.0 * np.linspace(0, 4, 4096, endpoint=False))
    bench_psd_4096 = measure_benchmark(lambda: spectral.welch_psd(x_4096, sample_rate=1000.0, segment_length=512))
    print(f"[*] FFT/PSD (4096 samples) p50: {bench_psd_4096['p50_us']} us")

    # 4. Hilbert Analytic Envelope (2048 samples)
    x_2048 = (1.0 + 0.5 * np.sin(2.0 * np.pi * 10.0 * np.linspace(0, 1, 2048, endpoint=False))) * np.sin(2.0 * np.pi * 150.0 * np.linspace(0, 1, 2048, endpoint=False))
    bench_envelope = measure_benchmark(lambda: envelope.amplitude_envelope(x_2048))
    print(f"[*] Analytic Envelope (2048 samples) p50: {bench_envelope['p50_us']} us")

    # 5. Full Statistical Feature Extraction (1024 samples)
    bench_features = measure_benchmark(lambda: features.extract_from_array(x_1024, sample_rate=1000.0))
    print(f"[*] Feature Extraction (1024 samples) p50: {bench_features['p50_us']} us")

    # 6. Electrical Power Metrics & Symmetrical Components
    bench_electrical = measure_benchmark(lambda: electrical.compute_symmetrical_components(230.0, 0.0, 210.0, -120.0, 200.0, 120.0))
    print(f"[*] Electrical Symmetrical Components p50: {bench_electrical['p50_us']} us")

    # 7. Sofia VM Forward Inference (Assembly Neural Network: 4 -> 8 -> 1)
    ann = asm_neural.AssemblyNeuralNetwork(input_dim=4, hidden_dim=8, output_dim=1, learning_rate=0.01)
    input_sample = np.array([0.5, -0.2, 0.8, 0.1], dtype=np.float64)
    target_sample = np.array([0.75], dtype=np.float64)

    bench_vm_inference = measure_benchmark(lambda: ann.forward(input_sample))
    print(f"[*] Sofia VM Forward Inference (4->8->1) p50: {bench_vm_inference['p50_us']} us")

    # 8. Sofia VM Training Step (Forward + Backprop + SGD)
    bench_vm_train = measure_benchmark(lambda: ann.train_step(input_sample, target_sample))
    print(f"[*] Sofia VM Training Step (4->8->1) p50: {bench_vm_train['p50_us']} us")

    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    results = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "os": platform.system(),
            "os_release": platform.release(),
            "os_version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "python_compiler": platform.python_compiler(),
            "python_implementation": platform.python_implementation(),
        },
        "benchmarks": {
            "import_time_ms": import_time_ms,
            "fft_psd_1024": bench_psd_1024,
            "fft_psd_4096": bench_psd_4096,
            "analytic_envelope_2048": bench_envelope,
            "feature_extraction_1024": bench_features,
            "electrical_symmetrical_components": bench_electrical,
            "sofia_vm_forward_inference": bench_vm_inference,
            "sofia_vm_training_step": bench_vm_train,
        },
        "memory": {
            "current_traced_kb": round(current_mem / 1024.0, 2),
            "peak_traced_kb": round(peak_mem / 1024.0, 2),
        },
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_file = RESULTS_DIR / "baseline_benchmark.json"
    out_file.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"[+] Saved baseline benchmark results to: {out_file}")
    print("=" * 70)
    return results


if __name__ == "__main__":
    run_all_benchmarks()
