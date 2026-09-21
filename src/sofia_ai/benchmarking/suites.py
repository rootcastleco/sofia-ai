"""The Sofia benchmark suite.

Run with::

    sofia benchmark
    sofia benchmark --suite features --iterations 100 --json

Suites:
    ``ingest``     telemetry ingest throughput
    ``features``   feature-extraction latency
    ``inference``  detector inference latency
    ``pipeline``   end-to-end edge runtime latency
    ``quantum``    quantum-inspired simulation vs a classical baseline
    ``all``        every suite

Every number is reported with the environment that produced it. No portability or
real-time claim is made from these measurements.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from ..features.extractor import extract_from_array
from ..inference import build_manifest_for, create_backend
from ..inference.detectors import DetectorConfig
from ..signal.windowing import WindowSpec, frame_signal
from ..telemetry.file_sources import InMemorySource
from ..telemetry.synthetic import (
    SyntheticMachineProfile,
    SyntheticVibrationSource,
    rotating_machine_signal,
)
from .core import BenchmarkSuite, Measurement

__all__ = ["DEFAULT_SAMPLE_RATE", "DEFAULT_WINDOW", "SUITES", "run_suite"]

DEFAULT_WINDOW: int = 1024
DEFAULT_SAMPLE_RATE: float = 1000.0
DEFAULT_ITERATIONS: int = 50

SUITES: tuple[str, ...] = ("ingest", "features", "inference", "pipeline", "quantum")


def run_suite(suite: str = "all", iterations: int = DEFAULT_ITERATIONS,
              *, window: int = DEFAULT_WINDOW,
              sample_rate: float = DEFAULT_SAMPLE_RATE) -> dict[str, Any]:
    """Run one or all benchmark suites and return the report payload."""
    bench = BenchmarkSuite(name="sofia-engine")
    selected = SUITES if suite in ("all", "", None) else (str(suite),)
    runners: dict[str, Callable[[BenchmarkSuite, int, int, float], None]] = {
        "ingest": _suite_ingest,
        "features": _suite_features,
        "inference": _suite_inference,
        "pipeline": _suite_pipeline,
        "quantum": _suite_quantum,
    }
    for name in selected:
        runner = runners.get(name)
        if runner is None:
            raise ValueError(f"unknown benchmark suite {name!r}; known: {SUITES}")
        runner(bench, iterations, window, sample_rate)
    return bench.to_dict()


# ---------------------------------------------------------------------------
# suites
# ---------------------------------------------------------------------------


def _suite_ingest(bench: BenchmarkSuite, iterations: int, window: int,
                  sample_rate: float) -> None:
    """Ingest throughput: reading and validating batches of samples."""
    source = SyntheticVibrationSource(SyntheticMachineProfile(seed=11), n_samples=100_000,
                                      batch_size=1024)
    source.open()
    config = {"batch_size": 1024, "source": "synthetic"}

    def read_batch() -> int:
        source._cursor = 0
        return len(source.read(max_records=1024))

    bench.run("ingest.batch_1024_samples", read_batch, iterations=iterations,
              input_size=1024, unit="s", config=config)

    samples = source.read(max_records=4096)
    memory = InMemorySource(samples, batch_size=256)
    memory.open()

    def read_memory() -> int:
        memory._cursor = 0
        return len(memory.read(max_records=256))

    bench.run("ingest.in_memory_256", read_memory, iterations=iterations,
              input_size=256, unit="s", config={"source": "memory"})
    source.close()


def _suite_features(bench: BenchmarkSuite, iterations: int, window: int,
                    sample_rate: float) -> None:
    """Feature extraction latency at several window lengths."""
    for length in (256, 1024, 4096):
        signal = rotating_machine_signal(length, sample_rate, shaft_hz=25.0,
                                         rng=np.random.default_rng(3))

        def extract_signal(sig: np.ndarray = signal) -> int:
            return extract_from_array(sig, sample_rate).size

        bench.run(f"features.extract_window_{length}", extract_signal,
                  iterations=iterations,
                  input_size=length, unit="s",
                  config={"window": length, "sample_rate": sample_rate,
                          "extractor": "sofia.core.vibration"})

    signal = rotating_machine_signal(window, sample_rate, rng=np.random.default_rng(3))
    spec = WindowSpec(length=256, hop=128)
    bench.run("features.frame_signal", lambda: frame_signal(signal, spec),
              iterations=iterations, input_size=window, unit="s",
              config={"window": 256, "hop": 128})


def _suite_inference(bench: BenchmarkSuite, iterations: int, window: int,
                     sample_rate: float) -> None:
    """Detector inference latency, measured on a pre-warmed detector."""
    signal = rotating_machine_signal(window, sample_rate, rng=np.random.default_rng(5))
    features = extract_from_array(signal, sample_rate)

    for name in ("threshold", "zscore", "mad", "ewma", "iqr", "cusum"):
        backend: Any = create_backend(name)
        if hasattr(backend, "config"):  # only detector backends carry DetectorConfig
            backend.config = DetectorConfig(feature="rms", window_size=64,
                                            warmup=8, min_samples=8, threshold=3.0)
        backend.load(build_manifest_for(
            backend, model_id=f"bench-{name}",
            metadata={"high": 2.0} if name == "threshold" else None,
        ))
        # Warm the rolling baseline so the measurement is steady-state.
        for _ in range(16):
            backend.infer(features)

        def _infer_once(bound: Any = backend) -> Any:
            return bound.infer(features)

        bench.run(f"inference.{name}", _infer_once,
                  iterations=iterations, input_size=features.size, unit="s",
                  config={"backend": name, "features": features.size})


def _suite_pipeline(bench: BenchmarkSuite, iterations: int, window: int,
                    sample_rate: float) -> None:
    """End-to-end window → features → inference latency."""
    from ..edge import EdgeRuntime, RuntimeConfig

    def build_runtime() -> EdgeRuntime:
        source = SyntheticVibrationSource(SyntheticMachineProfile(seed=17),
                                          n_samples=8192, batch_size=4096)
        backend: Any = create_backend("mad")
        backend.config = DetectorConfig(feature="rms", window_size=32, warmup=8,
                                        min_samples=8, threshold=3.0)
        backend.load(build_manifest_for(backend, model_id="bench-pipeline"))
        runtime = EdgeRuntime(
            source=source, backend=backend,
            config=RuntimeConfig(window_length=512, window_hop=256,
                                 sample_rate=sample_rate, buffer_capacity=16384,
                                 max_windows_per_tick=8),
        )
        runtime.start()
        return runtime

    runtime = build_runtime()
    bench.run("pipeline.tick", runtime.tick, iterations=iterations, input_size=512,
              unit="s", config={"window": 512, "hop": 256, "detector": "mad"})
    runtime.stop()

    def full_run() -> int:
        rt = build_runtime()
        events = rt.run(max_ticks=8, batch_size=4096)
        rt.stop()
        return len(events)

    bench.run("pipeline.run_8_ticks", full_run, iterations=max(5, iterations // 10),
              input_size=8192, unit="s",
              config={"ticks": 8, "window": 512, "hop": 256})


def _suite_quantum(bench: BenchmarkSuite, iterations: int, window: int,
                   sample_rate: float) -> None:
    """Quantum-inspired simulation against a classical baseline.

    This suite exists to test the claim, not to support it. If the quantum-inspired
    path is slower and no more accurate, the report says so.
    """
    from ..experimental.quantum import QuantumNeuralEngine, classical_baseline

    dimension = 64
    data = np.asarray(np.random.default_rng(9).normal(size=dimension), dtype=np.float64)
    engine = QuantumNeuralEngine(num_qubits=6, entanglement_depth=2, seed=4)

    bench.run("quantum.forward_dim64", lambda: engine.forward(data),
              iterations=iterations, input_size=dimension, unit="s",
              config={"num_qubits": 6, "depth": 2, "dimension": dimension})

    bench.run("quantum.classical_baseline_dim64",
              lambda: classical_baseline(data, dimension, seed=4),
              iterations=iterations, input_size=dimension, unit="s",
              config={"dimension": dimension})

    payload: dict[str, Any] = {
        "engine_output": [float(v) for v in engine.forward(data)[:8]],
        "baseline_output": [float(v) for v in classical_baseline(data, dimension, seed=4)[:8]],
    }
    bench.add(Measurement(
        name="quantum.output_sample",
        iterations=1,
        input_size=dimension,
        unit="dimensionless",
        mean=0.0,
        config=payload,
    ))

