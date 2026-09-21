"""Benchmark infrastructure: timing, percentile reporting and environment capture.

Design rules (mission §23):

* every reported number states hardware, OS, Python version, Sofia version/commit,
  input size and configuration;
* distributions are reported as p50/p95/p99, not just means;
* regression thresholds are derived from a measured baseline, never invented;
* benchmarks never claim "real-time" — they report measured distributions.
"""

from __future__ import annotations

import json
import os
import platform
import statistics
import subprocess
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Final

import numpy as np

from .. import __version__

__all__ = [
    "BenchmarkSuite",
    "Measurement",
    "current_commit",
    "environment",
    "format_report",
    "measure",
    "percentiles",
]

_WARMUP_RUNS: Final[int] = 3


@dataclass(frozen=True, slots=True)
class Measurement:
    """A single measured quantity.

    Attributes:
        name: Benchmark name.
        iterations: Number of measured runs.
        input_size: Declared input size (samples, windows or records).
        unit: Unit of the measured values (``s``, ``bytes``, ``records/s``).
        p50/p95/p99: Percentiles over measured runs.
        mean/min/max: Descriptive statistics.
        config: The configuration the measurement was taken under.
    """

    name: str
    iterations: int
    input_size: int
    unit: str = "s"
    p50: float = 0.0
    p95: float = 0.0
    p99: float = 0.0
    mean: float = 0.0
    min: float = 0.0
    max: float = 0.0
    std: float = 0.0
    config: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "iterations": self.iterations,
            "input_size": self.input_size,
            "unit": self.unit,
            "p50": self.p50,
            "p95": self.p95,
            "p99": self.p99,
            "mean": self.mean,
            "min": self.min,
            "max": self.max,
            "std": self.std,
            "config": dict(self.config),
        }


def percentiles(values: list[float] | np.ndarray) -> tuple[float, float, float]:
    """Return ``(p50, p95, p99)`` for a sample."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return 0.0, 0.0, 0.0
    p50, p95, p99 = np.percentile(arr, [50, 95, 99])
    return float(p50), float(p95), float(p99)


def measure(
    name: str,
    fn: Callable[[], Any],
    *,
    iterations: int = 50,
    input_size: int = 0,
    unit: str = "s",
    config: Mapping[str, Any] | None = None,
) -> Measurement:
    """Time ``fn`` for ``iterations`` runs after a short warm-up.

    Warm-up is excluded from the statistics because it measures import/JIT effects
    rather than steady-state behaviour.
    """
    for _ in range(_WARMUP_RUNS):
        fn()
    samples: list[float] = []
    for _ in range(max(1, int(iterations))):
        started = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - started)
    p50, p95, p99 = percentiles(samples)
    return Measurement(
        name=name,
        iterations=len(samples),
        input_size=int(input_size),
        unit=unit,
        p50=p50,
        p95=p95,
        p99=p99,
        mean=float(statistics.fmean(samples)) if samples else 0.0,
        min=float(min(samples)) if samples else 0.0,
        max=float(max(samples)) if samples else 0.0,
        std=float(statistics.pstdev(samples)) if len(samples) > 1 else 0.0,
        config=dict(config or {}),
    )


def current_commit() -> str:
    """Best-effort git commit. Returns ``"unknown"`` outside a checkout."""
    import shutil

    git = shutil.which("git")
    if git is None:
        return "unknown"
    try:
        result = subprocess.run(  # noqa: S603 - argv is a fixed, trusted command
            [git, "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5, check=False,
        )
    except (OSError, subprocess.SubprocessError):  # pragma: no cover
        return "unknown"
    if result.returncode != 0:
        return "unknown"
    return result.stdout.strip()[:12] or "unknown"


def environment() -> dict[str, Any]:
    """Capture the machine and runtime the benchmark ran on."""
    logical = os.cpu_count() or 1
    return {
        "sofia_version": __version__,
        "commit": current_commit(),
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "os": f"{platform.system()} {platform.release()}",
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "cpu_count_logical": logical,
        "numpy": np.__version__,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


@dataclass(slots=True)
class BenchmarkSuite:
    """Collects measurements and renders a reproducible report."""

    name: str = "sofia"
    results: list[Measurement] = field(default_factory=list)
    env: dict[str, Any] = field(default_factory=environment)

    def add(self, measurement: Measurement) -> Measurement:
        self.results.append(measurement)
        return measurement

    def run(self, name: str, fn: Callable[[], Any], **kwargs: Any) -> Measurement:
        return self.add(measure(name, fn, **kwargs))

    def to_dict(self) -> dict[str, Any]:
        return {
            "suite": self.name,
            "environment": self.env,
            "measurements": [m.to_dict() for m in self.results],
            "disclaimer": (
                "Numbers are valid only for the environment recorded above. "
                "They are not portability claims."
            ),
        }


def format_report(payload: Mapping[str, Any]) -> str:
    """Render a benchmark report as a fixed-width text table."""
    env = dict(payload.get("environment", {}) or {})
    lines: list[str] = [
        "SOFIA ENGINE BENCHMARK",
        "=" * 78,
        f"suite      : {payload.get('suite')}",
        f"sofia      : {env.get('sofia_version')} @ {env.get('commit')}",
        f"python     : {env.get('python')} ({env.get('python_implementation')})",
        f"os         : {env.get('os')} / {env.get('machine')}",
        f"cpu(logical): {env.get('cpu_count_logical')}",
        f"numpy      : {env.get('numpy')}",
        f"utc        : {env.get('timestamp_utc')}",
        "",
        f"{'benchmark':<34}{'n':>5}{'size':>9}{'p50':>12}{'p95':>12}{'p99':>12}",
        "-" * 78,
    ]
    for item in payload.get("measurements", []):
        unit = str(item.get("unit", "s"))
        scale, suffix = _scale_for(unit)
        lines.append(
            f"{item.get('name')!s:<34}"
            f"{int(item.get('iterations', 0)):>5}"
            f"{int(item.get('input_size', 0)):>9}"
            f"{item.get('p50', 0.0) * scale:>11.3f}{suffix:<1}"
            f"{item.get('p95', 0.0) * scale:>11.3f}{suffix:<1}"
            f"{item.get('p99', 0.0) * scale:>11.3f}{suffix:<1}"
        )
    lines.extend(["-" * 78, str(payload.get("disclaimer", ""))])
    return "\n".join(lines)


def _scale_for(unit: str) -> tuple[float, str]:
    return {
        "s": (1e3, "m"),
        "ms": (1.0, "m"),
        "us": (1e-3, "m"),
    }.get(unit, (1.0, ""))


def load_baseline(path: str) -> dict[str, Any]:
    """Load a previously recorded baseline for regression comparison."""
    with open(path, encoding="utf-8") as handle:
        return dict(json.load(handle))


def compare_to_baseline(
    payload: Mapping[str, Any], baseline: Mapping[str, Any], *, tolerance: float = 0.5
) -> list[dict[str, Any]]:
    """Compare p50 values against a baseline.

    Args:
        tolerance: Fractional regression allowed before a measurement is flagged.
            Defaults to 50 %, which is deliberately loose: CI machines vary.
    """
    old = {
        str(m.get("name")): float(m.get("p50", 0.0))
        for m in baseline.get("measurements", [])
    }
    regressions: list[dict[str, Any]] = []
    for measurement in payload.get("measurements", []):
        name = str(measurement.get("name"))
        previous = old.get(name)
        if previous is None:
            continue
        current = float(measurement.get("p50", 0.0))
        if previous > 0 and current > previous * (1.0 + tolerance):
            regressions.append({
                "name": name,
                "baseline_p50": previous,
                "current_p50": current,
                "ratio": current / previous,
            })
    return regressions
