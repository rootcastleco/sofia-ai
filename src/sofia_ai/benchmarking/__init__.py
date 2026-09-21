"""Benchmark suite. Every published number must come from here."""

from __future__ import annotations

from .core import (
    BenchmarkSuite,
    Measurement,
    compare_to_baseline,
    current_commit,
    environment,
    format_report,
    load_baseline,
    measure,
    percentiles,
)
from .suites import DEFAULT_ITERATIONS, DEFAULT_SAMPLE_RATE, DEFAULT_WINDOW, SUITES, run_suite

__all__ = [
    "DEFAULT_ITERATIONS",
    "DEFAULT_SAMPLE_RATE",
    "DEFAULT_WINDOW",
    "SUITES",
    "BenchmarkSuite",
    "Measurement",
    "compare_to_baseline",
    "current_commit",
    "environment",
    "format_report",
    "load_baseline",
    "measure",
    "percentiles",
    "run_suite",
]
