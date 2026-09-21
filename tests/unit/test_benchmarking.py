"""Benchmark suites must execute end-to-end and return a report payload."""

from __future__ import annotations

import pytest

from sofia_ai.benchmarking.suites import (
    DEFAULT_ITERATIONS,
    SUITES,
    run_suite,
)


@pytest.mark.parametrize("suite", SUITES)
def test_run_suite_each(suite: str) -> None:
    report = run_suite(suite, iterations=2)
    assert report["suite"] == "sofia-engine"
    assert isinstance(report["measurements"], list)
    assert report["disclaimer"]


def test_run_suite_all() -> None:
    report = run_suite("all", iterations=2)
    assert len(report["measurements"]) > 0


def test_unknown_suite_rejected() -> None:
    with pytest.raises(ValueError, match="unknown benchmark suite"):
        run_suite("nope")


def test_default_iterations_is_positive() -> None:
    assert DEFAULT_ITERATIONS > 0
    assert isinstance(DEFAULT_ITERATIONS, int)
