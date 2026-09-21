"""Shared pytest fixtures and global determinism guards.

Two autouse fixtures enforce the determinism policy (mission §21):

* no test may leave the global ``random`` or ``numpy.random`` state mutated in a
  way that changes subsequent behaviour,
* no test may rely on a wall-clock value; :class:`FixedTimeSource` is provided.
"""

from __future__ import annotations

import random
from collections.abc import Iterator

import numpy as np
import pytest

from sofia_ai.core.time import FixedTimeSource


@pytest.fixture(autouse=True)
def _deterministic_globals() -> Iterator[None]:
    """Snapshot global RNG state and require tests not to reseed it."""
    random_state = random.getstate()
    numpy_state = np.random.get_state()
    yield
    # A test may legitimately use the global RNG, but it must not reseed it:
    # reseeding one test would change the behaviour of every later test.
    assert random.getstate() == random_state, (
        "test reseeded the global random module; inject a seed instead"
    )
    assert _states_equal(np.random.get_state(), numpy_state), (
        "test reseeded numpy.random; use numpy.random.default_rng(seed) instead"
    )


def _states_equal(a: object, b: object) -> bool:
    if not isinstance(a, tuple) or not isinstance(b, tuple):
        return a == b
    if len(a) != len(b):
        return False
    for x, y in zip(a, b, strict=True):
        if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
            if x.shape != y.shape or not np.array_equal(x, y):
                return False
        elif x != y:
            return False
    return True


@pytest.fixture
def clock() -> FixedTimeSource:
    """Deterministic clock."""
    return FixedTimeSource()


@pytest.fixture
def rng() -> np.random.Generator:
    """Owned RNG. Never touches the global state."""
    return np.random.default_rng(1234)


@pytest.fixture
def sine() -> np.ndarray:
    """A deterministic 25 Hz test signal at 1 kHz, 1024 samples."""
    t = np.arange(1024, dtype=np.float64) / 1000.0
    return np.sin(2.0 * np.pi * 25.0 * t)


@pytest.fixture
def samples() -> list:
    """A deterministic list of TelemetrySample objects."""
    from sofia_ai.core.contracts import TelemetrySample

    values = np.sin(2.0 * np.pi * 25.0 * np.arange(256, dtype=np.float64) / 1000.0)
    return [
        TelemetrySample(
            timestamp=1_700_000_000.0 + i / 1000.0,
            device_id="test-device",
            channel="vibration_x",
            value=float(v),
            unit="g",
            source="test",
            sequence_number=i,
        )
        for i, v in enumerate(values)
    ]
