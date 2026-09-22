"""Unit tests for data quality (SOFIA-DQ-001..003)."""

from __future__ import annotations

import pytest

from sofia_ai.core.quality import (
    QUALITY_CONFIDENCE_FACTOR,
    DataQuality,
    combine_quality,
    is_usable,
    quality_confidence_factor,
    worst_quality,
)


def test_all_states_exist() -> None:
    expected = {
        "GOOD", "DEGRADED", "SATURATED", "STALE", "MISSING", "INVALID",
        "OUT_OF_RANGE", "DUPLICATE", "ESTIMATED", "UNSYNCHRONIZED",
    }
    assert {q.value for q in DataQuality} == expected


def test_good_has_full_confidence() -> None:
    assert quality_confidence_factor(DataQuality.GOOD) == 1.0


def test_invalid_has_zero_confidence() -> None:
    assert quality_confidence_factor(DataQuality.INVALID) == 0.0


def test_confidence_is_monotonic_in_trust() -> None:
    order = [DataQuality.GOOD, DataQuality.ESTIMATED, DataQuality.STALE,
             DataQuality.UNSYNCHRONIZED, DataQuality.OUT_OF_RANGE,
             DataQuality.MISSING, DataQuality.DUPLICATE, DataQuality.INVALID]
    factors = [quality_confidence_factor(q) for q in order]
    assert factors == sorted(factors, reverse=True)


def test_every_state_is_mapped() -> None:
    assert set(QUALITY_CONFIDENCE_FACTOR) == set(DataQuality)


def test_worst_quality_propagates() -> None:
    assert worst_quality([DataQuality.GOOD, DataQuality.STALE]) is DataQuality.STALE
    assert worst_quality([DataQuality.INVALID, DataQuality.GOOD]) is DataQuality.INVALID


def test_combine_quality_alias() -> None:
    assert combine_quality([DataQuality.ESTIMATED]) is DataQuality.ESTIMATED


def test_empty_collection_is_good() -> None:
    assert worst_quality([]) is DataQuality.GOOD


def test_is_usable_rejects_invalid_and_missing() -> None:
    assert is_usable(DataQuality.GOOD)
    assert not is_usable(DataQuality.INVALID)
    assert not is_usable(DataQuality.MISSING)


def test_bad_data_cannot_produce_full_confidence() -> None:
    """The core guarantee: bad data must not yield a confident result."""
    for quality in DataQuality:
        if quality is not DataQuality.GOOD:
            assert quality_confidence_factor(quality) < 1.0


@pytest.mark.parametrize("quality", list(DataQuality))
def test_factor_is_a_probability(quality: DataQuality) -> None:
    assert 0.0 <= quality_confidence_factor(quality) <= 1.0
