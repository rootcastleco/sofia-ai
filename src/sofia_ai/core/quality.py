"""Explicit data-quality representation.

Bad data must never silently become a confident machine diagnosis. Every sample
carries a :class:`DataQuality`, every detector can see it, and every health event
records the quality it was derived from.

The :func:`quality_confidence_factor` mapping is the single place where quality is
translated into confidence attenuation, so the behaviour is testable and auditable.
"""

from __future__ import annotations

from enum import StrEnum

__all__ = [
    "QUALITY_CONFIDENCE_FACTOR",
    "DataQuality",
    "combine_quality",
    "is_usable",
    "quality_confidence_factor",
    "worst_quality",
]


class DataQuality(StrEnum):
    """Quality state of a telemetry sample or window."""

    GOOD = "GOOD"
    """Within range, fresh, ordered, and finite."""

    STALE = "STALE"
    """Older than the configured freshness window."""

    MISSING = "MISSING"
    """Expected but not received (gap-filled placeholder)."""

    INVALID = "INVALID"
    """Failed validation: NaN, inf, wrong type, or unparseable."""

    OUT_OF_RANGE = "OUT_OF_RANGE"
    """Finite but outside the configured engineering range."""

    DUPLICATE = "DUPLICATE"
    """Repeat of an earlier timestamp/sequence number."""

    ESTIMATED = "ESTIMATED"
    """Interpolated, reconstructed, or otherwise not directly measured."""

    UNSYNCHRONIZED = "UNSYNCHRONIZED"
    """Clock not synchronized; event time is not trustworthy."""


#: Deterministic attenuation applied to detector confidence by data quality.
QUALITY_CONFIDENCE_FACTOR: dict[DataQuality, float] = {
    DataQuality.GOOD: 1.0,
    DataQuality.ESTIMATED: 0.8,
    DataQuality.STALE: 0.6,
    DataQuality.UNSYNCHRONIZED: 0.5,
    DataQuality.MISSING: 0.3,
    DataQuality.OUT_OF_RANGE: 0.3,
    DataQuality.DUPLICATE: 0.2,
    DataQuality.INVALID: 0.0,
}

#: Severity ordering used by :func:`worst_quality`.
_QUALITY_ORDER: tuple[DataQuality, ...] = (
    DataQuality.GOOD,
    DataQuality.ESTIMATED,
    DataQuality.UNSYNCHRONIZED,
    DataQuality.STALE,
    DataQuality.OUT_OF_RANGE,
    DataQuality.DUPLICATE,
    DataQuality.MISSING,
    DataQuality.INVALID,
)


def quality_confidence_factor(quality: DataQuality) -> float:
    """Return the multiplicative confidence attenuation for a quality state."""
    return QUALITY_CONFIDENCE_FACTOR[quality]


def worst_quality(qualities: list[DataQuality] | tuple[DataQuality, ...]) -> DataQuality:
    """Return the lowest-trust quality present in a collection."""
    if not qualities:
        return DataQuality.GOOD
    index = max(_QUALITY_ORDER.index(q) for q in qualities)
    return _QUALITY_ORDER[index]


def combine_quality(qualities: list[DataQuality] | tuple[DataQuality, ...]) -> DataQuality:
    """Alias of :func:`worst_quality`; a window inherits its worst sample."""
    return worst_quality(qualities)


def is_usable(quality: DataQuality) -> bool:
    """Whether a sample of this quality may contribute to inference at all."""
    return quality not in (DataQuality.INVALID, DataQuality.MISSING)
