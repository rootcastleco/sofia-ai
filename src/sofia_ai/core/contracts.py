"""Versioned domain contracts for Sofia Engine.

These dataclasses are the single source of truth for data exchanged between
layers. They are:

* validated at construction — invalid state cannot be constructed,
* explicitly versioned — ``CONTRACT_VERSION`` is embedded in every serialized form,
* serialization-stable — :mod:`sofia_ai.core.serialization` round-trips them,
* finite-only — NaN/inf are rejected, never propagated.

Immutability: contracts that are pure metadata are frozen. :class:`SignalWindow`
carries a mutable ndarray and is therefore mutable, but its metadata is not.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Final

import numpy as np

from .errors import ValidationError
from .quality import DataQuality
from .time import validate_timestamp
from .units import unit_alias
from .validation import (
    as_float_array,
    validate_identifier,
    validate_sample_rate,
    validate_tags,
)

__all__ = [
    "CONTRACT_VERSION",
    "DataQuality",
    "FeatureVector",
    "MachineState",
    "Severity",
    "SignalWindow",
    "TelemetryFrame",
    "TelemetrySample",
    "stable_id",
]

CONTRACT_VERSION: Final[str] = "2.0"


class MachineState(StrEnum):
    """Machine operating state — an explicit input to the policy engine."""

    UNKNOWN = "UNKNOWN"
    OFF = "OFF"
    STARTUP = "STARTUP"
    IDLE = "IDLE"
    RUNNING = "RUNNING"
    DEGRADED = "DEGRADED"
    FAULT = "FAULT"
    MAINTENANCE = "MAINTENANCE"
    EMERGENCY_STOP = "EMERGENCY_STOP"


class Severity(StrEnum):
    """Ordered severity for health events."""

    INFO = "INFO"
    NOTICE = "NOTICE"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"

    @property
    def rank(self) -> int:
        return _SEVERITY_RANK[self]


_SEVERITY_RANK: Final[dict[Severity, int]] = {
    Severity.INFO: 0,
    Severity.NOTICE: 1,
    Severity.WARNING: 2,
    Severity.CRITICAL: 3,
}


def stable_id(*parts: str) -> str:
    """Deterministic short identifier derived from its parts.

    Used for event ids so that identical inputs produce identical identifiers,
    which makes tests and deduplication deterministic.
    """
    digest = hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()
    return digest[:16]


@dataclass(frozen=True, slots=True)
class TelemetrySample:
    """A single measurement from a single channel.

    Attributes:
        timestamp: Event time — when the device produced the sample (epoch seconds, UTC).
        device_id: Equipment identifier.
        channel: Channel/measurement name, e.g. ``vibration_x``.
        value: Engineering value, must be finite.
        unit: Engineering unit symbol, validated against the unit registry.
        source: Transport identifier that produced this sample.
        quality: Data-quality state.
        sequence_number: Device-side monotonic counter, if available.
        tags: Free-form but bounded metadata.
        ingestion_time: When Sofia received the sample (epoch seconds, UTC).
    """

    timestamp: float
    device_id: str
    channel: str
    value: float
    unit: str
    source: str = "unknown"
    quality: DataQuality = DataQuality.GOOD
    sequence_number: int | None = None
    tags: Mapping[str, str] = field(default_factory=dict)
    ingestion_time: float | None = None

    def __post_init__(self) -> None:
        validate_timestamp(self.timestamp, name="timestamp")
        if self.ingestion_time is not None:
            validate_timestamp(self.ingestion_time, name="ingestion_time")
        # frozen dataclass: normalization must go through object.__setattr__.
        # Tags are always validated: a hostile payload must not be able to attach
        # unbounded or control-character metadata to a sample.
        object.__setattr__(self, "tags", dict(validate_tags(self.tags, "tags")))
        normalized_unit = unit_alias(self.unit)
        if normalized_unit != self.unit:
            object.__setattr__(self, "unit", normalized_unit)
        for attr in ("device_id", "channel", "source"):
            object.__setattr__(self, attr, validate_identifier(getattr(self, attr), attr))
        # Accept a plain string for quality so JSON round-trips are lossless.
        if not isinstance(self.quality, DataQuality):
            object.__setattr__(self, "quality", DataQuality(str(self.quality).upper()))
        # Value may be non-finite only when quality explicitly says it is invalid.
        if not math.isfinite(float(self.value)) and self.quality is not DataQuality.INVALID:
            raise ValidationError(
                "Sample value must be finite unless quality is INVALID",
                details={"channel": self.channel, "value": repr(self.value)},
            )

    @property
    def is_valid(self) -> bool:
        return self.quality is DataQuality.GOOD

    def with_quality(self, quality: DataQuality) -> TelemetrySample:
        """Return a copy with a different quality flag."""
        return TelemetrySample(
            timestamp=self.timestamp,
            device_id=self.device_id,
            channel=self.channel,
            value=self.value,
            unit=self.unit,
            source=self.source,
            quality=quality,
            sequence_number=self.sequence_number,
            tags=dict(self.tags),
            ingestion_time=self.ingestion_time,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "device_id": self.device_id,
            "channel": self.channel,
            "value": float(self.value),
            "unit": self.unit,
            "source": self.source,
            "quality": self.quality.value,
            "sequence_number": self.sequence_number,
            "tags": dict(self.tags),
            "ingestion_time": self.ingestion_time,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TelemetrySample:
        return cls(
            timestamp=float(data["timestamp"]),
            device_id=str(data["device_id"]),
            channel=str(data["channel"]),
            value=float(data["value"]),
            unit=str(data["unit"]),
            source=str(data.get("source", "unknown")),
            quality=DataQuality(data.get("quality", DataQuality.GOOD.value)),
            sequence_number=None if data.get("sequence_number") is None
            else int(data["sequence_number"]),
            tags=dict(data.get("tags", {})),
            ingestion_time=None if data.get("ingestion_time") is None
            else float(data["ingestion_time"]),
        )


@dataclass(frozen=True, slots=True)
class TelemetryFrame:
    """A bounded batch of samples sharing a device and a nominal time."""

    device_id: str
    timestamp: float
    samples: tuple[TelemetrySample, ...]
    source: str = "unknown"

    def __post_init__(self) -> None:
        validate_timestamp(self.timestamp, name="timestamp")
        object.__setattr__(self, "device_id", validate_identifier(self.device_id, "device_id"))
        object.__setattr__(self, "source", validate_identifier(self.source, "source"))

    @property
    def channels(self) -> tuple[str, ...]:
        return tuple(s.channel for s in self.samples)

    def to_dict(self) -> dict[str, Any]:
        return {
            "device_id": self.device_id,
            "timestamp": self.timestamp,
            "source": self.source,
            "samples": [s.to_dict() for s in self.samples],
        }


@dataclass(slots=True)
class SignalWindow:
    """A window of uniformly sampled signal values.

    The DSP axis is ``sample_rate`` plus ``start_index``. Wall-clock time is carried
    for provenance only and never used to derive frequency content.
    """

    values: np.ndarray
    sample_rate: float
    device_id: str
    channel: str
    unit: str
    start_index: int = 0
    start_time: float | None = None
    quality: DataQuality = DataQuality.GOOD
    quality_mask: tuple[DataQuality, ...] = ()

    def __post_init__(self) -> None:
        self.values = as_float_array(self.values, name="values")
        self.sample_rate = validate_sample_rate(self.sample_rate)
        self.device_id = validate_identifier(self.device_id, "device_id")
        self.channel = validate_identifier(self.channel, "channel")
        self.unit = unit_alias(self.unit)
        if self.start_index < 0:
            raise ValidationError(
                "start_index must be non-negative", details={"start_index": self.start_index}
            )
        if self.start_time is not None:
            validate_timestamp(self.start_time, name="start_time")
        if self.quality_mask and len(self.quality_mask) != len(self.values):
            raise ValidationError(
                "quality_mask length must match values length",
                details={"mask": len(self.quality_mask), "values": len(self.values)},
            )

    @property
    def length(self) -> int:
        return int(self.values.size)

    @property
    def duration_s(self) -> float:
        return self.length / self.sample_rate

    @property
    def end_index(self) -> int:
        return self.start_index + self.length

    @property
    def time_axis(self) -> np.ndarray:
        """Sample-index-derived time axis. Not derived from wall clock."""
        return self.start_index / self.sample_rate + np.arange(self.length) / self.sample_rate

    def to_dict(self) -> dict[str, Any]:
        return {
            "values": [float(v) for v in self.values],
            "sample_rate": self.sample_rate,
            "device_id": self.device_id,
            "channel": self.channel,
            "unit": self.unit,
            "start_index": self.start_index,
            "start_time": self.start_time,
            "quality": self.quality.value,
        }


@dataclass(frozen=True, slots=True)
class FeatureVector:
    """An ordered, named, versioned feature vector.

    Ordering is a fixed tuple: never derived from dict iteration, so two runs on
    the same input produce byte-identical vectors.
    """

    names: tuple[str, ...]
    values: tuple[float, ...]
    extractor_id: str
    extractor_version: str
    source_channel: str = ""
    sample_rate: float = 0.0
    quality: DataQuality = DataQuality.GOOD
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.names) != len(self.values):
            raise ValidationError(
                "FeatureVector names and values must have equal length",
                details={"names": len(self.names), "values": len(self.values)},
            )
        if len(set(self.names)) != len(self.names):
            raise ValidationError("FeatureVector names must be unique", details={})
        for name, value in zip(self.names, self.values, strict=True):
            if not math.isfinite(float(value)):
                raise ValidationError(
                    f"Feature {name!r} is not finite", details={"feature": name}
                )

    @property
    def size(self) -> int:
        return len(self.names)

    def as_array(self) -> np.ndarray:
        return np.asarray(self.values, dtype=np.float64)

    def as_dict(self) -> dict[str, float]:
        return dict(zip(self.names, self.values, strict=True))

    def select(self, names: tuple[str, ...]) -> FeatureVector:
        """Return a new vector restricted to ``names``, in the requested order."""
        index = {n: i for i, n in enumerate(self.names)}
        missing = [n for n in names if n not in index]
        if missing:
            raise ValidationError(
                f"Unknown feature(s): {missing}", details={"missing": missing}
            )
        values = tuple(self.values[index[n]] for n in names)
        return FeatureVector(
            names=names,
            values=values,
            extractor_id=self.extractor_id,
            extractor_version=self.extractor_version,
            source_channel=self.source_channel,
            sample_rate=self.sample_rate,
            quality=self.quality,
            metadata=dict(self.metadata),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "names": list(self.names),
            "values": [float(v) for v in self.values],
            "extractor_id": self.extractor_id,
            "extractor_version": self.extractor_version,
            "source_channel": self.source_channel,
            "sample_rate": self.sample_rate,
            "quality": self.quality.value,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> FeatureVector:
        return cls(
            names=tuple(data["names"]),
            values=tuple(float(v) for v in data["values"]),
            extractor_id=str(data["extractor_id"]),
            extractor_version=str(data["extractor_version"]),
            source_channel=str(data.get("source_channel", "")),
            sample_rate=float(data.get("sample_rate", 0.0)),
            quality=DataQuality(data.get("quality", DataQuality.GOOD.value)),
            metadata=dict(data.get("metadata", {})),
        )



