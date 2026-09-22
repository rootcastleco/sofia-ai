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
from .quality import DataQuality, SignalQuality
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
    "Feature",
    "FeatureVector",
    "InferenceRequest",
    "MachineState",
    "RuntimeFault",
    "Sample",
    "Severity",
    "SignalFrame",
    "SignalMetadata",
    "SignalQuality",
    "SignalWindow",
    "TelemetryFrame",
    "TelemetrySample",
    "stable_id",
]

CONTRACT_VERSION: Final[str] = "3.0"


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


#: Canonical alias for Sofia 3.x runtime specification.
Sample = TelemetrySample


@dataclass(frozen=True, slots=True)
class SignalMetadata:
    """Metadata describing an ingested physical signal channel."""

    sample_rate: float
    channel_name: str
    physical_unit: str
    sensor_id: str = ""
    calibration_id: str = ""
    scale_factor: float = 1.0

    def __post_init__(self) -> None:
        validate_sample_rate(self.sample_rate)
        object.__setattr__(self, "channel_name", validate_identifier(self.channel_name, "channel_name"))
        normalized = unit_alias(self.physical_unit)
        object.__setattr__(self, "physical_unit", normalized)
        if not math.isfinite(self.scale_factor) or self.scale_factor <= 0.0:
            raise ValidationError("scale_factor must be positive finite float", details={"scale_factor": self.scale_factor})


@dataclass(frozen=True, slots=True)
class SignalFrame:
    """An immutable, bounded frame of samples sharing metadata."""

    samples: tuple[Sample, ...]
    metadata: SignalMetadata | None = None

    def __post_init__(self) -> None:
        if len(self.samples) > 65536:
            raise ValidationError("SignalFrame exceeds maximum capacity (65536)", details={"size": len(self.samples)})

    @property
    def size(self) -> int:
        return len(self.samples)


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
class Feature:
    """A single named, versioned, unit-bearing feature."""

    name: str
    value: float
    unit: str
    uncertainty: float = 0.0
    quality: DataQuality = DataQuality.GOOD
    algorithm: str = ""
    algorithm_version: str = "1.0"

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", validate_identifier(self.name, "name"))
        if not math.isfinite(float(self.value)):
            raise ValidationError(
                f"Feature value for {self.name!r} must be finite",
                details={"name": self.name, "value": self.value},
            )
        if not math.isfinite(float(self.uncertainty)) or self.uncertainty < 0.0:
            raise ValidationError(
                f"Feature uncertainty for {self.name!r} must be non-negative finite float",
                details={"name": self.name, "uncertainty": self.uncertainty},
            )
        normalized = unit_alias(self.unit)
        object.__setattr__(self, "unit", normalized)


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
    schema_version: str = "3.0"
    uncertainties: tuple[float, ...] = ()
    features: tuple[Feature, ...] = ()

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

        unc = self.uncertainties
        if not unc or len(unc) != len(self.names):
            unc = tuple(0.0 for _ in self.names)
            object.__setattr__(self, "uncertainties", unc)

        if not self.features or len(self.features) != len(self.names):
            feats = tuple(
                Feature(
                    name=n,
                    value=v,
                    unit=str(self.metadata.get("unit", "dimensionless")),
                    uncertainty=u,
                    quality=self.quality,
                    algorithm=self.extractor_id,
                    algorithm_version=self.extractor_version,
                )
                for n, v, u in zip(self.names, self.values, unc, strict=True)
            )
            object.__setattr__(self, "features", feats)

    def validate_schema(self, expected_schema_version: str) -> bool:
        """Verify that vector matches expected schema version."""
        if self.schema_version != expected_schema_version:
            raise ValidationError(
                f"FeatureVector schema version mismatch: expected {expected_schema_version}, got {self.schema_version}",
                details={"expected": expected_schema_version, "actual": self.schema_version},
            )
        return True

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
        unc = tuple(self.uncertainties[index[n]] for n in names)
        feats = tuple(self.features[index[n]] for n in names)
        return FeatureVector(
            names=names,
            values=values,
            extractor_id=self.extractor_id,
            extractor_version=self.extractor_version,
            source_channel=self.source_channel,
            sample_rate=self.sample_rate,
            quality=self.quality,
            metadata=dict(self.metadata),
            schema_version=self.schema_version,
            uncertainties=unc,
            features=feats,
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
            "schema_version": self.schema_version,
            "uncertainties": list(self.uncertainties),
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
            schema_version=str(data.get("schema_version", "3.0")),
            uncertainties=tuple(float(u) for u in data.get("uncertainties", ())),
        )


@dataclass(frozen=True, slots=True)
class InferenceRequest:
    """Typed input payload for model inference."""

    model_id: str
    model_version: str
    features: FeatureVector
    context: Mapping[str, Any] = field(default_factory=dict)
    request_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_id", validate_identifier(self.model_id, "model_id"))
        if not self.request_id:
            object.__setattr__(self, "request_id", stable_id(self.model_id, self.model_version, str(id(self.features))))


@dataclass(frozen=True, slots=True)
class RuntimeFault:
    """Structured runtime fault representation."""

    fault_code: str
    subsystem: str
    severity: str
    message: str
    recoverable: bool = True
    timestamp: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "fault_code", validate_identifier(self.fault_code, "fault_code"))
        object.__setattr__(self, "subsystem", validate_identifier(self.subsystem, "subsystem"))
        if self.timestamp == 0.0:
            import time
            object.__setattr__(self, "timestamp", time.time())



