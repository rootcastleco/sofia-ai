"""Unit tests for Sofia Runtime v3 Core Contracts and Primitives."""

from __future__ import annotations

import pytest

from sofia_ai.core.contracts import (
    CONTRACT_VERSION,
    Feature,
    FeatureVector,
    InferenceRequest,
    RuntimeFault,
    Sample,
    SignalFrame,
    SignalMetadata,
    SignalQuality,
    TelemetrySample,
)
from sofia_ai.core.errors import ValidationError


def test_contract_version_is_v3() -> None:
    assert CONTRACT_VERSION == "3.0"


def test_sample_is_telemetry_sample() -> None:
    sample = Sample(
        timestamp=1700000000.0,
        device_id="pump-01",
        channel="vibration_x",
        value=2.45,
        unit="mm/s",
        quality=SignalQuality.GOOD,
    )
    assert isinstance(sample, TelemetrySample)
    assert sample.value == 2.45
    assert sample.unit == "mm/s"
    assert sample.quality == SignalQuality.GOOD


def test_signal_metadata_validation() -> None:
    meta = SignalMetadata(
        sample_rate=1000.0,
        channel_name="accel_z",
        physical_unit="m/s2",
        sensor_id="sn-10023",
        scale_factor=1.0,
    )
    assert meta.sample_rate == 1000.0
    assert meta.channel_name == "accel_z"
    assert meta.physical_unit == "m/s2"

    with pytest.raises(ValidationError, match="sample_rate must be positive"):
        SignalMetadata(sample_rate=-10.0, channel_name="accel_z", physical_unit="m/s2")

    with pytest.raises(ValidationError, match="scale_factor must be positive"):
        SignalMetadata(sample_rate=100.0, channel_name="accel_z", physical_unit="m/s2", scale_factor=-1.0)


def test_signal_frame_bounded_capacity() -> None:
    sample = Sample(
        timestamp=1700000000.0,
        device_id="pump-01",
        channel="accel_x",
        value=1.0,
        unit="g",
    )
    frame = SignalFrame(samples=(sample, sample))
    assert frame.size == 2

    # Over 65536 samples must raise ValidationError
    oversized = tuple(sample for _ in range(65537))
    with pytest.raises(ValidationError, match="SignalFrame exceeds maximum capacity"):
        SignalFrame(samples=oversized)


def test_signal_quality_extensions() -> None:
    assert SignalQuality.DEGRADED == "DEGRADED"
    assert SignalQuality.SATURATED == "SATURATED"
    from sofia_ai.core.quality import quality_confidence_factor

    assert quality_confidence_factor(SignalQuality.DEGRADED) == 0.7
    assert quality_confidence_factor(SignalQuality.SATURATED) == 0.2


def test_feature_primitive() -> None:
    feat = Feature(
        name="vibration_rms",
        value=3.14,
        unit="mm/s",
        uncertainty=0.05,
        quality=SignalQuality.GOOD,
        algorithm="rms",
        algorithm_version="2.0",
    )
    assert feat.name == "vibration_rms"
    assert feat.value == 3.14
    assert feat.uncertainty == 0.05
    assert feat.unit == "mm/s"

    with pytest.raises(ValidationError, match="must be finite"):
        Feature(name="bad_val", value=float("nan"), unit="mm/s")

    with pytest.raises(ValidationError, match="must be non-negative"):
        Feature(name="bad_unc", value=1.0, unit="mm/s", uncertainty=-0.5)


def test_feature_vector_versioning_and_schema() -> None:
    fv = FeatureVector(
        names=("rms", "peak"),
        values=(1.5, 4.2),
        extractor_id="test_extractor",
        extractor_version="1.0",
        schema_version="3.0",
        uncertainties=(0.1, 0.2),
    )
    assert fv.schema_version == "3.0"
    assert len(fv.features) == 2
    assert fv.features[0].name == "rms"
    assert fv.features[0].uncertainty == 0.1
    assert fv.features[1].name == "peak"
    assert fv.features[1].uncertainty == 0.2

    assert fv.validate_schema("3.0") is True
    with pytest.raises(ValidationError, match="schema version mismatch"):
        fv.validate_schema("2.0")


def test_inference_request_primitive() -> None:
    fv = FeatureVector(
        names=("f1",),
        values=(2.0,),
        extractor_id="test",
        extractor_version="1.0",
    )
    req = InferenceRequest(
        model_id="detector-01",
        model_version="1.0.0",
        features=fv,
    )
    assert req.model_id == "detector-01"
    assert req.model_version == "1.0.0"
    assert len(req.request_id) > 0


def test_runtime_fault_primitive() -> None:
    fault = RuntimeFault(
        fault_code="VM_MEM_OOB",
        subsystem="learning_asm",
        severity="ERROR",
        message="Memory access out of bounds at 0x10000",
    )
    assert fault.fault_code == "VM_MEM_OOB"
    assert fault.subsystem == "learning_asm"
    assert fault.recoverable is True
    assert fault.timestamp > 0.0
