"""Unit tests for core domain contracts (SOFIA-FR-001..012)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from sofia_ai.core.contracts import (
    CONTRACT_VERSION,
    FeatureVector,
    MachineState,
    Severity,
    SignalWindow,
    TelemetryFrame,
    TelemetrySample,
    stable_id,
)
from sofia_ai.core.errors import ValidationError
from sofia_ai.core.quality import DataQuality


class TestTelemetrySample:
    def test_required_fields(self) -> None:
        sample = TelemetrySample(
            timestamp=1_700_000_000.0, device_id="pump-01", channel="vib_x",
            value=1.5, unit="g",
        )
        assert sample.device_id == "pump-01"
        assert sample.channel == "vib_x"
        assert sample.value == 1.5
        assert sample.unit == "g"
        assert sample.quality is DataQuality.GOOD
        assert sample.source == "unknown"
        assert sample.sequence_number is None
        assert sample.tags == {}
        assert sample.ingestion_time is None

    def test_unit_alias_is_normalized(self) -> None:
        sample = TelemetrySample(timestamp=1.7e9, device_id="d", channel="c",
                                 value=1.0, unit="M/S2")
        assert sample.unit == "m/s2"

    def test_rejects_nan_with_good_quality(self) -> None:
        with pytest.raises(ValidationError):
            TelemetrySample(timestamp=1.7e9, device_id="d", channel="c",
                            value=float("nan"), unit="g")

    def test_rejects_inf_with_good_quality(self) -> None:
        with pytest.raises(ValidationError):
            TelemetrySample(timestamp=1.7e9, device_id="d", channel="c",
                            value=float("inf"), unit="g")

    def test_allows_nan_when_quality_invalid(self) -> None:
        sample = TelemetrySample(timestamp=1.7e9, device_id="d", channel="c",
                                 value=float("nan"), unit="g",
                                 quality=DataQuality.INVALID)
        assert not sample.is_valid

    def test_rejects_bad_timestamp(self) -> None:
        with pytest.raises(ValidationError):
            TelemetrySample(timestamp=0.0, device_id="d", channel="c", value=1.0,
                            unit="g")

    def test_rejects_control_characters_in_identifiers(self) -> None:
        with pytest.raises(ValidationError):
            TelemetrySample(timestamp=1.7e9, device_id="pump\n01", channel="c",
                            value=1.0, unit="g")

    def test_rejects_unknown_unit(self) -> None:
        with pytest.raises(ValidationError):
            TelemetrySample(timestamp=1.7e9, device_id="d", channel="c", value=1.0,
                            unit="furlongs/fortnight")

    def test_with_quality_returns_copy(self) -> None:
        sample = TelemetrySample(timestamp=1.7e9, device_id="d", channel="c",
                                 value=1.0, unit="g")
        marked = sample.with_quality(DataQuality.STALE)
        assert marked is not sample
        assert marked.quality is DataQuality.STALE
        assert sample.quality is DataQuality.GOOD

    def test_roundtrip_dict(self) -> None:
        sample = TelemetrySample(timestamp=1.7e9, device_id="d", channel="c",
                                 value=2.5, unit="mm/s", source="csv",
                                 quality=DataQuality.ESTIMATED,
                                 sequence_number=7, tags={"a": "b"})
        restored = TelemetrySample.from_dict(sample.to_dict())
        assert restored == sample
        assert restored.quality is DataQuality.ESTIMATED

    def test_quality_accepts_plain_string(self) -> None:
        """JSON round-trips must not lose the enum type."""
        sample = TelemetrySample(timestamp=1.7e9, device_id="d", channel="c",
                                 value=1.0, unit="g", quality="STALE")
        assert sample.quality is DataQuality.STALE


class TestTelemetryFrame:
    def test_channels_and_serialization(self) -> None:
        frame = TelemetryFrame(
            device_id="d",
            timestamp=1.7e9,
            samples=(
                TelemetrySample(timestamp=1.7e9, device_id="d", channel="a", value=1.0,
                                unit="g"),
                TelemetrySample(timestamp=1.7e9, device_id="d", channel="b", value=2.0,
                                unit="C"),
            ),
        )
        assert frame.channels == ("a", "b")
        assert "samples" in frame.to_dict()


class TestSignalWindow:
    def test_geometry(self) -> None:
        window = SignalWindow(values=np.zeros(512), sample_rate=1000.0,
                              device_id="d", channel="c", unit="g")
        assert window.length == 512
        assert math.isclose(window.duration_s, 0.512)
        assert window.end_index == 512

    def test_time_axis_uses_sample_index(self) -> None:
        window = SignalWindow(values=np.zeros(4), sample_rate=10.0, device_id="d",
                              channel="c", unit="g", start_index=20)
        axis = window.time_axis
        assert math.isclose(axis[0], 2.0)
        assert math.isclose(axis[-1], 2.3)

    def test_rejects_non_finite(self) -> None:
        with pytest.raises(ValidationError):
            SignalWindow(values=np.array([1.0, float("nan")]), sample_rate=1.0,
                         device_id="d", channel="c", unit="g")

    def test_rejects_bad_sample_rate(self) -> None:
        with pytest.raises(ValidationError):
            SignalWindow(values=np.zeros(4), sample_rate=0.0, device_id="d",
                         channel="c", unit="g")

    def test_quality_mask_length_must_match(self) -> None:
        with pytest.raises(ValidationError):
            SignalWindow(values=np.zeros(4), sample_rate=1.0, device_id="d",
                         channel="c", unit="g",
                         quality_mask=(DataQuality.GOOD, DataQuality.GOOD))


class TestFeatureVector:
    def test_names_values_must_match(self) -> None:
        with pytest.raises(ValidationError):
            FeatureVector(names=("a", "b"), values=(1.0,), extractor_id="e",
                          extractor_version="1")

    def test_names_must_be_unique(self) -> None:
        with pytest.raises(ValidationError):
            FeatureVector(names=("a", "a"), values=(1.0, 2.0), extractor_id="e",
                          extractor_version="1")

    def test_rejects_non_finite_value(self) -> None:
        with pytest.raises(ValidationError):
            FeatureVector(names=("a",), values=(float("inf"),), extractor_id="e",
                          extractor_version="1")

    def test_select_preserves_requested_order(self) -> None:
        vector = FeatureVector(names=("a", "b", "c"), values=(1.0, 2.0, 3.0),
                               extractor_id="e", extractor_version="1")
        selected = vector.select(("c", "a"))
        assert selected.names == ("c", "a")
        assert selected.values == (3.0, 1.0)

    def test_select_unknown_raises(self) -> None:
        vector = FeatureVector(names=("a",), values=(1.0,), extractor_id="e",
                               extractor_version="1")
        with pytest.raises(ValidationError):
            vector.select(("zzz",))

    def test_roundtrip(self) -> None:
        vector = FeatureVector(names=("a", "b"), values=(1.0, 2.0), extractor_id="e",
                               extractor_version="1", source_channel="c",
                               sample_rate=1000.0, quality=DataQuality.STALE)
        assert FeatureVector.from_dict(vector.to_dict()) == vector


class TestEnums:
    def test_severity_ordering(self) -> None:
        assert Severity.INFO.rank < Severity.NOTICE.rank < Severity.WARNING.rank
        assert Severity.WARNING.rank < Severity.CRITICAL.rank

    def test_machine_states_exist(self) -> None:
        assert MachineState.RUNNING.value == "RUNNING"
        assert MachineState.EMERGENCY_STOP.value == "EMERGENCY_STOP"


class TestStableId:
    def test_deterministic(self) -> None:
        assert stable_id("a", "b") == stable_id("a", "b")

    def test_distinct(self) -> None:
        assert stable_id("a", "b") != stable_id("a", "c")


def test_contract_version_is_declared() -> None:
    assert CONTRACT_VERSION == "2.0"
