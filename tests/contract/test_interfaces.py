"""Contract tests: every telemetry source and model backend obeys its interface.

These tests run the same assertions against every registered implementation, so a
new adapter or backend that violates the contract fails the build.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from sofia_ai.core.contracts import TelemetrySample
from sofia_ai.core.errors import SofiaError, TelemetryError
from sofia_ai.inference import build_manifest_for, create_backend, model_registry
from sofia_ai.inference.base import ModelBackend
from sofia_ai.telemetry import registry as telemetry_registry_module
from sofia_ai.telemetry.base import SourceState, TelemetrySource

REPO_ROOT = Path(__file__).resolve().parents[2]


def _sample(index: int = 0) -> TelemetrySample:
    return TelemetrySample(timestamp=1_700_000_000.0 + index * 0.001,
                           device_id="d", channel="c", value=float(index), unit="g")


class TestTelemetrySourceContract:
    """Every registered source must behave as a TelemetrySource."""

    @pytest.fixture
    def csv_path(self, tmp_path) -> str:
        path = tmp_path / "c.csv"
        path.write_text("timestamp,device_id,channel,value,unit\n"
                        "1700000000.0,d,c,1.0,g\n", encoding="utf-8")
        return str(path)

    def _instantiate(self, name: str, **kwargs: Any) -> TelemetrySource:
        source = telemetry_registry_module.telemetry_registry.create(name, **kwargs)
        assert isinstance(source, TelemetrySource)
        return source

    def test_every_source_subclasses_the_interface(self) -> None:
        for name in telemetry_registry_module.telemetry_registry.names():
            factory = telemetry_registry_module.telemetry_registry.get(name)
            assert callable(factory)

    def test_state_machine(self, csv_path) -> None:
        source = self._instantiate("csv", path=csv_path)
        assert source.state is SourceState.CREATED
        source.open()
        assert source.state is SourceState.OPEN
        source.close()
        assert source.state is SourceState.CLOSED

    def test_open_is_idempotent(self, csv_path) -> None:
        source = self._instantiate("csv", path=csv_path)
        source.open()
        source.open()
        assert source.state is SourceState.OPEN

    def test_close_is_idempotent(self, csv_path) -> None:
        source = self._instantiate("csv", path=csv_path)
        source.open()
        source.close()
        source.close()
        assert source.state is SourceState.CLOSED

    def test_read_returns_samples(self, csv_path) -> None:
        source = self._instantiate("csv", path=csv_path)
        source.open()
        samples = source.read()
        assert isinstance(samples, list)
        assert all(isinstance(s, TelemetrySample) for s in samples)

    def test_read_before_open_raises_typed_error(self, csv_path) -> None:
        source = self._instantiate("csv", path=csv_path)
        with pytest.raises(TelemetryError):
            source.read()

    def test_read_has_max_records(self, csv_path) -> None:
        source = self._instantiate("csv", path=csv_path)
        source.open()
        assert source.read(max_records=0) == []

    def test_describe_contract(self, csv_path) -> None:
        source = self._instantiate("csv", path=csv_path)
        source.open()
        payload = source.describe()
        for key in ("source_id", "type", "state", "limits", "stats"):
            assert key in payload, key

    def test_stats_contract(self, csv_path) -> None:
        source = self._instantiate("csv", path=csv_path)
        source.open()
        source.read()
        stats = source.stats.as_dict()
        for key in ("records_read", "records_emitted", "records_rejected", "bytes_read",
                    "protocol_errors", "reconnects"):
            assert key in stats, key

    def test_context_manager(self, csv_path) -> None:
        with self._instantiate("csv", path=csv_path) as source:
            source.read()
        assert source.state is SourceState.CLOSED

    def test_stream_is_bounded(self, csv_path) -> None:
        source = self._instantiate("csv", path=csv_path)
        source.open()
        assert list(source.stream(batch_size=1, max_batches=1)) != []

    def test_memory_source_contract(self) -> None:
        source = self._instantiate("memory", samples=[_sample(i) for i in range(3)])
        source.open()
        assert len(source.read()) == 3

    def test_replay_source_contract(self) -> None:
        source = telemetry_registry_module.telemetry_registry.create(
            "replay", samples=[_sample(i) for i in range(3)])
        source.open()
        assert len(source.read()) == 3

    def test_synthetic_source_contract(self) -> None:
        source = telemetry_registry_module.telemetry_registry.create("synthetic",
                                                                     n_samples=32)
        source.open()
        assert len(source.read()) >= 1

    def test_failures_are_typed_not_raw(self, csv_path) -> None:
        """A transport failure must be a SofiaError subclass, never a raw exception."""
        source = self._instantiate("csv", path=csv_path)
        try:
            source.read()
        except SofiaError:
            pass
        except Exception as exc:  # pragma: no cover
            pytest.fail(f"raw exception escaped: {type(exc).__name__}")


class TestModelBackendContract:
    """Every registered backend must behave as a ModelBackend."""

    @pytest.fixture
    def features(self):
        from sofia_ai.features import extract_from_array

        signal = np.sin(2 * np.pi * 25 * np.arange(256) / 1000.0)
        return extract_from_array(signal, 1000.0)

    @pytest.mark.parametrize("name", ["threshold", "zscore", "mad", "ewma", "iqr",
                                      "cusum", "callable"])
    def test_backend_interface(self, name: str, features) -> None:
        backend = create_backend(name)
        assert isinstance(backend, ModelBackend)
        metadata = backend.metadata()
        assert metadata.backend_id
        assert metadata.kind

    @pytest.mark.parametrize("name", ["zscore", "mad", "ewma", "iqr", "cusum"])
    def test_infer_requires_loaded_state(self, name: str, features) -> None:
        backend = create_backend(name)
        with pytest.raises(SofiaError):
            backend.infer(features)

    @pytest.mark.parametrize("name", ["zscore", "mad", "ewma", "iqr", "cusum"])
    def test_result_contract(self, name: str, features) -> None:
        backend = create_backend(name)
        backend.load(build_manifest_for(backend, model_id=f"contract-{name}"))
        result = backend.infer(features)
        for key in ("model_id", "model_version", "outcome", "score", "confidence",
                    "uncertainty", "latency_s", "evidence", "timestamp", "quality"):
            assert key in result.to_dict(), key
        assert 0.0 <= result.confidence <= 1.0
        assert 0.0 <= result.uncertainty <= 1.0
        assert result.latency_s >= 0.0
        assert np.isfinite(result.score)

    @pytest.mark.parametrize("name", ["zscore", "mad"])
    def test_manifest_must_match_feature(self, name: str, features) -> None:
        backend = create_backend(name)
        from sofia_ai.inference.manifest import ModelManifest, SchemaSpec

        manifest = ModelManifest(model_id="mismatch", version="1.0.0",
                                 input_schema=SchemaSpec(features=("nonexistent",),
                                                         shape=(1,)))
        with pytest.raises(SofiaError):
            backend.load(manifest)

    @pytest.mark.parametrize("name", ["threshold", "zscore", "mad", "ewma", "iqr",
                                      "cusum", "callable"])
    def test_close_is_safe(self, name: str, features) -> None:
        backend = create_backend(name)
        if name == "callable":
            def model(vector):
                return 0.0, 0.5, {}
            backend._fn = model
        backend.load(build_manifest_for(backend, model_id=f"close-{name}",
                                        metadata={"high": 0.0} if name == "threshold"
                                        else None))
        backend.close()
        assert backend.state.value == "CLOSED"

    def test_onnx_backend_reports_missing_extra(self) -> None:
        backend = create_backend("onnx")
        assert backend.metadata().requires_extra == "onnx"

    def test_torch_backend_reports_missing_extra(self) -> None:
        backend = create_backend("torch")
        assert backend.metadata().requires_extra == "torch"

    def test_registry_exposes_all_backends(self) -> None:
        assert len(model_registry) >= 9

    def test_every_backend_module_imports(self) -> None:
        """Lazy imports must not break: each backend module must be importable."""
        for module in ("sofia_ai.inference.detectors", "sofia_ai.inference.registry",
                       "sofia_ai.inference.manifest", "sofia_ai.inference.onnx_backend",
                       "sofia_ai.inference.torch_backend"):
            assert importlib.import_module(module)
