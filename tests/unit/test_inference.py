"""Unit tests for inference: backends, detectors and manifests (SOFIA-INF-*)."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from sofia_ai.core.contracts import DataQuality
from sofia_ai.core.errors import (
    ConfigurationError,
    InferenceError,
    ModelCompatibilityError,
    ModelError,
    ModelIntegrityError,
    ModelSchemaError,
    ValidationError,
)
from sofia_ai.features import extract_from_array
from sofia_ai.inference import (
    CallableBackend,
    DetectorConfig,
    InferenceOutcome,
    ModelManifest,
    SchemaSpec,
    build_manifest_for,
    create_backend,
    iqr_bounds,
    load_manifest,
    median_absolute_deviation,
    model_registry,
    register_model_backend,
    robust_zscore,
    save_manifest,
    validate_manifest,
    zscore,
)
from sofia_ai.inference.detectors import (
    CusumDetector,
    EwmaDetector,
    IqrDetector,
    MadDetector,
    ThresholdDetector,
    ZScoreDetector,
    cusum,
    ewma,
)


@pytest.fixture
def features() -> object:
    signal = np.sin(2 * np.pi * 25 * np.arange(256) / 1000.0)
    return extract_from_array(signal, 1000.0)


def _warm(detector, values, n: int = 40) -> object:
    """Feed a detector a stable baseline then return it."""
    for value in values:
        detector.infer(value)
    return detector


class TestPureFunctions:
    def test_zscore(self) -> None:
        assert math.isclose(zscore(3.0, 1.0, 1.0), 2.0)

    def test_zscore_zero_spread(self) -> None:
        assert zscore(1.0, 1.0, 0.0) == 0.0
        assert zscore(2.0, 1.0, 0.0) == float("inf")

    def test_robust_zscore(self) -> None:
        assert math.isclose(robust_zscore(3.0, 1.0, 1.0 / 1.4826), 2.0, rel_tol=1e-9)

    def test_mad(self) -> None:
        assert math.isclose(median_absolute_deviation([1.0, 1.0, 1.0, 5.0]), 0.0)

    def test_mad_of_spread(self) -> None:
        assert median_absolute_deviation([1.0, 2.0, 3.0, 4.0, 5.0]) > 0.0

    def test_ewma_of_constant(self) -> None:
        assert np.allclose(ewma(np.full(10, 2.0), 0.5), 2.0)

    def test_ewma_tracks_step(self) -> None:
        series = np.concatenate([np.zeros(20), np.full(20, 10.0)])
        smoothed = ewma(series, 0.2)
        assert smoothed[-1] > 9.0

    def test_iqr_bounds(self) -> None:
        low, high = iqr_bounds(np.arange(100, dtype=np.float64))
        assert low < high

    def test_cusum_detects_shift(self) -> None:
        series = np.concatenate([np.zeros(50), np.full(50, 5.0)])
        _, alarm = cusum(series, target=0.0, slack=0.5, threshold=5.0)
        assert alarm is not None
        assert alarm >= 50

    def test_cusum_no_alarm_in_control(self) -> None:
        series = np.zeros(100)
        _, alarm = cusum(series, target=0.0, slack=0.5, threshold=5.0)
        assert alarm is None


class TestDetectorConfig:
    def test_rejects_hop_gt_window(self) -> None:
        with pytest.raises(ConfigurationError):
            DetectorConfig(window_size=8, warmup=16)

    def test_rejects_bad_direction(self) -> None:
        with pytest.raises(ConfigurationError):
            DetectorConfig(direction="sideways")

    def test_rejects_bad_alpha(self) -> None:
        with pytest.raises(ValidationError):
            DetectorConfig(alpha=1.5)


class TestThresholdDetector:
    def test_fires_above_high(self, features) -> None:
        detector = ThresholdDetector(feature="rms", high=0.1)
        detector.load(build_manifest_for(detector, model_id="t",
                                         metadata={"high": 0.1}))
        result = detector.infer(features)
        assert result.is_anomalous

    def test_silent_below(self, features) -> None:
        detector = ThresholdDetector(feature="rms", high=100.0)
        detector.load(build_manifest_for(detector, model_id="t",
                                         metadata={"high": 100.0}))
        assert not detector.infer(features).is_anomalous

    def test_requires_limits(self, features) -> None:
        detector = ThresholdDetector()
        with pytest.raises(ModelError):
            detector.load(build_manifest_for(detector, model_id="t"))


class TestRollingDetectors:
    def _series(self, degrade_from: int, n: int = 60) -> list:
        rng = np.random.default_rng(3)
        out = []
        for i in range(n):
            signal = rng.normal(0, 0.1, 256)
            if i >= degrade_from:
                signal = signal * 12.0
            out.append(extract_from_array(signal, 1000.0))
        return out

    def test_mad_detects_shift(self) -> None:
        series = self._series(degrade_from=40, n=52)
        detector = MadDetector(DetectorConfig(feature="rms", window_size=32,
                                              warmup=8, min_samples=8, threshold=3.0))
        detector.load(build_manifest_for(detector, model_id="mad"))
        results = [detector.infer(v) for v in series]
        assert results[-1].is_anomalous
        assert results[-1].confidence > 0.5

    def test_mad_breakdown_point_is_fifty_percent(self) -> None:
        """Once >50% of the baseline is contaminated the median itself moves.

        This is documented behaviour, not a bug: it is why a rolling baseline
        must be refreshed on a confirmed regime change.
        """
        series = self._series(degrade_from=10, n=80)
        detector = MadDetector(DetectorConfig(feature="rms", window_size=32,
                                              warmup=8, min_samples=8, threshold=3.0))
        detector.load(build_manifest_for(detector, model_id="mad"))
        results = [detector.infer(v) for v in series]
        assert any(r.is_anomalous for r in results[:20])
        assert not results[-1].is_anomalous

    def test_zscore_is_masked_by_outliers(self) -> None:
        """Demonstrated limitation: sigma inflation hides the shift."""
        series = self._series(degrade_from=40, n=52)
        z_detector = ZScoreDetector(DetectorConfig(feature="rms", window_size=32,
                                                   warmup=8, min_samples=8, threshold=3.0))
        z_detector.load(build_manifest_for(z_detector, model_id="z"))
        z_results = [z_detector.infer(v) for v in series]

        mad_detector = MadDetector(DetectorConfig(feature="rms", window_size=32,
                                                  warmup=8, min_samples=8, threshold=3.0))
        mad_detector.load(build_manifest_for(mad_detector, model_id="mad"))
        mad_results = [mad_detector.infer(v) for v in series]

        assert abs(mad_results[-1].score) > abs(z_results[-1].score)

    def test_warmup_returns_unknown(self) -> None:
        series = self._series(degrade_from=1000, n=4)
        detector = MadDetector(DetectorConfig(feature="rms", window_size=32,
                                              warmup=16, min_samples=16))
        detector.load(build_manifest_for(detector, model_id="mad"))
        result = detector.infer(series[0])
        assert result.outcome is InferenceOutcome.UNKNOWN
        assert result.confidence == 0.0

    def test_iqr_detects(self) -> None:
        series = self._series(degrade_from=40, n=52)
        detector = IqrDetector(DetectorConfig(feature="rms", window_size=32,
                                              warmup=8, min_samples=8))
        detector.load(build_manifest_for(detector, model_id="iqr"))
        results = [detector.infer(v) for v in series]
        assert any(r.is_anomalous for r in results[40:])

    def test_ewma_detects(self) -> None:
        series = self._series(degrade_from=40, n=52)
        detector = EwmaDetector(DetectorConfig(feature="rms", window_size=32,
                                               warmup=8, min_samples=8, alpha=0.3))
        detector.load(build_manifest_for(detector, model_id="ewma"))
        results = [detector.infer(v) for v in series]
        assert results[-1].is_anomalous

    def test_cusum_detects(self) -> None:
        series = self._series(degrade_from=40, n=52)
        detector = CusumDetector(DetectorConfig(feature="rms", window_size=32,
                                                warmup=8, min_samples=8))
        detector.load(build_manifest_for(detector, model_id="cusum"))
        results = [detector.infer(v) for v in series]
        assert any(r.is_anomalous for r in results)

    def test_missing_feature_raises(self) -> None:
        detector = MadDetector(DetectorConfig(feature="nonexistent", window_size=4,
                                              warmup=1, min_samples=1))
        detector.load(build_manifest_for(detector, model_id="mad"))
        with pytest.raises(ModelError):
            detector.infer(extract_from_array(np.ones(64), 1000.0))


class TestQualityAttenuation:
    def test_bad_quality_lowers_confidence(self) -> None:
        detector = ThresholdDetector(feature="rms", high=0.1)
        detector.load(build_manifest_for(detector, model_id="t",
                                         metadata={"high": 0.1}))
        good = extract_from_array(np.ones(256), 1000.0)
        bad = extract_from_array(np.ones(256), 1000.0)
        bad = type(bad)(**{**bad.to_dict(), "quality": DataQuality.INVALID})
        assert detector.infer(bad).confidence < detector.infer(good).confidence


class TestManifest:
    def test_roundtrip(self, tmp_path) -> None:
        manifest = ModelManifest(model_id="m", version="1.2.3", kind="statistical",
                                 input_schema=SchemaSpec(features=("rms",), shape=(1,)))
        path = tmp_path / "m.json"
        save_manifest(manifest, str(path))
        assert load_manifest(str(path)).identity == "m@1.2.3"

    def test_rejects_bad_semver(self) -> None:
        with pytest.raises(ModelError):
            ModelManifest(model_id="m", version="1.2")

    def test_engine_compatibility(self) -> None:
        manifest = ModelManifest(model_id="m", version="1.0.0",
                                 engine_compatibility=">=3.0")
        with pytest.raises(ModelCompatibilityError):
            manifest.check_engine("2.0")

    def test_engine_ok(self) -> None:
        ModelManifest(model_id="m", version="1.0.0").check_engine("2.0")

    def test_input_order_enforced(self) -> None:
        manifest = ModelManifest(model_id="m", version="1.0.0",
                                 input_schema=SchemaSpec(features=("a", "b")))
        manifest.check_input(("a", "b", "c"))
        with pytest.raises(ModelSchemaError):
            manifest.check_input(("b", "a"))

    def test_missing_feature(self) -> None:
        manifest = ModelManifest(model_id="m", version="1.0.0",
                                 input_schema=SchemaSpec(features=("a",)))
        with pytest.raises(ModelSchemaError):
            manifest.check_input(("z",))

    def test_validate_rejects_bad_checksum(self) -> None:
        manifest = ModelManifest(model_id="m", version="1.0.0",
                                 artifact_checksum="abc")
        with pytest.raises(ModelIntegrityError):
            validate_manifest(manifest)

    def test_verify_artifact_missing(self, tmp_path) -> None:
        manifest = ModelManifest(model_id="m", version="1.0.0",
                                 artifact_path=str(tmp_path / "nope.bin"),
                                 artifact_checksum="0" * 64)
        with pytest.raises(ModelIntegrityError):
            manifest.verify_artifact()

    def test_verify_artifact_mismatch(self, tmp_path) -> None:
        path = tmp_path / "a.bin"
        path.write_bytes(b"data")
        manifest = ModelManifest(model_id="m", version="1.0.0",
                                 artifact_path=str(path), artifact_checksum="0" * 64)
        with pytest.raises(ModelIntegrityError):
            manifest.verify_artifact()

    def test_manifest_version_mismatch(self, tmp_path) -> None:
        path = tmp_path / "m.json"
        path.write_text(json.dumps({"manifest_version": "1.0", "model_id": "m",
                                    "version": "1.0.0"}), encoding="utf-8")
        with pytest.raises(ModelCompatibilityError):
            load_manifest(str(path))

    def test_load_missing_file(self, tmp_path) -> None:
        with pytest.raises(ModelError):
            load_manifest(str(tmp_path / "missing.json"))


class TestBackendLifecycle:
    def test_infer_before_load_raises(self, features) -> None:
        detector = MadDetector()
        with pytest.raises(InferenceError):
            detector.infer(features)

    def test_load_sets_state(self) -> None:
        detector = MadDetector()
        detector.load(build_manifest_for(detector, model_id="m"))
        assert detector.state.value == "LOADED"

    def test_close_sets_state(self) -> None:
        detector = MadDetector()
        detector.load(build_manifest_for(detector, model_id="m"))
        detector.close()
        assert detector.state.value == "CLOSED"

    def test_latency_is_measured(self, features) -> None:
        detector = ThresholdDetector(feature="rms", high=0.0)
        detector.load(build_manifest_for(detector, model_id="t",
                                         metadata={"high": 0.0}))
        assert detector.infer(features).latency_s >= 0.0

    def test_result_carries_model_identity(self, features) -> None:
        detector = ThresholdDetector(feature="rms", high=0.0)
        detector.load(build_manifest_for(detector, model_id="id-x",
                                         metadata={"high": 0.0}))
        result = detector.infer(features)
        assert result.model_id == "id-x"
        assert result.model_version == "1.0.0"

    def test_uncertainty_is_complement(self, features) -> None:
        detector = ThresholdDetector(feature="rms", high=0.0)
        detector.load(build_manifest_for(detector, model_id="t",
                                         metadata={"high": 0.0}))
        result = detector.infer(features)
        assert math.isclose(result.confidence + result.uncertainty, 1.0, abs_tol=1e-12)


class TestCallableBackend:
    def test_wraps_function(self, features) -> None:
        def model(vector):
            return 1.0, 0.5, {"note": "x"}

        backend = CallableBackend(model, anomaly_threshold=0.5)
        backend.load(build_manifest_for(backend, model_id="c"))
        result = backend.infer(features)
        assert result.is_anomalous
        assert result.evidence["note"] == "x"

    def test_requires_callable(self, features) -> None:
        backend = CallableBackend()
        with pytest.raises(ModelError):
            backend.load(build_manifest_for(backend, model_id="c"))

    def test_function_error_is_typed(self, features) -> None:
        def broken(_vector):
            raise RuntimeError("boom")

        backend = CallableBackend(broken)
        backend.load(build_manifest_for(backend, model_id="c"))
        with pytest.raises(InferenceError):
            backend.infer(features)


class TestRegistry:
    def test_builtin_backends_registered(self) -> None:
        for name in ("threshold", "zscore", "mad", "ewma", "iqr", "cusum", "callable",
                     "onnx", "torch"):
            assert name in model_registry

    def test_unknown_backend(self) -> None:
        with pytest.raises(ModelError):
            create_backend("nonexistent")

    def test_register_custom(self) -> None:
        register_model_backend("unit_test_backend",
                               MadDetector, overwrite=True)
        assert "unit_test_backend" in model_registry

    def test_double_register_without_overwrite(self) -> None:
        register_model_backend("dup_test", MadDetector)
        with pytest.raises(ModelError):
            register_model_backend("dup_test", MadDetector)

    def test_non_callable_rejected(self) -> None:
        with pytest.raises(ModelError):
            register_model_backend("bad", "not-callable")  # type: ignore[arg-type]
