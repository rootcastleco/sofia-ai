"""Unit tests for diagnostics: evidence, rules, events and health scoring."""

from __future__ import annotations

import math

import pytest

from sofia_ai.core.contracts import DataQuality, MachineState, Severity
from sofia_ai.core.errors import DiagnosticError, ValidationError
from sofia_ai.diagnostics import (
    AnomalyScoreRule,
    DataQualityRule,
    DiagnosticEngine,
    Evidence,
    EvidenceBundle,
    EvidenceKind,
    HealthEvent,
    OperatingStateRule,
    RuleContext,
    StalenessRule,
    TrendRule,
    combine_confidence,
    compute_health_score,
    default_rules,
    escalate_severity,
)


def _context(**overrides) -> RuleContext:
    base = {"device_id": "pump-01", "channel": "vibration_x", "score": 0.0, "confidence": 0.0}
    base.update(overrides)
    return RuleContext(**base)  # type: ignore[arg-type]


class TestEvidence:
    def test_deviation(self) -> None:
        item = Evidence(source="r", metric="m", observed=5.0, reference=2.0,
                        description="d")
        assert math.isclose(item.deviation, 3.0)
        assert math.isclose(item.relative_deviation, 1.5)

    def test_relative_deviation_zero_safe(self) -> None:
        item = Evidence(source="r", metric="m", observed=1.0, reference=0.0,
                        description="d")
        assert math.isclose(item.relative_deviation, 1.0)

    def test_requires_source_and_metric(self) -> None:
        with pytest.raises(ValidationError):
            Evidence(source="", metric="m", observed=1.0, reference=0.0, description="d")

    def test_weight_must_be_probability(self) -> None:
        with pytest.raises(ValidationError):
            Evidence(source="r", metric="m", observed=1.0, reference=0.0,
                     description="d", weight=2.0)

    def test_effective_confidence_uses_quality(self) -> None:
        item = Evidence(source="r", metric="m", observed=1.0, reference=0.0,
                        description="d", confidence=1.0, quality=DataQuality.STALE)
        assert item.effective_confidence < 1.0

    def test_serialization(self) -> None:
        item = Evidence(source="r", metric="m", observed=1.0, reference=0.0,
                        description="d")
        payload = item.to_dict()
        assert payload["kind"] == "RULE"
        assert payload["deviation"] == 1.0


class TestEvidenceBundle:
    def test_combine_is_noisy_or(self) -> None:
        a = Evidence(source="a", metric="m", observed=1.0, reference=0.0,
                     description="d", confidence=0.5, weight=1.0)
        b = Evidence(source="b", metric="m", observed=1.0, reference=0.0,
                     description="d", confidence=0.5, weight=1.0)
        assert math.isclose(combine_confidence((a, b)), 0.75)

    def test_empty_combine(self) -> None:
        assert combine_confidence(()) == 0.0

    def test_bundle_is_bounded(self) -> None:
        items = tuple(
            Evidence(source="a", metric="m", observed=1.0, reference=0.0, description="d")
            for _ in range(33)
        )
        with pytest.raises(ValidationError):
            EvidenceBundle(items)

    def test_add_and_filter(self) -> None:
        rule_item = Evidence(source="a", metric="m", observed=1.0, reference=0.0,
                             description="d", kind=EvidenceKind.RULE)
        model_item = Evidence(source="b", metric="m", observed=1.0, reference=0.0,
                              description="d", kind=EvidenceKind.MODEL)
        bundle = EvidenceBundle((rule_item,)).add(model_item)
        assert len(bundle) == 2
        assert len(bundle.filter_kind(EvidenceKind.MODEL)) == 1

    def test_strongest(self) -> None:
        weak = Evidence(source="a", metric="m", observed=1.0, reference=0.0,
                        description="d", confidence=0.1)
        strong = Evidence(source="b", metric="m", observed=1.0, reference=0.0,
                          description="d", confidence=0.9)
        assert EvidenceBundle((weak, strong)).strongest().source == "b"

    def test_worst_quality(self) -> None:
        good = Evidence(source="a", metric="m", observed=1.0, reference=0.0,
                        description="d", quality=DataQuality.GOOD)
        bad = Evidence(source="b", metric="m", observed=1.0, reference=0.0,
                       description="d", quality=DataQuality.STALE)
        assert EvidenceBundle((good, bad)).worst_quality() is DataQuality.STALE


class TestSeverityEscalation:
    def test_high_confidence_escalates(self) -> None:
        assert escalate_severity(Severity.INFO, 0.95) is Severity.CRITICAL

    def test_medium_confidence_escalates_to_warning(self) -> None:
        assert escalate_severity(Severity.INFO, 0.6) is Severity.WARNING

    def test_low_confidence_never_invents_severity(self) -> None:
        assert escalate_severity(Severity.INFO, 0.1) is Severity.INFO

    def test_never_lowers(self) -> None:
        assert escalate_severity(Severity.CRITICAL, 0.1) is Severity.CRITICAL


class TestRules:
    def test_anomaly_score_below_threshold_is_silent(self) -> None:
        rule = AnomalyScoreRule(warning_score=3.0)
        assert rule.evaluate(_context(score=1.0, confidence=0.9)) is None

    def test_anomaly_score_warning(self) -> None:
        rule = AnomalyScoreRule(warning_score=3.0, critical_score=6.0)
        item = rule.evaluate(_context(score=4.0, confidence=0.9))
        assert item is not None
        assert item.details["severity"] == "WARNING"

    def test_anomaly_score_critical(self) -> None:
        rule = AnomalyScoreRule(warning_score=3.0, critical_score=6.0)
        item = rule.evaluate(_context(score=10.0, confidence=0.9))
        assert item.details["severity"] == "CRITICAL"

    def test_anomaly_score_respects_confidence_floor(self) -> None:
        rule = AnomalyScoreRule(min_confidence=0.5)
        assert rule.evaluate(_context(score=100.0, confidence=0.1)) is None

    def test_anomaly_score_rejects_inverted_thresholds(self) -> None:
        with pytest.raises(ValidationError):
            AnomalyScoreRule(warning_score=6.0, critical_score=3.0)

    def test_trend_needs_history(self) -> None:
        rule = TrendRule()
        assert rule.evaluate(_context(feature_values={"rms__slope": 1.0, "rms__n": 2.0})) is None

    def test_trend_fires(self) -> None:
        rule = TrendRule(feature="rms", rate_per_sample=0.001, min_samples=8)
        item = rule.evaluate(_context(feature_values={"rms__slope": 0.05, "rms__n": 32.0}))
        assert item is not None
        assert "increasing" in item.description

    def test_operating_state_attenuates(self) -> None:
        rule = OperatingStateRule()
        assert rule.evaluate(_context(machine_state=MachineState.RUNNING)) is None
        assert rule.evaluate(_context(machine_state=MachineState.STARTUP)) is not None

    def test_data_quality_fires_on_bad(self) -> None:
        rule = DataQualityRule()
        assert rule.evaluate(_context(quality=DataQuality.GOOD)) is None
        assert rule.evaluate(_context(quality=DataQuality.STALE)) is not None

    def test_staleness(self) -> None:
        rule = StalenessRule(max_age_s=60.0)
        assert rule.evaluate(_context(age_s=1.0)) is None
        assert rule.evaluate(_context(age_s=120.0)) is not None

    def test_default_rules_are_present(self) -> None:
        assert len(default_rules()) == 5


class TestDiagnosticEngine:
    def test_no_evidence_returns_none(self) -> None:
        engine = DiagnosticEngine()
        assert engine.evaluate(_context()) is None

    def test_event_has_full_provenance(self) -> None:
        engine = DiagnosticEngine()
        event = engine.evaluate(_context(score=8.0, confidence=0.9))
        assert event is not None
        assert event.severity is Severity.CRITICAL
        assert event.device_id == "pump-01"
        assert event.channels == ("vibration_x",)
        assert event.rule_version
        assert event.diagnostic_source
        assert event.recommendation
        assert 0.0 <= event.confidence <= 1.0

    def test_uncertainty_is_complement(self) -> None:
        engine = DiagnosticEngine()
        event = engine.evaluate(_context(score=8.0, confidence=0.9))
        assert event is not None
        assert math.isclose(event.confidence + event.uncertainty, 1.0, abs_tol=1e-12)

    def test_event_id_is_deterministic(self) -> None:
        engine = DiagnosticEngine(clock=_FakeClock())
        a = engine.evaluate(_context(score=8.0, confidence=0.9))
        b = engine.evaluate(_context(score=8.0, confidence=0.9))
        assert a is not None and b is not None
        assert a.event_id == b.event_id

    def test_bad_quality_reduces_confidence(self) -> None:
        engine = DiagnosticEngine(clock=_FakeClock())
        good = engine.evaluate(_context(score=8.0, confidence=0.9))
        bad = engine.evaluate(_context(score=8.0, confidence=0.9,
                                       quality=DataQuality.STALE))
        assert good is not None and bad is not None
        assert bad.confidence < good.confidence

    def test_requires_rule_context(self) -> None:
        engine = DiagnosticEngine()
        with pytest.raises(DiagnosticError):
            engine.evaluate({"score": 1.0})  # type: ignore[arg-type]

    def test_engine_requires_rules(self) -> None:
        with pytest.raises(DiagnosticError):
            DiagnosticEngine(rules=())

    def test_evaluate_many(self) -> None:
        engine = DiagnosticEngine()
        events = engine.evaluate_many([
            _context(score=0.0),
            _context(score=8.0, confidence=0.9),
        ])
        assert len(events) == 1

    def test_extra_evidence_is_included(self) -> None:
        engine = DiagnosticEngine()
        extra = (Evidence(source="operator", metric="note", observed=1.0, reference=0.0,
                          description="operator observed noise", kind=EvidenceKind.OPERATOR),)
        event = engine.evaluate(_context(), extra_evidence=extra)
        assert event is not None
        assert "operator" in event.diagnostic_source

    def test_event_serialization(self) -> None:
        engine = DiagnosticEngine()
        event = engine.evaluate(_context(score=8.0, confidence=0.9))
        assert event is not None
        payload = event.to_dict()
        assert payload["contract_version"] == "2.0"
        assert payload["evidence"]


class _FakeClock:
    def wall(self) -> float:
        return 1_700_000_000.0


class TestHealthScore:
    def test_no_events_is_perfect(self) -> None:
        score = compute_health_score([])
        assert score.score == 100.0
        assert score.confidence == 1.0
        assert score.band_width == 0.0

    def test_critical_event_lowers_score(self) -> None:
        engine = DiagnosticEngine(clock=_FakeClock())
        event = engine.evaluate(_context(score=10.0, confidence=0.95))
        assert event is not None
        score = compute_health_score([event])
        assert score.score < 75.0

    def test_uncertain_events_widen_the_band(self) -> None:
        engine = DiagnosticEngine(clock=_FakeClock())
        certain = engine.evaluate(_context(score=10.0, confidence=0.99))
        uncertain = engine.evaluate(_context(score=10.0, confidence=0.55))
        assert certain is not None and uncertain is not None
        wide = compute_health_score([uncertain])
        narrow = compute_health_score([certain])
        assert wide.band_width > narrow.band_width

    def test_score_is_bounded(self) -> None:
        engine = DiagnosticEngine(clock=_FakeClock())
        events = [engine.evaluate(_context(score=100.0, confidence=1.0)) for _ in range(10)]
        score = compute_health_score([e for e in events if e is not None])
        assert 0.0 <= score.score <= 100.0
        assert 0.0 <= score.lower <= score.upper <= 100.0

    def test_serialization(self) -> None:
        payload = compute_health_score([]).to_dict()
        assert "uncertainty" in payload and "contributors" in payload


class TestHealthEvent:
    def test_is_actionable(self) -> None:
        engine = DiagnosticEngine(clock=_FakeClock())
        event = engine.evaluate(_context(score=8.0, confidence=0.9))
        assert event is not None
        assert event.is_actionable

    def test_low_confidence_not_actionable(self) -> None:
        bundle = EvidenceBundle((
            Evidence(source="r", metric="m", observed=1.0, reference=0.0,
                     description="d", confidence=0.1, weight=0.1),
        ))
        event = HealthEvent(
            event_id="x", severity=Severity.INFO, event_type="t", device_id="d",
            channels=("c",), evidence=bundle, confidence=0.1, recommendation="r",
            diagnostic_source="s", rule_version="1", timestamp=0.0,
        )
        assert not event.is_actionable
