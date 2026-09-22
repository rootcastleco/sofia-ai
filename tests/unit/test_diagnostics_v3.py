"""Unit tests for Diagnostics, Evidence, and Uncertainty (SOFIA-CORE-009, 010, 011)."""

from __future__ import annotations

import pytest

from sofia_ai.core.contracts import Severity, SignalQuality
from sofia_ai.diagnostics.engine import HealthEvent, HealthScore
from sofia_ai.diagnostics.evidence import DiagnosticEvidence, EvidenceBundle


def test_diagnostic_evidence_primitives() -> None:
    ev = DiagnosticEvidence(
        source="rule_iso_10816",
        metric="velocity_rms",
        observed=5.2,
        reference=2.8,
        description="Velocity RMS exceeds ISO 10816 Zone B limit (2.8 mm/s)",
        weight=0.9,
        confidence=0.95,
        quality=SignalQuality.GOOD,
    )
    assert ev.metric == "velocity_rms"
    assert ev.observed == 5.2
    assert ev.reference == 2.8
    assert ev.deviation == pytest.approx(2.4)
    assert ev.relative_deviation == pytest.approx(2.4 / 2.8)
    assert ev.effective_confidence == pytest.approx(0.95 * 1.0)


def test_evidence_quality_attenuation() -> None:
    # Test that degraded or saturated signal quality attenuates effective confidence
    ev_good = DiagnosticEvidence(
        source="sensor", metric="rms", observed=3.0, reference=2.0, description="ok",
        quality=SignalQuality.GOOD, confidence=1.0,
    )
    ev_degraded = DiagnosticEvidence(
        source="sensor", metric="rms", observed=3.0, reference=2.0, description="degraded",
        quality=SignalQuality.DEGRADED, confidence=1.0,
    )
    ev_saturated = DiagnosticEvidence(
        source="sensor", metric="rms", observed=3.0, reference=2.0, description="saturated",
        quality=SignalQuality.SATURATED, confidence=1.0,
    )

    assert ev_good.effective_confidence == 1.0
    assert ev_degraded.effective_confidence == 0.7
    assert ev_saturated.effective_confidence == 0.2


def test_health_score_with_uncertainty_band() -> None:
    # Create health score with explicit uncertainty band
    score = HealthScore(
        score=75.0,
        confidence=0.85,
        lower=68.0,
        upper=82.0,
        contributors=("bearing_wear", "voltage_unbalance"),
    )
    assert score.score == 75.0
    assert score.confidence == 0.85
    assert score.uncertainty == pytest.approx(0.15)
    assert score.band_width == pytest.approx(14.0)
    assert score.lower == 68.0
    assert score.upper == 82.0
    assert "bearing_wear" in score.contributors


def test_health_event_uncertainty_field() -> None:
    bundle = EvidenceBundle(items=(
        DiagnosticEvidence(source="rule", metric="m", observed=1.0, reference=0.5, description="test"),
    ))
    event = HealthEvent(
        event_id="test_ev_01",
        severity=Severity.WARNING,
        event_type="anomaly_detected",
        device_id="pump-01",
        channels=("vib_x",),
        evidence=bundle,
        confidence=0.8,
        recommendation="Inspect pump impeller",
        diagnostic_source="rule_engine",
        rule_version="1.0",
        timestamp=1700000000.0,
    )
    assert event.confidence == 0.8
    assert event.uncertainty == pytest.approx(0.2)
    assert event.is_actionable is True
