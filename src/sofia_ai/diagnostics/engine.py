"""Health events, health scoring and the diagnostic engine.

The engine's job is narrow and explicit:

1. collect :class:`Evidence` from versioned rules,
2. combine confidence with the noisy-OR rule,
3. derive a severity from evidence (never from an LLM),
4. emit a :class:`HealthEvent` that records every input that produced it.

An uncertain result stays uncertain. There is no code path that turns an
uncertain model output into a definite diagnosis.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final

from ..core.contracts import (
    CONTRACT_VERSION,
    DataQuality,
    MachineState,
    Severity,
    stable_id,
)
from ..core.errors import DiagnosticError
from ..core.validation import validate_identifier, validate_probability
from .evidence import (
    MAX_EVIDENCE_PER_EVENT,
    Evidence,
    EvidenceBundle,
    escalate_severity,
)
from .rules import DiagnosticRule, RuleContext, default_rules

__all__ = [
    "EVENT_CONTRACT_VERSION",
    "DiagnosticEngine",
    "HealthEvent",
    "HealthScore",
    "compute_health_score",
]

EVENT_CONTRACT_VERSION: Final[str] = CONTRACT_VERSION


@dataclass(frozen=True, slots=True)
class HealthEvent:
    """A machine-health event with full provenance.

    Attributes:
        event_id: Deterministic identifier derived from the event content.
        severity: Derived from evidence.
        event_type: Short classification, e.g. ``anomaly_detected``.
        device_id: Equipment identifier.
        channels: Affected channels.
        evidence: The justifying evidence bundle.
        confidence: Aggregate confidence in ``[0, 1]``.
        uncertainty: ``1 - confidence``. Always present.
        recommendation: Non-authoritative suggested engineer action.
        diagnostic_source: Rule or model identifier.
        rule_version: Version of the producing logic.
        model_version: Model version, when a model contributed.
        timestamp: Event creation time (epoch seconds).
        machine_state: Machine state at the time of the event.
        quality: Worst data quality behind the event.
        metadata: Extra structured context.
    """

    event_id: str
    severity: Severity
    event_type: str
    device_id: str
    channels: tuple[str, ...]
    evidence: EvidenceBundle
    confidence: float
    recommendation: str
    diagnostic_source: str
    rule_version: str
    timestamp: float
    machine_state: MachineState = MachineState.UNKNOWN
    model_version: str = ""
    quality: DataQuality = DataQuality.GOOD
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "confidence", validate_probability(self.confidence, "confidence")
        )

    @property
    def uncertainty(self) -> float:
        return 1.0 - self.confidence

    @property
    def is_actionable(self) -> bool:
        """Whether confidence is high enough to justify an engineer's attention."""
        return self.severity.rank >= Severity.WARNING.rank and self.confidence >= 0.5

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_version": EVENT_CONTRACT_VERSION,
            "event_id": self.event_id,
            "severity": self.severity.value,
            "event_type": self.event_type,
            "device_id": self.device_id,
            "channels": list(self.channels),
            "evidence": self.evidence.to_list(),
            "confidence": self.confidence,
            "uncertainty": self.uncertainty,
            "recommendation": self.recommendation,
            "diagnostic_source": self.diagnostic_source,
            "rule_version": self.rule_version,
            "model_version": self.model_version,
            "timestamp": self.timestamp,
            "machine_state": self.machine_state.value,
            "quality": self.quality.value,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class HealthScore:
    """A 0-100 machine-health score with an explicit uncertainty band.

    ``score`` is the point estimate; ``lower``/``upper`` bound the credible
    interval implied by the confidence. Publishing a bare health number without
    the band would hide uncertainty, which Sofia does not do.
    """

    score: float
    confidence: float
    lower: float
    upper: float
    contributors: tuple[str, ...] = ()

    @property
    def uncertainty(self) -> float:
        return 1.0 - self.confidence

    @property
    def band_width(self) -> float:
        return self.upper - self.lower

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "confidence": self.confidence,
            "uncertainty": self.uncertainty,
            "lower": self.lower,
            "upper": self.upper,
            "contributors": list(self.contributors),
        }


def compute_health_score(
    events: Sequence[HealthEvent],
    *,
    decay_severity: bool = True,
) -> HealthScore:
    """Map health events onto a 0-100 score with an uncertainty band.

    100 means "no adverse evidence". Each event subtracts in proportion to its
    severity and confidence. The band widens as aggregate confidence falls, so a
    score derived from poor data is visibly imprecise.
    """
    score = 100.0
    contributors: list[str] = []
    confidence_product = 1.0

    penalties = {
        Severity.INFO: 2.0,
        Severity.NOTICE: 5.0,
        Severity.WARNING: 15.0,
        Severity.CRITICAL: 35.0,
    }
    for event in events:
        magnitude = penalties[event.severity]
        if decay_severity:
            magnitude *= event.confidence
        score -= magnitude
        contributors.append(f"{event.event_type}:{event.severity.value}")
        confidence_product *= 1.0 - event.confidence

    score = float(max(0.0, min(100.0, score)))
    aggregate_confidence = float(1.0 - confidence_product) if events else 1.0
    half_band = 50.0 * (1.0 - aggregate_confidence)
    return HealthScore(
        score=score,
        confidence=aggregate_confidence,
        lower=float(max(0.0, score - half_band)),
        upper=float(min(100.0, score + half_band)),
        contributors=tuple(contributors),
    )


@dataclass(slots=True)
class DiagnosticEngine:
    """Turns inference results into evidence-backed health events.

    Args:
        rules: Versioned diagnostic rules. Defaults to :func:`default_rules`.
        min_event_confidence: Events below this confidence are still emitted but
            marked non-actionable, never silently dropped.
        clock: Injectable wall clock for deterministic tests.
    """

    rules: tuple[DiagnosticRule, ...] = field(default_factory=default_rules)
    min_event_confidence: float = 0.3
    clock: Any = None

    def __post_init__(self) -> None:
        self.min_event_confidence = validate_probability(
            self.min_event_confidence, "min_event_confidence"
        )
        if not self.rules:
            raise DiagnosticError("DiagnosticEngine requires at least one rule", details={})

    def evaluate(
        self,
        context: RuleContext,
        *,
        event_type: str = "anomaly_detected",
        recommendation: str = "",
        model_version: str = "",
        extra_evidence: Sequence[Evidence] = (),
    ) -> HealthEvent | None:
        """Run all rules and build an event, or ``None`` if nothing fired."""
        if not isinstance(context, RuleContext):
            raise DiagnosticError("context must be a RuleContext", details={})

        collected: list[Evidence] = []
        for rule in self.rules:
            item = rule.evaluate(context)
            if item is not None:
                collected.append(item)
        collected.extend(extra_evidence)

        if not collected:
            return None
        if len(collected) > MAX_EVIDENCE_PER_EVENT:
            collected = sorted(
                collected, key=lambda e: e.effective_confidence * e.weight, reverse=True
            )[:MAX_EVIDENCE_PER_EVENT]

        bundle = EvidenceBundle(tuple(collected))
        confidence = bundle.combined_confidence()
        quality = bundle.worst_quality()

        # Severity comes from evidence only.
        proposed = Severity.NOTICE
        for item in bundle:
            details = item.details or {}
            candidate = details.get("severity")
            if isinstance(candidate, str):
                try:
                    proposed = _max_severity(proposed, Severity(candidate))
                except ValueError:  # pragma: no cover - validated upstream
                    continue
        severity = escalate_severity(proposed, confidence)

        timestamp = self.clock.wall() if self.clock is not None else time.time()
        device_id = validate_identifier(context.device_id, "device_id")
        event_id = stable_id(
            device_id, context.channel, event_type, severity.value,
            f"{confidence:.6f}", f"{context.score:.6f}",
        )
        return HealthEvent(
            event_id=event_id,
            severity=severity,
            event_type=event_type,
            device_id=device_id,
            channels=(context.channel,),
            evidence=bundle,
            confidence=confidence,
            recommendation=recommendation or _default_recommendation(severity, context),
            diagnostic_source=",".join(sorted({e.source for e in bundle})),
            rule_version=_join_versions(bundle),
            timestamp=timestamp,
            machine_state=context.machine_state,
            model_version=model_version,
            quality=quality,
            metadata={"score": context.score, "unit": context.unit},
        )

    def evaluate_many(
        self,
        contexts: Sequence[RuleContext],
        **kwargs: Any,
    ) -> list[HealthEvent]:
        """Evaluate several contexts, preserving order and skipping non-firing ones."""
        events: list[HealthEvent] = []
        for context in contexts:
            event = self.evaluate(context, **kwargs)
            if event is not None:
                events.append(event)
        return events


def _max_severity(a: Severity, b: Severity) -> Severity:
    return a if a.rank >= b.rank else b


def _join_versions(bundle: EvidenceBundle) -> str:
    versions = sorted({f"{e.source}@{e.source_version}" for e in bundle})
    return ";".join(versions)


def _default_recommendation(severity: Severity, context: RuleContext) -> str:
    if severity is Severity.CRITICAL:
        return (
            f"Review {context.channel} on {context.device_id}: sustained deviation from "
            f"baseline. Inspect before continued operation."
        )
    if severity is Severity.WARNING:
        return (
            f"Monitor {context.channel} on {context.device_id} and compare against the "
            f"recorded baseline before scheduling intervention."
        )
    return f"Log {context.channel} deviation for trend review."

