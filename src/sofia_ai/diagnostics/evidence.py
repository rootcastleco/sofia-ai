"""Diagnostic evidence: the atomic unit of justification.

An :class:`Evidence` records *what* was observed, against *what* reference, by
*which* source, at *what* version. A health event is only as credible as its
evidence, so evidence is never optional and never free text alone.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Final

from ..core.contracts import DataQuality, Severity
from ..core.errors import ValidationError
from ..core.validation import validate_probability

__all__ = [
    "MAX_EVIDENCE_PER_EVENT",
    "Evidence",
    "EvidenceBundle",
    "EvidenceKind",
    "combine_confidence",
    "escalate_severity",
]

MAX_EVIDENCE_PER_EVENT: Final[int] = 32


class EvidenceKind(StrEnum):
    """Where an evidence item came from."""

    RULE = "RULE"
    MODEL = "MODEL"
    SIGNAL = "SIGNAL"
    CONTEXT = "CONTEXT"
    OPERATOR = "OPERATOR"


@dataclass(frozen=True, slots=True)
class Evidence:
    """One justifying observation.

    Attributes:
        source: Identifier of the rule or model that produced this item.
        source_version: Version of that rule or model.
        kind: Origin classification.
        metric: Name of the observed quantity.
        observed: Observed value.
        reference: Reference/threshold/baseline value.
        description: Human-readable explanation.
        weight: Relative importance in ``[0, 1]``.
        confidence: Confidence in this item alone, in ``[0, 1]``.
        quality: Data quality of the underlying samples.
        details: Extra structured context.
    """

    source: str
    metric: str
    observed: float
    reference: float
    description: str
    source_version: str = "1.0.0"
    kind: EvidenceKind = EvidenceKind.RULE
    weight: float = 1.0
    confidence: float = 1.0
    quality: DataQuality = DataQuality.GOOD
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "weight", validate_probability(self.weight, "weight"))
        object.__setattr__(
            self, "confidence", validate_probability(self.confidence, "confidence")
        )
        if not self.source or not self.metric:
            raise ValidationError("Evidence requires a source and a metric", details={})

    @property
    def deviation(self) -> float:
        """Signed deviation from the reference."""
        return float(self.observed) - float(self.reference)

    @property
    def relative_deviation(self) -> float:
        """Deviation relative to the reference magnitude. Zero-safe."""
        reference = float(self.reference)
        denominator = abs(reference) if abs(reference) > 1e-12 else 1.0
        return self.deviation / denominator

    @property
    def effective_confidence(self) -> float:
        """Confidence after data-quality attenuation."""
        from ..core.quality import quality_confidence_factor

        return float(self.confidence * quality_confidence_factor(self.quality))

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "source_version": self.source_version,
            "kind": self.kind.value,
            "metric": self.metric,
            "observed": self.observed,
            "reference": self.reference,
            "deviation": self.deviation,
            "relative_deviation": self.relative_deviation,
            "description": self.description,
            "weight": self.weight,
            "confidence": self.confidence,
            "effective_confidence": self.effective_confidence,
            "quality": self.quality.value,
            "details": dict(self.details),
        }


@dataclass(frozen=True, slots=True)
class EvidenceBundle:
    """A bounded, immutable collection of evidence items."""

    items: tuple[Evidence, ...] = ()

    def __post_init__(self) -> None:
        if len(self.items) > MAX_EVIDENCE_PER_EVENT:
            raise ValidationError(
                f"an event may carry at most {MAX_EVIDENCE_PER_EVENT} evidence items",
                details={"count": len(self.items)},
            )

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self):  # type: ignore[no-untyped-def]
        return iter(self.items)

    def add(self, item: Evidence) -> EvidenceBundle:
        """Return a new bundle with ``item`` appended."""
        return EvidenceBundle((*self.items, item))

    def filter_kind(self, kind: EvidenceKind) -> EvidenceBundle:
        return EvidenceBundle(tuple(i for i in self.items if i.kind is kind))

    def strongest(self) -> Evidence | None:
        if not self.items:
            return None
        return max(self.items, key=lambda i: i.effective_confidence * i.weight)

    def combined_confidence(self) -> float:
        return combine_confidence(self.items)

    def worst_quality(self) -> DataQuality:
        from ..core.quality import worst_quality

        return worst_quality([i.quality for i in self.items]) if self.items else DataQuality.GOOD

    def to_list(self) -> list[dict[str, Any]]:
        return [i.to_dict() for i in self.items]


def combine_confidence(items: tuple[Evidence, ...] | list[Evidence]) -> float:
    """Combine per-item confidences into one number.

    Uses the noisy-OR combination ``1 - Π(1 - w_i · c_i)``, which is the standard
    rule for independent evidence sources: each item can only increase aggregate
    confidence, and a single weak item cannot dominate.
    """
    if not items:
        return 0.0
    survival = 1.0
    for item in items:
        survival *= 1.0 - float(item.weight) * float(item.effective_confidence)
    return float(max(0.0, min(1.0, 1.0 - survival)))


def escalate_severity(base: Severity, confidence: float, *,
                      warning_at: float = 0.5, critical_at: float = 0.85) -> Severity:
    """Raise or lower a rule's proposed severity by aggregate confidence.

    Confidence is required to escalate; it can never invent a severity that no
    evidence supports.
    """
    if confidence >= critical_at and base.rank < Severity.CRITICAL.rank:
        return Severity.CRITICAL
    if confidence >= warning_at and base.rank < Severity.WARNING.rank:
        return Severity.WARNING
    return base
