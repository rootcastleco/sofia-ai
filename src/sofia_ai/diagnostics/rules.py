"""Versioned diagnostic rules.

A rule converts structured observations (inference results, features, machine
state) into :class:`Evidence`. Rules are:

* **versioned** — every evidence item records the rule version that produced it,
* **pure** — a rule never mutates state and never performs I/O,
* **honest** — a rule may return ``None`` to mean "no opinion", which is the
  correct default for anything not actually observed.

Sofia ships generic, machine-agnostic rules. It does not ship rules that claim a
specific mechanical fault: those require validated evidence for a specific machine
class and must be supplied by the user.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

from ..core.contracts import DataQuality, MachineState, Severity
from ..core.errors import ValidationError
from ..core.validation import validate_positive_float
from .evidence import Evidence, EvidenceKind

__all__ = [
    "RULE_VERSION",
    "AnomalyScoreRule",
    "DataQualityRule",
    "DiagnosticRule",
    "OperatingStateRule",
    "RuleContext",
    "StalenessRule",
    "TrendRule",
    "default_rules",
]

RULE_VERSION: Final[str] = "1.0.0"


@dataclass(frozen=True, slots=True)
class RuleContext:
    """Everything a rule may see. Nothing else is available, by design."""

    device_id: str
    channel: str
    score: float
    confidence: float
    quality: DataQuality = DataQuality.GOOD
    machine_state: MachineState = MachineState.UNKNOWN
    feature_values: dict[str, float] | None = None
    age_s: float = 0.0
    unit: str = ""

    def feature(self, name: str, default: float = 0.0) -> float:
        return float((self.feature_values or {}).get(name, default))


class DiagnosticRule(ABC):
    """Abstract diagnostic rule."""

    #: Stable identifier recorded on every evidence item.
    rule_id: str = "rule"
    #: Semantic version of the rule logic.
    rule_version: str = RULE_VERSION

    @abstractmethod
    def evaluate(self, context: RuleContext) -> Evidence | None:
        """Return evidence, or ``None`` when the rule has no opinion."""

    @property
    def identity(self) -> str:
        return f"{self.rule_id}@{self.rule_version}"


class AnomalyScoreRule(DiagnosticRule):
    """Escalates an anomalous detector score into severity-weighted evidence.

    Severity thresholds are explicit and configurable. No mechanical cause is
    inferred — only that the channel deviated from its baseline.
    """

    rule_id = "anomaly_score"

    def __init__(self, *, warning_score: float = 3.0, critical_score: float = 6.0,
                 min_confidence: float = 0.5) -> None:
        self.warning_score = validate_positive_float(warning_score, "warning_score")
        self.critical_score = validate_positive_float(critical_score, "critical_score")
        self.min_confidence = min_confidence
        if self.critical_score < self.warning_score:
            raise ValidationError(
                "critical_score must be >= warning_score",
                details={"warning": warning_score, "critical": critical_score},
            )

    def evaluate(self, context: RuleContext) -> Evidence | None:
        magnitude = abs(context.score)
        if magnitude < self.warning_score:
            return None
        if context.confidence < self.min_confidence:
            return None
        severity = Severity.CRITICAL if magnitude >= self.critical_score else Severity.WARNING
        return Evidence(
            source=self.rule_id,
            source_version=self.rule_version,
            kind=EvidenceKind.MODEL,
            metric="anomaly_score",
            observed=context.score,
            reference=self.warning_score,
            description=(
                f"{context.channel} deviates from its baseline by {magnitude:.2f} "
                f"(severity {severity.value}). Detected deviation, not a diagnosed cause."
            ),
            weight=0.9 if severity is Severity.CRITICAL else 0.6,
            confidence=context.confidence,
            quality=context.quality,
            details={"severity": severity.value, "device_id": context.device_id,
                     "critical_score": self.critical_score},
        )


class TrendRule(DiagnosticRule):
    """Detects sustained directional drift in a feature.

    Requires a declared rate and a minimum number of observations. Returns ``None``
    when there is not enough history — an opinion from two samples is noise.
    """

    rule_id = "trend"

    def __init__(self, *, feature: str = "rms", rate_per_sample: float = 0.001,
                 min_samples: int = 8) -> None:
        if not feature:
            raise ValidationError("TrendRule requires a feature name", details={})
        self.feature = feature
        self.rate_per_sample = float(rate_per_sample)
        self.min_samples = max(2, int(min_samples))

    def evaluate(self, context: RuleContext) -> Evidence | None:
        history = context.feature_values or {}
        slope = history.get(f"{self.feature}__slope")
        count = history.get(f"{self.feature}__n", 0.0)
        if slope is None or count < self.min_samples:
            return None
        if abs(slope) < abs(self.rate_per_sample):
            return None
        direction = "increasing" if slope > 0 else "decreasing"
        return Evidence(
            source=self.rule_id,
            source_version=self.rule_version,
            kind=EvidenceKind.SIGNAL,
            metric=f"{self.feature}_trend",
            observed=float(slope),
            reference=self.rate_per_sample,
            description=(
                f"{self.feature} is {direction} at {slope:+.5f} per sample over "
                f"{int(count)} observations."
            ),
            weight=0.4,
            confidence=float(min(1.0, count / (2.0 * self.min_samples))),
            quality=context.quality,
            details={"direction": direction, "samples": int(count)},
        )


class OperatingStateRule(DiagnosticRule):
    """Context rule: anomaly evidence is weaker when the machine is not running.

    This is the rule that prevents a vibration anomaly observed during STARTUP from
    being reported with the same confidence as one observed at steady state.
    """

    rule_id = "operating_state"

    def __init__(self, *, attenuated_states: Sequence[MachineState] = (
            MachineState.STARTUP, MachineState.MAINTENANCE, MachineState.OFF)) -> None:
        self.attenuated_states = tuple(attenuated_states)

    def evaluate(self, context: RuleContext) -> Evidence | None:
        if context.machine_state not in self.attenuated_states:
            return None
        return Evidence(
            source=self.rule_id,
            source_version=self.rule_version,
            kind=EvidenceKind.CONTEXT,
            metric="machine_state",
            observed=float(list(MachineState).index(context.machine_state)),
            reference=float(list(MachineState).index(MachineState.RUNNING)),
            description=(
                f"Machine state is {context.machine_state.value}; anomaly evidence "
                f"from this state is attenuated."
            ),
            weight=0.3,
            confidence=0.6,
            quality=context.quality,
            details={"machine_state": context.machine_state.value},
        )


class DataQualityRule(DiagnosticRule):
    """Warns when a decision rests on non-GOOD data.

    This rule is the mechanism that prevents bad data from becoming a confident
    machine diagnosis (SOFIA-DQ-003).
    """

    rule_id = "data_quality"

    def evaluate(self, context: RuleContext) -> Evidence | None:
        if context.quality is DataQuality.GOOD:
            return None
        return Evidence(
            source=self.rule_id,
            source_version=self.rule_version,
            kind=EvidenceKind.CONTEXT,
            metric="data_quality",
            observed=0.0,
            reference=1.0,
            description=(
                f"Underlying telemetry quality is {context.quality.value}; "
                f"diagnostic confidence is reduced accordingly."
            ),
            weight=0.5,
            confidence=0.9,
            quality=context.quality,
            details={"quality": context.quality.value},
        )


class StalenessRule(DiagnosticRule):
    """Flags decisions derived from stale telemetry."""

    rule_id = "staleness"

    def __init__(self, *, max_age_s: float = 60.0) -> None:
        self.max_age_s = validate_positive_float(max_age_s, "max_age_s")

    def evaluate(self, context: RuleContext) -> Evidence | None:
        if context.age_s <= self.max_age_s:
            return None
        return Evidence(
            source=self.rule_id,
            source_version=self.rule_version,
            kind=EvidenceKind.CONTEXT,
            metric="sample_age_s",
            observed=context.age_s,
            reference=self.max_age_s,
            description=(
                f"Telemetry is {context.age_s:.1f}s old, above the {self.max_age_s:.1f}s "
                f"freshness limit."
            ),
            weight=0.4,
            confidence=0.8,
            quality=context.quality,
            details={},
        )


def default_rules() -> tuple[DiagnosticRule, ...]:
    """The machine-agnostic rule set shipped with Sofia."""
    return (
        AnomalyScoreRule(),
        TrendRule(),
        OperatingStateRule(),
        DataQualityRule(),
        StalenessRule(),
    )
