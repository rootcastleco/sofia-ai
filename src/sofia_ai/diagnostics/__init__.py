"""Diagnostics: evidence, versioned rules, health events and health scoring.

Diagnostics consume structured evidence from inference and produce health events.
Severity is always derived from evidence — never from a language model.
"""

from __future__ import annotations

from .engine import (
    DiagnosticEngine,
    HealthEvent,
    HealthScore,
    compute_health_score,
)
from .evidence import (
    Evidence,
    EvidenceBundle,
    EvidenceKind,
    combine_confidence,
    escalate_severity,
)
from .rules import (
    AnomalyScoreRule,
    DataQualityRule,
    DiagnosticRule,
    OperatingStateRule,
    RuleContext,
    StalenessRule,
    TrendRule,
    default_rules,
)

__all__ = [
    "AnomalyScoreRule",
    "DataQualityRule",
    "DiagnosticEngine",
    "DiagnosticRule",
    "Evidence",
    "EvidenceBundle",
    "EvidenceKind",
    "HealthEvent",
    "HealthScore",
    "OperatingStateRule",
    "RuleContext",
    "StalenessRule",
    "TrendRule",
    "combine_confidence",
    "compute_health_score",
    "default_rules",
    "escalate_severity",
]
