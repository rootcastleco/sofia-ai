"""Command requests and policy decisions.

This module defines the **only** representation of machine actuation intent in
Sofia. Nothing else in the codebase may construct an actuator call.

The default posture is DENY. An approval is an explicit, recorded, reviewable
decision that lists the policy rules that were satisfied.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Final

from ..core.contracts import MachineState
from ..core.errors import ValidationError
from ..core.validation import validate_identifier, validate_probability

__all__ = [
    "INTERLOCK_ACTIONS",
    "CommandDecision",
    "CommandPriority",
    "CommandRequest",
    "DecisionOutcome",
    "PolicyRuleResult",
]

#: Actions that can never be approved by software policy alone.
INTERLOCK_ACTIONS: Final[tuple[str, ...]] = (
    "emergency_stop_reset",
    "interlock_bypass",
    "safety_override",
    "guard_bypass",
)


class DecisionOutcome(StrEnum):
    """Result of a policy evaluation."""

    APPROVE = "APPROVE"
    DENY = "DENY"
    REQUIRE_APPROVAL = "REQUIRE_APPROVAL"


class CommandPriority(StrEnum):
    """Requested urgency. Does not bypass policy."""

    LOW = "LOW"
    NORMAL = "NORMAL"
    HIGH = "HIGH"


@dataclass(frozen=True, slots=True)
class CommandRequest:
    """A request to act on equipment.

    Attributes:
        command_id: Unique identifier, used for replay protection.
        action: Action name, validated against a policy allow-list.
        device_id: Target equipment.
        parameters: Typed, bounded parameters. No free-form command strings.
        origin: Who or what produced the request (``"rl"``, ``"operator"``, ``"api"``).
        confidence: Confidence of the recommending model, if any.
        model_version: Model version behind the recommendation.
        issued_at: Creation time (epoch seconds).
        machine_state: Machine state at issue time.
        operator_approved: Whether a human has explicitly approved.
        interlock_clear: Whether the physical interlock reports clear.
        priority: Requested urgency.
    """

    action: str
    device_id: str
    command_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    parameters: Mapping[str, Any] = field(default_factory=dict)
    origin: str = "unknown"
    confidence: float = 0.0
    model_version: str = ""
    issued_at: float = field(default_factory=time.time)
    machine_state: MachineState = MachineState.UNKNOWN
    operator_approved: bool = False
    interlock_clear: bool = False
    priority: CommandPriority = CommandPriority.NORMAL

    def __post_init__(self) -> None:
        object.__setattr__(self, "action", validate_identifier(self.action, "action").lower())
        object.__setattr__(self, "device_id", validate_identifier(self.device_id, "device_id"))
        object.__setattr__(self, "origin", validate_identifier(self.origin, "origin").lower())
        object.__setattr__(
            self, "confidence", validate_probability(self.confidence, "confidence")
        )
        if len(self.parameters) > 32:
            raise ValidationError(
                "a command may carry at most 32 parameters",
                details={"count": len(self.parameters)},
            )
        for key in self.parameters:
            validate_identifier(str(key), "parameter key")

    def with_approval(self, *, approved: bool = True) -> CommandRequest:
        """Return a copy with an explicit operator approval flag."""
        return CommandRequest(
            action=self.action,
            device_id=self.device_id,
            command_id=self.command_id,
            parameters=dict(self.parameters),
            origin=self.origin,
            confidence=self.confidence,
            model_version=self.model_version,
            issued_at=self.issued_at,
            machine_state=self.machine_state,
            operator_approved=approved,
            interlock_clear=self.interlock_clear,
            priority=self.priority,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "command_id": self.command_id,
            "action": self.action,
            "device_id": self.device_id,
            "parameters": dict(self.parameters),
            "origin": self.origin,
            "confidence": self.confidence,
            "model_version": self.model_version,
            "issued_at": self.issued_at,
            "machine_state": self.machine_state.value,
            "operator_approved": self.operator_approved,
            "interlock_clear": self.interlock_clear,
            "priority": self.priority.value,
        }


@dataclass(frozen=True, slots=True)
class PolicyRuleResult:
    """The outcome of one policy rule."""

    rule: str
    passed: bool
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"rule": self.rule, "passed": self.passed, "detail": self.detail}


@dataclass(frozen=True, slots=True)
class CommandDecision:
    """The verdict for a command request.

    ``approved`` is True only when every rule passed and the outcome is APPROVE.
    A decision always lists the rules that were evaluated, including failures, so
    a denial is explainable.
    """

    request: CommandRequest
    outcome: DecisionOutcome
    approved: bool
    rules: tuple[PolicyRuleResult, ...]
    reason: str
    decided_at: float = field(default_factory=time.time)
    policy_version: str = "1.0.0"

    def denied_rules(self) -> tuple[PolicyRuleResult, ...]:
        return tuple(r for r in self.rules if not r.passed)

    def to_dict(self) -> dict[str, Any]:
        return {
            "command_id": self.request.command_id,
            "action": self.request.action,
            "device_id": self.request.device_id,
            "outcome": self.outcome.value,
            "approved": self.approved,
            "reason": self.reason,
            "decided_at": self.decided_at,
            "policy_version": self.policy_version,
            "rules": [r.to_dict() for r in self.rules],
        }
