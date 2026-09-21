"""Command policy engine. Default DENY.

There is no bypass. Every command — from the API, from an RL agent, from a rule —
must be evaluated by :meth:`PolicyEngine.evaluate` and produce a
:class:`CommandDecision`. A ``True`` from :attr:`CommandDecision.approved` means
every configured rule passed.

Rules evaluated, in order:

1. ``action_allowlist``      — the action must be explicitly allowed
2. ``interlock_action``      — safety-critical actions are never approvable
3. ``machine_state``         — the machine must be in an allowed state
4. ``operator_approval``     — human approval when required
5. ``confidence_floor``      — model confidence must clear the floor
6. ``command_freshness``     — the request must be recent (replay window)
7. ``replay_protection``     — the command id must not have been consumed
8. ``rate_limit``            — bounded commands per minute
9. ``interlock_status``      — physical interlock must report clear when required
10. ``physical_limits``      — numeric parameters must be inside declared bounds

Sofia is **not** a certified safety system. This engine is a software control
boundary, not a SIL-rated safety function.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Final

from ..core.contracts import MachineState
from ..core.errors import CommandReplayError, PolicyDeniedError
from ..core.validation import validate_probability
from .commands import (
    INTERLOCK_ACTIONS,
    CommandDecision,
    CommandRequest,
    DecisionOutcome,
    PolicyRuleResult,
)

__all__ = ["POLICY_VERSION", "PolicyEngine", "PolicyRule", "ReplayGuard"]

POLICY_VERSION: Final[str] = "1.0.0"


@dataclass(slots=True)
class ReplayGuard:
    """Bounded nonce store with a TTL window (threat T-20).

    Args:
        ttl_s: How long a command id is remembered.
        capacity: Maximum remembered ids. Oldest are evicted first.
    """

    ttl_s: float = 60.0
    capacity: int = 4096
    _seen: dict[str, float] = field(default_factory=dict, init=False, repr=False)
    _order: deque[str] = field(default_factory=deque, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.ttl_s <= 0:
            raise PolicyDeniedError("replay ttl must be positive", details={})
        if self.capacity <= 0:
            raise PolicyDeniedError("replay capacity must be positive", details={})

    def check_and_consume(self, command_id: str, now: float) -> bool:
        """Return True if the id is fresh, consuming it. False if replayed."""
        self._evict(now)
        if command_id in self._seen:
            return False
        self._seen[command_id] = now
        self._order.append(command_id)
        if len(self._seen) > self.capacity:
            oldest = self._order.popleft()
            self._seen.pop(oldest, None)
        return True

    def _evict(self, now: float) -> None:
        while self._order:
            oldest = self._order[0]
            if now - self._seen.get(oldest, now) <= self.ttl_s:
                break
            self._order.popleft()
            self._seen.pop(oldest, None)

    def __len__(self) -> int:
        return len(self._seen)


class PolicyRule:
    """A single named policy check.

    Subclass to add organization-specific rules; register with
    :meth:`PolicyEngine.add_rule`. There is no way to remove the default rules,
    by design.
    """

    name: str = "rule"

    def evaluate(self, request: CommandRequest, engine: PolicyEngine) -> PolicyRuleResult:
        """Return the result of this rule."""
        raise NotImplementedError


@dataclass(slots=True)
class PolicyEngine:
    """Evaluates command requests. Default posture is DENY.

    Args:
        allowed_actions: Explicit allow-list. Empty means nothing is allowed.
        require_operator_approval: Require ``operator_approved``.
        min_confidence: Confidence floor for model-origin requests.
        max_commands_per_min: Rate ceiling.
        command_ttl_s: Freshness window.
        allowed_machine_states: States in which commands may be approved.
        require_interlock_clear: Require ``interlock_clear``.
        parameter_limits: ``parameter name -> (min, max)`` bounds.
        clock: Injectable wall clock for deterministic tests.
    """

    allowed_actions: tuple[str, ...] = ()
    default_decision: str = "deny"
    require_operator_approval: bool = True
    min_confidence: float = 0.9
    max_commands_per_min: int = 5
    command_ttl_s: float = 30.0
    allowed_machine_states: tuple[MachineState, ...] = ()
    require_interlock_clear: bool = True
    parameter_limits: Mapping[str, tuple[float, float]] = field(default_factory=dict)
    clock: Any = None

    _replay: ReplayGuard = field(init=False, repr=False)
    _recent: deque[float] = field(default_factory=deque, init=False, repr=False)
    _extra_rules: list[PolicyRule] = field(default_factory=list, init=False, repr=False)
    denied_count: int = field(default=0, init=False)
    approved_count: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.min_confidence = validate_probability(self.min_confidence, "min_confidence")
        if self.max_commands_per_min <= 0:
            raise PolicyDeniedError("max_commands_per_min must be positive", details={})
        if self.command_ttl_s <= 0:
            raise PolicyDeniedError("command_ttl_s must be positive", details={})
        self._replay = ReplayGuard(ttl_s=max(self.command_ttl_s, 60.0))

    # -- extension ---------------------------------------------------------

    def add_rule(self, rule: PolicyRule) -> None:
        """Add an additional rule. Default rules cannot be removed."""
        self._extra_rules.append(rule)

    # -- evaluation --------------------------------------------------------

    def evaluate(self, request: CommandRequest) -> CommandDecision:
        """Evaluate a request and return a decision. Never raises for a DENY."""
        now = self.clock.wall() if self.clock is not None else time.time()
        results: list[PolicyRuleResult] = [
            self._rule_default_posture(request),
            self._rule_allowlist(request),
            self._rule_interlock_action(request),
            self._rule_machine_state(request),
            self._rule_operator_approval(request),
            self._rule_confidence(request),
            self._rule_freshness(request, now),
            self._rule_replay(request, now),
            self._rule_rate_limit(request, now),
            self._rule_interlock_status(request),
            self._rule_physical_limits(request),
        ]
        for rule in self._extra_rules:
            results.append(rule.evaluate(request, self))

        failures = [r for r in results if not r.passed]
        if failures:
            self.denied_count += 1
            return CommandDecision(
                request=request,
                outcome=DecisionOutcome.DENY,
                approved=False,
                rules=tuple(results),
                reason="; ".join(f"{r.rule}: {r.detail}" for r in failures),
                decided_at=now,
                policy_version=POLICY_VERSION,
            )

        self.approved_count += 1
        return CommandDecision(
            request=request,
            outcome=DecisionOutcome.APPROVE,
            approved=True,
            rules=tuple(results),
            reason="all policy rules passed",
            decided_at=now,
        )

    def approve_or_raise(self, request: CommandRequest) -> CommandDecision:
        """Evaluate and raise :class:`PolicyDeniedError` unless approved.

        Useful as a guard at a call site where proceeding without approval would be
        a programming error.
        """
        decision = self.evaluate(request)
        if not decision.approved:
            raise PolicyDeniedError(
                f"command {request.action!r} denied: {decision.reason}",
                details={"command_id": request.command_id, "action": request.action},
            )
        return decision

    # -- rules -------------------------------------------------------------

    def _rule_default_posture(self, request: CommandRequest) -> PolicyRuleResult:
        """Under a deny posture, never approve an action against unknown equipment state.

        This is what ``default_decision`` actually controls: an explicit, testable
        rule rather than a decorative field.
        """
        if self.default_decision != "deny":
            return PolicyRuleResult(
                rule="default_posture", passed=True,
                detail=f"posture is {self.default_decision!r} (non-default)",
            )
        if request.machine_state is MachineState.UNKNOWN:
            return PolicyRuleResult(
                rule="default_posture", passed=False,
                detail="deny posture: machine state is UNKNOWN, so no command "
                       "can be approved",
            )
        return PolicyRuleResult(rule="default_posture", passed=True,
                                detail="deny posture, machine state known")

    def _rule_allowlist(self, request: CommandRequest) -> PolicyRuleResult:
        allowed = request.action in self.allowed_actions
        return PolicyRuleResult(
            rule="action_allowlist",
            passed=allowed,
            detail="" if allowed else
            f"action {request.action!r} is not in the allow-list "
            f"{list(self.allowed_actions)}",
        )

    def _rule_interlock_action(self, request: CommandRequest) -> PolicyRuleResult:
        blocked = request.action in INTERLOCK_ACTIONS
        return PolicyRuleResult(
            rule="interlock_action",
            passed=not blocked,
            detail="" if not blocked else
            f"action {request.action!r} is safety-critical and can never be approved "
            f"by software policy",
        )

    def _rule_machine_state(self, request: CommandRequest) -> PolicyRuleResult:
        if not self.allowed_machine_states:
            return PolicyRuleResult(
                rule="machine_state", passed=False,
                detail="no machine states are configured as commandable",
            )
        allowed = request.machine_state in self.allowed_machine_states
        return PolicyRuleResult(
            rule="machine_state", passed=allowed,
            detail="" if allowed else
            f"machine state {request.machine_state.value} is not in "
            f"{[s.value for s in self.allowed_machine_states]}",
        )

    def _rule_operator_approval(self, request: CommandRequest) -> PolicyRuleResult:
        if not self.require_operator_approval:
            return PolicyRuleResult(rule="operator_approval", passed=True,
                                    detail="not required by policy")
        return PolicyRuleResult(
            rule="operator_approval",
            passed=request.operator_approved,
            detail="" if request.operator_approved else "operator approval is required",
        )

    def _rule_confidence(self, request: CommandRequest) -> PolicyRuleResult:
        if request.origin not in ("model", "rl", "api"):
            return PolicyRuleResult(rule="confidence_floor", passed=True,
                                    detail=f"not applicable for origin {request.origin!r}")
        passed = request.confidence >= self.min_confidence
        return PolicyRuleResult(
            rule="confidence_floor", passed=passed,
            detail="" if passed else
            f"confidence {request.confidence:.3f} is below the floor "
            f"{self.min_confidence:.3f}",
        )

    def _rule_freshness(self, request: CommandRequest, now: float) -> PolicyRuleResult:
        age = now - request.issued_at
        passed = 0.0 <= age <= self.command_ttl_s
        return PolicyRuleResult(
            rule="command_freshness", passed=passed,
            detail="" if passed else
            f"command age {age:.3f}s is outside the {self.command_ttl_s:.3f}s window",
        )

    def _rule_replay(self, request: CommandRequest, now: float) -> PolicyRuleResult:
        fresh = self._replay.check_and_consume(request.command_id, now)
        return PolicyRuleResult(
            rule="replay_protection", passed=fresh,
            detail="" if fresh else f"command id {request.command_id} was already consumed",
        )

    def _rule_rate_limit(self, request: CommandRequest, now: float) -> PolicyRuleResult:
        while self._recent and now - self._recent[0] > 60.0:
            self._recent.popleft()
        passed = len(self._recent) < self.max_commands_per_min
        if passed:
            self._recent.append(now)
        return PolicyRuleResult(
            rule="rate_limit", passed=passed,
            detail="" if passed else
            f"{len(self._recent)} commands in the last minute, ceiling is "
            f"{self.max_commands_per_min}",
        )

    def _rule_interlock_status(self, request: CommandRequest) -> PolicyRuleResult:
        if not self.require_interlock_clear:
            return PolicyRuleResult(rule="interlock_status", passed=True,
                                    detail="not required by policy")
        return PolicyRuleResult(
            rule="interlock_status", passed=request.interlock_clear,
            detail="" if request.interlock_clear else "physical interlock is not clear",
        )

    def _rule_physical_limits(self, request: CommandRequest) -> PolicyRuleResult:
        if not self.parameter_limits:
            return PolicyRuleResult(rule="physical_limits", passed=True,
                                    detail="no limits configured")
        for name, (low, high) in self.parameter_limits.items():
            if name not in request.parameters:
                continue
            try:
                value = float(request.parameters[name])
            except (TypeError, ValueError):
                return PolicyRuleResult(
                    rule="physical_limits", passed=False,
                    detail=f"parameter {name!r} is not numeric",
                )
            if not (low <= value <= high):
                return PolicyRuleResult(
                    rule="physical_limits", passed=False,
                    detail=f"parameter {name}={value} is outside [{low}, {high}]",
                )
        return PolicyRuleResult(rule="physical_limits", passed=True, detail="within limits")

    # -- introspection -----------------------------------------------------

    def describe(self) -> dict[str, Any]:
        return {
            "policy_version": POLICY_VERSION,
            "allowed_actions": list(self.allowed_actions),
            "require_operator_approval": self.require_operator_approval,
            "min_confidence": self.min_confidence,
            "max_commands_per_min": self.max_commands_per_min,
            "command_ttl_s": self.command_ttl_s,
            "allowed_machine_states": [s.value for s in self.allowed_machine_states],
            "require_interlock_clear": self.require_interlock_clear,
            "parameter_limits": {k: list(v) for k, v in self.parameter_limits.items()},
            "approved_count": self.approved_count,
            "denied_count": self.denied_count,
        }


def assert_no_bypass(command_id: str) -> None:  # pragma: no cover - documentation hook
    """Placeholder documenting that no bypass exists.

    Kept as an explicit, greppable marker: any future code that wants to skip the
    policy engine must be reviewed here.
    """
    raise CommandReplayError(
        "no bypass is permitted; every command must pass PolicyEngine.evaluate",
        details={"command_id": command_id},
    )
