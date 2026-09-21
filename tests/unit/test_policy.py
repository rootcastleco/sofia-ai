"""Unit tests for the command safety boundary (SOFIA-SAFE-*).

The central assertion of these tests: the default posture is DENY, and no
combination of inputs produces an approval unless every rule passes.
"""

from __future__ import annotations

import pytest

from sofia_ai.core.contracts import MachineState
from sofia_ai.core.errors import CommandReplayError, PolicyDeniedError, ValidationError
from sofia_ai.decision import (
    INTERLOCK_ACTIONS,
    CommandDecision,
    CommandPriority,
    CommandRequest,
    DecisionOutcome,
    PolicyEngine,
    PolicyRule,
    PolicyRuleResult,
    ReplayGuard,
)


@pytest.fixture
def engine() -> PolicyEngine:
    return PolicyEngine(
        allowed_actions=("set_speed", "start_pump"),
        require_operator_approval=True,
        min_confidence=0.9,
        max_commands_per_min=3,
        command_ttl_s=30.0,
        allowed_machine_states=(MachineState.RUNNING,),
        require_interlock_clear=True,
        parameter_limits={"target_rpm": (0.0, 3000.0)},
        clock=_FakeClock(1_700_000_000.0),
    )


class _FakeClock:
    def __init__(self, now: float) -> None:
        self.now = now

    def wall(self) -> float:
        return self.now


def _request(**overrides) -> CommandRequest:
    base = {"action": "set_speed", "device_id": "pump-01"}
    base.update(overrides)
    return CommandRequest(**base)  # type: ignore[arg-type]


class TestDefaults:
    def test_empty_engine_denies_everything(self) -> None:
        decision = PolicyEngine().evaluate(_request())
        assert decision.outcome is DecisionOutcome.DENY
        assert not decision.approved

    def test_deny_is_default_when_nothing_configured(self) -> None:
        engine = PolicyEngine(allowed_actions=("set_speed",))
        decision = engine.evaluate(_request())
        assert not decision.approved

    def test_denial_is_explainable(self) -> None:
        decision = PolicyEngine().evaluate(_request())
        assert decision.reason
        assert decision.denied_rules()

    def test_denial_serializes(self) -> None:
        payload = PolicyEngine().evaluate(_request()).to_dict()
        assert payload["approved"] is False
        assert payload["rules"]


class TestRules:
    def test_unlisted_action_denied(self, engine: PolicyEngine) -> None:
        decision = engine.evaluate(_request(action="shutdown_grid"))
        assert not decision.approved
        assert any(r.rule == "action_allowlist" for r in decision.denied_rules())

    def test_interlock_action_never_approved(self, engine: PolicyEngine) -> None:
        for action in INTERLOCK_ACTIONS:
            decision = engine.evaluate(CommandRequest(
                action=action, device_id="pump-01", operator_approved=True,
                machine_state=MachineState.RUNNING, interlock_clear=True,
                confidence=1.0, issued_at=1_700_000_000.0))
            assert not decision.approved
            assert any(r.rule == "interlock_action" for r in decision.denied_rules())

    def test_wrong_machine_state_denied(self, engine: PolicyEngine) -> None:
        decision = engine.evaluate(_request(machine_state=MachineState.MAINTENANCE))
        assert any(r.rule == "machine_state" for r in decision.denied_rules())

    def test_operator_approval_required(self, engine: PolicyEngine) -> None:
        decision = engine.evaluate(_request(machine_state=MachineState.RUNNING))
        assert any(r.rule == "operator_approval" for r in decision.denied_rules())

    def test_confidence_floor(self, engine: PolicyEngine) -> None:
        decision = engine.evaluate(_request(
            machine_state=MachineState.RUNNING, operator_approved=True,
            interlock_clear=True, origin="model", confidence=0.5))
        assert any(r.rule == "confidence_floor" for r in decision.denied_rules())

    def test_confidence_floor_not_applied_to_operator(self, engine: PolicyEngine) -> None:
        decision = engine.evaluate(_request(
            machine_state=MachineState.RUNNING, operator_approved=True,
            interlock_clear=True, origin="operator", confidence=0.0,
            issued_at=1_700_000_000.0))
        assert not any(r.rule == "confidence_floor" for r in decision.denied_rules())

    def test_freshness_window(self, engine: PolicyEngine) -> None:
        decision = engine.evaluate(_request(
            machine_state=MachineState.RUNNING, operator_approved=True,
            interlock_clear=True, issued_at=1_699_000_000.0))
        assert any(r.rule == "command_freshness" for r in decision.denied_rules())

    def test_future_command_denied(self, engine: PolicyEngine) -> None:
        decision = engine.evaluate(_request(
            machine_state=MachineState.RUNNING, operator_approved=True,
            interlock_clear=True, issued_at=1_700_001_000.0))
        assert any(r.rule == "command_freshness" for r in decision.denied_rules())

    def test_interlock_status(self, engine: PolicyEngine) -> None:
        decision = engine.evaluate(_request(
            machine_state=MachineState.RUNNING, operator_approved=True,
            interlock_clear=False, issued_at=1_700_000_000.0))
        assert any(r.rule == "interlock_status" for r in decision.denied_rules())

    def test_physical_limits(self, engine: PolicyEngine) -> None:
        decision = engine.evaluate(_request(
            machine_state=MachineState.RUNNING, operator_approved=True,
            interlock_clear=True, origin="operator", issued_at=1_700_000_000.0,
            parameters={"target_rpm": 9000.0}))
        assert any(r.rule == "physical_limits" for r in decision.denied_rules())

    def test_non_numeric_parameter(self, engine: PolicyEngine) -> None:
        decision = engine.evaluate(_request(
            machine_state=MachineState.RUNNING, operator_approved=True,
            interlock_clear=True, origin="operator", issued_at=1_700_000_000.0,
            parameters={"target_rpm": "fast"}))
        assert any(r.rule == "physical_limits" for r in decision.denied_rules())


class TestApproval:
    def test_all_rules_pass(self, engine: PolicyEngine) -> None:
        decision = engine.evaluate(_request(
            machine_state=MachineState.RUNNING, operator_approved=True,
            interlock_clear=True, origin="operator", issued_at=1_700_000_000.0,
            parameters={"target_rpm": 1500.0}))
        assert decision.approved
        assert decision.outcome is DecisionOutcome.APPROVE
        assert all(r.passed for r in decision.rules)

    def test_replay_of_same_id_denied(self, engine: PolicyEngine) -> None:
        request = _request(machine_state=MachineState.RUNNING, operator_approved=True,
                           interlock_clear=True, origin="operator",
                           issued_at=1_700_000_000.0)
        assert engine.evaluate(request).approved
        decision = engine.evaluate(request)
        assert not decision.approved
        assert any(r.rule == "replay_protection" for r in decision.denied_rules())

    def test_rate_limit(self, engine: PolicyEngine) -> None:
        for i in range(3):
            engine.evaluate(_request(machine_state=MachineState.RUNNING,
                                     operator_approved=True, interlock_clear=True,
                                     origin="operator", issued_at=1_700_000_000.0,
                                     command_id=f"rate-{i}"))
        decision = engine.evaluate(_request(
            machine_state=MachineState.RUNNING, operator_approved=True,
            interlock_clear=True, origin="operator", issued_at=1_700_000_000.0,
            command_id="rate-overflow"))
        assert any(r.rule == "rate_limit" for r in decision.denied_rules())

    def test_approve_or_raise(self, engine: PolicyEngine) -> None:
        decision = engine.approve_or_raise(_request(
            machine_state=MachineState.RUNNING, operator_approved=True,
            interlock_clear=True, origin="operator", issued_at=1_700_000_000.0))
        assert decision.approved

    def test_approve_or_raise_denies(self, engine: PolicyEngine) -> None:
        with pytest.raises(PolicyDeniedError):
            engine.approve_or_raise(_request())

    def test_counters(self, engine: PolicyEngine) -> None:
        engine.evaluate(_request())
        assert engine.denied_count == 1


class TestCommandRequest:
    def test_action_is_normalized(self) -> None:
        assert CommandRequest(action="SET_SPEED", device_id="d").action == "set_speed"

    def test_rejects_control_characters(self) -> None:
        with pytest.raises(ValidationError):
            CommandRequest(action="set\npeed", device_id="d")

    def test_parameter_limit(self) -> None:
        with pytest.raises(ValidationError):
            CommandRequest(action="a", device_id="d",
                           parameters={f"k{i}": 1 for i in range(64)})

    def test_confidence_must_be_probability(self) -> None:
        with pytest.raises(ValidationError):
            CommandRequest(action="a", device_id="d", confidence=2.0)

    def test_with_approval(self) -> None:
        request = CommandRequest(action="a", device_id="d")
        assert request.with_approval().operator_approved

    def test_serialization(self) -> None:
        payload = CommandRequest(action="a", device_id="d",
                                 priority=CommandPriority.HIGH).to_dict()
        assert payload["priority"] == "HIGH"


class TestReplayGuard:
    def test_fresh_then_replayed(self) -> None:
        guard = ReplayGuard(ttl_s=10.0)
        assert guard.check_and_consume("a", 0.0)
        assert not guard.check_and_consume("a", 1.0)

    def test_ttl_expiry(self) -> None:
        guard = ReplayGuard(ttl_s=10.0)
        assert guard.check_and_consume("a", 0.0)
        assert guard.check_and_consume("a", 100.0)

    def test_capacity_is_bounded(self) -> None:
        guard = ReplayGuard(ttl_s=1000.0, capacity=4)
        for i in range(10):
            guard.check_and_consume(f"id-{i}", float(i))
        assert len(guard) <= 4

    def test_rejects_bad_ttl(self) -> None:
        with pytest.raises(PolicyDeniedError):
            ReplayGuard(ttl_s=0.0)


class TestExtension:
    def test_custom_rule_can_deny(self, engine: PolicyEngine) -> None:
        class DenyEverything(PolicyRule):
            name = "custom_deny"

            def evaluate(self, request, policy_engine):
                return PolicyRuleResult(rule="custom_deny", passed=False,
                                        detail="custom policy")

        engine.add_rule(DenyEverything())
        decision = engine.evaluate(_request(
            machine_state=MachineState.RUNNING, operator_approved=True,
            interlock_clear=True, origin="operator", issued_at=1_700_000_000.0))
        assert not decision.approved
        assert any(r.rule == "custom_deny" for r in decision.denied_rules())

    def test_describe(self, engine: PolicyEngine) -> None:
        payload = engine.describe()
        assert payload["policy_version"] == "1.0.0"
        assert "set_speed" in payload["allowed_actions"]


class TestNoBypass:
    def test_no_bypass_marker_raises(self) -> None:
        from sofia_ai.decision.policy import assert_no_bypass

        with pytest.raises(CommandReplayError):
            assert_no_bypass("x")

    def test_decision_cannot_be_forged(self, engine: PolicyEngine) -> None:
        """approved=True is only reachable through a passing evaluation."""
        decision: CommandDecision = engine.evaluate(_request())
        assert decision.approved is False
        assert decision.outcome is DecisionOutcome.DENY
