"""Unit tests for Security & PolicyEngine Hardening (SOFIA-SAFE-*, SOFIA-SEC-*).

Verifies:
1. SOFIA-SAFE-001: PolicyEngine default DENY posture, interlock immutability, parameter bounds.
2. SOFIA-SAFE-002: Replay protection via Nonce + TTL window (threat T-20).
3. SOFIA-SAFE-003: LLM cannot actuate (copilot isolation from physical commands).
4. SOFIA-SEC-001: Secret scrubbing in mappings, logs, and free text.
"""

from __future__ import annotations

import time

import pytest

from sofia_ai.copilot.base import CopilotRequest, OfflineProvider
from sofia_ai.core.contracts import MachineState
from sofia_ai.core.errors import CommandReplayError
from sofia_ai.decision.commands import (
    INTERLOCK_ACTIONS,
    CommandRequest,
    DecisionOutcome,
)
from sofia_ai.decision.policy import PolicyEngine, ReplayGuard, assert_no_bypass
from sofia_ai.observability.redaction import REDACTED, redact_mapping, redact_text


class TestPolicyEngineHardening:
    """Verification of SOFIA-SAFE-001: Default DENY and safety constraints."""

    def test_default_posture_is_deny(self) -> None:
        engine = PolicyEngine()
        assert engine.default_decision == "deny"

        # Any command without explicit allowlist configuration is denied
        req = CommandRequest(
            command_id="cmd-1",
            device_id="pump-01",
            action="start",
            machine_state=MachineState.IDLE,
            operator_approved=True,
            confidence=0.99,
            issued_at=time.time(),
        )
        decision = engine.evaluate(req)
        assert decision.outcome == DecisionOutcome.DENY
        assert not decision.approved
        assert "action_allowlist" in decision.reason

    def test_unknown_machine_state_always_denied(self) -> None:
        engine = PolicyEngine(
            allowed_actions=("start",),
            allowed_machine_states=(MachineState.IDLE,),
            require_operator_approval=False,
        )
        req = CommandRequest(
            command_id="cmd-2",
            device_id="pump-01",
            action="start",
            machine_state=MachineState.UNKNOWN,
            confidence=0.99,
            issued_at=time.time(),
        )
        decision = engine.evaluate(req)
        assert decision.outcome == DecisionOutcome.DENY
        assert "UNKNOWN" in decision.reason

    def test_interlock_actions_never_approvable(self) -> None:
        for action in INTERLOCK_ACTIONS:
            engine = PolicyEngine(
                allowed_actions=(action,),
                allowed_machine_states=(MachineState.RUNNING, MachineState.IDLE),
                require_operator_approval=False,
                require_interlock_clear=False,
            )
            req = CommandRequest(
                command_id=f"cmd-interlock-{action}",
                device_id="safety-gate-01",
                action=action,
                machine_state=MachineState.RUNNING,
                confidence=1.0,
                issued_at=time.time(),
            )
            decision = engine.evaluate(req)
            assert decision.outcome == DecisionOutcome.DENY
            assert "safety-critical" in decision.reason

    def test_parameter_physical_limits(self) -> None:
        engine = PolicyEngine(
            allowed_actions=("set_speed",),
            allowed_machine_states=(MachineState.RUNNING,),
            require_operator_approval=False,
            require_interlock_clear=False,
            parameter_limits={"rpm": (0.0, 3600.0)},
        )

        # Within bounds
        req_ok = CommandRequest(
            command_id="cmd-speed-1",
            device_id="motor-01",
            action="set_speed",
            machine_state=MachineState.RUNNING,
            confidence=0.95,
            issued_at=time.time(),
            parameters={"rpm": 1800.0},
        )
        assert engine.evaluate(req_ok).approved

        # Exceeds bounds
        req_over = CommandRequest(
            command_id="cmd-speed-2",
            device_id="motor-01",
            action="set_speed",
            machine_state=MachineState.RUNNING,
            confidence=0.95,
            issued_at=time.time(),
            parameters={"rpm": 4500.0},
        )
        dec_over = engine.evaluate(req_over)
        assert not dec_over.approved
        assert "outside [0.0, 3600.0]" in dec_over.reason

    def test_no_bypass_assertion(self) -> None:
        with pytest.raises(CommandReplayError) as exc_info:
            assert_no_bypass("bypass-attempt-01")
        assert "no bypass is permitted" in str(exc_info.value)


class TestReplayProtection:
    """Verification of SOFIA-SAFE-002: Nonce + TTL Replay Protection."""

    def test_nonce_replay_rejection(self) -> None:
        engine = PolicyEngine(
            allowed_actions=("start",),
            allowed_machine_states=(MachineState.IDLE,),
            require_operator_approval=False,
            require_interlock_clear=False,
            command_ttl_s=30.0,
        )
        now = time.time()
        req = CommandRequest(
            command_id="nonce-unique-12345",
            device_id="fan-01",
            action="start",
            machine_state=MachineState.IDLE,
            confidence=0.95,
            issued_at=now,
        )

        first = engine.evaluate(req)
        assert first.approved, f"First evaluation failed: {first.reason}"

        # Replayed command with identical command_id must be rejected
        second = engine.evaluate(req)
        assert not second.approved
        assert "replay_protection" in second.reason

    def test_stale_command_rejection(self) -> None:
        engine = PolicyEngine(
            allowed_actions=("start",),
            allowed_machine_states=(MachineState.IDLE,),
            require_operator_approval=False,
            command_ttl_s=10.0,
        )
        now = time.time()
        stale_req = CommandRequest(
            command_id="cmd-stale-999",
            device_id="fan-01",
            action="start",
            machine_state=MachineState.IDLE,
            confidence=0.95,
            issued_at=now - 20.0,  # 20s old, exceeds 10s TTL
        )
        decision = engine.evaluate(stale_req)
        assert not decision.approved
        assert "command_freshness" in decision.reason

    def test_replay_guard_bounded_capacity(self) -> None:
        guard = ReplayGuard(ttl_s=60.0, capacity=10)
        now = time.time()

        for i in range(15):
            assert guard.check_and_consume(f"cmd-{i}", now)

        # Capacity bounded to 10
        assert len(guard) <= 10


class TestLLMCannotActuate:
    """Verification of SOFIA-SAFE-003: LLM isolation from physical actuation."""

    def test_copilot_has_no_actuation_surface(self) -> None:
        provider = OfflineProvider()
        # Ensure complete method returns purely text
        completion = provider.complete("Diagnostic reading: RMS=2.4 mm/s")
        assert isinstance(completion, str)

        req = CopilotRequest(
            question="What is the condition of the motor?",
            device_id="motor-01",
            evidence=({"metric": "rms", "observed": 2.4, "source": "vibration"},),
        )
        assert not hasattr(provider, "actuate")
        assert not hasattr(provider, "execute_command")
        assert not hasattr(req, "action")


class TestSecretScrubbing:
    """Verification of SOFIA-SEC-001: Secret Scrubbing."""

    def test_redact_mapping_scrubs_sensitive_keys(self) -> None:
        payload = {
            "device_id": "pump-01",
            "api_key": "sk-real-secret-1234567890",
            "password": "super-secret-password",
            "token": "tok_xyz",
            "authorization": "Bearer confidential_token",
            "nested": {
                "secret_key": "nested-secret-value",
                "normal_field": 42,
            },
        }

        redacted = redact_mapping(payload)
        assert redacted["api_key"] == REDACTED
        assert redacted["password"] == REDACTED
        assert redacted["token"] == REDACTED
        assert redacted["authorization"] == REDACTED
        assert redacted["nested"]["secret_key"] == REDACTED
        assert redacted["nested"]["normal_field"] == 42
        assert redacted["device_id"] == "pump-01"

    def test_redact_text_scrubs_tokens_and_truncates(self) -> None:
        raw_text = "Connecting with api_key=nvapi-secret1234567890 and Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
        scrubbed = redact_text(raw_text)
        assert "nvapi-secret" not in scrubbed
        assert REDACTED in scrubbed

        # Truncation test with space-separated text
        long_text = "telemetry log entry " * 200
        truncated = redact_text(long_text, max_length=100)
        assert len(truncated) <= 150
        assert "[truncated]" in truncated
