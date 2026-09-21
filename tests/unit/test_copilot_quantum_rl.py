"""Unit tests for the copilot layer, experimental quantum module and RL buffers."""

from __future__ import annotations

import math

import numpy as np
import pytest

from sofia_ai.copilot import (
    CopilotRequest,
    CopilotResponse,
    EvidenceCopilot,
    NullProvider,
    OfflineProvider,
    summarize_evidence,
)
from sofia_ai.core.errors import InsufficientDataError, ValidationError
from sofia_ai.experimental.quantum import (
    MAX_QUBITS,
    QuantumNeuralEngine,
    QuantumState,
    classical_baseline,
    estimate_memory_bytes,
)
from sofia_ai.learning.rl import (
    Experience,
    PrioritizedReplayBuffer,
    ReplayBuffer,
    torch_available,
)

# ---------------------------------------------------------------------------
# Copilot
# ---------------------------------------------------------------------------


class TestCopilot:
    def test_grounded_summary(self) -> None:
        copilot = EvidenceCopilot()
        request = CopilotRequest(
            question="Why is the pump flagged?",
            device_id="pump-01",
            evidence=({"source": "anomaly_score", "metric": "anomaly_score",
                       "observed": 12.0, "reference": 3.0, "quality": "GOOD"},),
        )
        response = copilot.ask(request)
        assert response.grounded
        assert "anomaly_score" in response.text
        assert "pump-01" in response.text

    def test_states_it_is_not_a_diagnosis(self) -> None:
        copilot = EvidenceCopilot()
        response = copilot.ask(CopilotRequest(question="q", device_id="d"))
        assert "not a fault diagnosis" in response.text

    def test_no_evidence_is_stated(self) -> None:
        copilot = EvidenceCopilot()
        response = copilot.ask(CopilotRequest(question="q", device_id="d"))
        assert "none recorded" in response.text

    def test_provider_text_is_opt_in(self) -> None:
        class Provider:
            def complete(self, prompt, *, max_tokens=512):
                return "MODEL COMMENTARY"

        # Not enabled: provider text must not appear.
        copilot = EvidenceCopilot(provider=Provider())
        response = copilot.ask(CopilotRequest(question="q", device_id="d"))
        assert "MODEL COMMENTARY" not in response.text

        # Enabled: it appears under an explicit non-authoritative heading.
        enabled = EvidenceCopilot(provider=Provider(), allow_provider_text=True)
        response = enabled.ask(CopilotRequest(question="q", device_id="d"))
        assert "MODEL COMMENTARY" in response.text
        assert "Non-authoritative" in response.text

    def test_provider_failure_is_contained(self) -> None:
        class Broken:
            def complete(self, prompt, *, max_tokens=512):
                raise RuntimeError("api down")

        copilot = EvidenceCopilot(provider=Broken(), allow_provider_text=True)
        response = copilot.ask(CopilotRequest(question="q", device_id="d"))
        assert response.metadata.get("provider_error") == "RuntimeError"

    def test_offline_provider_is_deterministic(self) -> None:
        provider = OfflineProvider()
        assert provider.complete("a\nb\nc") == provider.complete("a\nb\nc")

    def test_null_provider_declines(self) -> None:
        assert NullProvider().complete("x") == ""

    def test_request_validates(self) -> None:
        with pytest.raises(ValueError):
            CopilotRequest(question="  ", device_id="d")

    def test_summarize_evidence(self) -> None:
        lines = summarize_evidence([{"source": "s", "metric": "m", "observed": 1.5,
                                     "reference": 1.0, "quality": "GOOD"}])
        assert len(lines) == 1
        assert "1.5" in lines[0]

    def test_response_serialization(self) -> None:
        payload = CopilotResponse(text="t", provider="p", grounded=True).to_dict()
        assert payload["grounded"] is True

    def test_copilot_emits_no_commands(self) -> None:
        """The copilot surface must expose no actuation path."""
        copilot = EvidenceCopilot()
        assert not hasattr(copilot, "command")
        assert not hasattr(copilot, "actuate")


# ---------------------------------------------------------------------------
# Quantum (experimental)
# ---------------------------------------------------------------------------


class TestQuantumEnvelope:
    def test_memory_estimate_grows_as_four_to_the_n(self) -> None:
        assert estimate_memory_bytes(8, 1) == 256 * 256 * 8
        assert estimate_memory_bytes(9, 1) == 4 * estimate_memory_bytes(8, 1)

    def test_qubits_ceiling(self) -> None:
        with pytest.raises(ValidationError):
            QuantumNeuralEngine(num_qubits=MAX_QUBITS + 1)

    def test_memory_ceiling(self) -> None:
        # 12 qubits at depth 3 is ~400 MB; depth 6 exceeds the 512 MiB ceiling.
        with pytest.raises(ValidationError):
            QuantumNeuralEngine(num_qubits=12, entanglement_depth=6)

    def test_superposition_is_unit_norm(self) -> None:
        engine = QuantumNeuralEngine(num_qubits=3, seed=1)
        state = engine.create_superposition()
        assert math.isclose(float(np.linalg.norm(state.amplitude)), 1.0, abs_tol=1e-12)

    def test_dimension(self) -> None:
        assert QuantumNeuralEngine(num_qubits=4, seed=1).dimension == 16

    def test_forward_is_normalized(self) -> None:
        engine = QuantumNeuralEngine(num_qubits=4, entanglement_depth=2, seed=1)
        output = engine.forward(np.random.default_rng(0).normal(size=16))
        assert math.isclose(float(np.sum(output)), 1.0, rel_tol=1e-6)
        assert float(np.min(output)) >= 0.0

    def test_forward_is_deterministic(self) -> None:
        data = np.random.default_rng(0).normal(size=16)
        a = QuantumNeuralEngine(num_qubits=4, seed=7).forward(data)
        b = QuantumNeuralEngine(num_qubits=4, seed=7).forward(data)
        assert np.allclose(a, b)

    def test_layer_bounds(self) -> None:
        engine = QuantumNeuralEngine(num_qubits=3, entanglement_depth=2, seed=1)
        with pytest.raises(ValidationError):
            engine.apply_entanglement(engine.create_superposition(), layer=5)

    def test_entropy_bounds(self) -> None:
        engine = QuantumNeuralEngine(num_qubits=4, seed=1)
        features = engine.get_quantum_features(np.random.default_rng(0).normal(size=16))
        assert 0.0 <= features["entropy"] <= math.log2(16) + 1e-9

    def test_optimize_no_longer_raises(self) -> None:
        """The 1.x optimize() path was dimensionally broken and always raised."""
        engine = QuantumNeuralEngine(num_qubits=3, entanglement_depth=2, seed=1)
        engine.optimize(np.ones(8) * 0.01, 0.01)
        assert engine.weights[0].shape == (8, 8)

    def test_optimize_validates_shape(self) -> None:
        engine = QuantumNeuralEngine(num_qubits=3, entanglement_depth=2, seed=1)
        with pytest.raises(ValidationError):
            engine.optimize(np.ones(4), 0.01)

    def test_measure_samples_a_distribution(self) -> None:
        engine = QuantumNeuralEngine(num_qubits=3, seed=1)
        index, probability = engine.measure(engine.create_superposition())
        assert 0 <= index < 8
        assert 0.0 <= probability <= 1.0

    def test_classical_baseline_shape(self) -> None:
        baseline = classical_baseline(np.random.default_rng(0).normal(size=16), 16, seed=1)
        assert baseline.shape == (16,)

    def test_describe_is_honest(self) -> None:
        payload = QuantumNeuralEngine(num_qubits=4, seed=1).describe()
        assert payload["quantum_hardware"] is False
        assert "quantum-inspired" in payload["classification"]

    def test_quantum_state_requires_amplitude(self) -> None:
        with pytest.raises(ValidationError):
            QuantumState(amplitude=np.array([]))

    def test_no_quantum_advantage_claim_in_docstrings(self) -> None:
        import sofia_ai.experimental.quantum.engine as engine_module

        text = (engine_module.__doc__ or "").lower()
        assert "no quantum hardware" in text
        assert "not a quantum" in text or "no quantum" in text


# ---------------------------------------------------------------------------
# RL buffers (no torch required)
# ---------------------------------------------------------------------------


class TestReplayBuffers:
    def test_capacity_is_enforced(self) -> None:
        buffer: ReplayBuffer[int] = ReplayBuffer(5)
        for i in range(10):
            buffer.push(i)
        assert len(buffer) == 5
        assert buffer.evicted == 5

    def test_sample_is_bounded_by_capacity(self) -> None:
        buffer: ReplayBuffer[int] = ReplayBuffer(4)
        buffer.extend(range(4))
        with pytest.raises(ValidationError):
            buffer.sample(5)

    def test_sample_is_deterministic_for_a_seed(self) -> None:
        a = ReplayBuffer(20, seed=5)
        b = ReplayBuffer(20, seed=5)
        a.extend(range(20))
        b.extend(range(20))
        assert a.sample(5) == b.sample(5)

    def test_reseed_does_not_touch_global_rng(self) -> None:
        buffer = ReplayBuffer(4, seed=1)
        buffer.reseed(2)
        assert buffer.seed == 2

    def test_prioritized_sampling(self) -> None:
        buffer = PrioritizedReplayBuffer(10, seed=1)
        for i in range(4):
            buffer.push_with_priority(
                Experience(state=np.zeros(4), action=0, reward=0.0,
                           next_state=np.zeros(4), done=False),
                priority=float(i + 1),
            )
        samples, indices = buffer.sample_prioritized(2)
        assert len(samples) == 2
        assert indices.size == 2

    def test_prioritized_validates_priority(self) -> None:
        buffer = PrioritizedReplayBuffer(4, seed=1)
        with pytest.raises(InsufficientDataError):
            buffer.push_with_priority(
                Experience(state=np.zeros(2), action=0, reward=0.0,
                           next_state=np.zeros(2), done=False),
                priority=-1.0,
            )

    def test_experience_validates_shapes(self) -> None:
        with pytest.raises(InsufficientDataError):
            Experience(state=np.zeros(4), action=0, reward=0.0,
                       next_state=np.zeros(2), done=False)

    def test_experience_rejects_non_finite(self) -> None:
        with pytest.raises(InsufficientDataError):
            Experience(state=np.array([float("nan")]), action=0, reward=0.0,
                       next_state=np.zeros(1), done=False)


@pytest.mark.skipif(not torch_available(), reason="torch extra not installed")
class TestRLWithTorch:
    def test_agent_is_constructible(self) -> None:
        from sofia_ai.learning.rl import SofiaRLAgent

        agent = SofiaRLAgent(state_size=4, action_size=2, seed=1)
        assert agent.act(np.zeros(4)) in (0, 1)


def test_torch_absence_is_reported() -> None:
    assert isinstance(torch_available(), bool)
