"""Unit tests for the advanced quantum computing emulator."""

from __future__ import annotations

import math

from sofia_ai.quantum import (
    QuantumCircuit,
    QuantumKernel,
    angle_encoding_circuit,
    compute_quantum_fidelity,
)


def test_quantum_circuit_hadamard_superposition() -> None:
    qc = QuantumCircuit(num_qubits=1)
    qc.h(0)
    probs = qc.statevector.probabilities
    # Equal superposition: |0> has prob 0.5, |1> has prob 0.5
    assert abs(probs[0] - 0.5) < 1e-7
    assert abs(probs[1] - 0.5) < 1e-7


def test_quantum_circuit_bell_state() -> None:
    # Create maximally entangled Bell state (|00> + |11>) / sqrt(2)
    qc = QuantumCircuit(num_qubits=2)
    qc.h(0).cx(0, 1)

    probs = qc.statevector.probabilities
    # |00> is idx 0, |11> is idx 3
    assert abs(probs[0] - 0.5) < 1e-7
    assert abs(probs[1] - 0.0) < 1e-7
    assert abs(probs[2] - 0.0) < 1e-7
    assert abs(probs[3] - 0.5) < 1e-7

    # Measure shots
    counts = qc.measure(shots=1000)
    assert "00" in counts
    assert "11" in counts
    assert counts["00"] + counts["11"] == 1000


def test_quantum_rotations_and_expectation() -> None:
    qc = QuantumCircuit(num_qubits=1)
    # Ry(pi) flips |0> to |1>
    qc.ry(0, math.pi)
    probs = qc.statevector.probabilities
    assert abs(probs[1] - 1.0) < 1e-7
    # <Z> on |1> is -1
    exp_z = qc.expectation_z(0)
    assert abs(exp_z - (-1.0)) < 1e-7


def test_angle_encoding_and_fidelity() -> None:
    x = [0.0, math.pi / 2.0]
    qc1 = angle_encoding_circuit(x)
    qc2 = angle_encoding_circuit(x)

    # Identical states have fidelity 1.0
    f_same = compute_quantum_fidelity(qc1.statevector, qc2.statevector)
    assert abs(f_same - 1.0) < 1e-7

    # Orthogonal states have fidelity 0.0
    y = [math.pi / 2.0, 0.0]
    qc3 = angle_encoding_circuit(y)
    f_diff = compute_quantum_fidelity(qc1.statevector, qc3.statevector)
    assert f_diff < 0.5


def test_quantum_kernel() -> None:
    kernel = QuantumKernel(num_qubits=3)
    vec_a = [0.1, 0.5, 0.9]
    vec_b = [0.1, 0.5, 0.9]
    k_self = kernel.evaluate(vec_a, vec_b)
    assert abs(k_self - 1.0) < 1e-6

    vec_c = [1.2, -0.8, 2.5]
    k_other = kernel.evaluate(vec_a, vec_c)
    assert 0.0 <= k_other <= 1.0
