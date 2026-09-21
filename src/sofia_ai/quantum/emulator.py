"""Advanced Quantum Computing Emulator.

Implements high-precision statevector simulation, universal quantum gate sets,
parameterized variational quantum circuits (VQC), quantum feature maps, and
quantum kernel estimation.

Mathematically rigorous: complex amplitudes in C^(2^n), unitary evolution,
projective measurements, and quantum fidelity calculation with zero external
dependencies (NumPy only).
"""

from __future__ import annotations

import cmath
import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Final

import numpy as np

__all__ = [
    "MAX_EMULATOR_QUBITS",
    "QuantumCircuit",
    "QuantumKernel",
    "QuantumStateVector",
    "angle_encoding_circuit",
    "compute_quantum_fidelity",
]

MAX_EMULATOR_QUBITS: Final[int] = 16

# Standard Single-Qubit Matrices (2x2 Complex)
GATE_H: Final[np.ndarray] = (1.0 / math.sqrt(2.0)) * np.array(
    [[1.0 + 0j, 1.0 + 0j], [1.0 + 0j, -1.0 + 0j]], dtype=np.complex128
)
GATE_X: Final[np.ndarray] = np.array([[0j, 1.0 + 0j], [1.0 + 0j, 0j]], dtype=np.complex128)
GATE_Y: Final[np.ndarray] = np.array([[0j, -1.0j], [1.0j, 0j]], dtype=np.complex128)
GATE_Z: Final[np.ndarray] = np.array([[1.0 + 0j, 0j], [0j, -1.0 + 0j]], dtype=np.complex128)
GATE_S: Final[np.ndarray] = np.array([[1.0 + 0j, 0j], [0j, 1.0j]], dtype=np.complex128)
GATE_T: Final[np.ndarray] = np.array(
    [[1.0 + 0j, 0j], [0j, cmath.exp(1j * math.pi / 4.0)]], dtype=np.complex128
)


@dataclass(slots=True)
class QuantumStateVector:
    """A pure quantum state vector in C^(2^n) satisfying sum(|alpha_i|^2) = 1."""

    num_qubits: int
    amplitudes: np.ndarray

    def __post_init__(self) -> None:
        dim = 1 << self.num_qubits
        if len(self.amplitudes) != dim:
            raise ValueError(f"Amplitudes length {len(self.amplitudes)} does not match 2^{self.num_qubits} = {dim}")

    @property
    def probabilities(self) -> np.ndarray:
        """Measurement probabilities for each computational basis state."""
        return np.abs(self.amplitudes) ** 2

    def sample_shots(self, shots: int = 1024, rng: np.random.Generator | None = None) -> dict[str, int]:
        """Simulate projective measurement collapse over computational basis."""
        probs = self.probabilities
        probs = probs / np.sum(probs)  # Numerical stability
        generator = rng or np.random.default_rng()
        dim = len(probs)
        indices = generator.choice(dim, size=shots, p=probs)
        counts = Counter(indices)

        fmt = f"0{self.num_qubits}b"
        return {format(k, fmt): v for k, v in sorted(counts.items())}


class QuantumCircuit:
    """Universal Quantum Circuit Emulation Engine."""

    def __init__(self, num_qubits: int, rng: np.random.Generator | None = None) -> None:
        if num_qubits < 1 or num_qubits > MAX_EMULATOR_QUBITS:
            raise ValueError(f"num_qubits must be in 1..{MAX_EMULATOR_QUBITS}, received {num_qubits}")
        self.num_qubits = num_qubits
        self.dim = 1 << num_qubits
        self.rng = rng or np.random.default_rng()

        # Initialize to ground state |0...0>
        self._state = np.zeros(self.dim, dtype=np.complex128)
        self._state[0] = 1.0 + 0j

    @property
    def statevector(self) -> QuantumStateVector:
        """Return the current quantum state vector."""
        return QuantumStateVector(self.num_qubits, self._state.copy())

    def reset(self) -> None:
        """Reset circuit to |0...0>."""
        self._state.fill(0j)
        self._state[0] = 1.0 + 0j

    def apply_single_qubit_gate(self, gate: np.ndarray, target: int) -> None:
        """Apply a 2x2 unitary matrix to target qubit."""
        if target < 0 or target >= self.num_qubits:
            raise ValueError(f"Target qubit {target} out of range [0, {self.num_qubits - 1}]")

        stride = 1 << target
        for i in range(0, self.dim, 2 * stride):
            for j in range(stride):
                idx0 = i + j
                idx1 = idx0 + stride
                v0 = self._state[idx0]
                v1 = self._state[idx1]
                self._state[idx0] = gate[0, 0] * v0 + gate[0, 1] * v1
                self._state[idx1] = gate[1, 0] * v0 + gate[1, 1] * v1

    def h(self, qubit: int) -> QuantumCircuit:
        """Hadamard gate."""
        self.apply_single_qubit_gate(GATE_H, qubit)
        return self

    def x(self, qubit: int) -> QuantumCircuit:
        """Pauli-X (NOT) gate."""
        self.apply_single_qubit_gate(GATE_X, qubit)
        return self

    def y(self, qubit: int) -> QuantumCircuit:
        """Pauli-Y gate."""
        self.apply_single_qubit_gate(GATE_Y, qubit)
        return self

    def z(self, qubit: int) -> QuantumCircuit:
        """Pauli-Z gate."""
        self.apply_single_qubit_gate(GATE_Z, qubit)
        return self

    def s(self, qubit: int) -> QuantumCircuit:
        """Phase (S) gate."""
        self.apply_single_qubit_gate(GATE_S, qubit)
        return self

    def t(self, qubit: int) -> QuantumCircuit:
        """T gate (pi/8 gate)."""
        self.apply_single_qubit_gate(GATE_T, qubit)
        return self

    def rx(self, qubit: int, theta: float) -> QuantumCircuit:
        """Rotation around X-axis by angle theta."""
        c = math.cos(theta / 2.0)
        s = math.sin(theta / 2.0)
        gate = np.array([[c + 0j, -1j * s], [-1j * s, c + 0j]], dtype=np.complex128)
        self.apply_single_qubit_gate(gate, qubit)
        return self

    def ry(self, qubit: int, theta: float) -> QuantumCircuit:
        """Rotation around Y-axis by angle theta."""
        c = math.cos(theta / 2.0)
        s = math.sin(theta / 2.0)
        gate = np.array([[c + 0j, -s + 0j], [s + 0j, c + 0j]], dtype=np.complex128)
        self.apply_single_qubit_gate(gate, qubit)
        return self

    def rz(self, qubit: int, theta: float) -> QuantumCircuit:
        """Rotation around Z-axis by angle theta."""
        gate = np.array(
            [[cmath.exp(-1j * theta / 2.0), 0j], [0j, cmath.exp(1j * theta / 2.0)]],
            dtype=np.complex128,
        )
        self.apply_single_qubit_gate(gate, qubit)
        return self

    def cx(self, control: int, target: int) -> QuantumCircuit:
        """Controlled-NOT (CNOT) entangling gate."""
        if control == target:
            raise ValueError("Control and target qubits must be distinct")
        c_mask = 1 << control
        t_mask = 1 << target

        for i in range(self.dim):
            # Only apply X to target when control is 1 and target is 0 (to swap pair once)
            if (i & c_mask) and not (i & t_mask):
                pair = i ^ t_mask
                temp = self._state[i]
                self._state[i] = self._state[pair]
                self._state[pair] = temp
        return self

    def cz(self, control: int, target: int) -> QuantumCircuit:
        """Controlled-Z entangling gate."""
        if control == target:
            raise ValueError("Control and target qubits must be distinct")
        mask = (1 << control) | (1 << target)
        for i in range(self.dim):
            if (i & mask) == mask:
                self._state[i] = -self._state[i]
        return self

    def expectation_z(self, qubit: int) -> float:
        """Compute exact expectation value <psi| Z_qubit |psi>."""
        q_mask = 1 << qubit
        exp_val = 0.0
        probs = self.statevector.probabilities
        for i in range(self.dim):
            sign = -1.0 if (i & q_mask) else 1.0
            exp_val += sign * float(probs[i])
        return exp_val

    def measure(self, shots: int = 1024) -> dict[str, int]:
        """Perform projective measurement sampling."""
        return self.statevector.sample_shots(shots=shots, rng=self.rng)


def angle_encoding_circuit(
    features: Sequence[float],
    rng: np.random.Generator | None = None,
) -> QuantumCircuit:
    """Encode classical feature vector into quantum state via angle encoding:

    |x> = Otimes_{i} [ cos(x_i) |0> + sin(x_i) |1> ]
    """
    n = len(features)
    qc = QuantumCircuit(num_qubits=min(n, MAX_EMULATOR_QUBITS), rng=rng)
    for i in range(qc.num_qubits):
        val = float(features[i])
        qc.ry(i, 2.0 * val)
    return qc


def compute_quantum_fidelity(state_a: QuantumStateVector, state_b: QuantumStateVector) -> float:
    """Calculate quantum state fidelity F = |<psi_a | psi_b>|^2."""
    if state_a.num_qubits != state_b.num_qubits:
        raise ValueError("States must have identical qubit counts")
    inner_prod = np.vdot(state_a.amplitudes, state_b.amplitudes)
    return float(np.abs(inner_prod) ** 2)


class QuantumKernel:
    """Quantum Kernel Estimator for Quantum Machine Learning & Anomaly Detection.

    Computes the transition amplitude between two feature-encoded quantum states:
        K(x, y) = |<psi(x) | psi(y)>|^2
    """

    def __init__(self, num_qubits: int) -> None:
        self.num_qubits = min(num_qubits, MAX_EMULATOR_QUBITS)

    def evaluate(self, x: Sequence[float], y: Sequence[float]) -> float:
        """Compute quantum kernel value K(x, y) in [0, 1]."""
        qc_x = angle_encoding_circuit(x[: self.num_qubits])
        qc_y = angle_encoding_circuit(y[: self.num_qubits])
        return compute_quantum_fidelity(qc_x.statevector, qc_y.statevector)
