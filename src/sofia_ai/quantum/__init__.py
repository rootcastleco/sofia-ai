"""Advanced Quantum Computing and Quantum-Inspired Emulation.

Provides statevector simulation, universal quantum gate sets, parameterized
circuits (VQC), quantum feature maps, and quantum kernel estimation.
"""

from __future__ import annotations

from .emulator import (
    MAX_EMULATOR_QUBITS,
    QuantumCircuit,
    QuantumKernel,
    QuantumStateVector,
    angle_encoding_circuit,
    compute_quantum_fidelity,
)

__all__ = [
    "MAX_EMULATOR_QUBITS",
    "QuantumCircuit",
    "QuantumKernel",
    "QuantumStateVector",
    "angle_encoding_circuit",
    "compute_quantum_fidelity",
]
