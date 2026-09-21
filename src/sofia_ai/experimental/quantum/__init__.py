"""Experimental quantum-inspired simulation.

Quantum-inspired only: no quantum hardware, no demonstrated advantage.
See :mod:`sofia_ai.experimental.quantum.engine` for the honest description.
"""

from __future__ import annotations

from .engine import (
    MAX_QUBITS,
    MEMORY_CEILING_BYTES,
    QuantumNeuralEngine,
    QuantumState,
    classical_baseline,
    estimate_memory_bytes,
)

__all__ = [
    "MAX_QUBITS",
    "MEMORY_CEILING_BYTES",
    "QuantumNeuralEngine",
    "QuantumState",
    "classical_baseline",
    "estimate_memory_bytes",
]
