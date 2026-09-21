"""Quantum-inspired simulation — EXPERIMENTAL.

This module preserves the pre-2.0 ``QuantumNeuralEngine`` API while relocating it
behind an explicit experimental boundary and enforcing a resource ceiling.

## What it actually computes

``forward(x)`` performs:

1. ``v0 = x / ||x||``                              (L2 normalization)
2. ``v_k = W_k v_{k-1} / ||W_k v_{k-1}||``         (normalized matrix product, k = depth)
3. ``p = |v_depth|²``                              (element-wise square)
4. ``out = p + 0.1 * sin(phi)``                    (phase term; phi is always 0 here)

Each ``W_k`` is a random orthogonal matrix obtained from the SVD of a Gaussian
matrix (``u @ v``). Therefore the transformation is a composition of random
rotations followed by an element-wise nonlinearity. It is a classical
deterministic map on a ``2**num_qubits``-dimensional real vector.

## What it is not

No quantum hardware, no quantum circuit, no tensor-product state space, no
entanglement, no measurement collapse. ``apply_entanglement`` is a matrix-vector
product; ``measure`` is ``numpy.random.choice`` over a probability vector. The
quantum vocabulary in the original code described operations that do not have
those physical semantics.

## Resource envelope

Memory is ``O(depth * 4**num_qubits)`` float64 values because each "gate" is a
dense ``(2**n, 2**n)`` matrix. The pre-2.0 configuration allowed up to 20 qubits,
i.e. 26 TB of weights at the default depth — configurations that cannot be
constructed on any machine. A hard ceiling is now enforced.

## Measured advantage

None demonstrated. ``benchmarks/bench_quantum_baseline.py`` compares this engine
against a classical random-projection baseline. See
``docs/experimental-quantum.md`` for the result and its interpretation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Final

import numpy as np

from ...core.errors import ValidationError
from ...core.validation import validate_positive_int

__all__ = [
    "MAX_QUBITS",
    "MEMORY_CEILING_BYTES",
    "QuantumNeuralEngine",
    "QuantumState",
    "classical_baseline",
    "estimate_memory_bytes",
]

#: Hard ceiling on simulated qubits. Enforced because weight memory is
#: ``depth * 4**n`` float64 values.
MAX_QUBITS: Final[int] = 12

#: Refuse to allocate more than this many bytes for simulated weights.
MEMORY_CEILING_BYTES: Final[int] = 512 * 1024 * 1024  # 512 MiB

_ITEMSIZE: Final[int] = 8  # float64


def estimate_memory_bytes(num_qubits: int, entanglement_depth: int) -> int:
    """Bytes required by the simulated weight matrices."""
    dimension = 2 ** int(num_qubits)
    total = int(entanglement_depth) * dimension * dimension * _ITEMSIZE
    return int(total)


@dataclass(slots=True)
class QuantumState:
    """A normalized real amplitude vector plus an unused phase vector.

    The ``phase`` field is retained for API compatibility. In this simulation it is
    always zero, so the ``sin(phase)`` contribution in ``_decode_output`` is always
    zero — retained so the preserved API produces identical values.
    """

    amplitude: np.ndarray
    phase: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))

    def __post_init__(self) -> None:
        self.amplitude = np.asarray(self.amplitude, dtype=np.float64)
        if self.phase.size == 0:
            self.phase = np.zeros_like(self.amplitude)
        else:
            self.phase = np.asarray(self.phase, dtype=np.float64)
        if self.amplitude.size == 0:
            raise ValidationError("QuantumState amplitude must not be empty", details={})

    @property
    def dimension(self) -> int:
        return int(self.amplitude.size)

    @property
    def probabilities(self) -> np.ndarray:
        """Born-rule-style probabilities ``|amplitude|²``."""
        p = np.abs(self.amplitude) ** 2
        total = float(np.sum(p))
        return p / total if total > 0 else p


class QuantumNeuralEngine:
    """Quantum-inspired dense-map simulator. EXPERIMENTAL — no quantum advantage.

    Args:
        num_qubits: Simulated qubit count. Capped at :data:`MAX_QUBITS`.
        entanglement_depth: Number of normalized matrix layers.
        optimization_steps: Retained for API compatibility; unused by ``forward``.
        seed: Seed for this engine's owned generator. The global RNG is untouched.
    """

    def __init__(
        self,
        num_qubits: int = 8,
        entanglement_depth: int = 3,
        optimization_steps: int = 50,
        *,
        seed: int = 0,
    ) -> None:
        num_qubits = validate_positive_int(num_qubits, "num_qubits")
        entanglement_depth = validate_positive_int(entanglement_depth, "entanglement_depth",
                                                   maximum=64)
        if num_qubits > MAX_QUBITS:
            raise ValidationError(
                f"num_qubits={num_qubits} exceeds the {MAX_QUBITS} ceiling: the weight "
                f"matrices are dense 2**n x 2**n, requiring "
                f"{estimate_memory_bytes(num_qubits, entanglement_depth)} bytes",
                details={"num_qubits": num_qubits, "ceiling": MAX_QUBITS,
                         "bytes": estimate_memory_bytes(num_qubits, entanglement_depth)},
            )
        needed = estimate_memory_bytes(num_qubits, entanglement_depth)
        if needed > MEMORY_CEILING_BYTES:
            raise ValidationError(
                f"requested simulation needs {needed} bytes, above the "
                f"{MEMORY_CEILING_BYTES} byte ceiling",
                details={"bytes": needed, "ceiling": MEMORY_CEILING_BYTES},
            )
        self.num_qubits = int(num_qubits)
        self.entanglement_depth = int(entanglement_depth)
        self.optimization_steps = int(optimization_steps)
        self.dimension = 2 ** self.num_qubits
        self._rng = np.random.default_rng(int(seed))
        self.weights: list[np.ndarray] = self._initialize_weights()
        self.biases: list[np.ndarray] = self._initialize_biases()

    # -- construction ------------------------------------------------------

    def _initialize_weights(self) -> list[np.ndarray]:
        """Random orthogonal matrices via SVD of a Gaussian matrix."""
        weights: list[np.ndarray] = []
        for _ in range(self.entanglement_depth):
            w = self._rng.standard_normal((self.dimension, self.dimension)) * 0.1
            u, _, vh = np.linalg.svd(w)
            weights.append(u @ vh)
        return weights

    def _initialize_biases(self) -> list[np.ndarray]:
        """Bias vectors. Always zero in this simulation."""
        return [np.zeros(self.dimension, dtype=np.float64) for _ in range(self.entanglement_depth)]

    # -- operations --------------------------------------------------------

    def create_superposition(self, state: np.ndarray | None = None) -> QuantumState:
        """Uniform superposition, or the normalized form of ``state``."""
        if state is None:
            amplitude = np.ones(self.dimension, dtype=np.float64) / math.sqrt(self.dimension)
            return QuantumState(amplitude=amplitude)
        arr = np.asarray(state, dtype=np.float64)
        norm = float(np.linalg.norm(arr))
        amplitude = arr / norm if norm > 0 else arr
        return QuantumState(amplitude=amplitude)

    def apply_entanglement(self, state: QuantumState, layer: int = 0) -> QuantumState:
        """Apply one normalized matrix layer.

        Mathematically: ``v <- W_layer @ v``, then L2 renormalization. This is a
        matrix-vector product, not an entangling operation in the quantum sense.
        """
        if not 0 <= layer < len(self.weights):
            raise ValidationError(
                f"layer {layer} is outside [0, {len(self.weights)})",
                details={"layer": layer, "depth": self.entanglement_depth},
            )
        new_amplitude = self.weights[layer] @ state.amplitude
        norm = float(np.linalg.norm(new_amplitude))
        if norm > 0:
            new_amplitude = new_amplitude / norm
        return QuantumState(amplitude=new_amplitude, phase=state.phase + self.biases[layer])

    def measure(self, state: QuantumState, *, rng: np.random.Generator | None = None
                ) -> tuple[int, float]:
        """Sample an index from the probability distribution.

        This is ``numpy.random.choice`` over ``|amplitude|²``. It is not a physical
        measurement: repeated calls do not collapse shared state.
        """
        probabilities = state.probabilities
        generator = rng if rng is not None else self._rng
        index = int(generator.choice(probabilities.size, p=probabilities))
        return index, float(probabilities[index])

    def forward(self, input_data: np.ndarray) -> np.ndarray:
        """Full map: encode → normalized matrix layers → decode."""
        encoded = self._encode_input(input_data)
        state = self.create_superposition(encoded)
        for layer in range(self.entanglement_depth):
            state = self.apply_entanglement(state, layer=layer)
        return self._decode_output(state)

    def _encode_input(self, input_data: np.ndarray) -> np.ndarray:
        """Pad/truncate to the simulated dimension and normalize."""
        arr = np.asarray(input_data, dtype=np.float64).ravel()
        padded = np.zeros(self.dimension, dtype=np.float64)
        if arr.size:
            padded[: min(arr.size, self.dimension)] = arr[: self.dimension]
        norm = float(np.linalg.norm(padded))
        return padded / norm if norm > 0 else padded

    def _decode_output(self, state: QuantumState) -> np.ndarray:
        """Probabilities plus the (always-zero) phase contribution."""
        probabilities = state.probabilities
        return probabilities + np.sin(state.phase) * 0.1

    # -- features ----------------------------------------------------------

    def get_quantum_features(self, input_data: np.ndarray) -> dict[str, Any]:
        """Probability distribution, Shannon entropy and a coherence measure.

        Note: the original code applied ``|·|²`` a second time to an already
        normalized probability vector. That is preserved as
        ``probability_distribution`` for compatibility; ``entropy`` is now computed
        on the actual probability distribution, which is the meaningful quantity.
        """
        output = self.forward(input_data)
        probabilities = np.clip(output, 0.0, None)
        total = float(np.sum(probabilities))
        if total > 0:
            probabilities = probabilities / total
        return {
            "probability_distribution": probabilities,
            "entropy": self._calculate_entropy(probabilities),
            "coherence": self._calculate_coherence(probabilities),
        }

    @staticmethod
    def _calculate_entropy(probabilities: np.ndarray) -> float:
        """Shannon entropy in bits of a normalized distribution."""
        p = np.asarray(probabilities, dtype=np.float64)
        p = p[p > 0]
        if p.size == 0:
            return 0.0
        return float(-np.sum(p * np.log2(p)))

    @staticmethod
    def _calculate_coherence(probabilities: np.ndarray) -> float:
        """L1 mass of the off-diagonal of the rank-1 density matrix.

        For a real positive vector this equals ``(sum p)^2 - sum p^2``, i.e. it
        measures how spread the distribution is, not quantum coherence.
        """
        p = np.asarray(probabilities, dtype=np.float64)
        if p.size == 0:
            return 0.0
        density = np.outer(p, p)
        off_diagonal = density - np.diag(np.diag(density))
        return float(np.sum(np.abs(off_diagonal)))

    def optimize(self, loss_gradient: np.ndarray, learning_rate: float = 0.01) -> None:
        """Rank-1 weight update.

        The pre-2.0 version computed ``np.outer(loss_gradient, weight.T)``, whose
        shape ``(D, D*D)`` cannot be subtracted from a ``(D, D)`` weight — the path
        always raised. The intended update is a rank-1 outer product, which is what
        is implemented here.
        """
        gradient = np.asarray(loss_gradient, dtype=np.float64).ravel()
        if gradient.size != self.dimension:
            raise ValidationError(
                f"loss_gradient must have {self.dimension} elements, got {gradient.size}",
                details={"expected": self.dimension, "got": int(gradient.size)},
            )
        lr = float(learning_rate)
        if not math.isfinite(lr):
            raise ValidationError("learning_rate must be finite", details={})
        for i in range(self.entanglement_depth):
            self.weights[i] = self.weights[i] - lr * np.outer(gradient, gradient)
            self.biases[i] = self.biases[i] - lr * gradient

    # -- introspection -----------------------------------------------------

    def describe(self) -> dict[str, Any]:
        return {
            "num_qubits": self.num_qubits,
            "dimension": self.dimension,
            "entanglement_depth": self.entanglement_depth,
            "weight_bytes": estimate_memory_bytes(self.num_qubits, self.entanglement_depth),
            "max_qubits": MAX_QUBITS,
            "classification": "quantum-inspired classical simulation",
            "quantum_hardware": False,
        }


def classical_baseline(input_data: np.ndarray, dimension: int, *, seed: int = 0) -> np.ndarray:
    """Classical reference: random Gaussian projection + softmax-like normalization.

    This is the baseline the quantum-inspired engine must beat. It costs
    ``O(dimension)`` rather than ``O(dimension**2)``.
    """
    rng = np.random.default_rng(int(seed))
    arr = np.asarray(input_data, dtype=np.float64).ravel()
    padded = np.zeros(int(dimension), dtype=np.float64)
    if arr.size:
        padded[: min(arr.size, dimension)] = arr[:dimension]
    norm = float(np.linalg.norm(padded))
    if norm > 0:
        padded = padded / norm
    projected = rng.standard_normal(int(dimension)) @ padded * np.ones(int(dimension)) \
        + padded * float(np.sqrt(int(dimension)))
    activated = np.abs(projected) ** 2
    total = float(np.sum(activated))
    return activated / total if total > 0 else activated
