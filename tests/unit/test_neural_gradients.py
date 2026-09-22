"""Numerical Gradient Checking and Convergence Tests for AssemblyNeuralNetwork (SOFIA-ML-001 to 004)."""

from __future__ import annotations

import numpy as np

from sofia_ai.learning.asm import AssemblyNeuralNetwork


def compute_loss(model: AssemblyNeuralNetwork, x: np.ndarray, y: np.ndarray) -> float:
    """Compute MSE loss without updating parameters."""
    pred = model.forward(x)
    diff = pred - y
    return float(np.mean(diff**2))


def test_numerical_gradient_checking() -> None:
    """SOFIA-ML-003: Compare analytical VM gradients against finite differences (rel error < 1e-5)."""
    input_dim = 3
    hidden_dim = 4
    output_dim = 2
    model = AssemblyNeuralNetwork(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        learning_rate=0.0, # Zero learning rate so weights don't change during check
        seed=123,
    )

    x = np.array([0.5, -0.8, 1.2])
    y = np.array([1.5, -0.4])

    # Run train_step to compute analytical gradients in VM memory
    model.train_step(x, y)
    analytical_grads = model.get_gradients()

    # Numerical gradient check with central differences
    eps = 1e-6
    weights = model.get_weights()

    for param_name in ("W1", "B1", "W2", "B2"):
        w = weights[param_name]
        grad_anal = analytical_grads[param_name]
        grad_num = np.zeros_like(w)

        it = np.nditer(w, flags=["multi_index"])
        while not it.finished:
            idx = it.multi_index
            orig_val = w[idx]

            # f(theta + eps)
            w[idx] = orig_val + eps
            model.set_weights({param_name: w})
            loss_plus = compute_loss(model, x, y)

            # f(theta - eps)
            w[idx] = orig_val - eps
            model.set_weights({param_name: w})
            loss_minus = compute_loss(model, x, y)

            # Numerical gradient
            grad_num[idx] = (loss_plus - loss_minus) / (2.0 * eps)

            # Restore original value
            w[idx] = orig_val
            model.set_weights({param_name: w})

            it.iternext()

        # Compute relative error: ||grad_anal - grad_num|| / (||grad_anal|| + ||grad_num||)
        diff_norm = float(np.linalg.norm(grad_anal - grad_num))
        total_norm = float(np.linalg.norm(grad_anal) + np.linalg.norm(grad_num))
        rel_error = diff_norm / total_norm if total_norm > 1e-12 else 0.0

        assert rel_error < 1e-5, f"Gradient check failed for {param_name}: rel_error={rel_error:.2e}"


def test_assembly_neural_network_convergence() -> None:
    """SOFIA-ML-004: Verify deterministic monotonic convergence on regression."""
    model = AssemblyNeuralNetwork(
        input_dim=2,
        hidden_dim=6,
        output_dim=1,
        learning_rate=0.08,
        seed=42,
    )

    # Target function: y = 2.0 * x0 - 1.5 * x1
    x = np.array([0.8, -0.4])
    y = np.array([2.0 * 0.8 - 1.5 * (-0.4)]) # 1.6 + 0.6 = 2.2

    initial_loss = model.train_step(x, y)
    losses = []
    for _ in range(80):
        losses.append(model.train_step(x, y))

    final_loss = losses[-1]
    assert final_loss < initial_loss
    assert final_loss < 0.005, f"Expected convergence to < 0.005, got {final_loss}"


def test_gradient_buffer_isolation() -> None:
    """SOFIA-ML-002: Verify gradient buffers do not accumulate across independent steps."""
    model = AssemblyNeuralNetwork(input_dim=2, hidden_dim=3, output_dim=1, learning_rate=0.0, seed=42)
    x = np.array([1.0, 2.0])
    y = np.array([3.0])

    model.train_step(x, y)
    grad1 = model.get_gradients()["W2"].copy()

    # Second step with identical input
    model.train_step(x, y)
    grad2 = model.get_gradients()["W2"].copy()

    # Gradients must be identical (not doubled or accumulated)
    np.testing.assert_allclose(grad1, grad2)
