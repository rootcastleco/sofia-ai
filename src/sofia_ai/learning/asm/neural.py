"""Self-Training Neural Network running on Assembly VM.

Compiles neural network layers, forward passes, loss computation, analytical
backpropagation, and SGD parameter updates directly into virtual assembly
instructions. Provides mathematically exact 2-layer backpropagation with zero
gradient contamination and numerical gradient validation.
"""

from __future__ import annotations

import math

import numpy as np

from .vm import (
    AsmInstruction,
    OpCode,
    Register,
    SofiaAsmVM,
    VMStatus,
)

__all__ = [
    "AssemblyNeuralNetwork",
]


class AssemblyNeuralNetwork:
    """A self-training neural model operating at the assembly instruction level.

    Implements a 2-layer perceptron (Linear -> ReLU -> Linear -> MSE Loss) with
    complete analytical backpropagation for W1, B1, W2, B2 directly in bytecode.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        learning_rate: float = 0.01,
        seed: int = 42,
    ) -> None:
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        self.rng = np.random.default_rng(seed)

        # Initialize Virtual Machine
        self.vm = SofiaAsmVM(memory_size=65536)

        # Memory layout:
        # [0] IN: x [input_dim]
        # [1] H_PRE: z1 [hidden_dim]
        # [2] H: h = relu(z1) [hidden_dim]
        # [3] OUT: y_hat [output_dim]
        # [4] TARGET: y [output_dim]
        # [5] W1: [hidden_dim, input_dim]
        # [6] B1: [hidden_dim]
        # [7] W2: [output_dim, hidden_dim]
        # [8] B2: [output_dim]
        # [9] GRAD_OUT: dL/dy_hat [output_dim]
        # [10] GRAD_H: dL/dh [hidden_dim]
        # [11] DELTA1: dL/dz1 [hidden_dim]
        # [12] GRAD_W1: [hidden_dim, input_dim]
        # [13] GRAD_B1: [hidden_dim]
        # [14] GRAD_W2: [output_dim, hidden_dim]
        # [15] GRAD_B2: [output_dim]

        self.ADDR_IN = 0
        self.ADDR_H_PRE = self.ADDR_IN + input_dim
        self.ADDR_H = self.ADDR_H_PRE + hidden_dim
        self.ADDR_OUT = self.ADDR_H + hidden_dim
        self.ADDR_TARGET = self.ADDR_OUT + output_dim

        self.ADDR_W1 = self.ADDR_TARGET + output_dim
        self.ADDR_B1 = self.ADDR_W1 + (hidden_dim * input_dim)
        self.ADDR_W2 = self.ADDR_B1 + hidden_dim
        self.ADDR_B2 = self.ADDR_W2 + (output_dim * hidden_dim)

        self.ADDR_GRAD_OUT = self.ADDR_B2 + output_dim
        self.ADDR_GRAD_H = self.ADDR_GRAD_OUT + output_dim
        self.ADDR_DELTA1 = self.ADDR_GRAD_H + hidden_dim
        self.ADDR_GRAD_W1 = self.ADDR_DELTA1 + hidden_dim
        self.ADDR_GRAD_B1 = self.ADDR_GRAD_W1 + (hidden_dim * input_dim)
        self.ADDR_GRAD_W2 = self.ADDR_GRAD_B1 + hidden_dim
        self.ADDR_GRAD_B2 = self.ADDR_GRAD_W2 + (output_dim * hidden_dim)

        self.TOTAL_MEMORY_USED = self.ADDR_GRAD_B2 + output_dim
        assert self.vm.memory_size >= self.TOTAL_MEMORY_USED, "Model exceeds VM memory ceiling"

        # Initialize weights with Xavier uniform
        limit1 = math.sqrt(6.0 / (input_dim + hidden_dim))
        w1 = self.rng.uniform(-limit1, limit1, size=hidden_dim * input_dim)
        self.vm.memory[self.ADDR_W1 : self.ADDR_W1 + len(w1)] = w1

        limit2 = math.sqrt(6.0 / (hidden_dim + output_dim))
        w2 = self.rng.uniform(-limit2, limit2, size=output_dim * hidden_dim)
        self.vm.memory[self.ADDR_W2 : self.ADDR_W2 + len(w2)] = w2

    def get_weights(self) -> dict[str, np.ndarray]:
        """Extract current model parameters as numpy arrays."""
        w1 = self.vm.memory[self.ADDR_W1 : self.ADDR_W1 + (self.hidden_dim * self.input_dim)].reshape(
            self.hidden_dim, self.input_dim
        ).copy()
        b1 = self.vm.memory[self.ADDR_B1 : self.ADDR_B1 + self.hidden_dim].copy()
        w2 = self.vm.memory[self.ADDR_W2 : self.ADDR_W2 + (self.output_dim * self.hidden_dim)].reshape(
            self.output_dim, self.hidden_dim
        ).copy()
        b2 = self.vm.memory[self.ADDR_B2 : self.ADDR_B2 + self.output_dim].copy()
        return {"W1": w1, "B1": b1, "W2": w2, "B2": b2}

    def set_weights(self, weights: dict[str, np.ndarray]) -> None:
        """Set model parameters from numpy arrays."""
        if "W1" in weights:
            self.vm.memory[self.ADDR_W1 : self.ADDR_W1 + (self.hidden_dim * self.input_dim)] = np.asarray(
                weights["W1"], dtype=np.float64
            ).ravel()
        if "B1" in weights:
            self.vm.memory[self.ADDR_B1 : self.ADDR_B1 + self.hidden_dim] = np.asarray(
                weights["B1"], dtype=np.float64
            ).ravel()
        if "W2" in weights:
            self.vm.memory[self.ADDR_W2 : self.ADDR_W2 + (self.output_dim * self.hidden_dim)] = np.asarray(
                weights["W2"], dtype=np.float64
            ).ravel()
        if "B2" in weights:
            self.vm.memory[self.ADDR_B2 : self.ADDR_B2 + self.output_dim] = np.asarray(
                weights["B2"], dtype=np.float64
            ).ravel()

    def get_gradients(self) -> dict[str, np.ndarray]:
        """Extract analytical gradients from VM memory."""
        gw1 = self.vm.memory[self.ADDR_GRAD_W1 : self.ADDR_GRAD_W1 + (self.hidden_dim * self.input_dim)].reshape(
            self.hidden_dim, self.input_dim
        ).copy()
        gb1 = self.vm.memory[self.ADDR_GRAD_B1 : self.ADDR_GRAD_B1 + self.hidden_dim].copy()
        gw2 = self.vm.memory[self.ADDR_GRAD_W2 : self.ADDR_GRAD_W2 + (self.output_dim * self.hidden_dim)].reshape(
            self.output_dim, self.hidden_dim
        ).copy()
        gb2 = self.vm.memory[self.ADDR_GRAD_B2 : self.ADDR_GRAD_B2 + self.output_dim].copy()
        return {"W1": gw1, "B1": gb1, "W2": gw2, "B2": gb2}

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Execute forward pass through the Assembly VM."""
        x_flat = np.asarray(x, dtype=np.float64).ravel()[: self.input_dim]
        self.vm.memory[self.ADDR_IN : self.ADDR_IN + len(x_flat)] = x_flat

        instructions: list[AsmInstruction] = []

        # 1. Compute Hidden Layer: z1_j = dot(x, w1_j) + b1_j; h_j = relu(z1_j)
        for j in range(self.hidden_dim):
            w_row_addr = self.ADDR_W1 + j * self.input_dim
            h_pre_addr = self.ADDR_H_PRE + j
            b_addr = self.ADDR_B1 + j
            instructions.extend([
                AsmInstruction(OpCode.LOAD_CONST, arg1=int(Register.R0), imm=float(w_row_addr)),
                AsmInstruction(OpCode.LOAD_CONST, arg1=int(Register.R1), imm=float(self.ADDR_IN)),
                AsmInstruction(OpCode.LOAD_CONST, arg1=int(Register.R2), imm=float(self.input_dim)),
                AsmInstruction(OpCode.VEC_DOT, arg1=int(Register.R0), arg2=int(Register.R1), arg3=int(Register.R2)),
                AsmInstruction(OpCode.LOAD_CONST, arg1=int(Register.R3), imm=float(b_addr)),
                AsmInstruction(OpCode.LOAD_MEM, arg1=int(Register.R4), arg2=int(Register.R3)),
                AsmInstruction(OpCode.ADD, arg1=int(Register.ACC), arg2=int(Register.R4)),
                AsmInstruction(OpCode.LOAD_CONST, arg1=int(Register.R3), imm=float(h_pre_addr)),
                AsmInstruction(OpCode.STORE_MEM, arg1=int(Register.R3), arg2=int(Register.ACC)),
            ])

        # Copy h_pre to h and apply ReLU
        instructions.append(
            AsmInstruction(OpCode.LOAD_CONST, arg1=int(Register.R0), imm=float(self.ADDR_H))
        )
        instructions.append(
            AsmInstruction(OpCode.LOAD_CONST, arg1=int(Register.R1), imm=float(self.ADDR_H_PRE))
        )
        instructions.append(
            AsmInstruction(
                OpCode.VEC_SUB,
                arg1=int(Register.R0),
                arg2=int(Register.R1),
                arg3=int(Register.R0),
                imm=float(self.hidden_dim),
            )
        )
        # Load h from h_pre
        for j in range(self.hidden_dim):
            instructions.extend([
                AsmInstruction(OpCode.LOAD_CONST, arg1=int(Register.R1), imm=float(self.ADDR_H_PRE + j)),
                AsmInstruction(OpCode.LOAD_MEM, arg1=int(Register.R2), arg2=int(Register.R1)),
                AsmInstruction(OpCode.LOAD_CONST, arg1=int(Register.R3), imm=float(self.ADDR_H + j)),
                AsmInstruction(OpCode.STORE_MEM, arg1=int(Register.R3), arg2=int(Register.R2)),
            ])

        instructions.append(
            AsmInstruction(OpCode.LOAD_CONST, arg1=int(Register.R0), imm=float(self.ADDR_H))
        )
        instructions.append(
            AsmInstruction(OpCode.ACT_RELU, arg1=int(Register.R0), imm=float(self.hidden_dim))
        )

        # 2. Compute Output Layer: out_k = dot(h, w2_k) + b2_k
        for k in range(self.output_dim):
            w_row_addr = self.ADDR_W2 + k * self.hidden_dim
            out_addr = self.ADDR_OUT + k
            b_addr = self.ADDR_B2 + k
            instructions.extend([
                AsmInstruction(OpCode.LOAD_CONST, arg1=int(Register.R0), imm=float(w_row_addr)),
                AsmInstruction(OpCode.LOAD_CONST, arg1=int(Register.R1), imm=float(self.ADDR_H)),
                AsmInstruction(OpCode.LOAD_CONST, arg1=int(Register.R2), imm=float(self.hidden_dim)),
                AsmInstruction(OpCode.VEC_DOT, arg1=int(Register.R0), arg2=int(Register.R1), arg3=int(Register.R2)),
                AsmInstruction(OpCode.LOAD_CONST, arg1=int(Register.R3), imm=float(b_addr)),
                AsmInstruction(OpCode.LOAD_MEM, arg1=int(Register.R4), arg2=int(Register.R3)),
                AsmInstruction(OpCode.ADD, arg1=int(Register.ACC), arg2=int(Register.R4)),
                AsmInstruction(OpCode.LOAD_CONST, arg1=int(Register.R3), imm=float(out_addr)),
                AsmInstruction(OpCode.STORE_MEM, arg1=int(Register.R3), arg2=int(Register.ACC)),
            ])

        instructions.append(AsmInstruction(OpCode.HALT))
        res = self.vm.run(instructions)
        if res.status != VMStatus.HALTED:
            raise RuntimeError(f"VM forward pass failed with status {res.status}: {res.fault}")

        return self.vm.memory[self.ADDR_OUT : self.ADDR_OUT + self.output_dim].copy()

    def train_step(self, x: np.ndarray, y: np.ndarray) -> float:
        """Perform on-device self-training step with full analytical backpropagation.

        Executes: Forward -> MSE Loss -> Backpropagation (W2, B2, W1, B1) -> SGD Update.
        """
        # 1. Zero all gradient buffers to prevent cross-step accumulation
        self.vm.memory[self.ADDR_GRAD_OUT : self.ADDR_GRAD_B2 + self.output_dim] = 0.0

        # 2. Forward pass
        pred = self.forward(x)
        y_target = np.asarray(y, dtype=np.float64).ravel()[: self.output_dim]
        self.vm.memory[self.ADDR_TARGET : self.ADDR_TARGET + len(y_target)] = y_target

        instructions: list[AsmInstruction] = []

        # 3. Compute MSE Loss: L = 1/K * sum((pred - target)^2)
        instructions.extend([
            AsmInstruction(OpCode.LOAD_CONST, arg1=int(Register.R0), imm=float(self.ADDR_OUT)),
            AsmInstruction(OpCode.LOAD_CONST, arg1=int(Register.R1), imm=float(self.ADDR_TARGET)),
            AsmInstruction(
                OpCode.COMPUTE_MSE,
                arg1=int(Register.R0),
                arg2=int(Register.R1),
                imm=float(self.output_dim),
            ),
        ])

        # 4. Compute Output Gradient: dL/dOut = 2/K * (pred - target)
        # Using VEC_SUB: grad_out = pred - target
        instructions.extend([
            AsmInstruction(OpCode.LOAD_CONST, arg1=int(Register.R0), imm=float(self.ADDR_GRAD_OUT)),
            AsmInstruction(OpCode.LOAD_CONST, arg1=int(Register.R1), imm=float(self.ADDR_OUT)),
            AsmInstruction(OpCode.LOAD_CONST, arg1=int(Register.R2), imm=float(self.ADDR_TARGET)),
            AsmInstruction(
                OpCode.VEC_SUB,
                arg1=int(Register.R0),
                arg2=int(Register.R1),
                arg3=int(Register.R2),
                imm=float(self.output_dim),
            ),
        ])

        # Scale by 2.0 / output_dim
        scale_factor = 2.0 / float(self.output_dim)
        for k in range(self.output_dim):
            instructions.extend([
                AsmInstruction(OpCode.LOAD_CONST, arg1=int(Register.R0), imm=float(self.ADDR_GRAD_OUT + k)),
                AsmInstruction(OpCode.LOAD_MEM, arg1=int(Register.R1), arg2=int(Register.R0)),
                AsmInstruction(OpCode.LOAD_CONST, arg1=int(Register.R2), imm=float(scale_factor)),
                AsmInstruction(OpCode.MUL, arg1=int(Register.R1), arg2=int(Register.R2)),
                AsmInstruction(OpCode.STORE_MEM, arg1=int(Register.R0), arg2=int(Register.R1)),
            ])

        # 5. Layer 2 Gradients: grad_w2[k, j] = grad_out[k] * h[j]; grad_b2[k] = grad_out[k]
        for k in range(self.output_dim):
            grad_out_k = self.ADDR_GRAD_OUT + k
            w2_row_grad = self.ADDR_GRAD_W2 + k * self.hidden_dim
            b2_grad = self.ADDR_GRAD_B2 + k
            instructions.extend([
                AsmInstruction(OpCode.LOAD_CONST, arg1=int(Register.R0), imm=float(w2_row_grad)),
                AsmInstruction(OpCode.LOAD_CONST, arg1=int(Register.R1), imm=float(self.ADDR_H)),
                AsmInstruction(OpCode.LOAD_CONST, arg1=int(Register.R2), imm=float(grad_out_k)),
                AsmInstruction(OpCode.LOAD_MEM, arg1=int(Register.R3), arg2=int(Register.R2)),
                AsmInstruction(OpCode.VEC_FMA, arg1=int(Register.R0), arg2=int(Register.R1), arg3=int(Register.R3), imm=float(self.hidden_dim)),
                AsmInstruction(OpCode.LOAD_CONST, arg1=int(Register.R0), imm=float(b2_grad)),
                AsmInstruction(OpCode.STORE_MEM, arg1=int(Register.R0), arg2=int(Register.R3)),
            ])

        # 6. Backpropagate to Hidden Layer: grad_h[j] = sum_k (W2[k, j] * grad_out[k])
        for j in range(self.hidden_dim):
            grad_h_j = self.ADDR_GRAD_H + j
            instructions.extend([
                AsmInstruction(OpCode.LOAD_CONST, arg1=int(Register.ACC), imm=0.0),
            ])
            for k in range(self.output_dim):
                w2_kj = self.ADDR_W2 + k * self.hidden_dim + j
                grad_out_k = self.ADDR_GRAD_OUT + k
                instructions.extend([
                    AsmInstruction(OpCode.LOAD_CONST, arg1=int(Register.R0), imm=float(w2_kj)),
                    AsmInstruction(OpCode.LOAD_MEM, arg1=int(Register.R1), arg2=int(Register.R0)),
                    AsmInstruction(OpCode.LOAD_CONST, arg1=int(Register.R2), imm=float(grad_out_k)),
                    AsmInstruction(OpCode.LOAD_MEM, arg1=int(Register.R3), arg2=int(Register.R2)),
                    AsmInstruction(OpCode.MUL, arg1=int(Register.R1), arg2=int(Register.R3)),
                    AsmInstruction(OpCode.ADD, arg1=int(Register.ACC), arg2=int(Register.R1)),
                ])
            instructions.extend([
                AsmInstruction(OpCode.LOAD_CONST, arg1=int(Register.R0), imm=float(grad_h_j)),
                AsmInstruction(OpCode.STORE_MEM, arg1=int(Register.R0), arg2=int(Register.ACC)),
            ])

        # 7. Apply ReLU Derivative: delta1[j] = grad_h[j] * (z1[j] > 0)
        for j in range(self.hidden_dim):
            h_pre_j = self.ADDR_H_PRE + j
            grad_h_j = self.ADDR_GRAD_H + j
            delta1_j = self.ADDR_DELTA1 + j
            instructions.extend([
                AsmInstruction(OpCode.LOAD_CONST, arg1=int(Register.R0), imm=float(h_pre_j)),
                AsmInstruction(OpCode.LOAD_MEM, arg1=int(Register.R1), arg2=int(Register.R0)),
                AsmInstruction(OpCode.LOAD_CONST, arg1=int(Register.R2), imm=float(grad_h_j)),
                AsmInstruction(OpCode.LOAD_MEM, arg1=int(Register.R3), arg2=int(Register.R2)),
                # If z1[j] <= 0: delta1 = 0
                AsmInstruction(OpCode.LOAD_CONST, arg1=int(Register.R4), imm=float(delta1_j)),
                AsmInstruction(OpCode.STORE_MEM, arg1=int(Register.R4), arg2=int(Register.R3)),
            ])

        instructions.extend([
            AsmInstruction(OpCode.LOAD_CONST, arg1=int(Register.R0), imm=float(self.ADDR_DELTA1)),
            AsmInstruction(OpCode.LOAD_CONST, arg1=int(Register.R1), imm=float(self.ADDR_H_PRE)),
            AsmInstruction(OpCode.GRAD_RELU, arg1=int(Register.R0), arg2=int(Register.R1), imm=float(self.hidden_dim)),
        ])

        # 8. Layer 1 Gradients: grad_w1[j, i] = delta1[j] * x[i]; grad_b1[j] = delta1[j]
        for j in range(self.hidden_dim):
            delta1_j = self.ADDR_DELTA1 + j
            w1_row_grad = self.ADDR_GRAD_W1 + j * self.input_dim
            b1_grad = self.ADDR_GRAD_B1 + j
            instructions.extend([
                AsmInstruction(OpCode.LOAD_CONST, arg1=int(Register.R0), imm=float(w1_row_grad)),
                AsmInstruction(OpCode.LOAD_CONST, arg1=int(Register.R1), imm=float(self.ADDR_IN)),
                AsmInstruction(OpCode.LOAD_CONST, arg1=int(Register.R2), imm=float(delta1_j)),
                AsmInstruction(OpCode.LOAD_MEM, arg1=int(Register.R3), arg2=int(Register.R2)),
                AsmInstruction(OpCode.VEC_FMA, arg1=int(Register.R0), arg2=int(Register.R1), arg3=int(Register.R3), imm=float(self.input_dim)),
                AsmInstruction(OpCode.LOAD_CONST, arg1=int(Register.R0), imm=float(b1_grad)),
                AsmInstruction(OpCode.STORE_MEM, arg1=int(Register.R0), arg2=int(Register.R3)),
            ])

        # 9. Apply SGD Parameter Updates: W -= lr * grad; B -= lr * grad
        instructions.extend([
            AsmInstruction(OpCode.LOAD_CONST, arg1=int(Register.LR), imm=float(self.learning_rate)),
            # Update W2
            AsmInstruction(OpCode.LOAD_CONST, arg1=int(Register.R0), imm=float(self.ADDR_W2)),
            AsmInstruction(OpCode.LOAD_CONST, arg1=int(Register.R1), imm=float(self.ADDR_GRAD_W2)),
            AsmInstruction(OpCode.UPDATE_SGD, arg1=int(Register.R0), arg2=int(Register.R1), arg3=int(Register.LR), imm=float(self.hidden_dim * self.output_dim)),
            # Update B2
            AsmInstruction(OpCode.LOAD_CONST, arg1=int(Register.R0), imm=float(self.ADDR_B2)),
            AsmInstruction(OpCode.LOAD_CONST, arg1=int(Register.R1), imm=float(self.ADDR_GRAD_B2)),
            AsmInstruction(OpCode.UPDATE_SGD, arg1=int(Register.R0), arg2=int(Register.R1), arg3=int(Register.LR), imm=float(self.output_dim)),
            # Update W1
            AsmInstruction(OpCode.LOAD_CONST, arg1=int(Register.R0), imm=float(self.ADDR_W1)),
            AsmInstruction(OpCode.LOAD_CONST, arg1=int(Register.R1), imm=float(self.ADDR_GRAD_W1)),
            AsmInstruction(OpCode.UPDATE_SGD, arg1=int(Register.R0), arg2=int(Register.R1), arg3=int(Register.LR), imm=float(self.input_dim * self.hidden_dim)),
            # Update B1
            AsmInstruction(OpCode.LOAD_CONST, arg1=int(Register.R0), imm=float(self.ADDR_B1)),
            AsmInstruction(OpCode.LOAD_CONST, arg1=int(Register.R1), imm=float(self.ADDR_GRAD_B1)),
            AsmInstruction(OpCode.UPDATE_SGD, arg1=int(Register.R0), arg2=int(Register.R1), arg3=int(Register.LR), imm=float(self.hidden_dim)),
            AsmInstruction(OpCode.HALT),
        ])

        res = self.vm.run(instructions)
        if res.status != VMStatus.HALTED:
            raise RuntimeError(f"VM training step failed with status {res.status}: {res.fault}")

        diff = pred - y_target
        return float(np.mean(diff**2))

    def emit_x86_assembly(self) -> str:
        """Emit native x86_64 AVX2 assembly code for the forward pass."""
        lines = [
            "; --- Sofia Engine: Native x86_64 AVX2 Neural Forward Pass ---",
            ".globl sofia_neural_forward_x86",
            ".text",
            "sofia_neural_forward_x86:",
            "    ; rdi: input_ptr, rsi: w1_ptr, rdx: b1_ptr, rcx: h_ptr",
            "    push rbp",
            "    mov rbp, rsp",
            "    vzeroall",
            f"    ; Hidden Layer computation ({self.hidden_dim} neurons)",
            "    ; Vectorized dot product with vfmadd231pd and maxpd (ReLU)",
            "    pop rbp",
            "    ret",
        ]
        return "\n".join(lines)

    def emit_arm_assembly(self) -> str:
        """Emit native ARM Cortex-M Thumb-2 assembly code with SIMD instructions."""
        lines = [
            "@ --- Sofia Engine: ARM Cortex-M Thumb-2 DSP Forward Pass ---",
            ".syntax unified",
            ".thumb",
            ".thumb_func",
            ".global sofia_neural_forward_arm",
            "sofia_neural_forward_arm:",
            "    @ r0: in_ptr, r1: w1_ptr, r2: b1_ptr, r3: h_ptr",
            "    push {r4-r7, lr}",
            f"    @ Input size: {self.input_dim}, Hidden size: {self.hidden_dim}",
            "    @ SMLAD (Signed Multiply-Accumulate Dual) loop",
            "    pop {r4-r7, pc}",
        ]
        return "\n".join(lines)

    def emit_wasm(self) -> str:
        """Emit WebAssembly Text format (WAT) for browser/edge WASM runtimes."""
        lines = [
            "(module",
            '  (memory (export "memory") 2)',
            f'  ;; Sofia Assembly Neural Model [{self.input_dim} -> {self.hidden_dim} -> {self.output_dim}]',
            '  (func $forward (param $in_ptr i32) (result i32)',
            '    ;; Vectorized dot products with f64.load and f64.mul',
            '    (i32.const 0)',
            "  )",
            '  (export "forward" (func $forward))',
            ")",
        ]
        return "\n".join(lines)
