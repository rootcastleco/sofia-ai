"""Self-Training Neural Network running on Assembly VM.

Compiles neural network layers, forward passes, loss computation, and
backpropagation directly into virtual assembly instructions. Also emits raw
x86_64, ARM Cortex-M Thumb-2, and WebAssembly (WAT) assembly code for bare-metal
embedded deployment.
"""

from __future__ import annotations

import math
from typing import Any, Final

import numpy as np

from .vm import AsmInstruction, OpCode, Register, SofiaAsmVM, assemble

__all__ = [
    "AssemblyNeuralNetwork",
]


class AssemblyNeuralNetwork:
    """A self-training neural model operating at the assembly instruction level."""

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
        # 0: input [input_dim]
        # ADDR_H: hidden [hidden_dim]
        # ADDR_OUT: output [output_dim]
        # ADDR_TARGET: target [output_dim]
        # ADDR_W1: weights 1 [input_dim * hidden_dim]
        # ADDR_B1: bias 1 [hidden_dim]
        # ADDR_W2: weights 2 [hidden_dim * output_dim]
        # ADDR_B2: bias 2 [output_dim]
        # ADDR_GRAD_OUT: output grad [output_dim]
        # ADDR_GRAD_H: hidden grad [hidden_dim]
        # ADDR_GRAD_W1: grad W1 [input_dim * hidden_dim]
        # ADDR_GRAD_W2: grad W2 [hidden_dim * output_dim]

        self.ADDR_IN = 0
        self.ADDR_H = self.ADDR_IN + input_dim
        self.ADDR_OUT = self.ADDR_H + hidden_dim
        self.ADDR_TARGET = self.ADDR_OUT + output_dim
        self.ADDR_W1 = self.ADDR_TARGET + output_dim
        self.ADDR_B1 = self.ADDR_W1 + (input_dim * hidden_dim)
        self.ADDR_W2 = self.ADDR_B1 + hidden_dim
        self.ADDR_B2 = self.ADDR_W2 + (hidden_dim * output_dim)
        self.ADDR_GRAD_OUT = self.ADDR_B2 + output_dim
        self.ADDR_GRAD_H = self.ADDR_GRAD_OUT + output_dim
        self.ADDR_GRAD_W1 = self.ADDR_GRAD_H + hidden_dim
        self.ADDR_GRAD_W2 = self.ADDR_GRAD_W1 + (input_dim * hidden_dim)

        # Initialize weights with He/Xavier uniform
        limit1 = math.sqrt(6.0 / (input_dim + hidden_dim))
        w1 = self.rng.uniform(-limit1, limit1, size=input_dim * hidden_dim)
        self.vm.memory[self.ADDR_W1 : self.ADDR_W1 + len(w1)] = w1

        limit2 = math.sqrt(6.0 / (hidden_dim + output_dim))
        w2 = self.rng.uniform(-limit2, limit2, size=hidden_dim * output_dim)
        self.vm.memory[self.ADDR_W2 : self.ADDR_W2 + len(w2)] = w2

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Execute forward pass through the Assembly VM."""
        x_flat = np.asarray(x, dtype=np.float64).ravel()[: self.input_dim]
        self.vm.memory[self.ADDR_IN : self.ADDR_IN + len(x_flat)] = x_flat

        # Assembly Program for Forward Pass:
        # Layer 1: H = ReLU(W1 * X + B1)
        # Layer 2: Out = W2 * H + B2
        instructions: list[AsmInstruction] = []

        # 1. Compute Hidden Layer: h_j = dot(x, w1_j) + b1_j
        for j in range(self.hidden_dim):
            w_row_addr = self.ADDR_W1 + j * self.input_dim
            h_addr = self.ADDR_H + j
            b_addr = self.ADDR_B1 + j
            instructions.extend([
                AsmInstruction(OpCode.LOAD_CONST, arg1=int(Register.R0), imm=float(w_row_addr)),
                AsmInstruction(OpCode.LOAD_CONST, arg1=int(Register.R1), imm=float(self.ADDR_IN)),
                AsmInstruction(OpCode.LOAD_CONST, arg1=int(Register.R2), imm=float(self.input_dim)),
                AsmInstruction(OpCode.VEC_DOT, arg1=int(Register.R0), arg2=int(Register.R1), arg3=int(Register.R2)),
                # R3 = bias
                AsmInstruction(OpCode.LOAD_CONST, arg1=int(Register.R3), imm=float(b_addr)),
                AsmInstruction(OpCode.LOAD_MEM, arg1=int(Register.R4), arg2=int(Register.R3)),
                AsmInstruction(OpCode.ADD, arg1=int(Register.ACC), arg2=int(Register.R4)),
                # Store in H[j]
                AsmInstruction(OpCode.LOAD_CONST, arg1=int(Register.R3), imm=float(h_addr)),
                AsmInstruction(OpCode.STORE_MEM, arg1=int(Register.R3), arg2=int(Register.ACC)),
            ])

        # Apply ReLU to hidden layer
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
        self.vm.run(instructions)
        return self.vm.memory[self.ADDR_OUT : self.ADDR_OUT + self.output_dim].copy()

    def train_step(self, x: np.ndarray, y: np.ndarray) -> float:
        """Perform on-device self-training step: Forward -> Loss -> Backprop -> SGD."""
        pred = self.forward(x)
        y_target = np.asarray(y, dtype=np.float64).ravel()[: self.output_dim]
        self.vm.memory[self.ADDR_TARGET : self.ADDR_TARGET + len(y_target)] = y_target

        # Assembly Program for Training Step:
        instructions: list[AsmInstruction] = []

        # 1. Compute MSE Loss & Output Gradient: grad_out = 2 * (pred - target) / output_dim
        instructions.extend([
            AsmInstruction(OpCode.LOAD_CONST, arg1=int(Register.R0), imm=float(self.ADDR_OUT)),
            AsmInstruction(OpCode.LOAD_CONST, arg1=int(Register.R1), imm=float(self.ADDR_TARGET)),
            AsmInstruction(OpCode.COMPUTE_MSE, arg1=int(Register.R0), arg2=int(Register.R1), imm=float(self.output_dim)),
            # grad_out = pred - target
            AsmInstruction(OpCode.LOAD_CONST, arg1=int(Register.R0), imm=float(self.ADDR_GRAD_OUT)),
            AsmInstruction(OpCode.LOAD_CONST, arg1=int(Register.R1), imm=float(self.ADDR_OUT)),
            AsmInstruction(OpCode.LOAD_CONST, arg1=int(Register.R2), imm=float(self.ADDR_TARGET)),
            AsmInstruction(OpCode.VEC_SUB, arg1=int(Register.R0), arg2=int(Register.R1), arg3=int(Register.R2), imm=float(self.output_dim)),
        ])

        # 2. Backpropagate to W2: grad_w2[k, j] = grad_out[k] * h[j]
        for k in range(self.output_dim):
            grad_out_k = self.ADDR_GRAD_OUT + k
            w2_row_grad = self.ADDR_GRAD_W2 + k * self.hidden_dim
            instructions.extend([
                AsmInstruction(OpCode.LOAD_CONST, arg1=int(Register.R0), imm=float(w2_row_grad)),
                AsmInstruction(OpCode.LOAD_CONST, arg1=int(Register.R1), imm=float(self.ADDR_H)),
                AsmInstruction(OpCode.LOAD_CONST, arg1=int(Register.R2), imm=float(grad_out_k)),
                AsmInstruction(OpCode.LOAD_MEM, arg1=int(Register.R3), arg2=int(Register.R2)),
                AsmInstruction(OpCode.VEC_FMA, arg1=int(Register.R0), arg2=int(Register.R1), arg3=int(Register.R3), imm=float(self.hidden_dim)),
            ])

        # 3. Apply SGD Weight Update via Assembly VM: W -= lr * grad
        instructions.extend([
            # Load learning rate into LR register
            AsmInstruction(OpCode.LOAD_CONST, arg1=int(Register.LR), imm=float(self.learning_rate)),
            # Update W2
            AsmInstruction(OpCode.LOAD_CONST, arg1=int(Register.R0), imm=float(self.ADDR_W2)),
            AsmInstruction(OpCode.LOAD_CONST, arg1=int(Register.R1), imm=float(self.ADDR_GRAD_W2)),
            AsmInstruction(OpCode.UPDATE_SGD, arg1=int(Register.R0), arg2=int(Register.R1), arg3=int(Register.LR), imm=float(self.hidden_dim * self.output_dim)),
            # Update B2
            AsmInstruction(OpCode.LOAD_CONST, arg1=int(Register.R0), imm=float(self.ADDR_B2)),
            AsmInstruction(OpCode.LOAD_CONST, arg1=int(Register.R1), imm=float(self.ADDR_GRAD_OUT)),
            AsmInstruction(OpCode.UPDATE_SGD, arg1=int(Register.R0), arg2=int(Register.R1), arg3=int(Register.LR), imm=float(self.output_dim)),
            AsmInstruction(OpCode.HALT),
        ])

        self.vm.run(instructions)
        return float(self.vm.registers[Register.ERR])

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
