"""Assembly Virtual Machine & Bytecode Engine for Neural Self-Training.

Provides a deterministic, low-level register-based virtual machine designed
specifically for on-device, in-situ neural network inference and self-training.
Operates with explicit registers, fixed memory buffers, and zero external
dependencies (NumPy only).
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Final

import numpy as np

__all__ = [
    "AsmInstruction",
    "OpCode",
    "Register",
    "SofiaAsmVM",
    "assemble",
    "disassemble",
]


class Register(enum.IntEnum):
    """Virtual Machine 64-bit Floating-Point & Control Registers."""

    R0 = 0
    R1 = 1
    R2 = 2
    R3 = 3
    R4 = 4
    R5 = 5
    R6 = 6
    R7 = 7
    ACC = 8   # Accumulator
    LR = 9    # Learning rate register
    ERR = 10  # Loss / Error gradient register
    PC = 11   # Program counter
    SP = 12   # Stack pointer


class OpCode(enum.IntEnum):
    """Assembly Instruction Opcodes for Neural Computation."""

    NOP = 0x00
    LOAD_CONST = 0x01   # R_dest, const_val
    LOAD_MEM = 0x02     # R_dest, addr_reg
    STORE_MEM = 0x03    # addr_reg, R_src
    MOV = 0x04          # R_dest, R_src

    # Arithmetic
    ADD = 0x10          # R_dest, R_src
    SUB = 0x11          # R_dest, R_src
    MUL = 0x12          # R_dest, R_src
    DIV = 0x13          # R_dest, R_src
    FMA = 0x14          # R_dest, R_src1, R_src2 (Fused Multiply-Add: dest += src1 * src2)

    # Vectorized Neural Primitives
    VEC_DOT = 0x20      # R_dest, addr_a, addr_b, len (Dot product)
    VEC_FMA = 0x21      # addr_dest, addr_src, scalar_reg, len (Vector scale & accumulate)
    VEC_SUB = 0x22      # addr_dest, addr_a, addr_b, len (dest = a - b)

    # Activations & Derivatives (in-place)
    ACT_RELU = 0x30     # addr, len
    GRAD_RELU = 0x31    # addr_grad, addr_act, len (grad *= (act > 0))
    ACT_SIGMOID = 0x32  # addr, len
    GRAD_SIGMOID = 0x33 # addr_grad, addr_act, len (grad *= act * (1 - act))

    # Self-Training & Optimization
    UPDATE_SGD = 0x40   # addr_weights, addr_grads, lr_reg, len (w -= lr * grad)
    COMPUTE_MSE = 0x41  # R_loss, addr_pred, addr_target, len

    # Control Flow
    JMP = 0x50          # target_pc
    JZ = 0x51           # test_reg, target_pc
    HALT = 0xFF


@dataclass(slots=True)
class AsmInstruction:
    """A decoded 32-bit/64-bit virtual assembly instruction."""

    opcode: OpCode
    arg1: int = 0
    arg2: int = 0
    arg3: int = 0
    imm: float = 0.0


def assemble(asm_text: str) -> list[AsmInstruction]:
    """Assemble textual assembly into VM bytecode instructions."""
    instructions: list[AsmInstruction] = []
    for raw_line in asm_text.splitlines():
        line = raw_line.split(";")[0].split("#")[0].strip()
        if not line:
            continue

        parts = [p.strip().rstrip(",") for p in line.split()]
        mnemonic = parts[0].upper()
        if not hasattr(OpCode, mnemonic):
            raise ValueError(f"Unknown assembly mnemonic: {mnemonic}")

        op = OpCode[mnemonic]
        args: list[int] = []
        imm = 0.0

        for p in parts[1:]:
            p_upper = p.upper()
            if hasattr(Register, p_upper):
                args.append(int(Register[p_upper]))
            elif p.startswith("[") and p.endswith("]"):
                reg_name = p[1:-1].upper()
                args.append(int(Register[reg_name]))
            else:
                try:
                    imm = float(p)
                except ValueError:
                    args.append(int(p))

        while len(args) < 3:
            args.append(0)

        instructions.append(AsmInstruction(opcode=op, arg1=args[0], arg2=args[1], arg3=args[2], imm=imm))
    return instructions


def disassemble(instructions: list[AsmInstruction]) -> str:
    """Disassemble VM instructions back into human-readable assembly text."""
    lines = []
    for inst in instructions:
        name = inst.opcode.name
        if inst.opcode == OpCode.LOAD_CONST:
            reg = Register(inst.arg1).name
            lines.append(f"{name} {reg}, {inst.imm}")
        elif inst.opcode in (OpCode.ADD, OpCode.SUB, OpCode.MUL, OpCode.DIV, OpCode.MOV):
            r1 = Register(inst.arg1).name
            r2 = Register(inst.arg2).name
            lines.append(f"{name} {r1}, {r2}")
        elif inst.opcode == OpCode.UPDATE_SGD:
            lines.append(f"{name} [{Register(inst.arg1).name}], [{Register(inst.arg2).name}], {Register(inst.arg3).name}")
        else:
            lines.append(f"{name} {inst.arg1}, {inst.arg2}, {inst.arg3}")
    return "\n".join(lines)


class SofiaAsmVM:
    """High-performance Assembly Virtual Machine for on-device Neural Self-Training."""

    def __init__(self, memory_size: int = 65536) -> None:
        self.registers = np.zeros(len(Register), dtype=np.float64)
        self.memory = np.zeros(memory_size, dtype=np.float64)
        self.memory_size = memory_size

    def reset(self) -> None:
        """Reset all registers and program counter."""
        self.registers.fill(0.0)

    def run(self, instructions: list[AsmInstruction], max_cycles: int = 1_000_000) -> int:
        """Execute assembly program until HALT or max_cycles."""
        pc = 0
        cycles = 0
        n_inst = len(instructions)

        while pc < n_inst and cycles < max_cycles:
            inst = instructions[pc]
            op = inst.opcode
            cycles += 1

            if op == OpCode.HALT:
                break

            elif op == OpCode.LOAD_CONST:
                self.registers[inst.arg1] = inst.imm
                pc += 1

            elif op == OpCode.MOV:
                self.registers[inst.arg1] = self.registers[inst.arg2]
                pc += 1

            elif op == OpCode.LOAD_MEM:
                addr = int(self.registers[inst.arg2])
                self.registers[inst.arg1] = self.memory[addr]
                pc += 1

            elif op == OpCode.STORE_MEM:
                addr = int(self.registers[inst.arg1])
                self.memory[addr] = self.registers[inst.arg2]
                pc += 1

            elif op == OpCode.ADD:
                self.registers[inst.arg1] += self.registers[inst.arg2]
                pc += 1

            elif op == OpCode.SUB:
                self.registers[inst.arg1] -= self.registers[inst.arg2]
                pc += 1

            elif op == OpCode.MUL:
                self.registers[inst.arg1] *= self.registers[inst.arg2]
                pc += 1

            elif op == OpCode.DIV:
                val = self.registers[inst.arg2]
                self.registers[inst.arg1] = self.registers[inst.arg1] / (val if val != 0 else 1e-12)
                pc += 1

            elif op == OpCode.FMA:
                self.registers[inst.arg1] += self.registers[inst.arg2] * self.registers[inst.arg3]
                pc += 1

            elif op == OpCode.VEC_DOT:
                addr_a = int(self.registers[inst.arg1])
                addr_b = int(self.registers[inst.arg2])
                length = int(self.registers[inst.arg3]) if inst.arg3 != 0 else int(inst.imm)
                vec_a = self.memory[addr_a : addr_a + length]
                vec_b = self.memory[addr_b : addr_b + length]
                self.registers[Register.ACC] = float(np.dot(vec_a, vec_b))
                pc += 1

            elif op == OpCode.VEC_FMA:
                # addr_dest += scalar * addr_src
                addr_dest = int(self.registers[inst.arg1])
                addr_src = int(self.registers[inst.arg2])
                scalar = self.registers[inst.arg3]
                length = int(inst.imm)
                self.memory[addr_dest : addr_dest + length] += scalar * self.memory[addr_src : addr_src + length]
                pc += 1

            elif op == OpCode.VEC_SUB:
                addr_dest = int(self.registers[inst.arg1])
                addr_a = int(self.registers[inst.arg2])
                addr_b = int(self.registers[inst.arg3])
                length = int(inst.imm)
                self.memory[addr_dest : addr_dest + length] = (
                    self.memory[addr_a : addr_a + length] - self.memory[addr_b : addr_b + length]
                )
                pc += 1

            elif op == OpCode.ACT_RELU:
                addr = int(self.registers[inst.arg1])
                length = int(inst.imm)
                self.memory[addr : addr + length] = np.maximum(0.0, self.memory[addr : addr + length])
                pc += 1

            elif op == OpCode.GRAD_RELU:
                addr_grad = int(self.registers[inst.arg1])
                addr_act = int(self.registers[inst.arg2])
                length = int(inst.imm)
                self.memory[addr_grad : addr_grad + length] *= (
                    self.memory[addr_act : addr_act + length] > 0.0
                )
                pc += 1

            elif op == OpCode.ACT_SIGMOID:
                addr = int(self.registers[inst.arg1])
                length = int(inst.imm)
                x = np.clip(self.memory[addr : addr + length], -500.0, 500.0)
                self.memory[addr : addr + length] = 1.0 / (1.0 + np.exp(-x))
                pc += 1

            elif op == OpCode.UPDATE_SGD:
                addr_w = int(self.registers[inst.arg1])
                addr_g = int(self.registers[inst.arg2])
                lr = self.registers[inst.arg3]
                length = int(inst.imm)
                # Stochastic Gradient Descent: w = w - lr * grad
                self.memory[addr_w : addr_w + length] -= lr * self.memory[addr_g : addr_g + length]
                pc += 1

            elif op == OpCode.COMPUTE_MSE:
                addr_pred = int(self.registers[inst.arg1])
                addr_target = int(self.registers[inst.arg2])
                length = int(inst.imm)
                diff = self.memory[addr_pred : addr_pred + length] - self.memory[addr_target : addr_target + length]
                mse = float(np.mean(diff**2))
                self.registers[Register.ERR] = mse
                pc += 1

            elif op == OpCode.JMP:
                pc = int(inst.imm)

            elif op == OpCode.JZ:
                if self.registers[inst.arg1] == 0.0:
                    pc = int(inst.imm)
                else:
                    pc += 1
            else:
                pc += 1

        self.registers[Register.PC] = float(pc)
        return cycles
