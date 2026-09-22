"""Assembly Virtual Machine & Bytecode Engine for Neural Self-Training.

Provides a deterministic, low-level register-based virtual machine designed
specifically for on-device, in-situ neural network inference and self-training.
Operates with explicit registers, fixed memory buffers, strict memory-bounds
checking, cycle accounting, and zero external dependencies (NumPy only).
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = [
    "AsmInstruction",
    "OpCode",
    "Register",
    "SofiaAsmVM",
    "VMExecutionResult",
    "VMFault",
    "VMStatus",
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


class VMStatus(enum.StrEnum):
    """Execution status of the Sofia Assembly VM."""

    HALTED = "HALTED"
    CYCLE_LIMIT = "CYCLE_LIMIT"
    INVALID_OPCODE = "INVALID_OPCODE"
    INVALID_REGISTER = "INVALID_REGISTER"
    MEMORY_FAULT = "MEMORY_FAULT"
    NUMERIC_FAULT = "NUMERIC_FAULT"
    INVALID_PROGRAM = "INVALID_PROGRAM"


@dataclass(slots=True)
class VMFault:
    """Detailed information regarding a VM runtime fault."""

    fault_type: VMStatus
    pc: int
    message: str
    instruction: AsmInstruction | None = None


@dataclass(slots=True)
class VMExecutionResult:
    """Structured result returned by VM execution."""

    status: VMStatus
    cycles: int
    fault: VMFault | None = None
    pc: int = 0
    registers: dict[str, float] = field(default_factory=dict)

    def __int__(self) -> int:
        return self.cycles

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, int):
            return self.cycles == other
        if isinstance(other, VMExecutionResult):
            return self.status == other.status and self.cycles == other.cycles
        return False


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

    def _check_reg(self, reg_idx: int) -> bool:
        return 0 <= reg_idx < len(Register)

    def _check_mem(self, addr: int, length: int = 1) -> bool:
        return length > 0 and addr >= 0 and (addr + length) <= self.memory_size

    def run(self, instructions: list[AsmInstruction], max_cycles: int = 1_000_000) -> VMExecutionResult:
        """Execute assembly program until HALT, cycle limit, or fault."""
        pc = 0
        cycles = 0
        n_inst = len(instructions)

        while pc < n_inst:
            if cycles >= max_cycles:
                return VMExecutionResult(
                    status=VMStatus.CYCLE_LIMIT,
                    cycles=cycles,
                    fault=VMFault(VMStatus.CYCLE_LIMIT, pc, f"Exceeded maximum cycle ceiling ({max_cycles})"),
                    pc=pc,
                    registers={r.name: float(self.registers[r]) for r in Register},
                )

            inst = instructions[pc]
            op = inst.opcode
            cycles += 1

            if op == OpCode.HALT:
                pc += 1
                break

            elif op == OpCode.LOAD_CONST:
                if not self._check_reg(inst.arg1):
                    return self._fault(VMStatus.INVALID_REGISTER, pc, f"Invalid register {inst.arg1}", inst, cycles)
                self.registers[inst.arg1] = inst.imm
                pc += 1

            elif op == OpCode.MOV:
                if not (self._check_reg(inst.arg1) and self._check_reg(inst.arg2)):
                    return self._fault(VMStatus.INVALID_REGISTER, pc, "Invalid register in MOV", inst, cycles)
                self.registers[inst.arg1] = self.registers[inst.arg2]
                pc += 1

            elif op == OpCode.LOAD_MEM:
                if not (self._check_reg(inst.arg1) and self._check_reg(inst.arg2)):
                    return self._fault(VMStatus.INVALID_REGISTER, pc, "Invalid register in LOAD_MEM", inst, cycles)
                addr = int(self.registers[inst.arg2])
                if not self._check_mem(addr, 1):
                    return self._fault(VMStatus.MEMORY_FAULT, pc, f"Memory access out of bounds at {addr}", inst, cycles)
                self.registers[inst.arg1] = self.memory[addr]
                pc += 1

            elif op == OpCode.STORE_MEM:
                if not (self._check_reg(inst.arg1) and self._check_reg(inst.arg2)):
                    return self._fault(VMStatus.INVALID_REGISTER, pc, "Invalid register in STORE_MEM", inst, cycles)
                addr = int(self.registers[inst.arg1])
                if not self._check_mem(addr, 1):
                    return self._fault(VMStatus.MEMORY_FAULT, pc, f"Memory store out of bounds at {addr}", inst, cycles)
                self.memory[addr] = self.registers[inst.arg2]
                pc += 1

            elif op == OpCode.ADD:
                if not (self._check_reg(inst.arg1) and self._check_reg(inst.arg2)):
                    return self._fault(VMStatus.INVALID_REGISTER, pc, "Invalid register in ADD", inst, cycles)
                self.registers[inst.arg1] += self.registers[inst.arg2]
                pc += 1

            elif op == OpCode.SUB:
                if not (self._check_reg(inst.arg1) and self._check_reg(inst.arg2)):
                    return self._fault(VMStatus.INVALID_REGISTER, pc, "Invalid register in SUB", inst, cycles)
                self.registers[inst.arg1] -= self.registers[inst.arg2]
                pc += 1

            elif op == OpCode.MUL:
                if not (self._check_reg(inst.arg1) and self._check_reg(inst.arg2)):
                    return self._fault(VMStatus.INVALID_REGISTER, pc, "Invalid register in MUL", inst, cycles)
                self.registers[inst.arg1] *= self.registers[inst.arg2]
                pc += 1

            elif op == OpCode.DIV:
                if not (self._check_reg(inst.arg1) and self._check_reg(inst.arg2)):
                    return self._fault(VMStatus.INVALID_REGISTER, pc, "Invalid register in DIV", inst, cycles)
                val = self.registers[inst.arg2]
                self.registers[inst.arg1] = self.registers[inst.arg1] / (val if val != 0 else 1e-12)
                pc += 1

            elif op == OpCode.FMA:
                if not (self._check_reg(inst.arg1) and self._check_reg(inst.arg2) and self._check_reg(inst.arg3)):
                    return self._fault(VMStatus.INVALID_REGISTER, pc, "Invalid register in FMA", inst, cycles)
                self.registers[inst.arg1] += self.registers[inst.arg2] * self.registers[inst.arg3]
                pc += 1

            elif op == OpCode.VEC_DOT:
                if not (self._check_reg(inst.arg1) and self._check_reg(inst.arg2)):
                    return self._fault(VMStatus.INVALID_REGISTER, pc, "Invalid register in VEC_DOT", inst, cycles)
                addr_a = int(self.registers[inst.arg1])
                addr_b = int(self.registers[inst.arg2])
                length = int(self.registers[inst.arg3]) if (inst.arg3 != 0 and self._check_reg(inst.arg3)) else int(inst.imm)
                if not (self._check_mem(addr_a, length) and self._check_mem(addr_b, length)):
                    return self._fault(VMStatus.MEMORY_FAULT, pc, f"VEC_DOT out of bounds at [{addr_a}, {addr_b}], len={length}", inst, cycles)
                vec_a = self.memory[addr_a : addr_a + length]
                vec_b = self.memory[addr_b : addr_b + length]
                self.registers[Register.ACC] = float(np.dot(vec_a, vec_b))
                pc += 1

            elif op == OpCode.VEC_FMA:
                addr_dest = int(self.registers[inst.arg1])
                addr_src = int(self.registers[inst.arg2])
                scalar = self.registers[inst.arg3]
                length = int(inst.imm)
                if not (self._check_mem(addr_dest, length) and self._check_mem(addr_src, length)):
                    return self._fault(VMStatus.MEMORY_FAULT, pc, f"VEC_FMA out of bounds at [{addr_dest}, {addr_src}], len={length}", inst, cycles)
                self.memory[addr_dest : addr_dest + length] += scalar * self.memory[addr_src : addr_src + length]
                pc += 1

            elif op == OpCode.VEC_SUB:
                addr_dest = int(self.registers[inst.arg1])
                addr_a = int(self.registers[inst.arg2])
                addr_b = int(self.registers[inst.arg3])
                length = int(inst.imm)
                if not (self._check_mem(addr_dest, length) and self._check_mem(addr_a, length) and self._check_mem(addr_b, length)):
                    return self._fault(VMStatus.MEMORY_FAULT, pc, "VEC_SUB out of bounds", inst, cycles)
                self.memory[addr_dest : addr_dest + length] = (
                    self.memory[addr_a : addr_a + length] - self.memory[addr_b : addr_b + length]
                )
                pc += 1

            elif op == OpCode.ACT_RELU:
                addr = int(self.registers[inst.arg1])
                length = int(inst.imm)
                if not self._check_mem(addr, length):
                    return self._fault(VMStatus.MEMORY_FAULT, pc, f"ACT_RELU out of bounds at {addr}, len={length}", inst, cycles)
                self.memory[addr : addr + length] = np.maximum(0.0, self.memory[addr : addr + length])
                pc += 1

            elif op == OpCode.GRAD_RELU:
                addr_grad = int(self.registers[inst.arg1])
                addr_act = int(self.registers[inst.arg2])
                length = int(inst.imm)
                if not (self._check_mem(addr_grad, length) and self._check_mem(addr_act, length)):
                    return self._fault(VMStatus.MEMORY_FAULT, pc, "GRAD_RELU out of bounds", inst, cycles)
                self.memory[addr_grad : addr_grad + length] *= (
                    self.memory[addr_act : addr_act + length] > 0.0
                )
                pc += 1

            elif op == OpCode.ACT_SIGMOID:
                addr = int(self.registers[inst.arg1])
                length = int(inst.imm)
                if not self._check_mem(addr, length):
                    return self._fault(VMStatus.MEMORY_FAULT, pc, "ACT_SIGMOID out of bounds", inst, cycles)
                x = np.clip(self.memory[addr : addr + length], -500.0, 500.0)
                self.memory[addr : addr + length] = 1.0 / (1.0 + np.exp(-x))
                pc += 1

            elif op == OpCode.UPDATE_SGD:
                addr_w = int(self.registers[inst.arg1])
                addr_g = int(self.registers[inst.arg2])
                lr = self.registers[inst.arg3]
                length = int(inst.imm)
                if not (self._check_mem(addr_w, length) and self._check_mem(addr_g, length)):
                    return self._fault(VMStatus.MEMORY_FAULT, pc, f"UPDATE_SGD out of bounds at [{addr_w}, {addr_g}], len={length}", inst, cycles)
                self.memory[addr_w : addr_w + length] -= lr * self.memory[addr_g : addr_g + length]
                pc += 1

            elif op == OpCode.COMPUTE_MSE:
                addr_pred = int(self.registers[inst.arg1])
                addr_target = int(self.registers[inst.arg2])
                length = int(inst.imm)
                if not (self._check_mem(addr_pred, length) and self._check_mem(addr_target, length)):
                    return self._fault(VMStatus.MEMORY_FAULT, pc, "COMPUTE_MSE out of bounds", inst, cycles)
                diff = self.memory[addr_pred : addr_pred + length] - self.memory[addr_target : addr_target + length]
                mse = float(np.mean(diff**2))
                self.registers[Register.ERR] = mse
                pc += 1

            elif op == OpCode.JMP:
                target_pc = int(inst.imm)
                if not (0 <= target_pc < n_inst):
                    return self._fault(VMStatus.INVALID_PROGRAM, pc, f"JMP out of bounds to {target_pc}", inst, cycles)
                pc = target_pc

            elif op == OpCode.JZ:
                target_pc = int(inst.imm)
                if self.registers[inst.arg1] == 0.0:
                    if not (0 <= target_pc < n_inst):
                        return self._fault(VMStatus.INVALID_PROGRAM, pc, f"JZ out of bounds to {target_pc}", inst, cycles)
                    pc = target_pc
                else:
                    pc += 1

            else:
                return self._fault(VMStatus.INVALID_OPCODE, pc, f"Unknown opcode: {op}", inst, cycles)

        self.registers[Register.PC] = float(pc)
        return VMExecutionResult(
            status=VMStatus.HALTED,
            cycles=cycles,
            fault=None,
            pc=pc,
            registers={r.name: float(self.registers[r]) for r in Register},
        )

    def _fault(self, fault_type: VMStatus, pc: int, message: str, inst: AsmInstruction, cycles: int) -> VMExecutionResult:
        fault = VMFault(fault_type=fault_type, pc=pc, message=message, instruction=inst)
        self.registers[Register.PC] = float(pc)
        return VMExecutionResult(
            status=fault_type,
            cycles=cycles,
            fault=fault,
            pc=pc,
            registers={r.name: float(self.registers[r]) for r in Register},
        )
