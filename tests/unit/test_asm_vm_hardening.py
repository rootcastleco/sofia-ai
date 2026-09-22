"""Unit tests for SofiaAsmVM Hardening and Fault Semantics (SOFIA-VM-001 to 006)."""

from __future__ import annotations

import numpy as np

from sofia_ai.learning.asm.vm import (
    AsmInstruction,
    OpCode,
    Register,
    SofiaAsmVM,
    VMStatus,
)


def test_vm_cycle_limit_fault() -> None:
    """SOFIA-VM-004: Infinite loop terminates with CYCLE_LIMIT."""
    vm = SofiaAsmVM(memory_size=1024)
    # JMP 0 (infinite loop)
    instructions = [
        AsmInstruction(OpCode.JMP, imm=0.0),
    ]
    res = vm.run(instructions, max_cycles=100)
    assert res.status == VMStatus.CYCLE_LIMIT
    assert res.cycles == 100
    assert res.fault is not None
    assert "Exceeded maximum cycle ceiling" in res.fault.message


def test_vm_memory_bounds_fault() -> None:
    """SOFIA-VM-003: Memory access beyond buffer terminates with MEMORY_FAULT."""
    vm = SofiaAsmVM(memory_size=128)
    # Attempt to load from address 500
    instructions = [
        AsmInstruction(OpCode.LOAD_CONST, arg1=int(Register.R0), imm=500.0),
        AsmInstruction(OpCode.LOAD_MEM, arg1=int(Register.R1), arg2=int(Register.R0)),
        AsmInstruction(OpCode.HALT),
    ]
    res = vm.run(instructions)
    assert res.status == VMStatus.MEMORY_FAULT
    assert res.fault is not None
    assert "Memory access out of bounds" in res.fault.message


def test_vm_vector_bounds_fault() -> None:
    """SOFIA-VM-003: Vector dot product past memory bounds terminates with MEMORY_FAULT."""
    vm = SofiaAsmVM(memory_size=64)
    # VEC_DOT with length 100 on memory size 64
    instructions = [
        AsmInstruction(OpCode.LOAD_CONST, arg1=int(Register.R0), imm=0.0),
        AsmInstruction(OpCode.LOAD_CONST, arg1=int(Register.R1), imm=10.0),
        AsmInstruction(OpCode.LOAD_CONST, arg1=int(Register.R2), imm=100.0),
        AsmInstruction(OpCode.VEC_DOT, arg1=int(Register.R0), arg2=int(Register.R1), arg3=int(Register.R2)),
        AsmInstruction(OpCode.HALT),
    ]
    res = vm.run(instructions)
    assert res.status == VMStatus.MEMORY_FAULT
    assert res.fault is not None


def test_vm_invalid_register_fault() -> None:
    """SOFIA-VM-001: Out-of-range register terminates with INVALID_REGISTER."""
    vm = SofiaAsmVM(memory_size=128)
    instructions = [
        AsmInstruction(OpCode.LOAD_CONST, arg1=99, imm=42.0),
    ]
    res = vm.run(instructions)
    assert res.status == VMStatus.INVALID_REGISTER
    assert res.fault is not None


def test_vm_invalid_jump_fault() -> None:
    """SOFIA-VM-001: Jump past program length terminates with INVALID_PROGRAM."""
    vm = SofiaAsmVM(memory_size=128)
    instructions = [
        AsmInstruction(OpCode.JMP, imm=100.0),
    ]
    res = vm.run(instructions)
    assert res.status == VMStatus.INVALID_PROGRAM
    assert res.fault is not None


def test_vm_determinism() -> None:
    """SOFIA-VM-006: Identical program and memory produce bit-identical results."""
    program = [
        AsmInstruction(OpCode.LOAD_CONST, arg1=int(Register.R0), imm=10.5),
        AsmInstruction(OpCode.LOAD_CONST, arg1=int(Register.R1), imm=2.5),
        AsmInstruction(OpCode.ADD, arg1=int(Register.R0), arg2=int(Register.R1)),
        AsmInstruction(OpCode.HALT),
    ]
    vm1 = SofiaAsmVM(memory_size=256)
    vm2 = SofiaAsmVM(memory_size=256)

    res1 = vm1.run(program)
    res2 = vm2.run(program)

    assert res1.status == VMStatus.HALTED
    assert res2.status == VMStatus.HALTED
    assert res1.cycles == res2.cycles
    np.testing.assert_array_equal(vm1.registers, vm2.registers)
    np.testing.assert_array_equal(vm1.memory, vm2.memory)
