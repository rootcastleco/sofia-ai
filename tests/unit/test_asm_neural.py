"""Unit tests for Assembly Virtual Machine and Neural Self-Training Engine."""

from __future__ import annotations

import numpy as np
import pytest

from sofia_ai.learning.asm import (
    AsmInstruction,
    AssemblyNeuralNetwork,
    OpCode,
    Register,
    SofiaAsmVM,
    assemble,
    disassemble,
)


def test_asm_vm_basic_arithmetic() -> None:
    vm = SofiaAsmVM(memory_size=1024)
    # R0 = 15.5, R1 = 4.5, ACC = R0 + R1 = 20.0
    instructions = [
        AsmInstruction(OpCode.LOAD_CONST, arg1=int(Register.R0), imm=15.5),
        AsmInstruction(OpCode.LOAD_CONST, arg1=int(Register.R1), imm=4.5),
        AsmInstruction(OpCode.ADD, arg1=int(Register.R0), arg2=int(Register.R1)),
        AsmInstruction(OpCode.HALT),
    ]
    cycles = vm.run(instructions)
    assert cycles == 4
    assert vm.registers[Register.R0] == pytest.approx(20.0)


def test_asm_vm_vector_ops() -> None:
    vm = SofiaAsmVM(memory_size=1024)
    # Memory: a = [1.0, 2.0, 3.0], b = [4.0, 5.0, 6.0]
    # Dot product: 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32.0
    vm.memory[0:3] = np.array([1.0, 2.0, 3.0])
    vm.memory[3:6] = np.array([4.0, 5.0, 6.0])

    instructions = [
        AsmInstruction(OpCode.LOAD_CONST, arg1=int(Register.R0), imm=0.0),
        AsmInstruction(OpCode.LOAD_CONST, arg1=int(Register.R1), imm=3.0),
        AsmInstruction(OpCode.LOAD_CONST, arg1=int(Register.R2), imm=3.0),
        AsmInstruction(
            OpCode.VEC_DOT,
            arg1=int(Register.R0),
            arg2=int(Register.R1),
            arg3=int(Register.R2),
        ),
        AsmInstruction(OpCode.HALT),
    ]
    vm.run(instructions)
    assert vm.registers[Register.ACC] == pytest.approx(32.0)


def test_asm_vm_activations_and_sgd() -> None:
    vm = SofiaAsmVM(memory_size=1024)
    # Test ReLU: [-2.0, 0.0, 3.5] -> [0.0, 0.0, 3.5]
    vm.memory[10:13] = np.array([-2.0, 0.0, 3.5])
    instructions = [
        AsmInstruction(OpCode.LOAD_CONST, arg1=int(Register.R0), imm=10.0),
        AsmInstruction(OpCode.ACT_RELU, arg1=int(Register.R0), imm=3.0),
        # Test SGD: weights = [10.0, 20.0], grads = [1.0, 2.0], lr = 0.1
        # Expected: weights -= 0.1 * [1.0, 2.0] -> [9.9, 19.8]
        AsmInstruction(OpCode.LOAD_CONST, arg1=int(Register.LR), imm=0.1),
        AsmInstruction(OpCode.LOAD_CONST, arg1=int(Register.R1), imm=20.0),  # w addr
        AsmInstruction(OpCode.LOAD_CONST, arg1=int(Register.R2), imm=30.0),  # g addr
        AsmInstruction(
            OpCode.UPDATE_SGD,
            arg1=int(Register.R1),
            arg2=int(Register.R2),
            arg3=int(Register.LR),
            imm=2.0,
        ),
        AsmInstruction(OpCode.HALT),
    ]
    vm.memory[20:22] = np.array([10.0, 20.0])
    vm.memory[30:32] = np.array([1.0, 2.0])

    vm.run(instructions)
    np.testing.assert_allclose(vm.memory[10:13], [0.0, 0.0, 3.5])
    np.testing.assert_allclose(vm.memory[20:22], [9.9, 19.8])


def test_assemble_and_disassemble() -> None:
    asm_text = """
    LOAD_CONST R0, 42.0
    LOAD_CONST R1, 8.0
    ADD R0, R1
    HALT
    """
    instructions = assemble(asm_text)
    assert len(instructions) == 4
    assert instructions[0].opcode == OpCode.LOAD_CONST
    assert instructions[0].arg1 == int(Register.R0)
    assert instructions[0].imm == 42.0

    disassembled = disassemble(instructions)
    assert "LOAD_CONST R0, 42.0" in disassembled
    assert "HALT" in disassembled


def test_assembly_neural_network_forward() -> None:
    model = AssemblyNeuralNetwork(input_dim=4, hidden_dim=8, output_dim=2, seed=42)
    x = np.array([0.5, -0.2, 1.0, 0.3])
    out = model.forward(x)
    assert out.shape == (2,)
    assert not np.isnan(out).any()


def test_assembly_neural_network_self_training() -> None:
    # Verify that assembly train_step decreases MSE loss over training epochs
    model = AssemblyNeuralNetwork(
        input_dim=3, hidden_dim=6, output_dim=1, learning_rate=0.05, seed=123
    )
    x = np.array([1.0, 0.5, -0.5])
    y = np.array([2.5])

    initial_loss = model.train_step(x, y)
    for _ in range(50):
        final_loss = model.train_step(x, y)

    # Loss must reduce significantly during assembly-level self training
    assert final_loss < initial_loss
    assert final_loss < 0.1


def test_assembly_emitters() -> None:
    model = AssemblyNeuralNetwork(input_dim=4, hidden_dim=8, output_dim=2)
    x86_code = model.emit_x86_assembly()
    arm_code = model.emit_arm_assembly()
    wasm_code = model.emit_wasm()

    assert "sofia_neural_forward_x86" in x86_code
    assert "vzeroall" in x86_code
    assert "sofia_neural_forward_arm" in arm_code
    assert ".thumb" in arm_code
    assert "(module" in wasm_code
    assert "(export \"forward\"" in wasm_code
