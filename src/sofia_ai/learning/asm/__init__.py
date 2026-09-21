"""Assembly-level neural self-training and virtual execution engine."""

from __future__ import annotations

from .neural import AssemblyNeuralNetwork
from .vm import (
    AsmInstruction,
    OpCode,
    Register,
    SofiaAsmVM,
    assemble,
    disassemble,
)

__all__ = [
    "AsmInstruction",
    "AssemblyNeuralNetwork",
    "OpCode",
    "Register",
    "SofiaAsmVM",
    "assemble",
    "disassemble",
]
