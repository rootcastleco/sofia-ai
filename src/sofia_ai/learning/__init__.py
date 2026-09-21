"""Learning and self-training subsystem for Sofia AI.

Features:
- Assembly Virtual Machine & Self-Training Neural Network (:mod:`sofia_ai.learning.asm`)
- Automated Fine-Tuning Engine (:mod:`sofia_ai.learning.finetune`)
- Reinforcement Learning Engine (:mod:`sofia_ai.learning.rl`)
"""

from __future__ import annotations

from .asm import (
    AsmInstruction,
    AssemblyNeuralNetwork,
    OpCode,
    Register,
    SofiaAsmVM,
    assemble,
    disassemble,
)
from .finetune import (
    AutoFineTuner,
    DatasetCurator,
    FineTuneJob,
    FineTuneStatus,
)

__all__ = [
    "AsmInstruction",
    "AssemblyNeuralNetwork",
    "AutoFineTuner",
    "DatasetCurator",
    "FineTuneJob",
    "FineTuneStatus",
    "OpCode",
    "Register",
    "SofiaAsmVM",
    "assemble",
    "disassemble",
]
