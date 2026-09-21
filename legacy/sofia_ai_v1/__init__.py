"""
Sofia AI - Quantum-Level Neural NLP System

This module provides the core functionality for the Sofia AI model,
combining quantum-inspired neural networks with advanced NLP capabilities.
"""

__version__ = "1.0.0"
__author__ = "Sofia AI Team"

from sofia_ai.core.quantum_engine import QuantumNeuralEngine
from sofia_ai.core.nlp_processor import NLPProcessor
from sofia_ai.models.sofia_model import SofiaModel
from sofia_ai.utils.config import QuantumConfig

__all__ = [
    "SofiaModel",
    "QuantumNeuralEngine",
    "NLPProcessor",
    "QuantumConfig",
]
