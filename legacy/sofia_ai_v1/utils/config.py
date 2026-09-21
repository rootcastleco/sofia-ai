"""
Configuration utilities for Sofia AI

This module provides configuration classes and utilities
for setting up quantum parameters and model configurations.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import json


@dataclass
class QuantumConfig:
    """
    Configuration for quantum neural network parameters.
    
    Attributes:
        qubits: Number of qubits in the quantum system
        entanglement_depth: Depth of entanglement layers
        optimization_steps: Number of optimization iterations
        learning_rate: Learning rate for optimization
        coherence_threshold: Minimum coherence level to maintain
        measurement_shots: Number of measurement samples
    """
    qubits: int = 8
    entanglement_depth: int = 3
    optimization_steps: int = 50
    learning_rate: float = 0.01
    coherence_threshold: float = 0.5
    measurement_shots: int = 100
    use_gpu: bool = False
    precision: str = 'float32'
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.qubits < 1 or self.qubits > 20:
            raise ValueError("Qubits must be between 1 and 20")
        if self.entanglement_depth < 1 or self.entanglement_depth > 10:
            raise ValueError("Entanglement depth must be between 1 and 10")
        if self.learning_rate <= 0 or self.learning_rate > 1:
            raise ValueError("Learning rate must be between 0 and 1")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'qubits': self.qubits,
            'entanglement_depth': self.entanglement_depth,
            'optimization_steps': self.optimization_steps,
            'learning_rate': self.learning_rate,
            'coherence_threshold': self.coherence_threshold,
            'measurement_shots': self.measurement_shots,
            'use_gpu': self.use_gpu,
            'precision': self.precision
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'QuantumConfig':
        """Create configuration from dictionary."""
        return cls(**config_dict)
    
    def save(self, filepath: str):
        """Save configuration to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'QuantumConfig':
        """Load configuration from JSON file."""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


@dataclass
class NLPConfig:
    """
    Configuration for NLP processing parameters.
    
    Attributes:
        vocab_size: Size of vocabulary
        embedding_dim: Dimension of token embeddings
        max_sequence_length: Maximum sequence length
        num_heads: Number of attention heads
        num_layers: Number of transformer layers
        dropout_rate: Dropout rate for regularization
        languages: List of supported languages
    """
    vocab_size: int = 50000
    embedding_dim: int = 768
    max_sequence_length: int = 512
    num_heads: int = 12
    num_layers: int = 12
    dropout_rate: float = 0.1
    languages: List[str] = field(default_factory=lambda: ['en'])
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.vocab_size < 1000:
            raise ValueError("Vocabulary size must be at least 1000")
        if self.embedding_dim < 64 or self.embedding_dim > 4096:
            raise ValueError("Embedding dimension must be between 64 and 4096")
        if self.dropout_rate < 0 or self.dropout_rate > 0.9:
            raise ValueError("Dropout rate must be between 0 and 0.9")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'vocab_size': self.vocab_size,
            'embedding_dim': self.embedding_dim,
            'max_sequence_length': self.max_sequence_length,
            'num_heads': self.num_heads,
            'num_layers': self.num_layers,
            'dropout_rate': self.dropout_rate,
            'languages': self.languages
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'NLPConfig':
        """Create configuration from dictionary."""
        return cls(**config_dict)


@dataclass
class ModelConfig:
    """
    Complete configuration for Sofia AI model.
    
    Combines quantum and NLP configurations with additional
    model-specific parameters.
    """
    quantum: QuantumConfig = field(default_factory=QuantumConfig)
    nlp: NLPConfig = field(default_factory=NLPConfig)
    model_name: str = 'sofia-base'
    version: str = '1.0.0'
    description: str = 'Sofia AI - Quantum-Level Neural NLP System'
    
    # Training parameters
    batch_size: int = 32
    num_epochs: int = 100
    early_stopping_patience: int = 10
    
    # Inference parameters
    temperature: float = 1.0
    top_p: float = 0.9
    top_k: int = 50
    
    # Resource management
    max_memory_gb: float = 16.0
    num_workers: int = 4
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'quantum': self.quantum.to_dict(),
            'nlp': self.nlp.to_dict(),
            'model_name': self.model_name,
            'version': self.version,
            'description': self.description,
            'batch_size': self.batch_size,
            'num_epochs': self.num_epochs,
            'early_stopping_patience': self.early_stopping_patience,
            'temperature': self.temperature,
            'top_p': self.top_p,
            'top_k': self.top_k,
            'max_memory_gb': self.max_memory_gb,
            'num_workers': self.num_workers
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ModelConfig':
        """Create configuration from dictionary."""
        quantum_config = QuantumConfig.from_dict(config_dict.get('quantum', {}))
        nlp_config = NLPConfig.from_dict(config_dict.get('nlp', {}))
        
        return cls(
            quantum=quantum_config,
            nlp=nlp_config,
            model_name=config_dict.get('model_name', 'sofia-base'),
            version=config_dict.get('version', '1.0.0'),
            description=config_dict.get('description', 'Sofia AI - Quantum-Level Neural NLP System'),
            batch_size=config_dict.get('batch_size', 32),
            num_epochs=config_dict.get('num_epochs', 100),
            early_stopping_patience=config_dict.get('early_stopping_patience', 10),
            temperature=config_dict.get('temperature', 1.0),
            top_p=config_dict.get('top_p', 0.9),
            top_k=config_dict.get('top_k', 50),
            max_memory_gb=config_dict.get('max_memory_gb', 16.0),
            num_workers=config_dict.get('num_workers', 4)
        )
    
    def save(self, filepath: str):
        """Save configuration to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'ModelConfig':
        """Load configuration from JSON file."""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
    
    def get_optimized_config(self, use_case: str = 'general') -> 'ModelConfig':
        """
        Get optimized configuration for specific use cases.
        
        Args:
            use_case: One of 'general', 'realtime', 'accuracy', 'low_resource'
            
        Returns:
            Optimized ModelConfig
        """
        config = ModelConfig(
            quantum=QuantumConfig(**self.quantum.to_dict()),
            nlp=NLPConfig(**self.nlp.to_dict()),
            model_name=self.model_name,
            version=self.version
        )
        
        if use_case == 'realtime':
            config.quantum.qubits = 6
            config.quantum.entanglement_depth = 2
            config.nlp.max_sequence_length = 256
            config.batch_size = 1
        elif use_case == 'accuracy':
            config.quantum.qubits = 12
            config.quantum.entanglement_depth = 5
            config.quantum.optimization_steps = 100
            config.nlp.embedding_dim = 1024
            config.nlp.num_layers = 24
        elif use_case == 'low_resource':
            config.quantum.qubits = 4
            config.quantum.entanglement_depth = 1
            config.nlp.vocab_size = 10000
            config.nlp.embedding_dim = 256
            config.max_memory_gb = 4.0
        
        return config
