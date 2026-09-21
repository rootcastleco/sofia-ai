"""
Test suite for Sofia AI

Run with: pytest tests/
"""

import pytest
import numpy as np
from sofia_ai import SofiaModel, QuantumConfig, NLPProcessor
from sofia_ai.utils.config import ModelConfig


class TestQuantumEngine:
    """Tests for Quantum Neural Engine."""
    
    def test_initialization(self):
        """Test quantum engine initialization."""
        from sofia_ai.core.quantum_engine import QuantumNeuralEngine
        
        engine = QuantumNeuralEngine(num_qubits=4, entanglement_depth=2)
        assert engine.num_qubits == 4
        assert engine.entanglement_depth == 2
        assert engine.dimension == 16
    
    def test_superposition(self):
        """Test superposition state creation."""
        from sofia_ai.core.quantum_engine import QuantumNeuralEngine
        
        engine = QuantumNeuralEngine(num_qubits=3)
        state = engine.create_superposition()
        
        assert len(state.amplitude) == 8
        assert len(state.phase) == 8
        assert np.isclose(np.linalg.norm(state.amplitude), 1.0)
    
    def test_forward_pass(self):
        """Test forward pass through quantum network."""
        from sofia_ai.core.quantum_engine import QuantumNeuralEngine
        
        engine = QuantumNeuralEngine(num_qubits=4, entanglement_depth=2)
        input_data = np.random.randn(16)
        output = engine.forward(input_data)
        
        assert len(output) == 16
        assert np.all(output >= 0) or np.all(output <= 1)


class TestNLPProcessor:
    """Tests for NLP Processor."""
    
    def test_tokenization(self):
        """Test text tokenization."""
        processor = NLPProcessor()
        tokens = processor.tokenize("Hello world!")
        
        assert len(tokens) > 0
        assert tokens[0].text == "hello"
    
    def test_entity_recognition(self):
        """Test named entity recognition."""
        processor = NLPProcessor()
        text = "John Smith works at Google in New York."
        entities = processor.recognize_entities(text)
        
        assert len(entities) > 0
    
    def test_sentiment_analysis(self):
        """Test sentiment analysis."""
        processor = NLPProcessor()
        
        positive_text = "This is excellent and amazing!"
        sentiment = processor.analyze_sentiment(positive_text)
        assert sentiment['positive'] > sentiment['negative']
        
        negative_text = "This is terrible and awful!"
        sentiment = processor.analyze_sentiment(negative_text)
        assert sentiment['negative'] > sentiment['positive']
    
    def test_intent_detection(self):
        """Test intent detection."""
        processor = NLPProcessor()
        
        intent, confidence = processor.extract_intent("Hello, how are you?")
        assert intent == 'greeting'
        assert confidence > 0
        
        intent, confidence = processor.extract_intent("What is the weather?")
        assert intent == 'question'
    
    def test_full_processing(self):
        """Test complete text processing pipeline."""
        processor = NLPProcessor()
        result = processor.process("Hello! I love this product.")
        
        assert result.original_text == "Hello! I love this product."
        assert len(result.tokens) > 0
        assert result.sentiment is not None
        assert result.intent is not None


class TestSofiaModel:
    """Tests for Sofia AI Model."""
    
    def test_model_initialization(self):
        """Test model initialization."""
        model = SofiaModel()
        assert model is not None
        assert model.config is not None
    
    def test_custom_config(self):
        """Test model with custom configuration."""
        config = ModelConfig(
            quantum=QuantumConfig(qubits=6, entanglement_depth=2),
            nlp=NLPProcessor(vocab_size=10000, embedding_dim=256)
        )
        model = SofiaModel(config=config)
        assert model.config.quantum.qubits == 6
    
    def test_text_processing(self):
        """Test text processing."""
        model = SofiaModel()
        response = model.process("Hello, how can you help me?")
        
        assert response.text == "Hello, how can you help me?"
        assert response.intent is not None
        assert response.confidence >= 0
        assert response.processing_time_ms > 0
    
    def test_chat_functionality(self):
        """Test chat functionality."""
        model = SofiaModel()
        
        greeting_response = model.chat("Hello!")
        assert "Hello" in greeting_response or "hello" in greeting_response.lower()
        
        thanks_response = model.chat("Thank you!")
        assert "welcome" in thanks_response.lower()
    
    def test_analysis(self):
        """Test comprehensive analysis."""
        model = SofiaModel()
        analysis = model.analyze("I love working with AI technology!")
        
        assert 'original_text' in analysis
        assert 'sentiment' in analysis
        assert 'intent' in analysis
        assert 'entities' in analysis
    
    def test_capabilities(self):
        """Test getting model capabilities."""
        model = SofiaModel()
        capabilities = model.get_capabilities()
        
        assert 'nlp_tasks' in capabilities
        assert 'quantum_features' in capabilities
        assert len(capabilities['nlp_tasks']) > 0


class TestConfiguration:
    """Tests for configuration classes."""
    
    def test_quantum_config(self):
        """Test QuantumConfig."""
        config = QuantumConfig(qubits=8, entanglement_depth=3)
        assert config.qubits == 8
        assert config.entanglement_depth == 3
        
        config_dict = config.to_dict()
        assert 'qubits' in config_dict
    
    def test_model_config_serialization(self):
        """Test ModelConfig serialization."""
        import tempfile
        import os
        
        config = ModelConfig()
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_path = f.name
        
        try:
            config.save(temp_path)
            loaded_config = ModelConfig.load(temp_path)
            
            assert loaded_config.quantum.qubits == config.quantum.qubits
            assert loaded_config.nlp.embedding_dim == config.nlp.embedding_dim
        finally:
            os.unlink(temp_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
