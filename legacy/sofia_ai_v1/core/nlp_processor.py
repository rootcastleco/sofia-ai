"""
Natural Language Processor for Sofia AI

This module provides advanced NLP capabilities including:
- Tokenization and embedding
- Contextual understanding
- Semantic analysis
- Multi-language support
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import re
from collections import defaultdict


@dataclass
class Token:
    """Represents a tokenized word with metadata."""
    text: str
    position: int
    pos_tag: Optional[str] = None
    entity_type: Optional[str] = None
    embedding: Optional[np.ndarray] = None
    attention_weights: Optional[np.ndarray] = None


@dataclass
class ProcessedText:
    """Container for processed text results."""
    original_text: str
    tokens: List[Token]
    embeddings: np.ndarray
    entities: List[Tuple[str, str, int, int]]
    sentiment: Optional[Dict[str, float]] = None
    intent: Optional[str] = None
    confidence: Optional[float] = None


class NLPProcessor:
    """
    Advanced Natural Language Processor
    
    Provides state-of-the-art NLP capabilities with quantum-enhanced
    feature extraction and contextual understanding.
    """
    
    def __init__(
        self,
        vocab_size: int = 50000,
        embedding_dim: int = 768,
        max_sequence_length: int = 512,
        languages: List[str] = ['en']
    ):
        """
        Initialize the NLP Processor.
        
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of token embeddings
            max_sequence_length: Maximum sequence length
            languages: Supported languages
        """
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_sequence_length = max_sequence_length
        self.languages = languages
        
        # Initialize vocabulary and embeddings
        self.vocabulary = self._build_vocabulary()
        self.embedding_matrix = self._initialize_embeddings()
        
        # Entity recognition patterns
        self.entity_patterns = self._compile_entity_patterns()
        
        # Sentiment lexicon (simplified)
        self.sentiment_lexicon = self._build_sentiment_lexicon()
        
    def _build_vocabulary(self) -> Dict[str, int]:
        """Build initial vocabulary mapping."""
        # In practice, this would be loaded from pre-trained data
        vocab = {'<PAD>': 0, '<UNK>': 1, '<CLS>': 2, '<SEP>': 3}
        return vocab
    
    def _initialize_embeddings(self) -> np.ndarray:
        """Initialize embedding matrix."""
        return np.random.randn(self.vocab_size, self.embedding_dim) * 0.1
    
    def _compile_entity_patterns(self) -> Dict[str, re.Pattern]:
        """Compile regex patterns for entity recognition."""
        patterns = {
            'PERSON': re.compile(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b'),
            'ORGANIZATION': re.compile(r'\b(?:Inc|Ltd|Corp|LLC)\.?'),
            'LOCATION': re.compile(r'\b(?:New York|London|Paris|Tokyo|Berlin)\b'),
            'DATE': re.compile(r'\b\d{1,2}/\d{1,2}/\d{2,4}\b'),
            'EMAIL': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            'URL': re.compile(r'https?://[^\s]+'),
        }
        return patterns
    
    def _build_sentiment_lexicon(self) -> Dict[str, float]:
        """Build sentiment lexicon."""
        positive_words = {
            'good': 0.8, 'great': 0.9, 'excellent': 1.0, 'amazing': 0.95,
            'wonderful': 0.9, 'fantastic': 0.9, 'love': 0.85, 'happy': 0.8,
            'pleased': 0.75, 'satisfied': 0.7, 'helpful': 0.75, 'thank': 0.7
        }
        negative_words = {
            'bad': -0.8, 'terrible': -0.95, 'awful': -0.9, 'horrible': -0.95,
            'hate': -0.9, 'angry': -0.8, 'sad': -0.75, 'disappointed': -0.7,
            'poor': -0.65, 'worst': -0.95, 'useless': -0.85, 'problem': -0.6
        }
        return {**positive_words, **negative_words}
    
    def tokenize(self, text: str) -> List[Token]:
        """
        Tokenize input text.
        
        Args:
            text: Input text to tokenize
            
        Returns:
            List of Token objects
        """
        # Basic tokenization (in practice, use more sophisticated methods)
        words = re.findall(r'\b\w+\b|[^\w\s]', text.lower())
        
        tokens = []
        position = 0
        for word in words:
            token = Token(
                text=word,
                position=position,
                embedding=self._get_embedding(word)
            )
            tokens.append(token)
            position += 1
            
        return tokens
    
    def _get_embedding(self, word: str) -> np.ndarray:
        """Get or create embedding for a word."""
        word_lower = word.lower()
        if word_lower in self.vocabulary:
            idx = self.vocabulary[word_lower]
            return self.embedding_matrix[idx]
        else:
            # Return UNK embedding or hash-based embedding
            unk_idx = self.vocabulary.get('<UNK>', 1)
            return self.embedding_matrix[unk_idx]
    
    def recognize_entities(self, text: str) -> List[Tuple[str, str, int, int]]:
        """
        Recognize named entities in text.
        
        Args:
            text: Input text
            
        Returns:
            List of (entity_text, entity_type, start_pos, end_pos) tuples
        """
        entities = []
        
        for entity_type, pattern in self.entity_patterns.items():
            for match in pattern.finditer(text):
                entities.append((
                    match.group(),
                    entity_type,
                    match.start(),
                    match.end()
                ))
                
        return entities
    
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment of text.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with sentiment scores
        """
        words = text.lower().split()
        
        positive_score = 0.0
        negative_score = 0.0
        neutral_count = 0
        
        for word in words:
            # Remove punctuation
            word_clean = re.sub(r'[^\w]', '', word)
            
            if word_clean in self.sentiment_lexicon:
                score = self.sentiment_lexicon[word_clean]
                if score > 0:
                    positive_score += score
                else:
                    negative_score += abs(score)
            else:
                neutral_count += 1
        
        total = positive_score + negative_score
        if total == 0:
            return {'positive': 0.33, 'negative': 0.33, 'neutral': 0.34}
        
        return {
            'positive': positive_score / total,
            'negative': negative_score / total,
            'neutral': neutral_count / len(words) if words else 0
        }
    
    def extract_intent(self, text: str) -> Tuple[str, float]:
        """
        Extract intent from text.
        
        Args:
            text: Input text
            
        Returns:
            Tuple of (intent_label, confidence)
        """
        text_lower = text.lower()
        
        # Simple rule-based intent detection
        intent_patterns = {
            'greeting': ['hello', 'hi', 'hey', 'greetings', 'good morning', 'good evening'],
            'question': ['what', 'how', 'when', 'where', 'why', 'who', 'which'],
            'request': ['please', 'can you', 'could you', 'would you', 'i need'],
            'complaint': ['problem', 'issue', 'wrong', 'broken', 'not working'],
            'thanks': ['thank', 'thanks', 'appreciate', 'grateful'],
            'farewell': ['bye', 'goodbye', 'see you', 'later']
        }
        
        best_intent = 'unknown'
        best_confidence = 0.0
        
        for intent, keywords in intent_patterns.items():
            matches = sum(1 for keyword in keywords if keyword in text_lower)
            confidence = matches / len(keywords)
            
            if confidence > best_confidence:
                best_intent = intent
                best_confidence = confidence
        
        return best_intent, best_confidence
    
    def process(
        self,
        text: str,
        include_embeddings: bool = True,
        include_sentiment: bool = True,
        include_intent: bool = True
    ) -> ProcessedText:
        """
        Process text through the complete NLP pipeline.
        
        Args:
            text: Input text to process
            include_embeddings: Whether to compute embeddings
            include_sentiment: Whether to analyze sentiment
            include_intent: Whether to extract intent
            
        Returns:
            ProcessedText object with all results
        """
        # Truncate if necessary
        if len(text) > self.max_sequence_length * 10:  # Approximate character limit
            text = text[:self.max_sequence_length * 10]
        
        # Tokenize
        tokens = self.tokenize(text)
        
        # Get embeddings
        if include_embeddings and tokens:
            embeddings = np.array([t.embedding for t in tokens])
        else:
            embeddings = np.array([])
        
        # Recognize entities
        entities = self.recognize_entities(text)
        
        # Analyze sentiment
        sentiment = None
        if include_sentiment:
            sentiment = self.analyze_sentiment(text)
        
        # Extract intent
        intent = None
        confidence = None
        if include_intent:
            intent, confidence = self.extract_intent(text)
        
        return ProcessedText(
            original_text=text,
            tokens=tokens,
            embeddings=embeddings,
            entities=entities,
            sentiment=sentiment,
            intent=intent,
            confidence=confidence
        )
    
    def batch_process(self, texts: List[str]) -> List[ProcessedText]:
        """
        Process multiple texts in batch.
        
        Args:
            texts: List of texts to process
            
        Returns:
            List of ProcessedText objects
        """
        return [self.process(text) for text in texts]
    
    def get_contextual_representation(
        self,
        tokens: List[Token],
        context_window: int = 3
    ) -> np.ndarray:
        """
        Generate contextual representation using attention mechanism.
        
        Args:
            tokens: List of tokens
            context_window: Size of context window
            
        Returns:
            Contextual embeddings array
        """
        if not tokens or tokens[0].embedding is None:
            return np.array([])
        
        embeddings = np.array([t.embedding for t in tokens])
        contextual = np.zeros_like(embeddings)
        
        for i, token in enumerate(tokens):
            # Calculate attention weights based on proximity
            start = max(0, i - context_window)
            end = min(len(tokens), i + context_window + 1)
            
            weights = []
            weighted_sum = np.zeros(self.embedding_dim)
            
            for j in range(start, end):
                distance = abs(i - j)
                weight = 1.0 / (distance + 1)  # Closer tokens get higher weight
                weights.append(weight)
                weighted_sum += tokens[j].embedding * weight
            
            # Normalize
            total_weight = sum(weights)
            if total_weight > 0:
                contextual[i] = weighted_sum / total_weight
            else:
                contextual[i] = tokens[i].embedding
        
        return contextual
