"""Preserved pre-2.0 NLP processor. Deprecated.

Why this file exists:
    ``sofia_ai.core.nlp_processor.NLPProcessor`` was part of the public 1.x API.
    It is preserved here — same class name, same method names, same return shapes —
    so existing imports keep working. It is **not** the future of Sofia; it is a
    rule-based text utility, and the 2.0 documentation no longer describes it as a
    transformer system.

Defects fixed relative to the 1.x implementation (see ``specs/.../audit.md``):

* **S1-06** — the vocabulary was never populated, so every real token resolved to
  ``<UNK>`` and every embedding was identical. Embeddings are now deterministic
  hash-based vectors, so distinct tokens get distinct representations.
* **S2-04** — a ``50000 x 768`` float64 matrix (~307 MB) was allocated in
  ``__init__``. Allocation is now lazy, float32, and size-capped.
* **S2-03** — no global RNG mutation. The processor owns a seeded generator.

None of the public method signatures changed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final

import numpy as np

from ..core.errors import ValidationError

__all__ = ["MAX_EMBEDDING_BYTES", "NLPProcessor", "ProcessedText", "Token"]

#: Refuse to allocate an embedding matrix larger than this (64 MiB).
MAX_EMBEDDING_BYTES: Final[int] = 64 * 1024 * 1024


@dataclass(slots=True)
class Token:
    """A tokenized word with metadata."""

    text: str
    position: int
    pos_tag: str | None = None
    entity_type: str | None = None
    embedding: np.ndarray | None = None
    attention_weights: np.ndarray | None = None


@dataclass(slots=True)
class ProcessedText:
    """Container for processed text results."""

    original_text: str
    tokens: list[Token]
    embeddings: np.ndarray
    entities: list[tuple[str, str, int, int]]
    sentiment: dict[str, float] | None = None
    intent: str | None = None
    confidence: float | None = None


class NLPProcessor:
    """Rule-based text processor. Deprecated; preserved for compatibility.

    Provides tokenization, regex entity recognition, lexicon sentiment analysis,
    keyword intent detection and deterministic hash embeddings. It is not a learned
    language model and makes no accuracy claims.
    """

    def __init__(
        self,
        vocab_size: int = 50000,
        embedding_dim: int = 768,
        max_sequence_length: int = 512,
        languages: list[str] | None = None,
        *,
        seed: int = 0,
        lazy: bool = True,
    ) -> None:
        if vocab_size < 1000:
            raise ValidationError("vocab_size must be at least 1000",
                                  details={"vocab_size": vocab_size})
        if not 64 <= embedding_dim <= 4096:
            raise ValidationError("embedding_dim must be within [64, 4096]",
                                  details={"embedding_dim": embedding_dim})
        if max_sequence_length <= 0:
            raise ValidationError("max_sequence_length must be positive", details={})

        self.vocab_size = int(vocab_size)
        self.embedding_dim = int(embedding_dim)
        self.max_sequence_length = int(max_sequence_length)
        self.languages = list(languages) if languages else ["en"]
        self.lazy = bool(lazy)
        self._rng = np.random.default_rng(int(seed))
        self._seed = int(seed)

        self.vocabulary: dict[str, int] = self._build_vocabulary()
        self.embedding_matrix: np.ndarray | None = None
        self.entity_patterns = self._compile_entity_patterns()
        self.sentiment_lexicon = self._build_sentiment_lexicon()
        if not self.lazy:
            self._ensure_embeddings()

    # -- construction ------------------------------------------------------

    def _build_vocabulary(self) -> dict[str, int]:
        """Special tokens only; real words use deterministic hash indices."""
        return {"<PAD>": 0, "<UNK>": 1, "<CLS>": 2, "<SEP>": 3}

    def _ensure_embeddings(self) -> np.ndarray:
        """Allocate the embedding matrix on demand, float32, size-capped."""
        if self.embedding_matrix is not None:
            return self.embedding_matrix
        rows = min(self.vocab_size, 1 + MAX_EMBEDDING_BYTES // (self.embedding_dim * 4))
        if rows < self.vocab_size:
            self.vocab_size = int(rows)
        matrix = self._rng.standard_normal(
            (self.vocab_size, self.embedding_dim)
        ).astype(np.float32) * 0.1
        self.embedding_matrix = matrix
        return matrix

    def _compile_entity_patterns(self) -> dict[str, re.Pattern[str]]:
        return {
            "PERSON": re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b"),
            "ORGANIZATION": re.compile(r"\b(?:Inc|Ltd|Corp|LLC)\.?"),
            "LOCATION": re.compile(r"\b(?:New York|London|Paris|Tokyo|Berlin)\b"),
            "DATE": re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b"),
            "EMAIL": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
            "URL": re.compile(r"https?://[^\s]+"),
        }

    def _build_sentiment_lexicon(self) -> dict[str, float]:
        positive_words = {
            "good": 0.8, "great": 0.9, "excellent": 1.0, "amazing": 0.95,
            "wonderful": 0.9, "fantastic": 0.9, "love": 0.85, "happy": 0.8,
            "pleased": 0.75, "satisfied": 0.7, "helpful": 0.75, "thank": 0.7,
        }
        negative_words = {
            "bad": -0.8, "terrible": -0.95, "awful": -0.9, "horrible": -0.95,
            "hate": -0.9, "angry": -0.8, "sad": -0.75, "disappointed": -0.7,
            "poor": -0.65, "worst": -0.95, "useless": -0.85, "problem": -0.6,
        }
        return {**positive_words, **negative_words}

    # -- embeddings --------------------------------------------------------

    def _index_for(self, word: str) -> int:
        """Deterministic index for a word.

        Known special tokens keep their ids; every other word hashes into the
        embedding table. This is what makes two different words produce two
        different vectors — the 1.x code returned ``<UNK>`` for all of them.
        """
        lowered = word.lower()
        if lowered in self.vocabulary:
            return self.vocabulary[lowered]
        digest = __import__("hashlib").sha256(lowered.encode("utf-8")).digest()
        hashed = int.from_bytes(digest[:8], "big")
        return 1 + hashed % max(1, self.vocab_size - 1)

    def _get_embedding(self, word: str) -> np.ndarray:
        matrix = self._ensure_embeddings()
        return np.asarray(matrix[self._index_for(word)], dtype=np.float64)

    # -- API ---------------------------------------------------------------

    def tokenize(self, text: str) -> list[Token]:
        """Tokenize input text."""
        words = re.findall(r"\b\w+\b|[^\w\s]", str(text).lower())
        tokens: list[Token] = []
        for position, word in enumerate(words):
            tokens.append(
                Token(text=word, position=position, embedding=self._get_embedding(word))
            )
        return tokens

    def recognize_entities(self, text: str) -> list[tuple[str, str, int, int]]:
        """Regex entity recognition."""
        entities: list[tuple[str, str, int, int]] = []
        for entity_type, pattern in self.entity_patterns.items():
            for match in pattern.finditer(str(text)):
                entities.append((match.group(), entity_type, match.start(), match.end()))
        return entities

    def analyze_sentiment(self, text: str) -> dict[str, float]:
        """Lexicon sentiment. Returns positive/negative/neutral fractions."""
        words = str(text).lower().split()
        if not words:
            return {"positive": 0.0, "negative": 0.0, "neutral": 1.0}
        positive = 0.0
        negative = 0.0
        neutral = 0
        for word in words:
            cleaned = re.sub(r"[^\w]", "", word)
            score = self.sentiment_lexicon.get(cleaned)
            if score is None:
                neutral += 1
            elif score > 0:
                positive += score
            else:
                negative += abs(score)
        total = positive + negative
        if total == 0:
            return {"positive": 0.33, "negative": 0.33,
                    "neutral": neutral / len(words)}
        return {
            "positive": positive / total,
            "negative": negative / total,
            "neutral": neutral / len(words),
        }

    def extract_intent(self, text: str) -> tuple[str, float]:
        """Keyword intent detection. Returns ``(intent, confidence)``."""
        lowered = str(text).lower()
        intent_patterns = {
            "greeting": ["hello", "hi", "hey", "greetings", "good morning", "good evening"],
            "question": ["what", "how", "when", "where", "why", "who", "which"],
            "request": ["please", "can you", "could you", "would you", "i need"],
            "complaint": ["problem", "issue", "wrong", "broken", "not working"],
            "thanks": ["thank", "thanks", "appreciate", "grateful"],
            "farewell": ["bye", "goodbye", "see you", "later"],
        }
        best_intent = "unknown"
        best_confidence = 0.0
        for intent, keywords in intent_patterns.items():
            matches = sum(1 for keyword in keywords if keyword in lowered)
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
        include_intent: bool = True,
    ) -> ProcessedText:
        """Run the full text pipeline."""
        original = str(text)
        limit = self.max_sequence_length * 10
        if len(original) > limit:
            original = original[:limit]
        tokens = self.tokenize(original)
        embeddings = (
            np.array([t.embedding for t in tokens], dtype=np.float64)
            if (include_embeddings and tokens) else np.array([], dtype=np.float64)
        )
        sentiment = self.analyze_sentiment(original) if include_sentiment else None
        intent: str | None = None
        confidence: float | None = None
        if include_intent:
            intent, confidence = self.extract_intent(original)
        return ProcessedText(
            original_text=original,
            tokens=tokens,
            embeddings=embeddings,
            entities=self.recognize_entities(original),
            sentiment=sentiment,
            intent=intent,
            confidence=confidence,
        )

    def batch_process(self, texts: list[str]) -> list[ProcessedText]:
        return [self.process(text) for text in texts]

    def get_contextual_representation(
        self, tokens: list[Token], context_window: int = 3
    ) -> np.ndarray:
        """Distance-weighted average of neighbouring token embeddings."""
        if not tokens or tokens[0].embedding is None:
            return np.array([], dtype=np.float64)
        embeddings = np.array([t.embedding for t in tokens], dtype=np.float64)
        contextual = np.zeros_like(embeddings)
        for i in range(len(tokens)):
            start = max(0, i - context_window)
            end = min(len(tokens), i + context_window + 1)
            weights: list[float] = []
            weighted = np.zeros(self.embedding_dim, dtype=np.float64)
            for j in range(start, end):
                weight = 1.0 / (abs(i - j) + 1)
                weights.append(weight)
                weighted += embeddings[j] * weight
            total = sum(weights)
            contextual[i] = weighted / total if total > 0 else embeddings[i]
        return contextual

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"NLPProcessor(vocab={self.vocab_size}, dim={self.embedding_dim})"

