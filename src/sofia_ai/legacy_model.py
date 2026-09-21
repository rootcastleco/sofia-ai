"""Preserved pre-2.0 ``SofiaModel`` facade. Deprecated.

Kept so ``from sofia_ai import SofiaModel`` continues to work. Behaviour differences
from 1.x, all documented in ``migration.md``:

* persistence is **JSON + NPZ**, not pickle — 1.x ``.pkl`` files are deliberately
  not loadable, because unpickling an untrusted model file is arbitrary code
  execution;
* the embedding matrix is allocated lazily (no 307 MB at construction);
* ``chat`` no longer claims to be "quantum-powered".

This facade is not the Sofia Engine product surface. For machine health use
:mod:`sofia_ai.diagnostics` and :mod:`sofia_ai.inference`.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config.legacy import ModelConfig
from .copilot.legacy_nlp import NLPProcessor, ProcessedText
from .core.errors import ModelError
from .experimental.quantum import QuantumNeuralEngine

__all__ = ["LEGACY_MODEL_FORMAT", "ModelResponse", "SofiaModel"]

LEGACY_MODEL_FORMAT: str = "sofia-legacy-2.0"


@dataclass(slots=True)
class ModelResponse:
    """Container for a legacy model response."""

    text: str
    intent: str | None
    confidence: float
    sentiment: dict[str, float] | None
    entities: list[tuple[str, str, int, int]]
    quantum_features: dict[str, Any] | None
    processing_time_ms: float
    metadata: dict[str, Any]


class SofiaModel:
    """Deprecated 1.x facade: text processing backed by preserved components."""

    def __init__(self, config: ModelConfig | None = None) -> None:
        self.config = config or ModelConfig()
        self.quantum_engine = QuantumNeuralEngine(
            num_qubits=self.config.quantum.qubits,
            entanglement_depth=self.config.quantum.entanglement_depth,
            optimization_steps=self.config.quantum.optimization_steps,
        )
        self.nlp_processor = NLPProcessor(
            vocab_size=self.config.nlp.vocab_size,
            embedding_dim=self.config.nlp.embedding_dim,
            max_sequence_length=self.config.nlp.max_sequence_length,
            languages=self.config.nlp.languages,
        )
        self.is_trained = False
        self.model_version = self.config.version

    # -- inference ---------------------------------------------------------

    def process(self, text: str) -> ModelResponse:
        """Process text through the preserved pipeline."""
        started = time.perf_counter()
        result: ProcessedText = self.nlp_processor.process(
            text=text, include_embeddings=True, include_sentiment=True,
            include_intent=True,
        )
        quantum_features: dict[str, Any] | None = None
        if result.embeddings.size > 0:
            features = result.embeddings.ravel()
            quantum_features = self.quantum_engine.get_quantum_features(features)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return ModelResponse(
            text=result.original_text,
            intent=result.intent,
            confidence=result.confidence or 0.0,
            sentiment=result.sentiment,
            entities=result.entities,
            quantum_features=quantum_features,
            processing_time_ms=elapsed_ms,
            metadata={
                "num_tokens": len(result.tokens),
                "model_version": self.model_version,
                "config": self.config.model_name,
            },
        )

    def chat(self, message: str, context: list[str] | None = None) -> str:
        """Generate a conversational response."""
        response = self.process(message)
        if response.intent == "greeting":
            return "Hello! I'm Sofia. How can I help you today?"
        if response.intent == "question":
            return f"I understand you're asking about something. {self._generate_answer(response)}"
        if response.intent == "thanks":
            return "You're welcome! Feel free to ask me anything else."
        if response.intent == "farewell":
            return "Goodbye! Have a great day!"
        if response.sentiment and response.sentiment.get("negative", 0) > 0.5:
            return "I sense some frustration. Let me know how I can better assist you."
        return (
            "Thank you for your message. I've processed it with "
            f"{response.confidence:.2%} confidence. What else would you like to discuss?"
        )

    def _generate_answer(self, response: ModelResponse) -> str:
        if response.entities:
            entities = ", ".join(e[0] for e in response.entities[:3])
            return f"I found references to: {entities}."
        return "I'm processing your question with the preserved 1.x text pipeline."

    def analyze(self, text: str) -> dict[str, Any]:
        """Comprehensive text analysis."""
        response = self.process(text)
        return {
            "original_text": text,
            "intent": response.intent,
            "confidence": response.confidence,
            "sentiment": response.sentiment,
            "entities": [
                {"text": e[0], "type": e[1], "start": e[2], "end": e[3]}
                for e in response.entities
            ],
            "quantum_metrics": {
                "entropy": (
                    response.quantum_features.get("entropy")
                    if response.quantum_features else None
                ),
                "coherence": (
                    response.quantum_features.get("coherence")
                    if response.quantum_features else None
                ),
            },
            "processing_time_ms": response.processing_time_ms,
            "token_count": response.metadata["num_tokens"],
        }

    def batch_process(self, texts: list[str]) -> list[ModelResponse]:
        return [self.process(text) for text in texts]

    def get_capabilities(self) -> dict[str, Any]:
        return {
            "nlp_tasks": [
                "tokenization",
                "named_entity_recognition",
                "sentiment_analysis",
                "intent_detection",
            ],
            "quantum_features": [
                "superposition_encoding",
                "normalized_matrix_layers",
                "probabilistic_sampling",
            ],
            "supported_languages": self.config.nlp.languages,
            "max_context_length": self.config.nlp.max_sequence_length,
            "note": "quantum_features are a quantum-inspired simulation, not quantum hardware",
        }

    def train(self, training_data: list[dict[str, Any]],
              epochs: int | None = None) -> None:
        """Simplified training loop retained for API compatibility.

        This does not fit a useful model; it exists so 1.x callers keep working.
        """
        num_epochs = epochs or self.config.num_epochs
        for _ in range(max(0, int(num_epochs))):
            for example in training_data:
                text = str(example.get("text", ""))
                _ = self.nlp_processor.process(text)
        self.is_trained = True

    # -- persistence (no pickle) -------------------------------------------

    def save(self, filepath: str) -> str:
        """Save as JSON manifest plus NPZ weights. No pickle."""
        path = Path(filepath)
        weights_path = path.with_suffix(".npz")
        arrays: dict[str, np.ndarray] = {
            f"w{i}": np.asarray(w)
            for i, w in enumerate(self.quantum_engine.weights)
        }
        arrays.update(
            {f"b{i}": np.asarray(b) for i, b in enumerate(self.quantum_engine.biases)}
        )
        np.savez(weights_path, allow_pickle=False, **arrays)
        payload = {
            "format": LEGACY_MODEL_FORMAT,
            "config": self.config.to_dict(),
            "is_trained": self.is_trained,
            "weights_file": weights_path.name,
            "weights_count": len(self.quantum_engine.weights),
        }
        path.write_text(json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8")
        return str(path)

    @classmethod
    def load(cls, filepath: str) -> SofiaModel:
        """Load a JSON+NPZ model. 1.x pickle files are intentionally rejected."""
        path = Path(filepath)
        if path.suffix == ".pkl":
            raise ModelError(
                "Sofia 2.0 does not load pickle files: unpickling untrusted model "
                "files is arbitrary code execution. Re-save the model with save().",
                details={"path": str(path)},
            )
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ModelError(
                f"cannot read model {path}: {exc}", details={"path": str(path)}
            ) from exc
        if payload.get("format") != LEGACY_MODEL_FORMAT:
            raise ModelError(
                f"unsupported model format {payload.get('format')!r}",
                details={"expected": LEGACY_MODEL_FORMAT},
            )
        model = cls(config=ModelConfig.from_dict(payload["config"]))
        weights_file = path.parent / str(payload["weights_file"])
        if weights_file.is_file():
            with np.load(weights_file) as data:
                model.quantum_engine.weights = [
                    np.asarray(data[f"w{i}"]) for i in range(int(payload["weights_count"]))
                ]
                model.quantum_engine.biases = [
                    np.asarray(data[f"b{i}"]) for i in range(int(payload["weights_count"]))
                ]
        model.is_trained = bool(payload.get("is_trained", False))
        return model

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return (
            f"SofiaModel(version='{self.model_version}', "
            f"qubits={self.config.quantum.qubits}, trained={self.is_trained})"
        )
