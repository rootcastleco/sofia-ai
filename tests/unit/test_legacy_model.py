"""Deprecated 1.x SofiaModel facade: processing, chat, and persistence."""

from __future__ import annotations

import pytest

from sofia_ai.core.errors import ModelError
from sofia_ai.legacy_model import LEGACY_MODEL_FORMAT, SofiaModel


class TestSofiaModel:
    def test_process_returns_response(self) -> None:
        model = SofiaModel()
        assert model.is_trained is False
        response = model.process("Hello there!")
        assert response.text == "Hello there!"
        assert isinstance(response.confidence, float)
        assert "num_tokens" in response.metadata

    def test_chat_greeting(self) -> None:
        assert "Hello!" in SofiaModel().chat("Hi there")

    def test_chat_generic_fallback(self) -> None:
        reply = SofiaModel().chat("The pump fails at 60 Hz on Tuesdays")
        assert "confidence" in reply

    def test_batch_process(self) -> None:
        model = SofiaModel()
        responses = model.batch_process(["one", "two", "three"])
        assert len(responses) == 3

    def test_analyze_structure(self) -> None:
        payload = SofiaModel().analyze("What is the state of the motor?")
        assert payload["original_text"]
        assert "quantum_metrics" in payload
        assert "token_count" in payload

    def test_get_capabilities(self) -> None:
        caps = SofiaModel().get_capabilities()
        assert "nlp_tasks" in caps
        assert "quantum_features" in caps
        assert caps["supported_languages"] == ["en"]

    def test_train_marks_state(self) -> None:
        model = SofiaModel()
        model.train([{"text": "sample"}], epochs=1)
        assert model.is_trained is True

    def test_save_load_roundtrip(self, tmp_path) -> None:
        path = tmp_path / "m.json"
        model = SofiaModel()
        model.train([{"text": "train me"}], epochs=1)
        saved = model.save(str(path))
        assert saved.endswith("m.json")
        loaded = SofiaModel.load(saved)
        assert loaded.is_trained is True

    def test_save_produces_json_not_pickle(self, tmp_path) -> None:
        path = tmp_path / "m.json"
        saved = SofiaModel().save(str(path))
        text = path.read_text(encoding="utf-8")
        import json

        assert json.loads(text)["format"] == LEGACY_MODEL_FORMAT
        assert saved.endswith(".json")

    def test_load_rejects_pickle(self, tmp_path) -> None:
        path = tmp_path / "m.pkl"
        path.write_bytes(b"not really a pickle")
        with pytest.raises(ModelError, match="does not load pickle"):
            SofiaModel.load(str(path))

    def test_load_rejects_garbage(self, tmp_path) -> None:
        path = tmp_path / "m.json"
        path.write_text("{not json", encoding="utf-8")
        with pytest.raises(ModelError, match="cannot read model"):
            SofiaModel.load(str(path))

    def test_load_rejects_wrong_format(self, tmp_path) -> None:
        path = tmp_path / "m.json"
        import json

        path.write_text(json.dumps({"format": "other"}), encoding="utf-8")
        with pytest.raises(ModelError, match="unsupported model format"):
            SofiaModel.load(str(path))
