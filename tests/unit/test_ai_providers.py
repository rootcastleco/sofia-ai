"""Unit tests for AI Engine providers (NVIDIA and OpenRouter)."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from sofia_ai.copilot.base import CopilotRequest
from sofia_ai.copilot.providers import (
    AIEngine,
    NvidiaProvider,
    OpenRouterProvider,
)


def test_nvidia_provider_no_key() -> None:
    with patch.dict(os.environ, {}, clear=True):
        provider = NvidiaProvider(api_key=None)
        assert not provider.is_available
        with pytest.raises(ValueError, match="NVIDIA API key not found"):
            provider.complete("test")


def test_openrouter_provider_no_key() -> None:
    with patch.dict(os.environ, {}, clear=True):
        provider = OpenRouterProvider(api_key=None)
        assert not provider.is_available
        with pytest.raises(ValueError, match="OpenRouter API key not found"):
            provider.complete("test")


def test_nvidia_provider_mock_completion() -> None:
    provider = NvidiaProvider(api_key="nvapi-test-key")
    assert provider.is_available

    mock_resp = MagicMock()
    mock_resp.read.return_value = b'{"choices": [{"message": {"content": "NVIDIA analysis: Normal."}}]}'
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp):
        res = provider.complete("Check vibration")
        assert res == "NVIDIA analysis: Normal."


def test_openrouter_provider_mock_completion() -> None:
    provider = OpenRouterProvider(api_key="sk-or-test-key")
    assert provider.is_available

    mock_resp = MagicMock()
    mock_resp.read.return_value = b'{"choices": [{"message": {"content": "OpenRouter analysis: Normal."}}]}'
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp):
        res = provider.complete("Check vibration")
        assert res == "OpenRouter analysis: Normal."


def test_ai_engine_auto_resolution() -> None:
    with patch.dict(os.environ, {}, clear=True):
        engine = AIEngine(provider="auto")
        assert engine.provider_name == "offline"

        # Ask in offline mode
        req = CopilotRequest(
            question="What is the bearing health?",
            device_id="pump-01",
            evidence=({"metric": "rms", "observed": 2.5, "source": "sensor"},),
        )
        resp = engine.ask(req)
        assert resp.grounded
        assert "Device: pump-01" in resp.text
        assert "rms=2.5" in resp.text


def test_ai_engine_with_nvidia_key() -> None:
    with patch.dict(os.environ, {"NVIDIA_API_KEY": "nvapi-test"}, clear=True):
        engine = AIEngine(provider="auto")
        assert engine.provider_name == "nvidia"


def test_ai_engine_with_openrouter_key() -> None:
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-or-test"}, clear=True):
        engine = AIEngine(provider="auto")
        assert engine.provider_name == "openrouter"
