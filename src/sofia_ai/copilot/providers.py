"""AI Engine providers for Sofia Copilot.

Supports NVIDIA NIM (api.nvidia.com) and OpenRouter (openrouter.ai) as LLM
backends, implemented with Python standard library (urllib.request) to ensure
zero third-party dependencies in the core.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from typing import Any, Final

from ..security.secrets import secret
from .base import CopilotProvider, CopilotRequest, CopilotResponse, EvidenceCopilot

logger = logging.getLogger(__name__)

DEFAULT_NVIDIA_MODEL: Final[str] = "nvidia/llama-3.1-nemotron-70b-instruct"
DEFAULT_OPENROUTER_MODEL: Final[str] = "anthropic/claude-3.5-sonnet"
DEFAULT_TIMEOUT_SECONDS: Final[float] = 30.0


class NvidiaProvider:
    """NVIDIA NIM LLM Provider (https://build.nvidia.com).

    Uses standard library urllib.request to maintain zero-dependency core.
    API key is read from ``NVIDIA_API_KEY`` or ``SOFIA_NVIDIA_API_KEY``.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_NVIDIA_MODEL,
        base_url: str = "https://integrate.api.nvidia.com/v1",
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self.api_key = api_key or secret("NVIDIA_API_KEY") or os.environ.get("NVIDIA_API_KEY")
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    @property
    def is_available(self) -> bool:
        return bool(self.api_key and self.api_key.strip())

    def complete(self, prompt: str, *, max_tokens: int = 512, temperature: float = 0.2) -> str:
        """Call NVIDIA chat completion endpoint."""
        if not self.is_available:
            raise ValueError(
                "NVIDIA API key not found. Set NVIDIA_API_KEY or SOFIA_NVIDIA_API_KEY."
            )

        url = f"{self.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "User-Agent": "Sofia-Engine/2.0 (Rootcastle)",
        }
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are Sofia Copilot, an expert industrial condition monitoring "
                        "and technical diagnostics assistant developed by Rootcastle Engineering "
                        "& Innovation. You analyze evidence objectively based on physics, "
                        "ISO 10816/20816 vibration standards, and digital signal processing. "
                        "Never invent sensor readings. State limitations clearly."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
                choices = body.get("choices", [])
                if choices and "message" in choices[0]:
                    return str(choices[0]["message"].get("content", "")).strip()
                return ""
        except urllib.error.HTTPError as exc:
            err_body = exc.read().decode("utf-8", errors="replace")
            logger.error("NVIDIA API error %d: %s", exc.code, err_body)
            raise RuntimeError(f"NVIDIA API HTTP {exc.code}: {err_body}") from exc
        except Exception as exc:
            logger.error("NVIDIA connection failed: %s", exc)
            raise RuntimeError(f"NVIDIA connection error: {exc}") from exc


class OpenRouterProvider:
    """OpenRouter LLM Provider (https://openrouter.ai).

    Allows routing to Claude, GPT-4o, Llama 3, DeepSeek, Mistral, and more.
    API key is read from ``OPENROUTER_API_KEY`` or ``SOFIA_OPENROUTER_API_KEY``.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_OPENROUTER_MODEL,
        base_url: str = "https://openrouter.ai/api/v1",
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self.api_key = (
            api_key or secret("OPENROUTER_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
        )
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    @property
    def is_available(self) -> bool:
        return bool(self.api_key and self.api_key.strip())

    def complete(self, prompt: str, *, max_tokens: int = 512, temperature: float = 0.2) -> str:
        """Call OpenRouter chat completion endpoint."""
        if not self.is_available:
            raise ValueError(
                "OpenRouter API key not found. Set OPENROUTER_API_KEY or SOFIA_OPENROUTER_API_KEY."
            )

        url = f"{self.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "https://rootcastle.com/",
            "X-Title": "Sofia Engine (Rootcastle)",
            "User-Agent": "Sofia-Engine/2.0 (Rootcastle)",
        }
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are Sofia Copilot, an expert industrial condition monitoring "
                        "and technical diagnostics assistant developed by Rootcastle Engineering "
                        "& Innovation. You analyze evidence objectively based on physics, "
                        "ISO 10816/20816 vibration standards, and digital signal processing. "
                        "Never invent sensor readings. State limitations clearly."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
                choices = body.get("choices", [])
                if choices and "message" in choices[0]:
                    return str(choices[0]["message"].get("content", "")).strip()
                return ""
        except urllib.error.HTTPError as exc:
            err_body = exc.read().decode("utf-8", errors="replace")
            logger.error("OpenRouter API error %d: %s", exc.code, err_body)
            raise RuntimeError(f"OpenRouter API HTTP {exc.code}: {err_body}") from exc
        except Exception as exc:
            logger.error("OpenRouter connection failed: %s", exc)
            raise RuntimeError(f"OpenRouter connection error: {exc}") from exc


class AIEngine:
    """Unified Sofia AI Engine.

    Automatically resolves available LLM keys (NVIDIA, OpenRouter) or operates
    in deterministic offline mode.
    """

    def __init__(
        self,
        provider: str = "auto",
        model: str | None = None,
        allow_provider_text: bool = True,
    ) -> None:
        self.provider_name = provider.lower()
        self.allow_provider_text = allow_provider_text
        self.provider: CopilotProvider = self._resolve_provider(self.provider_name, model)
        self.copilot = EvidenceCopilot(
            provider=self.provider,
            allow_provider_text=allow_provider_text,
            name=f"sofia-ai-{self.provider_name}",
        )

    def _resolve_provider(self, name: str, model: str | None) -> CopilotProvider:
        from .base import OfflineProvider

        if name == "nvidia":
            return NvidiaProvider(model=model or DEFAULT_NVIDIA_MODEL)
        if name == "openrouter":
            return OpenRouterProvider(model=model or DEFAULT_OPENROUTER_MODEL)
        if name == "auto":
            # Priority: NVIDIA -> OpenRouter -> Offline
            nvidia = NvidiaProvider(model=model or DEFAULT_NVIDIA_MODEL)
            if nvidia.is_available:
                self.provider_name = "nvidia"
                return nvidia
            openrouter = OpenRouterProvider(model=model or DEFAULT_OPENROUTER_MODEL)
            if openrouter.is_available:
                self.provider_name = "openrouter"
                return openrouter
            self.provider_name = "offline"
            return OfflineProvider()

        self.provider_name = "offline"
        return OfflineProvider()

    def ask(self, request: CopilotRequest) -> CopilotResponse:
        """Process an engineering question through the copilot."""
        return self.copilot.ask(request)

    def explain(
        self,
        question: str,
        device_id: str,
        evidence: list[dict[str, Any]] | None = None,
        events: list[dict[str, Any]] | None = None,
    ) -> str:
        """Convenience method returning plain text response."""
        req = CopilotRequest(
            question=question,
            device_id=device_id,
            evidence=tuple(evidence or []),
            events=tuple(events or []),
        )
        return self.ask(req).text


__all__ = [
    "AIEngine",
    "DEFAULT_NVIDIA_MODEL",
    "DEFAULT_OPENROUTER_MODEL",
    "NvidiaProvider",
    "OpenRouterProvider",
]
