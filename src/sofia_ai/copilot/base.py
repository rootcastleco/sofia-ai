"""Optional technical copilot layer.

The copilot is **outside** the deterministic pipeline. It consumes structured
Sofia evidence and produces engineer-facing text. It never:

* invents machine state — it can only describe evidence it was given,
* emits commands — there is no actuation surface in this package,
* influences severity — severity is decided by :mod:`sofia_ai.diagnostics`.

Architecture::

    Telemetry / Models / Diagnostics --> Structured Evidence --> Engineering Copilot

Never::

    LLM --> unrestricted machine actuator

The default provider is a deterministic offline renderer, so the core project is
fully usable with no LLM and no network.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final, Protocol

from ..core.validation import validate_identifier

__all__ = [
    "PROVIDER_CONTRACT_VERSION",
    "CopilotProvider",
    "CopilotRequest",
    "CopilotResponse",
    "EngineeringCopilot",
    "NullProvider",
    "OfflineProvider",
    "summarize_evidence",
]

PROVIDER_CONTRACT_VERSION: Final[str] = "1.0"


class CopilotProvider(Protocol):
    """A text-generation provider."""

    def complete(self, prompt: str, *, max_tokens: int = 512) -> str:
        """Return a completion for ``prompt``."""
        ...


class NullProvider:
    """Provider that always declines. Used when no LLM is configured."""

    __slots__ = ()

    def complete(self, prompt: str, *, max_tokens: int = 512) -> str:
        return ""


class OfflineProvider:
    """Deterministic, dependency-free renderer. The default provider.

    It renders the structured facts it is given and adds no claims of its own.
    """

    __slots__ = ()

    def complete(self, prompt: str, *, max_tokens: int = 512) -> str:
        lines = [line for line in prompt.splitlines() if line.strip()]
        return "\n".join(lines[: max(1, max_tokens // 8)])


@dataclass(frozen=True, slots=True)
class CopilotRequest:
    """A request for an engineering explanation.

    Args:
        question: The engineer's question.
        device_id: Equipment identifier.
        evidence: Structured evidence records. The only permitted source of facts.
        events: Health events to explain.
        audience: Intended reader, e.g. ``maintenance``.
    """

    question: str
    device_id: str
    evidence: tuple[Mapping[str, Any], ...] = ()
    events: tuple[Mapping[str, Any], ...] = ()
    audience: str = "maintenance"

    def __post_init__(self) -> None:
        if not self.question.strip():
            raise ValueError("question must not be empty")
        object.__setattr__(
            self, "device_id", validate_identifier(self.device_id, "device_id")
        )


@dataclass(frozen=True, slots=True)
class CopilotResponse:
    """A copilot answer with its provenance attached."""

    text: str
    provider: str
    grounded: bool
    sources: tuple[str, ...] = ()
    provider_contract: str = PROVIDER_CONTRACT_VERSION
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "provider": self.provider,
            "grounded": self.grounded,
            "sources": list(self.sources),
            "provider_contract": self.provider_contract,
            "metadata": dict(self.metadata),
        }


def summarize_evidence(evidence: Sequence[Mapping[str, Any]]) -> list[str]:
    """Render evidence records as factual bullet lines. No interpretation."""
    lines: list[str] = []
    for item in evidence:
        metric = str(item.get("metric", "unknown"))
        observed = item.get("observed")
        reference = item.get("reference")
        source = str(item.get("source", "unknown"))
        quality = str(item.get("quality", "GOOD"))
        lines.append(
            f"- {source} observed {metric}={_fmt(observed)} "
            f"(reference {_fmt(reference)}, data quality {quality})"
        )
    return lines


def _fmt(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{value:.4g}"
    return str(value)


class EngineeringCopilot(ABC):
    """Base class for copilot implementations."""

    @abstractmethod
    def ask(self, request: CopilotRequest) -> CopilotResponse:
        """Answer a request using only the supplied evidence."""


@dataclass(slots=True)
class EvidenceCopilot(EngineeringCopilot):
    """Default copilot: renders grounded explanations, optionally via a provider.

    Args:
        provider: Text provider. ``None`` selects :class:`OfflineProvider`.
        allow_provider_text: When False (default) the response is always the
            deterministic rendering. Set True only when the provider is trusted and
            the output is clearly labelled as non-authoritative.
    """

    provider: CopilotProvider | None = None
    allow_provider_text: bool = False
    name: str = "evidence-copilot"

    def ask(self, request: CopilotRequest) -> CopilotResponse:
        """Build a grounded answer.

        The deterministic rendering is always produced. If a provider is configured
        and enabled, its text is appended under an explicit non-authoritative
        heading.
        """
        facts = summarize_evidence(request.evidence)
        event_lines = self._render_events(request.events)
        sources = tuple(sorted({
            str(e.get("source", "unknown")) for e in request.evidence
        }))

        body: list[str] = [
            f"Device: {request.device_id}",
            f"Question: {request.question.strip()}",
            "",
            "Observed evidence (from Sofia pipeline):",
        ]
        body.extend(facts or ["- none recorded"])
        if event_lines:
            body.append("")
            body.append("Health events:")
            body.extend(event_lines)
        body.append("")
        body.append(
            "This summary restates Sofia evidence only. It is not a fault diagnosis "
            "and does not authorize any action."
        )
        text = "\n".join(body)

        provider_name = type(self.provider).__name__ if self.provider else "offline"
        metadata: dict[str, Any] = {
            "evidence_count": len(request.evidence),
            "event_count": len(request.events),
            "audience": request.audience,
        }

        if self.provider is not None and self.allow_provider_text:
            prompt = "\n".join([request.question, "", *facts])
            try:
                generated = self.provider.complete(prompt)
            except Exception as exc:
                metadata["provider_error"] = type(exc).__name__
                generated = ""
            if generated:
                text += "\n\n--- Non-authoritative model commentary ---\n" + generated
                metadata["provider_used"] = True

        return CopilotResponse(
            text=text,
            provider=provider_name,
            grounded=True,
            sources=sources,
            metadata=metadata,
        )

    @staticmethod
    def _render_events(events: Sequence[Mapping[str, Any]]) -> list[str]:
        lines: list[str] = []
        for event in events:
            severity = str(event.get("severity", "UNKNOWN"))
            etype = str(event.get("event_type", "unknown"))
            confidence = event.get("confidence", 0.0)
            lines.append(
                f"- {etype} severity={severity} confidence={_fmt(confidence)} "
                f"uncertainty={_fmt(1.0 - float(confidence or 0.0))}"
            )
        return lines
