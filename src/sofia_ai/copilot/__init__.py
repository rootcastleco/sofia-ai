"""Optional engineering copilot. Outside the deterministic pipeline.

Consumes structured Sofia evidence; cannot emit commands. The default provider is
offline and deterministic, so the core project needs no LLM. Supports NVIDIA NIM
and OpenRouter for LLM-powered diagnostics.
"""

from __future__ import annotations

from .base import (
    PROVIDER_CONTRACT_VERSION,
    CopilotProvider,
    CopilotRequest,
    CopilotResponse,
    EngineeringCopilot,
    EvidenceCopilot,
    NullProvider,
    OfflineProvider,
    summarize_evidence,
)
from .providers import (
    AIEngine,
    NvidiaProvider,
    OpenRouterProvider,
)

__all__ = [
    "PROVIDER_CONTRACT_VERSION",
    "AIEngine",
    "CopilotProvider",
    "CopilotRequest",
    "CopilotResponse",
    "EngineeringCopilot",
    "EvidenceCopilot",
    "NvidiaProvider",
    "NullProvider",
    "OfflineProvider",
    "OpenRouterProvider",
    "summarize_evidence",
]
