"""Optional engineering copilot. Outside the deterministic pipeline.

Consumes structured Sofia evidence; cannot emit commands. The default provider is
offline and deterministic, so the core project needs no LLM.
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

__all__ = [
    "PROVIDER_CONTRACT_VERSION",
    "CopilotProvider",
    "CopilotRequest",
    "CopilotResponse",
    "EngineeringCopilot",
    "EvidenceCopilot",
    "NullProvider",
    "OfflineProvider",
    "summarize_evidence",
]
