"""Automated fine-tuning subsystem for Sofia AI.

Curates edge telemetry into training datasets and triggers automated fine-tuning
jobs via NVIDIA NIM or OpenAI-compatible APIs without third-party dependencies.
"""

from __future__ import annotations

from .engine import (
    AutoFineTuner,
    DatasetCurator,
    FineTuneJob,
    FineTuneStatus,
    ProviderCapabilities,
)

__all__ = [
    "AutoFineTuner",
    "DatasetCurator",
    "FineTuneJob",
    "FineTuneStatus",
    "ProviderCapabilities",
]
