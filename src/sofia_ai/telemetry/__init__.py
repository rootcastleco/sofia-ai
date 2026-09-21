"""Telemetry layer: transport adapters behind a protocol-independent interface.

The core never imports a transport. Adapters for MQTT, Modbus and serial import
their client libraries lazily so the minimal install stays numpy-only.
"""

from __future__ import annotations

from .base import (
    SourceLimits,
    SourceState,
    SourceStats,
    TelemetrySink,
    TelemetrySource,
    check_payload_size,
)
from .file_sources import CsvSource, InMemorySource, JsonlSource
from .registry import Registry, register_telemetry_source, telemetry_registry
from .replay import ReplayMode, ReplaySource
from .synthetic import (
    MultiChannelSyntheticSource,
    SyntheticMachineProfile,
    SyntheticVibrationSource,
    inject_faults,
    rotating_machine_signal,
)

__all__ = [
    "CsvSource",
    "InMemorySource",
    "JsonlSource",
    "MultiChannelSyntheticSource",
    "Registry",
    "ReplayMode",
    "ReplaySource",
    "SourceLimits",
    "SourceState",
    "SourceStats",
    "SyntheticMachineProfile",
    "SyntheticVibrationSource",
    "TelemetrySink",
    "TelemetrySource",
    "check_payload_size",
    "inject_faults",
    "register_telemetry_source",
    "rotating_machine_signal",
    "telemetry_registry",
]
