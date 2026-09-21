"""Edge runtime: bounded buffers, offline persistence and local pipelines.

Everything here runs without network access. Connectivity loss degrades the
pipeline; it never stops it.
"""

from __future__ import annotations

from .buffer import BoundedRingBuffer, ChannelBuffer, OverflowPolicy
from .health import DEGRADED_THRESHOLDS, HealthMonitor, HealthReport, HealthStatus
from .reconnect import BackoffCalculator, ConnectionState, Reconnector, ReconnectPolicy
from .runtime import EdgeRuntime, RuntimeConfig, RuntimeStats
from .store_forward import StoreForward, StoreStats

__all__ = [
    "DEGRADED_THRESHOLDS",
    "BackoffCalculator",
    "BoundedRingBuffer",
    "ChannelBuffer",
    "ConnectionState",
    "EdgeRuntime",
    "HealthMonitor",
    "HealthReport",
    "HealthStatus",
    "OverflowPolicy",
    "ReconnectPolicy",
    "Reconnector",
    "RuntimeConfig",
    "RuntimeStats",
    "StoreForward",
    "StoreStats",
]
