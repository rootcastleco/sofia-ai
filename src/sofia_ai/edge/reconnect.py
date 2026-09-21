"""Reconnect handling with a hard attempt ceiling and a backoff ceiling.

A transport that never gives up and never backs off is a denial-of-service
against itself. This module bounds both.

The backoff sequence is deterministic when ``jitter`` is zero, which keeps tests
reproducible; jitter is opt-in for production fleets that need to avoid
thundering herds.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Final

__all__ = [
    "BackoffCalculator",
    "ConnectionState",
    "ReconnectPolicy",
    "Reconnector",
]

DEFAULT_MAX_ATTEMPTS: Final[int] = 8


class ConnectionState(StrEnum):
    """Connectivity state machine. Explicit, with a terminal error state."""

    CONNECTED = "CONNECTED"
    DISCONNECTED = "DISCONNECTED"
    RECONNECTING = "RECONNECTING"
    FAILED = "FAILED"


@dataclass(frozen=True, slots=True)
class ReconnectPolicy:
    """Bounded reconnect parameters."""

    max_attempts: int = DEFAULT_MAX_ATTEMPTS
    initial_backoff_s: float = 0.5
    max_backoff_s: float = 30.0
    multiplier: float = 2.0
    jitter: float = 0.0

    def __post_init__(self) -> None:
        if self.max_attempts <= 0:
            raise ValueError("max_attempts must be positive")
        if self.initial_backoff_s <= 0 or self.max_backoff_s <= 0:
            raise ValueError("backoff bounds must be positive")
        if self.initial_backoff_s > self.max_backoff_s:
            raise ValueError("initial_backoff_s must not exceed max_backoff_s")
        if self.multiplier < 1.0:
            raise ValueError("multiplier must be >= 1")
        if not 0.0 <= self.jitter <= 1.0:
            raise ValueError("jitter must be within [0, 1]")


class BackoffCalculator:
    """Exponential backoff with a ceiling and optional deterministic jitter."""

    def __init__(self, policy: ReconnectPolicy, *, rng: Any = None) -> None:
        self.policy = policy
        self._rng = rng

    def delay_for(self, attempt: int) -> float:
        """Delay in seconds before attempt number ``attempt`` (1-based)."""
        if attempt <= 0:
            return 0.0
        raw = self.policy.initial_backoff_s * (self.policy.multiplier ** (attempt - 1))
        capped = min(raw, self.policy.max_backoff_s)
        if self.policy.jitter > 0 and self._rng is not None:
            spread = capped * self.policy.jitter
            capped = capped - spread + 2.0 * spread * float(self._rng.random())
        return float(max(0.0, capped))

    def schedule(self) -> list[float]:
        """The full delay schedule. Bounded by ``max_attempts``."""
        return [self.delay_for(i) for i in range(1, self.policy.max_attempts + 1)]


@dataclass(slots=True)
class Reconnector:
    """Drives reconnect attempts against an injected connect callable.

    Args:
        policy: Bounded policy.
        connect: Zero-argument callable returning True on success.
        sleep: Injectable sleep, so tests advance time without waiting.
        rng: Optional RNG for jitter.
    """

    policy: ReconnectPolicy = field(default_factory=ReconnectPolicy)
    connect: Callable[[], bool] = field(default=lambda: True)
    sleep: Callable[[float], None] = field(default=lambda _seconds: None)
    rng: Any = None

    state: ConnectionState = field(default=ConnectionState.DISCONNECTED, init=False)
    attempts: int = field(default=0, init=False)
    total_delay_s: float = field(default=0.0, init=False)
    successes: int = field(default=0, init=False)
    failures: int = field(default=0, init=False)
    _backoff: BackoffCalculator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._backoff = BackoffCalculator(self.policy, rng=self.rng)

    def run(self) -> bool:
        """Attempt reconnection until success or the attempt ceiling.

        Returns:
            True if connected. On exhaustion the state is ``FAILED`` — an explicit
            error state, never a silent retry loop.
        """
        self.state = ConnectionState.RECONNECTING
        while self.attempts < self.policy.max_attempts:
            self.attempts += 1
            delay = self._backoff.delay_for(self.attempts)
            if delay > 0:
                self.sleep(delay)
                self.total_delay_s += delay
            try:
                connected = bool(self.connect())
            except Exception:
                connected = False
            if connected:
                self.successes += 1
                self.state = ConnectionState.CONNECTED
                return True
            self.failures += 1
        self.state = ConnectionState.FAILED
        return False

    def mark_disconnected(self) -> None:
        """Return to DISCONNECTED and reset the attempt counter."""
        self.state = ConnectionState.DISCONNECTED
        self.attempts = 0

    def describe(self) -> dict[str, Any]:
        return {
            "state": self.state.value,
            "attempts": self.attempts,
            "max_attempts": self.policy.max_attempts,
            "successes": self.successes,
            "failures": self.failures,
            "total_delay_s": self.total_delay_s,
            "schedule": self._backoff.schedule(),
        }
