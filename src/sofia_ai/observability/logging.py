"""Structured, rate-limited, secret-free logging.

Logs are emitted as single-line JSON objects so they can be ingested by any log
pipeline and cannot be forged by embedded newlines (threat T-16).

Rate limiting is per message key, so a failing sensor producing 10 000 identical
errors per second produces at most ``rate_limit_per_min`` log lines per minute and
an accurate ``suppressed`` count.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Final

from ..core.validation import validate_identifier

__all__ = [
    "REDACTED",
    "LogRateLimiter",
    "SofiaLogger",
    "configure_logging",
    "get_logger",
]

REDACTED: Final[str] = "***REDACTED***"

_RESERVED = {"asctime", "created", "exc_info", "exc_text", "filename", "funcName", "levelname",
             "levelno", "lineno", "module", "msecs", "message", "msg", "name", "pathname",
             "process", "processName", "relativeCreated", "stack_info", "thread", "threadName",
             "taskName"}


def _compact_details(details: Mapping[str, Any]) -> str:
    """Flatten a Sofia error ``details`` mapping into a readable hint."""
    parts = []
    for key, value in details.items():
        if isinstance(value, (str, int, float, bool)):
            parts.append(f"{key}={value}")
    return "details: " + ", ".join(parts) if parts else "details: <empty>"


@dataclass(slots=True)
class LogRateLimiter:
    """Token-per-minute limiter keyed by message key.

    Args:
        per_minute: Maximum emissions per key per minute.
        max_keys: Bounded key table so a hostile key space cannot grow memory.
    """

    per_minute: int = 30
    max_keys: int = 1024

    _counts: dict[str, tuple[float, int]] = field(default_factory=dict, init=False,
                                                  repr=False)

    def __post_init__(self) -> None:
        if self.per_minute <= 0:
            raise ValueError("per_minute must be positive")

    def allow(self, key: str, now: float) -> tuple[bool, int]:
        """Return ``(allowed, suppressed_count)`` for a key."""
        window_start = now - 60.0
        self._prune(window_start)
        state = self._counts.get(key)
        if state is None or state[0] < window_start:
            if len(self._counts) >= self.max_keys and key not in self._counts:
                # Table full: suppress rather than grow unbounded.
                return False, (state[1] if state else 0) + 1
            self._counts[key] = (now, 1)
            return True, 0
        count = state[1] + 1
        self._counts[key] = (state[0], count)
        return count <= self.per_minute, max(0, count - self.per_minute)

    def _prune(self, cutoff: float) -> None:
        expired = [k for k, (ts, _) in self._counts.items() if ts < cutoff]
        for key in expired:
            self._counts.pop(key, None)

    def __len__(self) -> int:
        return len(self._counts)


@dataclass(slots=True)
class SofiaLogger:
    """Structured logger wrapper.

    Args:
        name: Logger name.
        level: Minimum level.
        rate_limit_per_min: Per-key emission ceiling.
        redact: Redact secret-like mapping keys.
        correlation_id: Attached to every record.
        device_id: Attached to every record.
    """

    name: str = "sofia"
    level: str = "INFO"
    rate_limit_per_min: int = 30
    redact: bool = True
    correlation_id: str = ""
    device_id: str = ""
    _limiter: LogRateLimiter = field(init=False, repr=False)
    _logger: logging.Logger = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.name = validate_identifier(self.name.replace(".", "_"), "logger.name")
        self._logger = logging.getLogger(f"sofia.{self.name}")
        self._logger.setLevel(getattr(logging, self.level.upper(), logging.INFO))
        self._limiter = LogRateLimiter(per_minute=max(1, int(self.rate_limit_per_min)))

    # -- core --------------------------------------------------------------

    def log(self, level: str, event: str, **fields: Any) -> bool:
        """Emit one structured record. Returns True if emitted (not rate-limited)."""
        allowed, suppressed = self._limiter.allow(f"{level}:{event}", time.monotonic())
        if not allowed:
            return False
        payload = self._build(event, level, suppressed, fields)
        self._logger.log(
            getattr(logging, level.upper(), logging.INFO),
            json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str),
        )
        return True

    def _build(self, event: str, level: str, suppressed: int,
               fields: Mapping[str, Any]) -> dict[str, Any]:
        record: dict[str, Any] = {
            "ts": time.time(),
            "level": level.upper(),
            "logger": self.name,
            "event": event,
        }
        if self.correlation_id:
            record["correlation_id"] = self.correlation_id
        if self.device_id:
            record["device_id"] = self.device_id
        if suppressed:
            record["suppressed"] = suppressed
        for key, value in fields.items():
            record_key = f"f_{key}" if key in _RESERVED else key
            record[record_key] = self._sanitize(record_key, value)
        return record

    def _sanitize(self, key: str, value: Any) -> Any:
        from ..security.secrets import is_secret_key

        if self.redact and is_secret_key(key):
            return REDACTED
        if isinstance(value, Mapping):
            return {
                k: (REDACTED if self.redact and is_secret_key(str(k)) else self._sanitize(str(k), v))
                for k, v in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [self._sanitize(key, v) for v in value]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)

    # -- level helpers -----------------------------------------------------

    def debug(self, event: str, **fields: Any) -> bool:
        return self.log("DEBUG", event, **fields)

    def info(self, event: str, **fields: Any) -> bool:
        return self.log("INFO", event, **fields)

    def warning(self, event: str, **fields: Any) -> bool:
        return self.log("WARNING", event, **fields)

    def error(self, event: str, **fields: Any) -> bool:
        return self.log("ERROR", event, **fields)

    def exception(self, event: str, **fields: Any) -> bool:
        """Log at ERROR with the active exception's traceback attached."""
        import sys
        import traceback

        fields = dict(fields)
        exc = sys.exc_info()
        if exc[1] is not None:
            preview = repr(exc[1])
            if isinstance(exc[1], Exception):
                existing = getattr(exc[1], "details", None)
                if isinstance(existing, Mapping) and existing:
                    preview = preview + " " + _compact_details(existing)
            fields["exception"] = preview
            fields["traceback"] = "".join(
                traceback.format_exception(*exc)).rstrip()
        return self.log("ERROR", event, **fields)

    def critical(self, event: str, **fields: Any) -> bool:
        return self.log("CRITICAL", event, **fields)

    def bind(self, **fields: Any) -> SofiaLogger:
        """Return a copy with additional bound context."""
        clone = SofiaLogger(
            name=self.name,
            level=self.level,
            rate_limit_per_min=self.rate_limit_per_min,
            redact=self.redact,
            correlation_id=str(fields.get("correlation_id", self.correlation_id)),
            device_id=str(fields.get("device_id", self.device_id)),
        )
        return clone


def get_logger(name: str = "sofia", **kwargs: Any) -> SofiaLogger:
    """Get a Sofia structured logger."""
    return SofiaLogger(name=name, **kwargs)


def configure_logging(level: str = "INFO", *, structured: bool = True) -> None:
    """Configure the root ``sofia`` logger with a plain stderr handler.

    Args:
        level: Minimum level.
        structured: Emit raw JSON lines (True) or a human-readable prefix (False).
    """
    root = logging.getLogger("sofia")
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    for handler in list(root.handlers):
        root.removeHandler(handler)
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(message)s" if structured else "%(levelname)s %(message)s")
    )
    root.addHandler(handler)
    root.propagate = False
