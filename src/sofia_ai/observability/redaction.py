"""Secret redaction for logs, events and exported state.

Defence in depth: :mod:`sofia_ai.security.secrets` keeps secrets out of the
process, this module keeps them out of the output if they ever arrive anyway.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final

from ..core.validation import MAX_TEXT_LENGTH

__all__ = [
    "REDACTED",
    "SENSITIVE_KEYS",
    "contains_secret_marker",
    "redact_mapping",
    "redact_text",
    "redact_value",
]

REDACTED: Final[str] = "***REDACTED***"

#: Keys whose values are always redacted, regardless of nesting depth.
SENSITIVE_KEYS: Final[frozenset[str]] = frozenset({
    "password", "passwd", "pwd", "secret", "token", "api_key", "apikey", "api-key",
    "access_key", "secret_key", "private_key", "credential", "credentials",
    "authorization", "auth", "bearer", "session_id", "cookie", "csrf",
    "connection_string", "dsn", "sas", "signature",
})


def redact_value(value: Any) -> Any:
    """Redact a scalar or recursively redact a container."""
    if isinstance(value, Mapping):
        return redact_mapping(value)
    if isinstance(value, (list, tuple)):
        return [redact_value(v) for v in value]
    if isinstance(value, str):
        return redact_text(value)
    return value


def redact_mapping(data: Mapping[str, Any]) -> dict[str, Any]:
    """Return a copy with sensitive keys and secret-like strings redacted."""
    out: dict[str, Any] = {}
    for key, value in data.items():
        lowered = str(key).strip().lower()
        if lowered in SENSITIVE_KEYS or _looks_sensitive(lowered):
            out[key] = REDACTED
        else:
            out[key] = redact_value(value)
    return out


def _looks_sensitive(lowered_key: str) -> bool:
    return any(
        token in lowered_key
        for token in ("password", "secret", "token", "api_key", "apikey", "credential",
                      "authorization", "private_key", "passphrase")
    )


def redact_text(text: str, *, max_length: int = MAX_TEXT_LENGTH) -> str:
    """Redact key=value and bearer-token patterns inside free text.

    Also truncates at ``max_length`` so a hostile string cannot bloat a record.
    """
    import re

    scrubbed = text
    patterns = (
        r"(?i)\b(bearer|token|api[_-]?key|password|secret)\s*[=: ]\s*\S+",
        r"(?i)\b[A-Za-z0-9+/]{40,}={0,2}",
    )
    replacements = (
        r"\1=" + REDACTED,
        REDACTED,
    )
    for pattern, replacement in zip(patterns, replacements, strict=True):
        scrubbed = re.sub(pattern, replacement, scrubbed)
    if len(scrubbed) > max_length:
        scrubbed = scrubbed[:max_length] + "...[truncated]"
    return scrubbed


def contains_secret_marker(value: Any) -> bool:
    """Whether a value contains the redaction marker."""
    if isinstance(value, str):
        return REDACTED in value
    if isinstance(value, Mapping):
        return any(contains_secret_marker(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return any(contains_secret_marker(v) for v in value)
    return False
