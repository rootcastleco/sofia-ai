"""Secret resolution. Secrets live in the environment, never in source or config files.

Rules:
    * configuration may name an environment variable, never a value,
    * :func:`require_secret` raises when the variable is missing,
    * :func:`secret` returns ``None`` when absent (for optional credentials),
    * resolved values are never logged; :mod:`sofia_ai.observability.redaction`
      provides defence in depth.
"""

from __future__ import annotations

import os
from typing import Final

from ..core.errors import SecretMissingError

__all__ = [
    "SECRET_KEY_HINTS",
    "env_name",
    "has_secret",
    "is_secret_key",
    "require_secret",
    "secret",
]

SOFIA_PREFIX: Final[str] = "SOFIA_"

#: Substrings that mark a mapping key as carrying secret material.
SECRET_KEY_HINTS: Final[tuple[str, ...]] = (
    "password",
    "passwd",
    "secret",
    "token",
    "api_key",
    "apikey",
    "api-key",
    "access_key",
    "private_key",
    "credential",
    "authorization",
    "auth",
    "bearer",
    "session",
    "cookie",
    "cert_passphrase",
)


def env_name(name: str) -> str:
    """Return the environment variable name for a logical secret name."""
    stripped = name.strip().upper()
    if not stripped:
        raise SecretMissingError("secret name must not be empty", details={})
    return stripped if stripped.startswith(SOFIA_PREFIX) else f"{SOFIA_PREFIX}{stripped}"


def secret(name: str | None) -> str | None:
    """Return a secret from the environment, or ``None`` if absent or blank."""
    if not name:
        return None
    value = os.environ.get(env_name(name))
    if value is None:
        value = os.environ.get(name.strip())
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def require_secret(name: str | None) -> str:
    """Return a secret, raising :class:`SecretMissingError` if it is absent."""
    if not name:
        raise SecretMissingError("a secret environment variable name is required", details={})
    resolved = secret(name)
    if resolved is None:
        raise SecretMissingError(
            f"required secret {env_name(name)!r} is not set in the environment",
            details={"env": env_name(name)},
        )
    return resolved


def has_secret(name: str | None) -> bool:
    """Whether a secret is present."""
    return secret(name) is not None


def is_secret_key(key: str) -> bool:
    """Whether a mapping key looks like it carries secret material."""
    lowered = str(key).strip().lower()
    return any(hint in lowered for hint in SECRET_KEY_HINTS)
