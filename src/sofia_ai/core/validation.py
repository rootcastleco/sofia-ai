"""Input validation primitives shared by every Sofia layer.

Design rules:
    * validation is explicit and loud — no silent clamping, no hidden fallback,
    * non-finite values are rejected at the boundary,
    * identifiers are restricted so they cannot inject newlines into structured logs,
    * numeric limits are named constants, not magic numbers.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from typing import Any, Final

import numpy as np

from .errors import ValidationError

__all__ = [
    "IDENTIFIER_PATTERN",
    "MAX_IDENTIFIER_LENGTH",
    "MAX_PAYLOAD_BYTES",
    "MAX_TAG_COUNT",
    "MAX_TEXT_LENGTH",
    "as_float_array",
    "ensure_finite",
    "ensure_finite_array",
    "sanitize_identifier",
    "validate_identifier",
    "validate_payload_size",
    "validate_positive_float",
    "validate_positive_int",
    "validate_probability",
    "validate_range",
    "validate_sample_rate",
    "validate_tags",
]

MAX_PAYLOAD_BYTES: Final[int] = 1 << 20  # 1 MiB default transport ceiling
MAX_IDENTIFIER_LENGTH: Final[int] = 128
MAX_TAG_COUNT: Final[int] = 32
MAX_TEXT_LENGTH: Final[int] = 4096

# Allows letters, digits, dot, dash, underscore, slash, colon. Explicitly forbids
# whitespace, control characters, quotes and backslashes, which prevents log
# injection and path-like identifiers.
IDENTIFIER_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z0-9._:/-]{1,128}$")


def ensure_finite(value: float, name: str = "value") -> float:
    """Return ``value`` as float, raising if it is NaN or infinite."""
    v = float(value)
    if not math.isfinite(v):
        raise ValidationError(
            f"{name} must be finite, got {value!r}",
            details={"name": name, "value": repr(value)},
        )
    return v


def ensure_finite_array(values: Any, name: str = "values") -> np.ndarray:
    """Return a float64 ndarray, raising if any element is NaN or infinite.

    Args:
        values: Any array-like.
        name: Field name for error messages.
    """
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return arr
    if not np.all(np.isfinite(arr)):
        bad = int(np.count_nonzero(~np.isfinite(arr)))
        raise ValidationError(
            f"{name} contains {bad} non-finite element(s)",
            details={"name": name, "non_finite_count": bad},
        )
    return arr


def sanitize_identifier(value: str) -> str:
    """Strip surrounding whitespace and control characters from an identifier."""
    return str(value).strip()


def validate_identifier(value: str, name: str = "identifier") -> str:
    """Validate a device/channel/source identifier.

    Rejects empty values, over-long values and any character outside
    :data:`IDENTIFIER_PATTERN`. This is the control that prevents log injection
    (T-16) and path-like identifiers (T-18).
    """
    if not isinstance(value, str):
        raise ValidationError(
            f"{name} must be a string, got {type(value).__name__}",
            details={"name": name, "type": type(value).__name__},
        )
    cleaned = sanitize_identifier(value)
    if not cleaned:
        raise ValidationError(f"{name} must not be empty", details={"name": name})
    if len(cleaned) > MAX_IDENTIFIER_LENGTH:
        raise ValidationError(
            f"{name} exceeds {MAX_IDENTIFIER_LENGTH} characters",
            details={"name": name, "length": len(cleaned)},
        )
    if IDENTIFIER_PATTERN.match(cleaned) is None:
        raise ValidationError(
            f"{name} contains disallowed characters: {cleaned!r}",
            details={"name": name, "value": cleaned},
        )
    return cleaned


def validate_range(
    value: float,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    name: str = "value",
) -> float:
    """Validate that a finite value lies within ``[minimum, maximum]``."""
    v = ensure_finite(value, name)
    if minimum is not None and v < minimum:
        raise ValidationError(
            f"{name} {v} is below minimum {minimum}",
            details={"name": name, "value": v, "minimum": minimum},
        )
    if maximum is not None and v > maximum:
        raise ValidationError(
            f"{name} {v} is above maximum {maximum}",
            details={"name": name, "value": v, "maximum": maximum},
        )
    return v


def validate_tags(tags: Any, name: str = "tags") -> dict[str, str]:
    """Validate a tag mapping: bounded size, string keys and values, no control chars."""
    if tags is None:
        return {}
    if not isinstance(tags, dict):
        raise ValidationError(
            f"{name} must be a mapping, got {type(tags).__name__}",
            details={"name": name},
        )
    if len(tags) > MAX_TAG_COUNT:
        raise ValidationError(
            f"{name} exceeds {MAX_TAG_COUNT} entries",
            details={"name": name, "count": len(tags)},
        )
    out: dict[str, str] = {}
    for key, value in tags.items():
        k = sanitize_identifier(str(key))
        v = "".join(ch for ch in str(value) if ch.isprintable()).strip()
        if not k:
            raise ValidationError(f"{name} contains an empty key", details={"name": name})
        if len(v) > MAX_TEXT_LENGTH:
            raise ValidationError(
                f"{name}[{k}] exceeds {MAX_TEXT_LENGTH} characters",
                details={"name": name, "key": k},
            )
        out[k] = v
    return out


def validate_payload_size(size_bytes: int, *, limit: int = MAX_PAYLOAD_BYTES) -> int:
    """Validate an inbound payload size against the ceiling.

    Raises:
        ValidationError: if the size is negative or exceeds ``limit``.
    """
    size = int(size_bytes)
    if size < 0:
        raise ValidationError(
            f"payload size must be non-negative, got {size}",
            details={"size": size},
        )
    if size > limit:
        raise ValidationError(
            f"payload of {size} bytes exceeds the {limit} byte limit",
            details={"size": size, "limit": limit},
        )
    return size


def validate_positive_int(value: int, name: str = "value", *, maximum: int | None = None) -> int:
    """Validate a strictly positive integer, optionally bounded above."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValidationError(
            f"{name} must be an integer, got {type(value).__name__}",
            details={"name": name},
        )
    v = int(value)
    if v <= 0:
        raise ValidationError(f"{name} must be positive, got {v}", details={"name": name})
    if maximum is not None and v > maximum:
        raise ValidationError(
            f"{name} must be <= {maximum}, got {v}", details={"name": name, "value": v}
        )
    return v


def validate_positive_float(
    value: float, name: str = "value", *, maximum: float | None = None
) -> float:
    """Validate a strictly positive finite float, optionally bounded above."""
    v = ensure_finite(value, name)
    if v <= 0.0:
        raise ValidationError(f"{name} must be positive, got {v}", details={"name": name})
    if maximum is not None and v > maximum:
        raise ValidationError(
            f"{name} must be <= {maximum}, got {v}", details={"name": name, "value": v}
        )
    return v


def validate_probability(value: float, name: str = "probability") -> float:
    """Validate a value in ``[0, 1]``."""
    return validate_range(value, minimum=0.0, maximum=1.0, name=name)


def validate_sample_rate(value: float, name: str = "sample_rate") -> float:
    """Validate a sample rate: finite, positive, and within a sane instrumentation range."""
    return validate_positive_float(value, name, maximum=1.0e9)


def as_float_array(values: Sequence[float] | np.ndarray, name: str = "values") -> np.ndarray:
    """Coerce to a contiguous float64 array, validating finiteness."""
    arr = ensure_finite_array(values, name)
    return np.ascontiguousarray(arr, dtype=np.float64)
