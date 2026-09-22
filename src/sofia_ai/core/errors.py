"""Typed error hierarchy for Sofia Engine.

Every error raised by Sofia derives from :class:`SofiaError`. Callers may catch the
base class to handle "anything Sofia-related" without hiding unrelated bugs.

Rules enforced across the codebase:
    * no bare ``except Exception: pass``
    * errors carry a stable ``code`` suitable for logs and metrics
    * errors never contain secret material (see ``observability.redaction``)
"""

from __future__ import annotations

from typing import Any, ClassVar

__all__ = [
    "BufferOverflowError",
    "CommandReplayError",
    "ConfigurationError",
    "DiagnosticError",
    "DimensionError",
    "FeatureError",
    "FeatureNotSupportedError",
    "InferenceError",
    "InsufficientDataError",
    "ModelCompatibilityError",
    "ModelError",
    "ModelIntegrityError",
    "ModelSchemaError",
    "PayloadTooLargeError",
    "PluginError",
    "PolicyDeniedError",
    "SecretMissingError",
    "SignalError",
    "SignatureError",
    "SofiaError",
    "StorageError",
    "TelemetryError",
    "TelemetryProtocolError",
    "TimeoutError",
    "TimestampError",
    "TransportClosedError",
    "UnitError",
    "UnsafePathError",
    "ValidationError",
]


class SofiaError(Exception):
    """Base class for all Sofia errors.

    Args:
        message: Human-readable description, safe to log.
        code: Stable machine-readable error code.
        details: Optional structured context. Must not contain secrets.
    """

    default_code: ClassVar[str] = "SOFIA_ERR"

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code or self.default_code
        self.details: dict[str, Any] = dict(details or {})

    def to_dict(self) -> dict[str, Any]:
        """Serialize for structured logging."""
        return {"error": type(self).__name__, "code": self.code, "message": self.message,
                "details": self.details}

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"[{self.code}] {self.message}"


# --- core -------------------------------------------------------------------


class ValidationError(SofiaError):
    """Input failed validation (type, range, finiteness, identifier syntax)."""

    default_code = "VALIDATION"


class UnitError(ValidationError):
    """Unknown unit, or conversion between incompatible dimensions."""

    default_code = "UNIT"


class DimensionError(UnitError):
    """Attempted conversion between two different physical dimensions."""

    default_code = "UNIT_DIMENSION"


class TimestampError(ValidationError):
    """Timestamp is non-finite, out of range, or inconsistent with the clock model."""

    default_code = "TIMESTAMP"


class ConfigurationError(SofiaError):
    """Configuration failed schema or range validation."""

    default_code = "CONFIG"


class FeatureNotSupportedError(SofiaError):
    """Operation or method is not supported by the provider, model, or hardware profile."""

    default_code = "FEATURE_NOT_SUPPORTED"


# --- telemetry --------------------------------------------------------------


class TelemetryError(SofiaError):
    """Base class for telemetry transport failures."""

    default_code = "TELEMETRY"


class TelemetryProtocolError(TelemetryError):
    """A frame or record violated the protocol grammar."""

    default_code = "TELEMETRY_PROTOCOL"


class PayloadTooLargeError(TelemetryError):
    """A payload exceeded the configured byte ceiling."""

    default_code = "TELEMETRY_PAYLOAD"


class TransportClosedError(TelemetryError):
    """An operation was attempted on a closed or failed transport."""

    default_code = "TELEMETRY_CLOSED"


class TimeoutError(TelemetryError):
    """An operation exceeded its deadline."""

    default_code = "TELEMETRY_TIMEOUT"


# --- signal / features ------------------------------------------------------


class SignalError(SofiaError):
    """Signal processing received invalid parameters."""

    default_code = "SIGNAL"


class InsufficientDataError(SignalError):
    """Not enough samples to compute the requested result."""

    default_code = "SIGNAL_INSUFFICIENT"


class FeatureError(SofiaError):
    """Feature extraction failed."""

    default_code = "FEATURE"


# --- inference --------------------------------------------------------------


class ModelError(SofiaError):
    """Base class for model lifecycle failures."""

    default_code = "MODEL"


class ModelSchemaError(ModelError):
    """Model input/output schema does not match the manifest or the request."""

    default_code = "MODEL_SCHEMA"


class ModelIntegrityError(ModelError):
    """Checksum or signature verification failed."""

    default_code = "MODEL_INTEGRITY"


class ModelCompatibilityError(ModelError):
    """Model requires a different engine, preprocessing or schema version."""

    default_code = "MODEL_COMPAT"


class InferenceError(ModelError):
    """A backend failed during inference."""

    default_code = "MODEL_INFERENCE"


# --- diagnostics / decision -------------------------------------------------


class DiagnosticError(SofiaError):
    """A diagnostic rule or engine invocation failed."""

    default_code = "DIAGNOSTIC"


class PolicyDeniedError(SofiaError):
    """Raised when a command is requested through a channel that requires approval.

    In normal operation the policy engine returns a ``DENY`` decision rather than
    raising. This error is reserved for programmatic misuse, such as attempting to
    bypass the policy engine.
    """

    default_code = "POLICY_DENIED"


class CommandReplayError(PolicyDeniedError):
    """A command identifier was already consumed inside the replay window."""

    default_code = "POLICY_REPLAY"


# --- edge / storage ---------------------------------------------------------


class BufferOverflowError(SofiaError):
    """A bounded structure was pushed beyond its policy limit."""

    default_code = "BUFFER_OVERFLOW"


class StorageError(SofiaError):
    """Local persistence failed (write, rotation, or read-back)."""

    default_code = "STORAGE"


# --- security ---------------------------------------------------------------


class UnsafePathError(SofiaError):
    """A resolved path escaped its permitted root (path traversal)."""

    default_code = "SECURITY_PATH"


class SignatureError(SofiaError):
    """Signature material was absent, malformed, or did not verify."""

    default_code = "SECURITY_SIGNATURE"


class SecretMissingError(SofiaError):
    """A required secret was not present in the environment."""

    default_code = "SECURITY_SECRET"


class PluginError(ModelError):
    """A plugin failed to load, or its name was not allow-listed."""

    default_code = "PLUGIN"
