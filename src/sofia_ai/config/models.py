"""Typed, validated, externalized, versioned configuration.

Design rules:
    * configuration is data, validated by a schema before use,
    * no credential is ever stored in a committed config file — secrets are resolved
      from the environment by :mod:`sofia_ai.security.secrets`,
    * every field has a documented bound; out-of-range values raise at load time,
    * the config object is immutable after validation so runtime mutation cannot
      create inconsistent state.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Final

from ..core.errors import ConfigurationError, ValidationError
from ..core.validation import (
    MAX_PAYLOAD_BYTES,
)
from ..core.validation import (
    validate_identifier as _vi,
)
from ..core.validation import (
    validate_positive_float as _vpf,
)
from ..core.validation import (
    validate_positive_int as _vpi,
)
from ..core.validation import (
    validate_probability as _vp,
)

__all__ = [
    "CONFIG_VERSION",
    "ENV_PREFIX",
    "BufferConfig",
    "ChannelConfig",
    "DeviceConfig",
    "LoggingConfig",
    "MetricsConfig",
    "PolicyConfig",
    "ReconnectConfig",
    "SofiaConfig",
    "StoreForwardConfig",
    "TransportConfig",
    "WindowConfig",
    "config_from_dict",
    "load_config",
]

CONFIG_VERSION: Final[str] = "2.0"
ENV_PREFIX: Final[str] = "SOFIA_"


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ConfigurationError(
            f"{name} must be a mapping, got {type(value).__name__}", details={"name": name}
        )
    return value


def _as_int(value: Any, name: str, *, maximum: int | None = None) -> int:
    """Validate a positive int, reporting failures as ConfigurationError."""
    try:
        return _vpi(value, name, maximum=maximum)
    except ValidationError as exc:
        raise ConfigurationError(exc.message, details=exc.details) from exc


def _as_float(value: Any, name: str, *, maximum: float | None = None) -> float:
    """Validate a positive float, reporting failures as ConfigurationError."""
    try:
        return _vpf(value, name, maximum=maximum)
    except ValidationError as exc:
        raise ConfigurationError(exc.message, details=exc.details) from exc


def _as_probability(value: Any, name: str) -> float:
    """Validate ``[0, 1]``, reporting failures as ConfigurationError."""
    try:
        return _vp(value, name)
    except ValidationError as exc:
        raise ConfigurationError(exc.message, details=exc.details) from exc


def _as_identifier(value: Any, name: str) -> str:
    """Validate an identifier, reporting failures as ConfigurationError."""
    try:
        return _vi(str(value), name)
    except ValidationError as exc:
        raise ConfigurationError(exc.message, details=exc.details) from exc


def _as_unit(value: Any, name: str = "unit") -> str:
    """Resolve a unit symbol, reporting failures as ConfigurationError."""
    from ..core.units import UnitError, unit_alias

    try:
        return unit_alias(str(value))
    except UnitError as exc:
        raise ConfigurationError(exc.message, details=exc.details) from exc


def _get(section: Mapping[str, Any], key: str, default: Any) -> Any:
    value = section.get(key, default)
    return default if value is None else value


@dataclass(frozen=True, slots=True)
class BufferConfig:
    """Bounded ingest buffer."""

    max_samples: int = 4096
    overflow_policy: str = "drop_oldest"

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "max_samples", _as_int(self.max_samples, "buffer.max_samples",
                                                       maximum=10_000_000)
        )
        if self.overflow_policy not in ("drop_oldest", "drop_newest", "reject"):
            raise ConfigurationError(
                f"buffer.overflow_policy must be drop_oldest|drop_newest|reject, "
                f"got {self.overflow_policy!r}",
                details={"value": self.overflow_policy},
            )

    @classmethod
    def from_dict(cls, data: Any) -> BufferConfig:
        d = _require_mapping(data, "buffer")
        return cls(max_samples=int(_get(d, "max_samples", 4096)),
                   overflow_policy=str(_get(d, "overflow_policy", "drop_oldest")))


@dataclass(frozen=True, slots=True)
class WindowConfig:
    """Window assembly parameters."""

    length: int = 1024
    hop: int = 512
    sample_rate: float = 1000.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "length", _as_int(self.length, "window.length", maximum=1 << 20)
        )
        object.__setattr__(
            self, "hop", _as_int(self.hop, "window.hop", maximum=1 << 20)
        )
        object.__setattr__(
            self, "sample_rate", _as_float(self.sample_rate, "window.sample_rate",
                                                         maximum=1e9)
        )
        if self.hop > self.length:
            raise ConfigurationError(
                "window.hop must not exceed window.length",
                details={"length": self.length, "hop": self.hop},
            )

    @classmethod
    def from_dict(cls, data: Any) -> WindowConfig:
        d = _require_mapping(data, "window")
        return cls(length=int(_get(d, "length", 1024)),
                   hop=int(_get(d, "hop", 512)),
                   sample_rate=float(_get(d, "sample_rate", 1000.0)))


@dataclass(frozen=True, slots=True)
class TransportConfig:
    """Transport-level limits applied to every telemetry adapter."""

    max_payload_bytes: int = MAX_PAYLOAD_BYTES
    read_timeout_s: float = 5.0
    connect_timeout_s: float = 10.0
    max_retries: int = 3
    rate_limit_per_s: float = 0.0  # 0 = unlimited

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "max_payload_bytes",
            _as_int(self.max_payload_bytes, "transport.max_payload_bytes",
                                  maximum=1 << 30),
        )
        object.__setattr__(
            self, "read_timeout_s",
            _as_float(self.read_timeout_s, "transport.read_timeout_s", maximum=3600.0),
        )
        object.__setattr__(
            self, "connect_timeout_s",
            _as_float(self.connect_timeout_s, "transport.connect_timeout_s",
                                    maximum=3600.0),
        )
        object.__setattr__(
            self, "max_retries",
            _as_int(self.max_retries, "transport.max_retries", maximum=64),
        )
        if self.rate_limit_per_s < 0:
            raise ConfigurationError(
                "transport.rate_limit_per_s must be >= 0",
                details={"value": self.rate_limit_per_s},
            )

    @classmethod
    def from_dict(cls, data: Any) -> TransportConfig:
        d = _require_mapping(data, "transport")
        return cls(max_payload_bytes=int(_get(d, "max_payload_bytes", MAX_PAYLOAD_BYTES)),
                   read_timeout_s=float(_get(d, "read_timeout_s", 5.0)),
                   connect_timeout_s=float(_get(d, "connect_timeout_s", 10.0)),
                   max_retries=int(_get(d, "max_retries", 3)),
                   rate_limit_per_s=float(_get(d, "rate_limit_per_s", 0.0)))


@dataclass(frozen=True, slots=True)
class LoggingConfig:
    """Structured logging controls."""

    level: str = "INFO"
    rate_limit_per_min: int = 30
    structured: bool = True
    redact: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "level", str(self.level).upper())
        if self.level not in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
            raise ConfigurationError(
                f"logging.level must be a standard level name, got {self.level!r}",
                details={"value": self.level},
            )
        object.__setattr__(
            self, "rate_limit_per_min",
            _as_int(self.rate_limit_per_min, "logging.rate_limit_per_min",
                                  maximum=1_000_000),
        )

    @classmethod
    def from_dict(cls, data: Any) -> LoggingConfig:
        d = _require_mapping(data, "logging")
        return cls(level=str(_get(d, "level", "INFO")).upper(),
                   rate_limit_per_min=int(_get(d, "rate_limit_per_min", 30)),
                   structured=bool(_get(d, "structured", True)),
                   redact=bool(_get(d, "redact", True)))


@dataclass(frozen=True, slots=True)
class MetricsConfig:
    """Metrics retention controls."""

    max_samples: int = 10_000
    enabled: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "max_samples",
            _as_int(self.max_samples, "metrics.max_samples", maximum=1_000_000),
        )

    @classmethod
    def from_dict(cls, data: Any) -> MetricsConfig:
        d = _require_mapping(data, "metrics")
        return cls(max_samples=int(_get(d, "max_samples", 10_000)),
                   enabled=bool(_get(d, "enabled", True)))


@dataclass(frozen=True, slots=True)
class StoreForwardConfig:
    """Bounded local persistence for offline operation."""

    enabled: bool = False
    directory: str = "sofia_store"
    max_bytes: int = 64 * 1024 * 1024
    max_files: int = 8
    flush_every: int = 128

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "max_bytes",
            _as_int(self.max_bytes, "store_forward.max_bytes", maximum=1 << 40),
        )
        object.__setattr__(
            self, "max_files",
            _as_int(self.max_files, "store_forward.max_files", maximum=1024),
        )
        object.__setattr__(
            self, "flush_every",
            _as_int(self.flush_every, "store_forward.flush_every", maximum=1 << 20),
        )
        if not self.directory:
            raise ConfigurationError("store_forward.directory must not be empty", details={})

    @classmethod
    def from_dict(cls, data: Any) -> StoreForwardConfig:
        d = _require_mapping(data, "store_forward")
        return cls(enabled=bool(_get(d, "enabled", False)),
                   directory=str(_get(d, "directory", "sofia_store")),
                   max_bytes=int(_get(d, "max_bytes", 64 * 1024 * 1024)),
                   max_files=int(_get(d, "max_files", 8)),
                   flush_every=int(_get(d, "flush_every", 128)))


@dataclass(frozen=True, slots=True)
class ReconnectConfig:
    """Bounded reconnect behaviour. There is always a ceiling."""

    max_attempts: int = 8
    initial_backoff_s: float = 0.5
    max_backoff_s: float = 30.0
    multiplier: float = 2.0
    jitter: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "max_attempts",
            _as_int(self.max_attempts, "reconnect.max_attempts", maximum=64),
        )
        object.__setattr__(
            self, "initial_backoff_s",
            _as_float(self.initial_backoff_s, "reconnect.initial_backoff_s",
                                    maximum=3600.0),
        )
        object.__setattr__(
            self, "max_backoff_s",
            _as_float(self.max_backoff_s, "reconnect.max_backoff_s", maximum=3600.0),
        )
        if self.multiplier < 1.0:
            raise ConfigurationError(
                "reconnect.multiplier must be >= 1", details={"value": self.multiplier}
            )
        if not (0.0 <= self.jitter <= 1.0):
            raise ConfigurationError(
                "reconnect.jitter must be within [0, 1]", details={"value": self.jitter}
            )
        if self.initial_backoff_s > self.max_backoff_s:
            raise ConfigurationError(
                "reconnect.initial_backoff_s must not exceed reconnect.max_backoff_s",
                details={"initial": self.initial_backoff_s, "max": self.max_backoff_s},
            )

    @classmethod
    def from_dict(cls, data: Any) -> ReconnectConfig:
        d = _require_mapping(data, "reconnect")
        return cls(max_attempts=int(_get(d, "max_attempts", 8)),
                   initial_backoff_s=float(_get(d, "initial_backoff_s", 0.5)),
                   max_backoff_s=float(_get(d, "max_backoff_s", 30.0)),
                   multiplier=float(_get(d, "multiplier", 2.0)),
                   jitter=float(_get(d, "jitter", 0.0)))


@dataclass(frozen=True, slots=True)
class ChannelConfig:
    """Per-channel engineering definition."""

    name: str
    unit: str
    minimum: float | None = None
    maximum: float | None = None
    sample_rate: float | None = None
    max_age_s: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _as_identifier(self.name, "channel.name"))


        object.__setattr__(self, "unit", _as_unit(self.unit))
        if self.minimum is not None and self.maximum is not None and self.minimum > self.maximum:
            raise ConfigurationError(
                f"channel {self.name}: minimum exceeds maximum",
                    details={"channel": self.name},
                )
        if self.sample_rate is not None:
            object.__setattr__(
                self, "sample_rate",
                _as_float(self.sample_rate, "channel.sample_rate", maximum=1e9),
            )
        if self.max_age_s is not None:
            object.__setattr__(
                self, "max_age_s",
                _as_float(self.max_age_s, "channel.max_age_s", maximum=86400.0 * 30),
            )

    @classmethod
    def from_dict(cls, data: Any) -> ChannelConfig:
        d = _require_mapping(data, "channel")
        if "name" not in d or "unit" not in d:
            raise ConfigurationError("channel requires 'name' and 'unit'", details={})
        return cls(name=str(d["name"]),
                   unit=str(d["unit"]),
                   minimum=None if d.get("minimum") is None else float(d["minimum"]),
                   maximum=None if d.get("maximum") is None else float(d["maximum"]),
                   sample_rate=None if d.get("sample_rate") is None else float(d["sample_rate"]),
                   max_age_s=None if d.get("max_age_s") is None else float(d["max_age_s"]))


@dataclass(frozen=True, slots=True)
class DeviceConfig:
    """A known device and its channel definitions."""

    device_id: str
    channels: tuple[ChannelConfig, ...] = ()
    tags: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "device_id", _as_identifier(self.device_id, "device_id"))
        object.__setattr__(self, "tags", dict(self.tags or {}))

    def channel(self, name: str) -> ChannelConfig | None:
        for ch in self.channels:
            if ch.name == name:
                return ch
        return None

    @classmethod
    def from_dict(cls, data: Any) -> DeviceConfig:
        d = _require_mapping(data, "device")
        if "device_id" not in d:
            raise ConfigurationError("device requires 'device_id'", details={})
        channels = tuple(ChannelConfig.from_dict(c) for c in d.get("channels", []) or [])
        return cls(device_id=str(d["device_id"]), channels=channels,
                   tags=dict(d.get("tags", {}) or {}))


@dataclass(frozen=True, slots=True)
class PolicyConfig:
    """Command safety policy. Default posture is deny."""

    default_decision: str = "deny"
    allowed_actions: tuple[str, ...] = ()
    require_operator_approval: bool = True
    min_confidence: float = 0.9
    max_commands_per_min: int = 5
    command_ttl_s: float = 30.0
    allowed_machine_states: tuple[str, ...] = ()
    interlock_required: bool = True

    def __post_init__(self) -> None:
        if self.default_decision not in ("deny", "allow"):
            raise ConfigurationError(
                "policy.default_decision must be deny|allow",
                details={"value": self.default_decision},
            )
        object.__setattr__(
            self, "min_confidence",
            _as_probability(self.min_confidence, "policy.min_confidence"),
        )
        object.__setattr__(
            self, "max_commands_per_min",
            _as_int(self.max_commands_per_min, "policy.max_commands_per_min",
                                  maximum=10_000),
        )
        object.__setattr__(
            self, "command_ttl_s",
            _as_float(self.command_ttl_s, "policy.command_ttl_s", maximum=3600.0),
        )
        for action in self.allowed_actions:
            _as_identifier(action, "policy.allowed_actions")

    @classmethod
    def from_dict(cls, data: Any) -> PolicyConfig:
        d = _require_mapping(data, "policy")
        return cls(default_decision=str(_get(d, "default_decision", "deny")).lower(),
                   allowed_actions=tuple(str(a) for a in _get(d, "allowed_actions", ()) or ()),
                   require_operator_approval=bool(_get(d, "require_operator_approval", True)),
                   min_confidence=float(_get(d, "min_confidence", 0.9)),
                   max_commands_per_min=int(_get(d, "max_commands_per_min", 5)),
                   command_ttl_s=float(_get(d, "command_ttl_s", 30.0)),
                   allowed_machine_states=tuple(
                       str(s) for s in _get(d, "allowed_machine_states", ()) or ()
                   ),
                   interlock_required=bool(_get(d, "interlock_required", True)))


@dataclass(frozen=True, slots=True)
class SofiaConfig:
    """Top-level validated configuration."""

    version: str = CONFIG_VERSION
    device_id: str = "default"
    buffer: BufferConfig = field(default_factory=BufferConfig)
    window: WindowConfig = field(default_factory=WindowConfig)
    transport: TransportConfig = field(default_factory=TransportConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    store_forward: StoreForwardConfig = field(default_factory=StoreForwardConfig)
    reconnect: ReconnectConfig = field(default_factory=ReconnectConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    devices: tuple[DeviceConfig, ...] = ()
    device_allowlist: tuple[str, ...] = ()
    offline: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "device_id", _as_identifier(self.device_id, "device_id"))
        for device in self.devices:
            if not isinstance(device, DeviceConfig):
                raise ConfigurationError("devices must contain DeviceConfig instances",
                                         details={})
        for name in self.device_allowlist:
            _as_identifier(name, "device_allowlist")

    def device(self, device_id: str) -> DeviceConfig | None:
        for dev in self.devices:
            if dev.device_id == device_id:
                return dev
        return None

    def channel(self, device_id: str, channel: str) -> ChannelConfig | None:
        dev = self.device(device_id)
        return None if dev is None else dev.channel(channel)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "device_id": self.device_id,
            "offline": self.offline,
            "buffer": {"max_samples": self.buffer.max_samples,
                       "overflow_policy": self.buffer.overflow_policy},
            "window": {"length": self.window.length, "hop": self.window.hop,
                       "sample_rate": self.window.sample_rate},
            "transport": {"max_payload_bytes": self.transport.max_payload_bytes,
                          "read_timeout_s": self.transport.read_timeout_s,
                          "connect_timeout_s": self.transport.connect_timeout_s,
                          "max_retries": self.transport.max_retries,
                          "rate_limit_per_s": self.transport.rate_limit_per_s},
            "logging": {"level": self.logging.level,
                        "rate_limit_per_min": self.logging.rate_limit_per_min,
                        "structured": self.logging.structured, "redact": self.logging.redact},
            "metrics": {"max_samples": self.metrics.max_samples,
                        "enabled": self.metrics.enabled},
            "store_forward": {"enabled": self.store_forward.enabled,
                              "directory": self.store_forward.directory,
                              "max_bytes": self.store_forward.max_bytes,
                              "max_files": self.store_forward.max_files,
                              "flush_every": self.store_forward.flush_every},
            "reconnect": {"max_attempts": self.reconnect.max_attempts,
                          "initial_backoff_s": self.reconnect.initial_backoff_s,
                          "max_backoff_s": self.reconnect.max_backoff_s,
                          "multiplier": self.reconnect.multiplier,
                          "jitter": self.reconnect.jitter},
            "policy": {"default_decision": self.policy.default_decision,
                       "allowed_actions": list(self.policy.allowed_actions),
                       "require_operator_approval": self.policy.require_operator_approval,
                       "min_confidence": self.policy.min_confidence,
                       "max_commands_per_min": self.policy.max_commands_per_min,
                       "command_ttl_s": self.policy.command_ttl_s,
                       "allowed_machine_states": list(self.policy.allowed_machine_states),
                       "interlock_required": self.policy.interlock_required},
            "devices": [
                {"device_id": d.device_id, "tags": dict(d.tags),
                 "channels": [
                     {"name": c.name, "unit": c.unit, "minimum": c.minimum,
                      "maximum": c.maximum, "sample_rate": c.sample_rate,
                      "max_age_s": c.max_age_s}
                     for c in d.channels
                 ]}
                for d in self.devices
            ],
            "device_allowlist": list(self.device_allowlist),
        }


def config_from_dict(data: Mapping[str, Any]) -> SofiaConfig:
    """Build a validated :class:`SofiaConfig` from a mapping."""
    d = _require_mapping(data, "config")
    version = str(_get(d, "version", CONFIG_VERSION))
    if version.split(".", maxsplit=1)[0] != CONFIG_VERSION.split(".")[0]:
        raise ConfigurationError(
            f"Configuration major version {version!r} is incompatible with {CONFIG_VERSION!r}",
            details={"version": version, "expected": CONFIG_VERSION},
        )
    return SofiaConfig(
        version=version,
        device_id=str(_get(d, "device_id", "default")),
        buffer=BufferConfig.from_dict(d.get("buffer")),
        window=WindowConfig.from_dict(d.get("window")),
        transport=TransportConfig.from_dict(d.get("transport")),
        logging=LoggingConfig.from_dict(d.get("logging")),
        metrics=MetricsConfig.from_dict(d.get("metrics")),
        store_forward=StoreForwardConfig.from_dict(d.get("store_forward")),
        reconnect=ReconnectConfig.from_dict(d.get("reconnect")),
        policy=PolicyConfig.from_dict(d.get("policy")),
        devices=tuple(DeviceConfig.from_dict(x) for x in d.get("devices", []) or []),
        device_allowlist=tuple(str(x) for x in d.get("device_allowlist", ()) or ()),
        offline=bool(_get(d, "offline", True)),
    )


def load_config(path: str | None = None, *, apply_env: bool = True) -> SofiaConfig:
    """Load configuration from a JSON file, then apply environment overrides.

    Environment overrides are restricted to a declared allow-list so a hostile
    environment cannot inject arbitrary config keys.

    Args:
        path: JSON config path. ``None`` yields defaults.
        apply_env: Whether to apply ``SOFIA_*`` environment overrides.
    """
    raw: dict[str, Any] = {}
    if path is not None:
        from ..core.serialization import load_json

        loaded = load_json(path)
        if not isinstance(loaded, dict):
            raise ConfigurationError("Configuration root must be a JSON object", details={})
        raw = dict(loaded)
    if apply_env:
        raw.update(_env_overrides())
    return config_from_dict(raw)


_ENV_OVERRIDES: Final[Mapping[str, str]] = {
    "SOFIA_DEVICE_ID": "device_id",
    "SOFIA_OFFLINE": "offline",
}


def _env_overrides() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for env_key, config_key in _ENV_OVERRIDES.items():
        value = os.environ.get(env_key)
        if value is None:
            continue
        if config_key == "offline":
            out[config_key] = value.strip().lower() in ("1", "true", "yes", "on")
        else:
            out[config_key] = value.strip()
    return out
