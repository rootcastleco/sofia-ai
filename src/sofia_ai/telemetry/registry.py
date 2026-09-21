"""Adapter registry.

New protocols and new model backends integrate by registration, not by editing
the core. Registration is explicit: there is no filesystem scanning and no
``importlib`` call driven by untrusted data (threat T-17).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Final, TypeVar

from ..core.errors import PluginError
from ..core.validation import validate_identifier

__all__ = ["Registry", "register_telemetry_source", "telemetry_registry"]

T = TypeVar("T")


class Registry:
    """A name → factory registry with allow-list semantics.

    Args:
        name: Registry name for error messages.
        allowed_prefix: If set, every registered name must start with it. This is
            the control that prevents a plugin name from resolving to arbitrary
            modules.
    """

    def __init__(self, name: str, *, allowed_prefix: str | None = None) -> None:
        self.name = name
        self.allowed_prefix = allowed_prefix
        self._factories: dict[str, Callable[..., object]] = {}

    def register(self, key: str, factory: Callable[..., T], *, overwrite: bool = False) -> None:
        """Register a factory under ``key``."""
        name = self._validate_key(key)
        if name in self._factories and not overwrite:
            raise PluginError(
                f"{self.name}: {name!r} is already registered",
                details={"registry": self.name, "key": name},
            )
        if not callable(factory):
            raise PluginError(
                f"{self.name}: factory for {name!r} is not callable",
                details={"registry": self.name, "key": name},
            )
        self._factories[name] = factory

    def unregister(self, key: str) -> None:
        self._factories.pop(self._validate_key(key), None)

    def get(self, key: str) -> Callable[..., object]:
        name = self._validate_key(key)
        factory = self._factories.get(name)
        if factory is None:
            raise PluginError(
                f"{self.name}: {name!r} is not registered. Known: {sorted(self._factories)}",
                details={"registry": self.name, "key": name,
                         "known": sorted(self._factories)},
            )
        return factory

    def create(self, key: str, **kwargs: object) -> object:
        factory = self.get(key)
        try:
            return factory(**kwargs)
        except PluginError:
            raise
        except Exception as exc:
            raise PluginError(
                f"{self.name}: factory {key!r} failed: {exc}",
                details={"registry": self.name, "key": key},
            ) from exc

    def _validate_key(self, key: str) -> str:
        name = validate_identifier(key, f"{self.name}.key").lower()
        if self.allowed_prefix and not name.startswith(self.allowed_prefix):
            raise PluginError(
                f"{self.name}: key {name!r} must start with {self.allowed_prefix!r}",
                details={"registry": self.name, "key": name},
            )
        return name

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._factories))

    def as_mapping(self) -> Mapping[str, Callable[..., object]]:
        return dict(self._factories)

    def __contains__(self, key: str) -> bool:
        return key.lower() in self._factories

    def __len__(self) -> int:
        return len(self._factories)


#: Global telemetry source registry. Populated by :func:`register_builtin_sources`.
telemetry_registry: Final[Registry] = Registry("telemetry")


def register_telemetry_source(key: str, factory: Callable[..., object],
                              *, overwrite: bool = False) -> None:
    """Register a telemetry source factory under ``key``."""
    telemetry_registry.register(key, factory, overwrite=overwrite)


def register_builtin_sources() -> None:
    """Register the sources that ship with Sofia.

    File and synthetic sources are always available. Protocol adapters are
    registered lazily so their optional dependencies are not imported here.
    """
    from .file_sources import CsvSource, InMemorySource, JsonlSource
    from .replay import ReplaySource
    from .synthetic import SyntheticVibrationSource

    register_telemetry_source("csv", CsvSource)
    register_telemetry_source("jsonl", JsonlSource)
    register_telemetry_source("memory", InMemorySource)
    register_telemetry_source("replay", ReplaySource)
    register_telemetry_source("synthetic", SyntheticVibrationSource)
    register_telemetry_source("mqtt", _lazy_mqtt)
    register_telemetry_source("modbus", _lazy_modbus)
    register_telemetry_source("serial", _lazy_serial)


def _lazy_mqtt(**kwargs: object) -> object:
    from .mqtt import MqttSource

    return MqttSource(**kwargs)  # type: ignore[arg-type]


def _lazy_modbus(**kwargs: object) -> object:
    from .modbus import ModbusSource

    return ModbusSource(**kwargs)  # type: ignore[arg-type]


def _lazy_serial(**kwargs: object) -> object:
    from .serial_adapter import SerialSource

    return SerialSource(**kwargs)  # type: ignore[arg-type]


register_builtin_sources()
