"""Modbus TCP and RTU telemetry adapter (optional extra: ``sofia-engine[modbus]``).

Design notes:
    * register parsing is length-checked *before* indexing — a truncated frame is a
      :class:`TelemetryProtocolError`, never an ``IndexError`` (threat T-02);
    * reading is bounded: a declared register map, never an unbounded scan;
    * the client is injectable so unit tests need no PLC;
    * secrets are never read from configuration.

Integration against physical Modbus hardware is **NOT VERIFIED** in this repository.
"""

from __future__ import annotations

import struct
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Final

from ..core.contracts import TelemetrySample
from ..core.errors import ConfigurationError, TelemetryError, TelemetryProtocolError
from ..core.validation import validate_positive_int
from .base import SourceLimits, TelemetrySource

__all__ = [
    "ByteOrder",
    "ModbusConfig",
    "ModbusSource",
    "RegisterSpec",
    "RegisterType",
    "decode_registers",
]

_MAX_REGISTERS_PER_READ: Final[int] = 125  # Modbus protocol limit


class RegisterType(StrEnum):
    """Decoded register interpretation."""

    UINT16 = "UINT16"
    INT16 = "INT16"
    UINT32 = "UINT32"
    INT32 = "INT32"
    FLOAT32 = "FLOAT32"


class ByteOrder(StrEnum):
    """Multi-register byte/word ordering."""

    BIG_WORD_BIG_BYTE = "BIG_WORD_BIG_BYTE"
    BIG_WORD_LITTLE_BYTE = "BIG_WORD_LITTLE_BYTE"
    LITTLE_WORD_BIG_BYTE = "LITTLE_WORD_BIG_BYTE"
    LITTLE_WORD_LITTLE_BYTE = "LITTLE_WORD_LITTLE_BYTE"


@dataclass(frozen=True, slots=True)
class RegisterSpec:
    """One logical channel mapped onto Modbus registers."""

    name: str
    address: int
    register_type: RegisterType = RegisterType.FLOAT32
    unit: str = "dimensionless"
    scale: float = 1.0
    offset: float = 0.0
    byte_order: ByteOrder = ByteOrder.BIG_WORD_BIG_BYTE

    def __post_init__(self) -> None:
        if self.address < 0 or self.address > 0xFFFF:
            raise ConfigurationError(
                f"register address {self.address} out of range",
                details={"name": self.name, "address": self.address},
            )

    @property
    def register_count(self) -> int:
        return 1 if self.register_type in (RegisterType.UINT16, RegisterType.INT16) else 2


@dataclass(frozen=True, slots=True)
class ModbusConfig:
    """Modbus connection settings."""

    host: str = "localhost"
    port: int = 502
    unit_id: int = 1
    timeout_s: float = 5.0
    retries: int = 3
    mode: str = "tcp"

    def __post_init__(self) -> None:
        if not self.host:
            raise ConfigurationError("modbus.host must not be empty", details={})
        if not 1 <= self.port <= 65535:
            raise ConfigurationError(f"modbus.port {self.port} out of range",
                                     details={"port": self.port})
        if not 0 <= self.unit_id <= 247:
            raise ConfigurationError(f"modbus.unit_id {self.unit_id} out of range",
                                     details={"unit_id": self.unit_id})
        if self.mode not in ("tcp", "rtu"):
            raise ConfigurationError(f"modbus.mode must be tcp|rtu, got {self.mode!r}",
                                     details={"mode": self.mode})


def decode_registers(registers: Sequence[int], spec: RegisterSpec) -> float:
    """Decode raw 16-bit registers into an engineering value.

    Raises:
        TelemetryProtocolError: if fewer registers are supplied than the type needs.
    """
    needed = spec.register_count
    if len(registers) < needed:
        raise TelemetryProtocolError(
            f"{spec.name}: need {needed} register(s), got {len(registers)}",
            details={"channel": spec.name, "needed": needed, "got": len(registers)},
        )
    regs = [int(r) & 0xFFFF for r in registers[:needed]]

    if spec.register_type is RegisterType.UINT16:
        raw = float(regs[0])
    elif spec.register_type is RegisterType.INT16:
        raw = float(struct.unpack(">h", struct.pack(">H", regs[0]))[0])
    else:
        hi, lo = regs[0], regs[1]
        if spec.byte_order is ByteOrder.LITTLE_WORD_BIG_BYTE:
            hi, lo = lo, hi
        if spec.byte_order in (ByteOrder.BIG_WORD_LITTLE_BYTE,
                               ByteOrder.LITTLE_WORD_LITTLE_BYTE):
            hi = _swap_bytes(hi)
            lo = _swap_bytes(lo)
        payload = struct.pack(">HH", hi, lo)
        if spec.register_type is RegisterType.FLOAT32:
            raw = float(struct.unpack(">f", payload)[0])
        elif spec.register_type is RegisterType.UINT32:
            raw = float(struct.unpack(">I", payload)[0])
        else:
            raw = float(struct.unpack(">i", payload)[0])
    return raw * spec.scale + spec.offset


def _swap_bytes(value: int) -> int:
    return ((value & 0xFF) << 8) | ((value >> 8) & 0xFF)


class ModbusSource(TelemetrySource):
    """Poll a declared Modbus register map.

    Args:
        config: Connection settings.
        registers: Declared register map. Bounded by construction.
        client_factory: Injectable client. Must expose
            ``read_holding_registers(address, count, unit=...)``.
    """

    def __init__(
        self,
        config: ModbusConfig | None = None,
        registers: Sequence[RegisterSpec] = (),
        *,
        client_factory: Callable[[ModbusConfig], Any] | None = None,
        device_id: str = "modbus-device",
        source_id: str = "modbus",
        limits: SourceLimits | None = None,
        clock: Any | None = None,
    ) -> None:
        super().__init__(source_id=source_id, limits=limits)
        self.config = config or ModbusConfig()
        self.registers = tuple(registers)
        self._client_factory = client_factory
        self.device_id = device_id
        self._clock = clock
        self._client: Any = None
        self.failures: int = 0
        if not self.registers:
            raise ConfigurationError("ModbusSource requires at least one RegisterSpec",
                                     details={})

    def _open(self) -> None:
        self._client = self._make_client()

    def _make_client(self) -> Any:
        if self._client_factory is not None:
            return self._client_factory(self.config)
        try:
            from pymodbus.client import ModbusTcpClient
        except ImportError as exc:
            raise TelemetryError(
                "pymodbus is not installed. Install the 'modbus' extra: "
                "pip install sofia-engine[modbus]",
                details={"extra": "modbus"},
            ) from exc
        return ModbusTcpClient(self.config.host, port=self.config.port,
                               timeout=self.config.timeout_s)

    def _read_registers(self, address: int, count: int) -> Sequence[int]:
        validate_positive_int(count, "count", maximum=_MAX_REGISTERS_PER_READ)
        response = self._client.read_holding_registers(address, count=count,
                                                       unit=self.config.unit_id)
        if response is None:
            raise TelemetryProtocolError(
                f"no response for register {address}", details={"address": address}
            )
        if getattr(response, "isError", lambda: False)():
            raise TelemetryProtocolError(
                f"Modbus error reading register {address}", details={"address": address}
            )
        registers = getattr(response, "registers", None)
        if not isinstance(registers, (list, tuple)):
            raise TelemetryProtocolError(
                f"malformed Modbus response for register {address}",
                details={"address": address},
            )
        if len(registers) != count:
            raise TelemetryProtocolError(
                f"Modbus short frame: asked {count}, got {len(registers)}",
                details={"address": address, "expected": count, "got": len(registers)},
            )
        return registers

    def read(self, *, max_records: int | None = None) -> list[TelemetrySample]:
        self._require_open()
        now = self._now()
        out: list[TelemetrySample] = []
        limit = len(self.registers) if max_records is None else int(max_records)
        for spec in list(self.registers)[: max(0, limit)]:
            try:
                raw = self._read_registers(spec.address, spec.register_count)
                value = decode_registers(raw, spec)
            except (TelemetryProtocolError, TelemetryError) as exc:
                self.failures += 1
                self.stats.protocol_errors += 1
                self.state = type(self.state).DEGRADED
                raise TelemetryProtocolError(
                    f"Modbus read failed for {spec.name}: {exc}",
                    details={"channel": spec.name, "address": spec.address},
                ) from exc
            except Exception as exc:
                self.failures += 1
                self.state = type(self.state).FAILED
                raise TelemetryError(
                    f"Modbus transport failure on {spec.name}: {exc}",
                    details={"channel": spec.name},
                ) from exc
            out.append(
                TelemetrySample(
                    timestamp=now,
                    device_id=self.device_id,
                    channel=spec.name,
                    value=value,
                    unit=spec.unit,
                    source=self.source_id,
                )
            )
            self.stats.records_read += 1
        self.stats.records_emitted += len(out)
        self.stats.bytes_read += len(out) * 4
        return out

    def _now(self) -> float:
        import time

        return self._clock.wall() if self._clock is not None else time.time()

    def _close(self) -> None:
        if self._client is not None and hasattr(self._client, "close"):
            self._client.close()
        self._client = None

    def describe(self) -> dict[str, Any]:
        base = super().describe()
        base.update({
            "host": self.config.host,
            "port": self.config.port,
            "mode": self.config.mode,
            "channels": [r.name for r in self.registers],
            "failures": self.failures,
        })
        return base
