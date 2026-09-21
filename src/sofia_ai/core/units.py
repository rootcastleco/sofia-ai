"""Engineering units as a first-class concept.

Industrial AI without unit discipline is dangerous. This module provides:

* a unit registry mapping symbols to a physical dimension and a SI conversion,
* explicit, dimension-checked conversion,
* a :class:`Quantity` value type that refuses to add incompatible units.

There is **no implicit conversion** anywhere in Sofia: a value of ``10`` with unit
``"g"`` is never silently treated as ``"m/s2"``. All conversions go through
:func:`convert`, which raises :class:`DimensionError` on mismatch.

The registry is intentionally small and explicit rather than a general parser:
a wrong unit symbol should fail loudly at configuration time.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

from .errors import DimensionError, UnitError

__all__ = [
    "DIMENSIONS",
    "UNITS",
    "Quantity",
    "UnitError",
    "UnitSpec",
    "convert",
    "dimension_of",
    "from_si",
    "is_known_unit",
    "to_si",
    "unit_alias",
]


@dataclass(frozen=True, slots=True)
class UnitSpec:
    """A registered engineering unit.

    Attributes:
        symbol: Canonical symbol, e.g. ``"mm/s"``.
        dimension: Physical dimension name, e.g. ``"velocity"``.
        to_si_factor: Multiplicative factor to the SI base unit.
        si_offset: Additive offset applied *before* the factor (used for °C/°F).
        aliases: Alternative accepted spellings.
    """

    symbol: str
    dimension: str
    to_si_factor: float
    si_offset: float = 0.0
    aliases: tuple[str, ...] = ()

    def to_si(self, value: float) -> float:
        return (value + self.si_offset) * self.to_si_factor

    def from_si(self, value: float) -> float:
        return value / self.to_si_factor - self.si_offset


def _spec(
    symbol: str,
    dimension: str,
    factor: float,
    offset: float = 0.0,
    aliases: tuple[str, ...] = (),
) -> tuple[str, UnitSpec]:
    return symbol, UnitSpec(symbol, dimension, factor, offset, aliases)


_G: Final[float] = 9.80665

UNITS: Final[Mapping[str, UnitSpec]] = dict(
    [
        # --- dimensionless / ratio -----------------------------------------
        _spec("dimensionless", "dimensionless", 1.0, 0.0, ("1", "ratio", "-", "")),
        _spec("percent", "dimensionless", 0.01, 0.0, ("%",)),
        # --- acceleration ---------------------------------------------------
        _spec("m/s2", "acceleration", 1.0, 0.0, ("m/s^2", "m/s**2", "mps2")),
        _spec("g", "acceleration", _G, 0.0, ("gn", "g-force")),
        _spec("mm/s2", "acceleration", 1e-3, 0.0, ("mm/s^2",)),
        _spec("in/s2", "acceleration", 0.0254, 0.0, ("in/s^2",)),
        # --- velocity -------------------------------------------------------
        _spec("m/s", "velocity", 1.0),
        _spec("mm/s", "velocity", 1e-3, 0.0, ("mmps",)),
        _spec("in/s", "velocity", 0.0254, 0.0, ("ips",)),
        _spec("um/s", "velocity", 1e-6),
        # --- displacement ---------------------------------------------------
        _spec("m", "displacement", 1.0),
        _spec("mm", "displacement", 1e-3),
        _spec("um", "displacement", 1e-6, 0.0, ("micron", "microns", "um_pk")),
        _spec("mil", "displacement", 2.54e-5, 0.0, ("mils",)),
        _spec("in", "displacement", 0.0254, 0.0, ("inch",)),
        # --- temperature ----------------------------------------------------
        _spec("C", "temperature", 1.0, 273.15, ("degC", "celsius", "°C")),
        _spec("K", "temperature", 1.0, 0.0, ("kelvin",)),
        _spec("F", "temperature", 5.0 / 9.0, 459.67, ("degF", "fahrenheit", "°F")),
        # --- rotation -------------------------------------------------------
        _spec("RPM", "frequency", 1.0 / 60.0, 0.0, ("rpm",)),
        _spec("Hz", "frequency", 1.0, 0.0, ("hz",)),
        _spec("rad/s", "angular_velocity", 1.0, 0.0, ("rad_s",)),
        _spec("deg/s", "angular_velocity", math.pi / 180.0, 0.0),
        _spec("deg", "angle", math.pi / 180.0, 0.0, ("degree",)),
        _spec("rad", "angle", 1.0, 0.0),
        # --- electrical -----------------------------------------------------
        _spec("V", "voltage", 1.0, 0.0, ("volt",)),
        _spec("mV", "voltage", 1e-3, 0.0),
        _spec("A", "current", 1.0, 0.0, ("amp",)),
        _spec("mA", "current", 1e-3, 0.0),
        _spec("W", "power", 1.0, 0.0),
        _spec("kW", "power", 1e3, 0.0),
        _spec("ohm", "resistance", 1.0, 0.0),
        # --- pressure -------------------------------------------------------
        _spec("Pa", "pressure", 1.0, 0.0),
        _spec("kPa", "pressure", 1e3, 0.0),
        _spec("MPa", "pressure", 1e6, 0.0),
        _spec("bar", "pressure", 1e5, 0.0),
        _spec("psi", "pressure", 6894.757293168361, 0.0),
        # --- flow -----------------------------------------------------------
        _spec("m3/s", "flow", 1.0, 0.0, ("m^3/s",)),
        _spec("l/min", "flow", 1.0 / 60000.0, 0.0, ("L/min", "lpm")),
        _spec("m3/h", "flow", 1.0 / 3600.0, 0.0, ("m^3/h",)),
        # --- time -----------------------------------------------------------
        _spec("s", "time", 1.0, 0.0, ("sec", "second")),
        _spec("ms", "time", 1e-3, 0.0),
        _spec("us", "time", 1e-6, 0.0),
        # --- position / distance (GPS-ish) ----------------------------------
        _spec("deg_lat", "angle", 1.0, 0.0),
        _spec("deg_lon", "angle", 1.0, 0.0),
        # --- counters / digital ---------------------------------------------
        _spec("count", "count", 1.0, 0.0, ("counts", "counter")),
        _spec("bool", "digital", 1.0, 0.0, ("digital", "state")),
    ]
)

_ALIAS_INDEX: Final[Mapping[str, str]] = {
    alias.lower(): symbol
    for symbol, spec in UNITS.items()
    for alias in (symbol, *spec.aliases)
}

DIMENSIONS: Final[tuple[str, ...]] = tuple(sorted({s.dimension for s in UNITS.values()}))


def unit_alias(symbol: str) -> str:
    """Resolve a symbol or alias to its canonical unit symbol.

    Raises:
        UnitError: if the symbol is not registered.
    """
    key = symbol.strip().lower()
    canonical = _ALIAS_INDEX.get(key)
    if canonical is None:
        raise UnitError(
            f"Unknown unit {symbol!r}. Registered units: {sorted(UNITS)}",
            details={"unit": symbol},
        )
    return canonical


def is_known_unit(symbol: str) -> bool:
    """Return True if ``symbol`` (or one of its aliases) is registered."""
    return symbol.strip().lower() in _ALIAS_INDEX


def dimension_of(symbol: str) -> str:
    """Return the physical dimension of ``symbol``."""
    return UNITS[unit_alias(symbol)].dimension


def to_si(value: float, symbol: str) -> float:
    """Convert a value from ``symbol`` to the SI base unit of its dimension."""
    return UNITS[unit_alias(symbol)].to_si(float(value))


def from_si(value: float, symbol: str) -> float:
    """Convert a value from the SI base unit of a dimension to ``symbol``."""
    return UNITS[unit_alias(symbol)].from_si(float(value))


def convert(value: float, source: str, target: str) -> float:
    """Convert ``value`` from the ``source`` unit to the ``target`` unit.

    Raises:
        UnitError: if either unit is unknown.
        DimensionError: if the units belong to different physical dimensions.
    """
    src = UNITS[unit_alias(source)]
    dst = UNITS[unit_alias(target)]
    if src.dimension != dst.dimension:
        raise DimensionError(
            f"Cannot convert {src.dimension} ({source}) to {dst.dimension} ({target})",
            details={"source": source, "target": target,
                     "source_dimension": src.dimension, "target_dimension": dst.dimension},
        )
    return dst.from_si(src.to_si(value))


@dataclass(frozen=True, slots=True)
class Quantity:
    """A scalar value with an explicit unit.

    Arithmetic is dimension-checked. ``Quantity(1.0, "g") + Quantity(1.0, "mm/s")``
    raises :class:`DimensionError`. Addition returns the result in the unit of the
    left operand, using an explicit conversion.
    """

    value: float
    unit: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "unit", unit_alias(self.unit))
        if not math.isfinite(float(self.value)):
            raise UnitError(
                f"Quantity value must be finite, got {self.value!r}",
                details={"value": self.value},
            )

    @property
    def dimension(self) -> str:
        return UNITS[self.unit].dimension

    @property
    def si(self) -> float:
        return to_si(self.value, self.unit)

    def to(self, target: str) -> Quantity:
        """Return an equivalent quantity expressed in ``target``."""
        return Quantity(convert(self.value, self.unit, target), target)

    def __add__(self, other: Quantity) -> Quantity:
        if not isinstance(other, Quantity):
            raise UnitError("Quantity can only be added to Quantity")
        return Quantity(self.value + convert(other.value, other.unit, self.unit), self.unit)

    def __sub__(self, other: Quantity) -> Quantity:
        if not isinstance(other, Quantity):
            raise UnitError("Quantity can only be subtracted from Quantity")
        return Quantity(self.value - convert(other.value, other.unit, self.unit), self.unit)

    def __mul__(self, factor: float) -> Quantity:
        return Quantity(self.value * float(factor), self.unit)

    def __rmul__(self, factor: float) -> Quantity:
        return self.__mul__(factor)

    def __truediv__(self, factor: float) -> Quantity:
        if float(factor) == 0.0:
            raise UnitError("Division by zero")
        return Quantity(self.value / float(factor), self.unit)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Quantity):
            return NotImplemented
        if self.unit == other.unit:
            return math.isclose(self.value, other.value, rel_tol=1e-12, abs_tol=1e-12)
        if self.dimension != other.dimension:
            return False
        return math.isclose(self.si, other.si, rel_tol=1e-12, abs_tol=1e-12)

    def __lt__(self, other: Quantity) -> bool:
        return self.si < other.si

    def __le__(self, other: Quantity) -> bool:
        return self.si <= other.si

    def __hash__(self) -> int:
        return hash((self.unit, round(self.value, 12)))

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"Quantity({self.value!r}, {self.unit!r})"
