"""Unit tests for engineering units (SOFIA-UNIT-001..004)."""

from __future__ import annotations

import math

import pytest

from sofia_ai.core.errors import DimensionError, UnitError
from sofia_ai.core.units import (
    DIMENSIONS,
    UNITS,
    Quantity,
    convert,
    dimension_of,
    from_si,
    is_known_unit,
    to_si,
    unit_alias,
)


class TestRegistry:
    def test_known_units_are_declared(self) -> None:
        for symbol in ("g", "m/s2", "mm/s", "um", "mil", "C", "F", "K", "RPM", "Hz",
                       "V", "mV", "A", "mA", "kW", "bar", "psi", "l/min"):
            assert is_known_unit(symbol), symbol

    def test_unknown_unit_raises(self) -> None:
        with pytest.raises(UnitError):
            unit_alias("parsec")

    def test_alias_resolution(self) -> None:
        assert unit_alias("m/s^2") == "m/s2"
        assert unit_alias("IPS") == "in/s"
        assert unit_alias("degC") == "C"

    def test_dimensions_are_consistent(self) -> None:
        assert "acceleration" in DIMENSIONS
        assert "velocity" in DIMENSIONS
        assert dimension_of("mm/s") == "velocity"
        assert dimension_of("g") == "acceleration"


class TestConversion:
    def test_g_to_ms2(self) -> None:
        assert math.isclose(convert(1.0, "g", "m/s2"), 9.80665, rel_tol=1e-12)

    def test_roundtrip_identity(self) -> None:
        value = 3.75
        assert math.isclose(convert(convert(value, "g", "mm/s2"), "mm/s2", "g"), value,
                            rel_tol=1e-9)

    def test_celsius_fahrenheit(self) -> None:
        assert math.isclose(convert(100.0, "C", "F"), 212.0, abs_tol=1e-9)
        assert math.isclose(convert(32.0, "F", "C"), 0.0, abs_tol=1e-9)
        assert math.isclose(convert(0.0, "C", "K"), 273.15, abs_tol=1e-12)
        assert math.isclose(convert(-40.0, "C", "F"), -40.0, abs_tol=1e-9)

    def test_rpm_to_hz(self) -> None:
        assert math.isclose(convert(60.0, "RPM", "Hz"), 1.0, rel_tol=1e-12)

    def test_psi_to_bar(self) -> None:
        assert math.isclose(convert(14.5038, "psi", "bar"), 1.0, rel_tol=1e-4)

    def test_cross_dimension_raises(self) -> None:
        with pytest.raises(DimensionError):
            convert(1.0, "g", "mm/s")

    def test_cross_dimension_current_voltage(self) -> None:
        with pytest.raises(DimensionError):
            convert(1.0, "A", "V")

    def test_si_helpers(self) -> None:
        assert math.isclose(to_si(1.0, "mm/s"), 1e-3)
        assert math.isclose(from_si(1e-3, "mm/s"), 1.0)


class TestQuantity:
    def test_addition_same_unit(self) -> None:
        assert Quantity(1.0, "g") + Quantity(2.0, "g") == Quantity(3.0, "g")

    def test_addition_converts(self) -> None:
        result = Quantity(1.0, "g") + Quantity(4.903325, "m/s2")
        assert math.isclose(result.value, 1.5, rel_tol=1e-6)

    def test_addition_cross_dimension_raises(self) -> None:
        with pytest.raises(DimensionError):
            Quantity(1.0, "g") + Quantity(1.0, "mm/s")

    def test_subtraction(self) -> None:
        assert Quantity(3.0, "g") - Quantity(1.0, "g") == Quantity(2.0, "g")

    def test_scaling(self) -> None:
        assert Quantity(2.0, "g") * 3 == Quantity(6.0, "g")
        assert 3 * Quantity(2.0, "g") == Quantity(6.0, "g")
        assert Quantity(6.0, "g") / 2 == Quantity(3.0, "g")

    def test_division_by_zero_raises(self) -> None:
        with pytest.raises(UnitError):
            Quantity(1.0, "g") / 0.0

    def test_to(self) -> None:
        assert math.isclose(Quantity(1.0, "g").to("m/s2").value, 9.80665, rel_tol=1e-12)

    def test_equality_across_units(self) -> None:
        assert Quantity(1.0, "g") == Quantity(9.80665, "m/s2")

    def test_inequality_across_dimensions(self) -> None:
        assert Quantity(1.0, "g") != Quantity(1.0, "mm/s")

    def test_rejects_non_finite(self) -> None:
        with pytest.raises(UnitError):
            Quantity(float("nan"), "g")

    def test_ordering(self) -> None:
        assert Quantity(1.0, "m/s") < Quantity(2000.0, "mm/s")

    def test_repr(self) -> None:
        assert "g" in repr(Quantity(1.0, "g"))


def test_registry_size_is_bounded() -> None:
    """The registry must stay small and explicit, not a general parser."""
    assert 40 <= len(UNITS) <= 200
