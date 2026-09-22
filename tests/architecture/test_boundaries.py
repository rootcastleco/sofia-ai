"""Architecture Boundary Tests.

Verifies SOFIA-CORE-015:
The deterministic scientific runtime (sofia_ai.core, sofia_ai.signal, sofia_ai.features)
must not depend on or import any prohibited external libraries:
torch, onnxruntime, fastapi, paho.mqtt, pymodbus, serial, requests, or network clients.
"""

from __future__ import annotations

import importlib
import sys
from typing import Final

PROHIBITED_CORE_MODULES: Final[tuple[str, ...]] = (
    "torch",
    "onnxruntime",
    "fastapi",
    "paho",
    "pymodbus",
    "serial",
    "requests",
    "openai",
    "anthropic",
)

CORE_PACKAGES: Final[tuple[str, ...]] = (
    "sofia_ai.core",
    "sofia_ai.core.contracts",
    "sofia_ai.core.quality",
    "sofia_ai.core.units",
    "sofia_ai.core.time",
    "sofia_ai.core.validation",
    "sofia_ai.core.errors",
    "sofia_ai.signal.spectral",
    "sofia_ai.signal.envelope",
    "sofia_ai.signal.filters",
    "sofia_ai.signal.electrical",
    "sofia_ai.signal.acoustic",
    "sofia_ai.features.multidomain",
    "sofia_ai.features.statistical",
    "sofia_ai.features.spectral",
    "sofia_ai.features.extractor",
    "sofia_ai.features.rotating",
)


def test_core_packages_import_cleanly() -> None:
    """Ensure all core packages import without requiring any optional dependencies."""
    for pkg_name in CORE_PACKAGES:
        mod = importlib.import_module(pkg_name)
        assert mod is not None


def test_no_prohibited_modules_in_core() -> None:
    """Ensure that importing core packages does not pull prohibited modules into sys.modules."""
    # First verify that no prohibited modules are loaded by core imports
    for prohibited in PROHIBITED_CORE_MODULES:
        loaded = [m for m in sys.modules if m == prohibited or m.startswith(f"{prohibited}.")]
        assert not loaded, f"Prohibited module {prohibited!r} was imported by core: {loaded}"


def test_core_capabilities_manifest() -> None:
    """Verify that the capability manifest honestly reflects offline_first."""
    import sofia_ai

    assert sofia_ai.CAPABILITIES["offline_first"] is True
    assert sofia_ai.CAPABILITIES["functional_safety_certified"] is False
