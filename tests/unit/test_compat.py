"""Deprecated 1.x API shims must keep resolving with a DeprecationWarning."""

from __future__ import annotations

import importlib
import re
import warnings

import pytest


def _deprecated_error() -> DeprecationWarning:
    with pytest.warns(DeprecationWarning) as captured:
        importlib.import_module("sofia_ai.compat")
        from sofia_ai import compat

        compat.__getattr__("SofiaModel")
    return captured.list[0]


def test_import_sofia_ai_emits_no_warning() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        import sofia_ai  # noqa: F401


def test_known_names_resolve_and_warn() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        from sofia_ai import compat

        for name in ("SofiaModel", "QuantumNeuralEngine", "QuantumState",
                     "NLPProcessor", "Token", "ProcessedText",
                     "QuantumConfig", "NLPConfig", "ModelConfig",
                     "SofiaRLAgent", "QNetwork", "ReplayBuffer"):
            compat.__getattr__(name)
        assert len(caught) == 12
        for warning in caught:
            assert issubclass(warning.category, DeprecationWarning)
            assert "2.0.0" in str(warning.message)


def test_unknown_name_raises_attribute_error() -> None:
    from sofia_ai import compat

    with pytest.raises(AttributeError, match="no attribute 'nope'"):
        compat.__getattr__("nope")


def test_dir_includes_deprecated_names() -> None:
    from sofia_ai import compat

    for name in ("SofiaModel", "NLPProcessor", "SofiaRLAgent"):
        assert name in compat.__dir__()


def test_not_in_public_all() -> None:
    from sofia_ai import compat

    assert "SofiaModel" in compat.__all__
    assert "QuantumConfig" in compat.__all__


def test__import_legacy_model_shim() -> None:
    import sofia_ai.legacy_model  # noqa: F401


def test_deprecated_map_has_reasonable_targets() -> None:
    from sofia_ai import compat

    assert compat.DEPRECATED["SofiaModel"] == "sofia_ai.diagnostics / sofia_ai.inference"
    assert "experimental.quantum" in compat.DEPRECATED["QuantumNeuralEngine"]
    assert compat.DEPRECATED["SofiaRLAgent"].endswith("learning.rl.SofiaRLAgent")
    assert re.match(r"^\d+\.\d+\.\d+$", compat.REMOVAL_VERSION)


def test__import_is_cheap() -> None:
    """Importing sofia_ai must not pull torch or other heavy deps."""
    import sys

    assert "torch" not in sys.modules
