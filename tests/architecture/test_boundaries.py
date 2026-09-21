"""Architecture boundary tests.

These tests enforce the module dependency rules from
``specs/001-sofia-engine-modernization/architecture.md``. A violation fails the
build, which is what keeps the boundaries real rather than aspirational.
"""

from __future__ import annotations

import ast
from collections.abc import Iterator
from pathlib import Path

import pytest

SRC_ROOT = Path(__file__).resolve().parents[2] / "src" / "sofia_ai"

#: Modules that must never be imported outside the minimal install.
FORBIDDEN_FRAMEWORKS = {
    "torch": "torch",
    "onnxruntime": "onnx",
    "paho": "mqtt",
    "pymodbus": "modbus",
    "serial": "serial",
    "fastapi": "api",
    "uvicorn": "api",
}

#: Modules that may import optional frameworks (adapters and experimental code).
ALLOWED_FRAMEWORK_USERS = frozenset({
    "inference/onnx_backend.py",
    "inference/torch_backend.py",
    "learning/rl/agent.py",
    "learning/rl/networks.py",
    "telemetry/mqtt.py",
    "telemetry/modbus.py",
    "telemetry/serial_adapter.py",
    "cli/main.py",
})


def _python_files(root: Path) -> Iterator[Path]:
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        yield path


def _relative(path: Path) -> str:
    return path.relative_to(SRC_ROOT).as_posix()


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            found.add(node.module)
    return found


def _source_of(path: Path) -> str:
    return path.read_text(encoding="utf-8")


class TestCoreIsolation:
    def test_core_imports_nothing_outside_core(self) -> None:
        """core may import its own modules, but nothing from another Sofia layer."""
        for path in _python_files(SRC_ROOT / "core"):
            tree = ast.parse(_source_of(path), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.level and node.level > 1:
                    pytest.fail(
                        f"core/{_relative(path)} imports outside core "
                        f"(level={node.level})")

    def test_core_has_no_framework_dependencies(self) -> None:
        for path in _python_files(SRC_ROOT / "core"):
            for module in _imported_modules(path):
                if module in FORBIDDEN_FRAMEWORKS and module != "serial":
                    pytest.fail(
                        f"core/{_relative(path)} imports {module}, which would make "
                        f"the minimal install heavier"
                    )


class TestDeterministicPipelineHasNoFrameworks:
    @pytest.mark.parametrize("package", ["signal", "features", "diagnostics", "decision",
                                         "config", "observability", "security", "edge"])
    def test_package_imports_no_optional_framework(self, package: str) -> None:
        for path in _python_files(SRC_ROOT / package):
            relative = _relative(path)
            if relative in ALLOWED_FRAMEWORK_USERS:
                continue
            for module in _imported_modules(path):
                if module in FORBIDDEN_FRAMEWORKS:
                    pytest.fail(
                        f"{relative} imports {module}; install "
                        f"sofia-engine[{FORBIDDEN_FRAMEWORKS[module]}] only for adapters"
                    )


class TestExperimentalQuarantine:
    def test_nothing_outside_experimental_imports_it(self) -> None:
        allowed = ("experimental/", "compat", "benchmarking/")
        for path in _python_files(SRC_ROOT):
            relative = _relative(path)
            if relative.startswith(allowed):
                continue
            modules = _imported_modules(path)
            if any(m == "experimental" or m.startswith("experimental.")
                   or m.startswith("sofia_ai.experimental.") for m in modules):
                pytest.fail(f"{relative} imports the experimental quantum module")


class TestNoUnsafeDeserialization:
    @pytest.mark.parametrize("pattern", ["pickle.load", "pickle.loads",
                                         "yaml.unsafe_load", "marshal.load"])
    def test_pattern_absent(self, pattern: str) -> None:
        for path in _python_files(SRC_ROOT):
            if pattern in _source_of(path):
                pytest.fail(f"{_relative(path)} uses {pattern}")

    def test_no_eval_or_exec(self) -> None:
        for path in _python_files(SRC_ROOT):
            tree = ast.parse(_source_of(path), filename=str(path))
            for node in ast.walk(tree):
                if (isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Name)
                        and node.func.id in ("eval", "exec", "compile")):
                    pytest.fail(
                        f"{_relative(path)} calls {node.func.id}()")

    def test_no_shell_true(self) -> None:
        for path in _python_files(SRC_ROOT):
            if "shell=True" in _source_of(path):
                pytest.fail(f"{_relative(path)} uses subprocess with shell=True")


class TestNoSilentExceptionSwallowing:
    def test_no_bare_except_pass(self) -> None:
        for path in _python_files(SRC_ROOT):
            tree = ast.parse(_source_of(path), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.ExceptHandler):
                    if node.type is None:
                        pytest.fail(f"{_relative(path)} has a bare except")
                    body = [n for n in node.body if not isinstance(n, ast.Pass)]
                    if not body:
                        pytest.fail(f"{_relative(path)} swallows an exception with pass")


class TestNoGlobalRngMutation:
    @pytest.mark.parametrize("call", ["random.seed", "np.random.seed", "numpy.random.seed"])
    def test_no_global_seeding_in_library(self, call: str) -> None:
        for path in _python_files(SRC_ROOT):
            relative = _relative(path)
            if relative.startswith("learning/rl"):
                continue  # RL owns its own generator; seeded explicitly and documented
            if call in _source_of(path):
                pytest.fail(
                    f"{relative} seeds the global RNG ({call}); inject a Generator")


class TestNoImportTimeWork:
    def test_package_init_files_are_cheap(self) -> None:
        """Package __init__ files must not perform I/O or heavy imports."""
        for path in _python_files(SRC_ROOT):
            if path.name != "__init__.py":
                continue
            tree = ast.parse(_source_of(path), filename=str(path))
            for node in tree.body:
                if isinstance(node, (ast.AsyncFunctionDef,)):
                    pytest.fail(f"{_relative(path)} defines async work at import time")
                if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                    func = node.value.func
                    name = getattr(func, "id", None) or getattr(func, "attr", None)
                    if name in ("open", "connect", "load"):
                        pytest.fail(f"{_relative(path)} performs I/O at import time")


class TestCopilotIsolation:
    def test_copilot_has_no_actuation_surface(self) -> None:
        for path in _python_files(SRC_ROOT / "copilot"):
            source = _source_of(path)
            for forbidden in ("PolicyEngine", "CommandRequest", "actuate", "command("):
                if forbidden in source:
                    pytest.fail(
                        f"copilot/{_relative(path)} references {forbidden}; the "
                        f"copilot must stay outside the control path")


class TestSafetyBoundary:
    def test_policy_default_is_deny(self) -> None:
        source = _source_of(SRC_ROOT / "decision" / "policy.py")
        assert 'default_decision: str = "deny"' in source

    def test_interlock_actions_declared(self) -> None:
        from sofia_ai.decision import INTERLOCK_ACTIONS

        assert "interlock_bypass" in INTERLOCK_ACTIONS
        assert "emergency_stop_reset" in INTERLOCK_ACTIONS

    def test_no_bypass_helper_exists(self) -> None:
        source = _source_of(SRC_ROOT / "decision" / "policy.py")
        assert "assert_no_bypass" in source


class TestBoundedStructures:
    def test_every_dataclass_buffer_has_capacity(self) -> None:
        """Ring buffers and reservoirs must declare an explicit ceiling."""
        for name in ("edge/buffer.py", "observability/metrics.py",
                     "decision/policy.py", "observability/logging.py"):
            source = _source_of(SRC_ROOT / name)
            assert "capacity" in source or "max_samples" in source or "max_keys" in source

    def test_store_forward_has_byte_ceiling(self) -> None:
        source = _source_of(SRC_ROOT / "edge" / "store_forward.py")
        assert "max_bytes" in source
        assert "max_files" in source


class TestDocstringDiscipline:
    def test_modules_declare_intent(self) -> None:
        for path in _python_files(SRC_ROOT):
            if path.name == "__init__.py" and not _source_of(path).strip():
                continue
            tree = ast.parse(_source_of(path), filename=str(path))
            if tree.body and isinstance(tree.body[0], ast.Expr) \
                    and isinstance(tree.body[0].value, ast.Constant):
                continue
            if not _source_of(path).strip():
                continue
            pytest.fail(f"{_relative(path)} has no module docstring")
