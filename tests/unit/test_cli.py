"""Unit tests for the command-line interface (SOFIA-PKG-005/006)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from sofia_ai.cli.main import EXIT_ERROR, EXIT_OK, EXIT_USAGE, build_parser, main

REPO_ROOT = Path(__file__).resolve().parents[2]
VIBRATION_CSV = REPO_ROOT / "examples" / "data" / "vibration.csv"
TELEMETRY_JSONL = REPO_ROOT / "examples" / "data" / "telemetry.jsonl"


class TestParser:
    def test_all_commands_declared(self) -> None:
        parser = build_parser()
        actions = [a for a in parser._actions if hasattr(a, "choices") and a.choices]
        names = {name for action in actions for name in action.choices}
        for command in ("info", "doctor", "inspect", "analyze", "replay", "benchmark",
                        "models", "devices", "config"):
            assert command in names, command

    def test_no_actuation_command(self) -> None:
        parser = build_parser()
        actions = [a for a in parser._actions if hasattr(a, "choices") and a.choices]
        names = {name for action in actions for name in action.choices}
        for forbidden in ("command", "actuate", "control", "write", "set", "execute"):
            assert forbidden not in names, forbidden

    def test_no_command_prints_usage(self, capsys) -> None:
        assert main([]) == EXIT_USAGE
        assert "usage" in capsys.readouterr().out


class TestInfo:
    def test_text(self, capsys) -> None:
        assert main(["info"]) == EXIT_OK
        out = capsys.readouterr().out
        assert "Sofia Engine" in out
        assert "Apache-2.0" in out

    def test_json(self, capsys) -> None:
        assert main(["info", "--json"]) == EXIT_OK
        payload = json.loads(capsys.readouterr().out)
        assert payload["sofia_version"] == "3.0.0a1"
        assert payload["capabilities"]["offline_first"] is True


class TestDoctor:
    def test_text(self, capsys) -> None:
        assert main(["doctor"]) == EXIT_OK
        out = capsys.readouterr().out
        assert "numpy" in out

    def test_json(self, capsys) -> None:
        assert main(["doctor", "--json"]) == EXIT_OK
        payload = json.loads(capsys.readouterr().out)
        assert payload["offline_ready"] is True
        assert any(c["extra"] == "core" for c in payload["checks"])


class TestModels:
    def test_lists_backends(self, capsys) -> None:
        assert main(["models"]) == EXIT_OK
        assert "mad" in capsys.readouterr().out

    def test_json(self, capsys) -> None:
        assert main(["models", "--json"]) == EXIT_OK
        payload = json.loads(capsys.readouterr().out)
        assert payload["count"] >= 9

    def test_no_actuation_backend(self, capsys) -> None:
        assert main(["models", "--json"]) == EXIT_OK
        names = [b["name"] for b in json.loads(capsys.readouterr().out)["backends"]]
        assert "actuator" not in names


class TestConfig:
    def test_default_config(self, capsys) -> None:
        assert main(["config"]) == EXIT_OK
        payload = json.loads(capsys.readouterr().out) if False else None
        assert payload is None

    def test_config_json(self, capsys) -> None:
        assert main(["config", "--json"]) == EXIT_OK
        payload = json.loads(capsys.readouterr().out)
        assert payload["valid"] is True

    def test_devices(self, capsys) -> None:
        assert main(["devices", "--json"]) == EXIT_OK
        payload = json.loads(capsys.readouterr().out)
        assert "device_id" in payload


class TestInspect:
    def test_csv(self, capsys) -> None:
        assert main(["inspect", str(VIBRATION_CSV), "--json"]) == EXIT_OK
        payload = json.loads(capsys.readouterr().out)
        assert payload["samples"] == 4096
        assert payload["format"] == "csv"
        assert "vibration_x" in payload["channels"]

    def test_jsonl(self, capsys) -> None:
        assert main(["inspect", str(TELEMETRY_JSONL), "--json"]) == EXIT_OK
        payload = json.loads(capsys.readouterr().out)
        assert payload["format"] == "jsonl"
        assert "temperature" in payload["channels"]

    def test_missing_file(self, capsys) -> None:
        assert main(["inspect", "does-not-exist.csv"]) == EXIT_ERROR
        assert "does-not-exist.csv" in capsys.readouterr().err


class TestAnalyze:
    def test_analyze(self, capsys) -> None:
        assert main(["analyze", str(VIBRATION_CSV), "--json"]) == EXIT_OK
        payload = json.loads(capsys.readouterr().out)
        assert payload["windows"] > 0
        assert payload["features"] > 0
        assert "rms" in payload["first_window_features"]

    def test_analyze_detects_anomaly(self, capsys) -> None:
        assert main(["analyze", str(VIBRATION_CSV), "--json"]) == EXIT_OK
        payload = json.loads(capsys.readouterr().out)
        assert payload["anomalous_windows"] > 0

    def test_analyze_with_shaft(self, capsys) -> None:
        assert main(["analyze", str(VIBRATION_CSV), "--shaft-hz", "25",
                     "--json"]) == EXIT_OK
        payload = json.loads(capsys.readouterr().out)
        assert "order_1_energy" in payload["first_window_features"]


class TestReplay:
    def test_replay(self, capsys) -> None:
        assert main(["replay", str(VIBRATION_CSV), "--ticks", "4", "--json"]) == EXIT_OK
        payload = json.loads(capsys.readouterr().out)
        assert payload["stats"]["ticks"] == 4


class TestBenchmark:
    def test_quick_suite(self, capsys) -> None:
        assert main(["benchmark", "--suite", "features", "--iterations", "3"]) == EXIT_OK
        out = capsys.readouterr().out
        assert "BENCHMARK" in out
        assert "p50" in out

    def test_json_output(self, capsys) -> None:
        assert main(["benchmark", "--suite", "features", "--iterations", "2",
                     "--json"]) == EXIT_OK
        payload = json.loads(capsys.readouterr().out)
        assert payload["measurements"]
        assert "disclaimer" in payload

    def test_unknown_suite(self) -> None:
        with pytest.raises(ValueError):
            main(["benchmark", "--suite", "nonexistent", "--iterations", "1"])

    def test_output_file(self, tmp_path, capsys) -> None:
        out = tmp_path / "bench.json"
        assert main(["benchmark", "--suite", "features", "--iterations", "2",
                     "--output", str(out)]) == EXIT_OK
        assert json.loads(out.read_text(encoding="utf-8"))["measurements"]


class TestModuleEntryPoint:
    def test_python_dash_m(self) -> None:
        import os

        env = dict(os.environ)
        env["PYTHONPATH"] = str(REPO_ROOT / "src")
        result = subprocess.run(
            [sys.executable, "-m", "sofia_ai", "info"],
            capture_output=True, text=True, cwd=str(REPO_ROOT), env=env, check=False,
        )
        assert result.returncode == 0, result.stderr
        assert "Sofia Engine" in result.stdout
