"""Unit tests for typed configuration (§39) and packaging metadata."""

from __future__ import annotations

import json

import pytest

from sofia_ai.config import (
    BufferConfig,
    ChannelConfig,
    DeviceConfig,
    LoggingConfig,
    MetricsConfig,
    PolicyConfig,
    ReconnectConfig,
    SofiaConfig,
    StoreForwardConfig,
    TransportConfig,
    WindowConfig,
    config_from_dict,
    load_config,
)
from sofia_ai.core.errors import ConfigurationError


class TestSections:
    def test_buffer_defaults(self) -> None:
        config = BufferConfig()
        assert config.max_samples == 4096
        assert config.overflow_policy == "drop_oldest"

    def test_buffer_rejects_bad_policy(self) -> None:
        with pytest.raises(ConfigurationError):
            BufferConfig(overflow_policy="explode")

    def test_window_rejects_hop_gt_length(self) -> None:
        with pytest.raises(ConfigurationError):
            WindowConfig(length=8, hop=16)

    def test_transport_bounds(self) -> None:
        with pytest.raises(ConfigurationError):
            TransportConfig(max_payload_bytes=0)
        with pytest.raises(ConfigurationError):
            TransportConfig(read_timeout_s=0)
        with pytest.raises(ConfigurationError):
            TransportConfig(max_retries=-1)

    def test_logging_level(self) -> None:
        assert LoggingConfig(level="debug").level == "DEBUG"
        with pytest.raises(ConfigurationError):
            LoggingConfig(level="VERBOSE")

    def test_reconnect_backoff_bounds(self) -> None:
        with pytest.raises(ConfigurationError):
            ReconnectConfig(initial_backoff_s=10.0, max_backoff_s=5.0)
        with pytest.raises(ConfigurationError):
            ReconnectConfig(multiplier=0.5)
        with pytest.raises(ConfigurationError):
            ReconnectConfig(jitter=2.0)

    def test_store_forward(self) -> None:
        with pytest.raises(ConfigurationError):
            StoreForwardConfig(directory="")

    def test_metrics(self) -> None:
        assert MetricsConfig().max_samples == 10_000

    def test_policy_config(self) -> None:
        config = PolicyConfig()
        assert config.default_decision == "deny"
        assert config.min_confidence == 0.9
        with pytest.raises(ConfigurationError):
            PolicyConfig(default_decision="maybe")
        with pytest.raises(ConfigurationError):
            PolicyConfig(min_confidence=2.0)

    def test_channel_config_validates_unit(self) -> None:
        with pytest.raises(ConfigurationError):
            ChannelConfig(name="c", unit="parsec")

    def test_channel_inverted_range(self) -> None:
        with pytest.raises(ConfigurationError):
            ChannelConfig(name="c", unit="g", minimum=10.0, maximum=1.0)

    def test_device_lookup(self) -> None:
        device = DeviceConfig(device_id="d", channels=(
            ChannelConfig(name="vibration_x", unit="g"),))
        assert device.channel("vibration_x") is not None
        assert device.channel("nope") is None


class TestSofiaConfig:
    def test_defaults(self) -> None:
        config = SofiaConfig()
        assert config.offline is True
        assert config.version == "2.0"

    def test_roundtrip_dict(self) -> None:
        config = config_from_dict({"device_id": "d",
                                   "window": {"length": 256, "hop": 128}})
        assert config.window.length == 256
        assert config.window.hop == 128
        restored = config_from_dict(config.to_dict())
        assert restored.window.length == 256

    def test_rejects_major_version(self) -> None:
        with pytest.raises(ConfigurationError):
            config_from_dict({"version": "1.0"})

    def test_device_and_channel_lookup(self) -> None:
        config = config_from_dict({
            "devices": [{"device_id": "d", "channels": [
                {"name": "vibration_x", "unit": "g", "minimum": -10.0,
                 "maximum": 10.0}]}],
        })
        assert config.device("d") is not None
        assert config.channel("d", "vibration_x") is not None
        assert config.channel("d", "nope") is None
        assert config.device("other") is None

    def test_load_from_file(self, tmp_path) -> None:
        path = tmp_path / "c.json"
        path.write_text(json.dumps({"device_id": "from-file"}), encoding="utf-8")
        assert load_config(str(path)).device_id == "from-file"

    def test_load_without_file(self) -> None:
        assert load_config().device_id == "default"

    def test_env_override(self, tmp_path, monkeypatch) -> None:
        path = tmp_path / "c.json"
        path.write_text(json.dumps({"device_id": "file"}), encoding="utf-8")
        monkeypatch.setenv("SOFIA_DEVICE_ID", "env")
        assert load_config(str(path)).device_id == "env"
        monkeypatch.setenv("SOFIA_OFFLINE", "false")
        assert load_config(str(path)).offline is False

    def test_env_override_is_allowlisted(self, monkeypatch) -> None:
        """An arbitrary SOFIA_* variable must not inject a config key."""
        monkeypatch.setenv("SOFIA_WINDOW_LENGTH", "1")
        assert load_config().window.length == 1024


class TestPackaging:
    def test_metadata_is_consistent(self) -> None:
        import tomllib
        from pathlib import Path

        path = Path(__file__).resolve().parents[2] / "pyproject.toml"
        payload = tomllib.loads(path.read_text(encoding="utf-8"))
        project = payload["project"]
        assert project["name"] == "sofia-engine"
        assert project["license"] == "Apache-2.0"
        assert "rootcastleco/sofia-rl" in project["urls"]["Repository"]
        assert project["requires-python"] == ">=3.11"
        assert "License" not in " ".join(project["classifiers"])  # PEP 639
        assert project["license"] == "Apache-2.0"  # SPDX, single source of truth

    def test_core_dependency_is_minimal(self) -> None:
        import tomllib
        from pathlib import Path

        path = Path(__file__).resolve().parents[2] / "pyproject.toml"
        payload = tomllib.loads(path.read_text(encoding="utf-8"))
        assert payload["project"]["dependencies"] == ["numpy>=1.24"]

    def test_extras_exist(self) -> None:
        import tomllib
        from pathlib import Path

        path = Path(__file__).resolve().parents[2] / "pyproject.toml"
        extras = tomllib.loads(path.read_text(encoding="utf-8"))["project"][
            "optional-dependencies"]
        for name in ("mqtt", "modbus", "serial", "onnx", "torch", "industrial",
                     "dev", "docs"):
            assert name in extras, name

    def test_cli_entry_point(self) -> None:
        import tomllib
        from pathlib import Path

        path = Path(__file__).resolve().parents[2] / "pyproject.toml"
        scripts = tomllib.loads(path.read_text(encoding="utf-8"))["project"]["scripts"]
        assert scripts["sofia"] == "sofia_ai.cli.main:main"

    def test_version_matches_package(self) -> None:
        import sofia_ai

        assert sofia_ai.__version__ == "2.0.0"

    def test_license_file_is_apache(self) -> None:
        from pathlib import Path

        head = Path(__file__).resolve().parents[2] / "LICENSE"
        assert "Apache License" in head.read_text(encoding="utf-8")[:200]
