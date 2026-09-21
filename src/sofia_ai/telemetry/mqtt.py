"""MQTT telemetry adapter (optional extra: ``pip install sofia-engine[mqtt]``).

The MQTT client library is imported lazily, so the minimal install never pulls it
in. All transport failures are converted to Sofia typed errors. Credentials are
resolved from the environment, never from configuration files.

This adapter is unit-tested against an injected fake client; no live broker is
required. Integration against a real broker is **NOT VERIFIED** in this repository.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from typing import Any, Final

from ..core.contracts import TelemetrySample
from ..core.errors import (
    ConfigurationError,
    PayloadTooLargeError,
    TelemetryError,
    TelemetryProtocolError,
    TransportClosedError,
)
from ..core.quality import DataQuality
from ..core.units import is_known_unit
from ..security.secrets import require_secret, secret
from .base import SourceLimits, TelemetrySource

__all__ = ["MqttConfig", "MqttPublishSink", "MqttSource"]

_DEFAULT_TOPICS: Final[tuple[str, ...]] = ("sofia/telemetry/#",)


class MqttConfig:
    """MQTT connection settings. Secrets come from the environment."""

    def __init__(
        self,
        *,
        host: str = "localhost",
        port: int = 1883,
        topics: Sequence[str] = _DEFAULT_TOPICS,
        client_id: str = "sofia-engine",
        username_env: str | None = None,
        password_env: str | None = None,
        tls_enabled: bool = False,
        ca_cert: str | None = None,
        keepalive_s: int = 60,
        qos: int = 0,
    ) -> None:
        if not host:
            raise ConfigurationError("mqtt.host must not be empty", details={})
        if not 1 <= port <= 65535:
            raise ConfigurationError(f"mqtt.port {port} out of range", details={"port": port})
        if not topics:
            raise ConfigurationError("mqtt.topics must not be empty", details={})
        if qos not in (0, 1, 2):
            raise ConfigurationError(f"mqtt.qos must be 0|1|2, got {qos}", details={"qos": qos})
        self.host = host
        self.port = port
        self.topics = tuple(topics)
        self.client_id = client_id
        self.username_env = username_env
        self.password_env = password_env
        self.tls_enabled = tls_enabled
        self.ca_cert = ca_cert
        self.keepalive_s = keepalive_s
        self.qos = qos

    def credentials(self) -> tuple[str | None, str | None]:
        """Resolve credentials from the environment. Never from config files."""
        username = secret(self.username_env) if self.username_env else None
        password = require_secret(self.password_env) if self.password_env else None
        return username, password

    def describe(self) -> dict[str, Any]:
        return {
            "host": self.host,
            "port": self.port,
            "topics": list(self.topics),
            "client_id": self.client_id,
            "tls_enabled": self.tls_enabled,
            "qos": self.qos,
            "username_env": self.username_env,
            "password_env": "***" if self.password_env else None,
        }


class MqttSource(TelemetrySource):
    """Subscribe to MQTT topics and decode telemetry payloads.

    Args:
        config: Connection settings.
        client_factory: Injectable client factory. Defaults to ``paho.mqtt.client``.
            Tests inject a fake so no broker is needed.
        decoder: Payload decoder. Default expects the Sofia JSON sample envelope.
    """

    def __init__(
        self,
        config: MqttConfig | None = None,
        *,
        client_factory: Callable[[], Any] | None = None,
        decoder: Callable[[bytes], TelemetrySample] | None = None,
        source_id: str = "mqtt",
        limits: SourceLimits | None = None,
        device_allowlist: Sequence[str] = (),
    ) -> None:
        super().__init__(source_id=source_id, limits=limits)
        self.config = config or MqttConfig()
        self._client_factory = client_factory
        self._decoder = decoder or decode_sofia_payload
        self._allowlist = tuple(device_allowlist)
        self._client: Any = None
        self._inbox: list[TelemetrySample] = []
        self._max_inbox = 10_000

    def _open(self) -> None:
        client = self._make_client()
        self._client = client
        self._attach_callbacks(client)
        username, password = self.config.credentials()
        if username is not None:
            self._set_credentials(client, username, password)
        if self.config.tls_enabled:
            self._configure_tls(client)
        try:
            self._connect(client)
        except Exception as exc:
            self.state = type(self.state).FAILED
            raise TelemetryError(
                f"MQTT connect to {self.config.host}:{self.config.port} failed: {exc}",
                details={"host": self.config.host, "port": self.config.port},
            ) from exc
        for topic in self.config.topics:
            self._subscribe(client, topic)

    # -- paho isolation ----------------------------------------------------

    def _make_client(self) -> Any:
        return make_mqtt_client(self.config.client_id, self._client_factory)

    def _attach_callbacks(self, client: Any) -> None:
        def on_message(_client: Any, _userdata: Any, message: Any) -> None:
            self._on_message(message)

        client.on_message = on_message

    def _set_credentials(self, client: Any, username: str | None, password: str | None) -> None:
        if hasattr(client, "username_pw_set"):
            client.username_pw_set(username, password)

    def _configure_tls(self, client: Any) -> None:
        if hasattr(client, "tls_set"):
            client.tls_set(ca_certs=self.config.ca_cert)

    def _connect(self, client: Any) -> None:
        client.connect(self.config.host, self.config.port, self.config.keepalive_s)
        if hasattr(client, "loop_start"):
            client.loop_start()

    def _subscribe(self, client: Any, topic: str) -> None:
        client.subscribe(topic, qos=self.config.qos)

    # -- data path ---------------------------------------------------------

    def _on_message(self, message: Any) -> None:
        payload = getattr(message, "payload", b"")
        size = len(payload)
        self.stats.bytes_read += size
        try:
            if size > self.limits.max_payload_bytes:
                raise PayloadTooLargeError(
                    f"MQTT payload of {size} bytes exceeds "
                    f"{self.limits.max_payload_bytes} byte limit",
                    details={"size": size},
                )
            sample = self._decoder(payload)
            if self._allowlist and sample.device_id not in self._allowlist:
                self.stats.records_rejected += 1
                return
            if len(self._inbox) >= self._max_inbox:
                self._inbox.pop(0)
                self.stats.records_rejected += 1
            self._inbox.append(sample)
            self.stats.records_read += 1
        except (TelemetryError, ValueError, KeyError, TypeError):
            self.stats.protocol_errors += 1
            self.stats.records_rejected += 1

    def read(self, *, max_records: int | None = None) -> list[TelemetrySample]:
        if self.state is not type(self.state).OPEN:
            raise TransportClosedError(
                f"MQTT source {self.source_id!r} is {self.state.value}",
                details={"state": self.state.value},
            )
        if self._client is not None and hasattr(self._client, "loop"):
            self._client.loop(timeout=0.0)
        count = len(self._inbox) if max_records is None else int(max_records)
        if count <= 0:
            return []
        out = self._inbox[:count]
        self._inbox = self._inbox[count:]
        self.stats.records_emitted += len(out)
        return out

    def _close(self) -> None:
        if self._client is None:
            return
        if hasattr(self._client, "loop_stop"):
            self._client.loop_stop()
        if hasattr(self._client, "disconnect"):
            self._client.disconnect()
        self._client = None

    @property
    def pending(self) -> int:
        return len(self._inbox)


class MqttPublishSink:
    """Bounded publisher for Sofia health events (optional extra)."""

    def __init__(
        self,
        config: MqttConfig,
        *,
        topic: str = "sofia/events",
        client_factory: Callable[[], Any] | None = None,
    ) -> None:
        self.config = config
        self.topic = topic
        self._client_factory = client_factory
        self._client: Any = None
        self.published = 0
        self.failures = 0

    def open(self) -> None:
        client = make_mqtt_client(self.config.client_id, self._client_factory)
        username, password = self.config.credentials()
        if username is not None and hasattr(client, "username_pw_set"):
            client.username_pw_set(username, password)
        if self.config.tls_enabled and hasattr(client, "tls_set"):
            client.tls_set(ca_certs=self.config.ca_cert)
        try:
            client.connect(self.config.host, self.config.port, self.config.keepalive_s)
            if hasattr(client, "loop_start"):
                client.loop_start()
        except Exception as exc:
            self.failures += 1
            raise TelemetryError(
                f"MQTT publish connect failed: {exc}",
                details={"host": self.config.host, "port": self.config.port},
            ) from exc
        self._client = client

    def publish(self, payload: dict[str, Any]) -> bool:
        if self._client is None:
            self.failures += 1
            return False
        try:
            data = json.dumps(payload, sort_keys=True, allow_nan=False).encode("utf-8")
            self._client.publish(self.topic, data, qos=self.config.qos)
        except (TypeError, ValueError):
            self.failures += 1
            return False
        else:
            self.published += 1
            return True

    def close(self) -> None:
        if self._client is not None and hasattr(self._client, "disconnect"):
            self._client.disconnect()
        self._client = None


def make_mqtt_client(client_id: str, factory: Callable[[], Any] | None = None) -> Any:
    """Create an MQTT client, importing ``paho`` lazily.

    Raises:
        TelemetryError: if ``paho-mqtt`` is absent and no factory is supplied.
    """
    if factory is not None:
        return factory()
    try:
        import paho.mqtt.client as mqtt
    except ImportError as exc:
        raise TelemetryError(
            "paho-mqtt is not installed. Install the 'mqtt' extra: "
            "pip install sofia-engine[mqtt]",
            details={"extra": "mqtt"},
        ) from exc
    return mqtt.Client(client_id=client_id)


def decode_sofia_payload(payload: bytes) -> TelemetrySample:
    """Decode the canonical Sofia MQTT JSON envelope into a sample.

    Accepted envelope::

        {"timestamp": 1700000000.0, "device_id": "...", "channel": "...",
         "value": 1.0, "unit": "g", "quality": "GOOD", "sequence_number": 1}

    Raises:
        TelemetryProtocolError: on any structural or semantic violation.
    """
    try:
        text = payload.decode("utf-8") if isinstance(payload, bytes) else str(payload)
    except UnicodeDecodeError as exc:
        raise TelemetryProtocolError("MQTT payload is not valid UTF-8", details={}) from exc
    try:
        record = json.loads(text)
    except ValueError as exc:
        raise TelemetryProtocolError("MQTT payload is not valid JSON", details={}) from exc
    if not isinstance(record, dict):
        raise TelemetryProtocolError("MQTT payload is not a JSON object", details={})
    if "value" not in record:
        raise TelemetryProtocolError("MQTT payload has no 'value' field", details={})
    unit = str(record.get("unit", "dimensionless"))
    if not is_known_unit(unit):
        raise TelemetryProtocolError(f"MQTT payload has unknown unit {unit!r}",
                                    details={"unit": unit})
    quality_raw = str(record.get("quality", "GOOD")).upper()
    try:
        quality = DataQuality(quality_raw)
    except ValueError as exc:
        raise TelemetryProtocolError(
            f"MQTT payload has unknown quality {quality_raw!r}", details={}
        ) from exc
    timestamp = record.get("timestamp")
    if timestamp is None:
        timestamp = time.time()
    return TelemetrySample(
        timestamp=float(timestamp),
        device_id=str(record.get("device_id", "unknown")),
        channel=str(record.get("channel", "value")),
        value=float(record["value"]),
        unit=unit,
        source="mqtt",
        quality=quality,
        sequence_number=None if record.get("sequence_number") is None
        else int(record["sequence_number"]),
        tags=dict(record.get("tags", {}) or {}),
    )
