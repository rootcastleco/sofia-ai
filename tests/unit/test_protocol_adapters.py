"""Unit tests for MQTT, Modbus and serial adapters with injected fakes.

No broker, PLC or serial port is required: every transport object is injected.
Integration against real hardware is NOT VERIFIED in this repository.
"""

from __future__ import annotations

import json
import struct

import pytest

from sofia_ai.core.errors import (
    ConfigurationError,
    PayloadTooLargeError,
    TelemetryError,
    TelemetryProtocolError,
)
from sofia_ai.core.quality import DataQuality
from sofia_ai.telemetry.base import SourceLimits
from sofia_ai.telemetry.modbus import (
    ModbusConfig,
    ModbusSource,
    RegisterSpec,
    RegisterType,
    decode_registers,
)
from sofia_ai.telemetry.mqtt import (
    MqttConfig,
    MqttPublishSink,
    MqttSource,
    decode_sofia_payload,
)
from sofia_ai.telemetry.serial_adapter import (
    SerialConfig,
    SerialSource,
    decode_delimited_record,
)

# ---------------------------------------------------------------------------
# MQTT
# ---------------------------------------------------------------------------


class FakeMqttClient:
    """Minimal stand-in for a paho client."""

    def __init__(self) -> None:
        self.on_message = None
        self.subscribed: list[str] = []
        self.connected = False
        self.published: list[tuple[str, bytes]] = []
        self.credentials = None

    def username_pw_set(self, username, password) -> None:
        self.credentials = (username, password)

    def tls_set(self, ca_certs=None) -> None:
        self.tls = ca_certs

    def connect(self, host, port, keepalive) -> None:
        self.connected = True

    def loop_start(self) -> None:
        pass

    def loop_stop(self) -> None:
        pass

    def loop(self, timeout=0.0) -> None:
        pass

    def subscribe(self, topic, qos=0) -> None:
        self.subscribed.append(topic)

    def publish(self, topic, payload, qos=0) -> None:
        self.published.append((topic, payload))

    def disconnect(self) -> None:
        self.connected = False


class FakeMessage:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload


def _envelope(**overrides) -> bytes:
    base = {"timestamp": 1_700_000_000.0, "device_id": "d", "channel": "c",
            "value": 1.5, "unit": "g"}
    base.update(overrides)
    return json.dumps(base).encode("utf-8")


class TestMqttPayload:
    def test_decodes_envelope(self) -> None:
        sample = decode_sofia_payload(_envelope())
        assert sample.value == 1.5
        assert sample.unit == "g"

    def test_rejects_non_json(self) -> None:
        with pytest.raises(TelemetryProtocolError):
            decode_sofia_payload(b"not json")

    def test_rejects_non_object(self) -> None:
        with pytest.raises(TelemetryProtocolError):
            decode_sofia_payload(b"[1,2]")

    def test_rejects_missing_value(self) -> None:
        with pytest.raises(TelemetryProtocolError):
            decode_sofia_payload(json.dumps({"device_id": "d"}).encode())

    def test_rejects_unknown_unit(self) -> None:
        with pytest.raises(TelemetryProtocolError):
            decode_sofia_payload(_envelope(unit="parsec"))

    def test_rejects_unknown_quality(self) -> None:
        with pytest.raises(TelemetryProtocolError):
            decode_sofia_payload(_envelope(quality="MAYBE"))

    def test_quality_is_parsed(self) -> None:
        assert decode_sofia_payload(_envelope(quality="STALE")).quality is DataQuality.STALE


class TestMqttConfig:
    def test_requires_host(self) -> None:
        with pytest.raises(ConfigurationError):
            MqttConfig(host="")

    def test_port_range(self) -> None:
        with pytest.raises(ConfigurationError):
            MqttConfig(port=0)

    def test_qos_range(self) -> None:
        with pytest.raises(ConfigurationError):
            MqttConfig(qos=5)

    def test_credentials_from_environment(self, monkeypatch) -> None:
        monkeypatch.setenv("SOFIA_MQTT_USER", "u")
        monkeypatch.setenv("SOFIA_MQTT_PASS", "p")
        config = MqttConfig(username_env="mqtt_user", password_env="mqtt_pass")
        assert config.credentials() == ("u", "p")

    def test_describe_hides_password(self) -> None:
        payload = MqttConfig(password_env="x").describe()
        assert payload["password_env"] == "***"


class TestMqttSource:
    def test_subscribe_and_decode(self) -> None:
        client = FakeMqttClient()
        source = MqttSource(MqttConfig(topics=("sofia/#",)), client_factory=lambda: client)
        source.open()
        assert client.subscribed == ["sofia/#"]
        client.on_message(client, None, FakeMessage(_envelope(value=2.5)))
        samples = source.read()
        assert len(samples) == 1
        assert samples[0].value == 2.5

    def test_oversized_payload_rejected(self) -> None:
        client = FakeMqttClient()
        source = MqttSource(client_factory=lambda: client,
                            limits=SourceLimits(max_payload_bytes=10))
        source.open()
        client.on_message(client, None, FakeMessage(b"x" * 100))
        assert source.read() == []
        assert source.stats.protocol_errors == 1

    def test_device_allowlist(self) -> None:
        client = FakeMqttClient()
        source = MqttSource(client_factory=lambda: client, device_allowlist=("allowed",))
        source.open()
        client.on_message(client, None, FakeMessage(_envelope(device_id="allowed")))
        client.on_message(client, None, FakeMessage(_envelope(device_id="other")))
        assert len(source.read()) == 1
        assert source.stats.records_rejected == 1

    def test_connect_failure_is_typed(self) -> None:
        class FailClient(FakeMqttClient):
            def connect(self, host, port, keepalive):
                raise OSError("no route to host")

        source = MqttSource(client_factory=FailClient)
        with pytest.raises(TelemetryError):
            source.open()
        assert source.state.value == "FAILED"

    def test_read_when_not_open(self) -> None:
        source = MqttSource(client_factory=FakeMqttClient)
        with pytest.raises(TelemetryError):
            source.read()

    def test_inbox_is_bounded(self) -> None:
        client = FakeMqttClient()
        source = MqttSource(client_factory=lambda: client)
        source.open()
        source._max_inbox = 3
        for _ in range(10):
            client.on_message(client, None, FakeMessage(_envelope()))
        assert source.pending == 3

    def test_publish_sink(self) -> None:
        client = FakeMqttClient()
        sink = MqttPublishSink(MqttConfig(), topic="sofia/events",
                                client_factory=lambda: client)
        sink.open()
        assert sink.publish({"a": 1})
        assert sink.published == 1
        assert client.published[0][0] == "sofia/events"
        sink.close()

    def test_publish_without_client(self) -> None:
        sink = MqttPublishSink(MqttConfig())
        assert not sink.publish({"a": 1})
        assert sink.failures == 1

    def test_missing_extra_raises_typed_error(self) -> None:
        source = MqttSource()
        with pytest.raises(TelemetryError):
            source.open()


# ---------------------------------------------------------------------------
# Modbus
# ---------------------------------------------------------------------------


class FakeModbusResponse:
    def __init__(self, registers, is_error=False) -> None:
        self.registers = registers
        self._is_error = is_error

    def isError(self) -> bool:
        return self._is_error


class FakeModbusClient:
    def __init__(self, registers=None, error=False, short=False) -> None:
        self.registers = registers or [0, 0]
        self.error = error
        self.short = short
        self.calls: list[tuple[int, int]] = []

    def read_holding_registers(self, address, count=1, unit=1):
        self.calls.append((address, count))
        if self.error:
            return FakeModbusResponse([], is_error=True)
        if self.short:
            return FakeModbusResponse([])
        return FakeModbusResponse(self.registers[:count])

    def close(self) -> None:
        pass


class TestModbusDecoding:
    def test_uint16(self) -> None:
        spec = RegisterSpec(name="t", address=0, register_type=RegisterType.UINT16)
        assert decode_registers([42], spec) == 42.0

    def test_int16_negative(self) -> None:
        spec = RegisterSpec(name="t", address=0, register_type=RegisterType.INT16)
        assert decode_registers([0xFFFF], spec) == -1.0

    def test_float32(self) -> None:
        spec = RegisterSpec(name="t", address=0, register_type=RegisterType.FLOAT32)
        packed = struct.pack(">f", 1.5)
        registers = struct.unpack(">HH", packed)
        assert decode_registers(list(registers), spec) == pytest.approx(1.5, rel=1e-6)

    def test_short_frame_is_protocol_error(self) -> None:
        spec = RegisterSpec(name="t", address=0, register_type=RegisterType.FLOAT32)
        with pytest.raises(TelemetryProtocolError):
            decode_registers([1], spec)

    def test_scale_and_offset(self) -> None:
        spec = RegisterSpec(name="t", address=0, register_type=RegisterType.UINT16,
                            scale=0.1, offset=-40.0)
        assert decode_registers([250], spec) == pytest.approx(-15.0)

    def test_address_range(self) -> None:
        with pytest.raises(ConfigurationError):
            RegisterSpec(name="t", address=0x10000)


class TestModbusSource:
    def _source(self, **kwargs) -> ModbusSource:
        registers = (RegisterSpec(name="temp", address=0, register_type=RegisterType.UINT16,
                                  unit="C"),)
        return ModbusSource(registers=registers, device_id="plc-01", **kwargs)

    def test_reads_declared_registers(self) -> None:
        client = FakeModbusClient(registers=[125])
        source = self._source(client_factory=lambda _cfg: client)
        source.open()
        samples = source.read()
        assert len(samples) == 1
        assert samples[0].value == 125.0
        assert samples[0].unit == "C"

    def test_short_frame_is_typed_error(self) -> None:
        source = self._source(client_factory=lambda _cfg: FakeModbusClient(short=True))
        source.open()
        with pytest.raises(TelemetryProtocolError):
            source.read()
        assert source.state.value == "DEGRADED"

    def test_error_response_is_typed_error(self) -> None:
        source = self._source(client_factory=lambda _cfg: FakeModbusClient(error=True))
        source.open()
        with pytest.raises(TelemetryProtocolError):
            source.read()

    def test_transport_exception_is_normalized(self) -> None:
        class Broken:
            def read_holding_registers(self, address, count=1, unit=1):
                raise OSError("timeout")

            def close(self):
                pass

        source = self._source(client_factory=lambda _cfg: Broken())
        source.open()
        with pytest.raises(TelemetryError):
            source.read()
        assert source.state.value == "FAILED"

    def test_requires_registers(self) -> None:
        with pytest.raises(ConfigurationError):
            ModbusSource(registers=())

    def test_config_validation(self) -> None:
        with pytest.raises(ConfigurationError):
            ModbusConfig(unit_id=300)
        with pytest.raises(ConfigurationError):
            ModbusConfig(mode="can")

    def test_describe(self) -> None:
        source = self._source(client_factory=lambda _cfg: FakeModbusClient())
        source.open()
        payload = source.describe()
        assert payload["channels"] == ["temp"]

    def test_missing_extra_raises_typed_error(self) -> None:
        source = self._source()
        with pytest.raises(TelemetryError):
            source.open()


# ---------------------------------------------------------------------------
# Serial
# ---------------------------------------------------------------------------


class FakeSerialPort:
    def __init__(self, chunks) -> None:
        self._chunks = list(chunks)
        self.closed = False

    @property
    def in_waiting(self) -> int:
        return len(self._chunks[0]) if self._chunks else 0

    def read(self, size=1) -> bytes:
        if not self._chunks:
            return b""
        return self._chunks.pop(0)

    def close(self) -> None:
        self.closed = True


class TestSerialDecoding:
    def test_json_frame(self) -> None:
        payload = json.dumps({"timestamp": 1.7e9, "device_id": "d", "channel": "c",
                              "value": 2.0, "unit": "g"}).encode()
        sample = decode_delimited_record(payload)
        assert sample.value == 2.0

    def test_csv_frame(self) -> None:
        sample = decode_delimited_record(b"dev,chan,3.5,mm/s")
        assert sample.value == 3.5
        assert sample.unit == "mm/s"

    def test_rejects_bad_csv_field_count(self) -> None:
        with pytest.raises(TelemetryProtocolError):
            decode_delimited_record(b"a,b,c")

    def test_rejects_unknown_unit(self) -> None:
        with pytest.raises(TelemetryProtocolError):
            decode_delimited_record(b"a,b,1.0,parsec")

    def test_rejects_non_numeric(self) -> None:
        with pytest.raises(TelemetryProtocolError):
            decode_delimited_record(b"a,b,xyz,g")

    def test_rejects_empty(self) -> None:
        with pytest.raises(TelemetryProtocolError):
            decode_delimited_record(b"   ")


class TestSerialSource:
    def test_reads_framed_records(self) -> None:
        frame = json.dumps({"timestamp": 1.7e9, "device_id": "d", "channel": "c",
                            "value": 1.0, "unit": "g"}).encode() + b"\n"
        port = FakeSerialPort([frame, frame])
        source = SerialSource(SerialConfig(), port_factory=lambda _cfg: port)
        source.open()
        samples = source.read()
        assert len(samples) == 2

    def test_bad_frame_is_counted_not_raised(self) -> None:
        port = FakeSerialPort([b"{bad}\n", json.dumps(
            {"timestamp": 1.7e9, "device_id": "d", "channel": "c", "value": 1.0,
             "unit": "g"}).encode() + b"\n"])
        source = SerialSource(SerialConfig(), port_factory=lambda _cfg: port)
        source.open()
        samples = source.read()
        assert len(samples) == 1
        assert source.stats.protocol_errors == 1

    def test_timeout_is_not_a_transport_failure(self) -> None:
        """No data available must not crash the pipeline."""
        port = FakeSerialPort([])
        source = SerialSource(SerialConfig(), port_factory=lambda _cfg: port)
        source.open()
        assert source.read() == []
        assert source.read_timeout_count >= 1

    def test_oversized_frame_rejected(self) -> None:
        port = FakeSerialPort([b"a" * 100])
        source = SerialSource(SerialConfig(max_frame_bytes=10),
                              port_factory=lambda _cfg: port)
        source.open()
        with pytest.raises(PayloadTooLargeError):
            source.read()

    def test_transport_exception_normalized(self) -> None:
        class Broken:
            in_waiting = 1

            def read(self, size=1):
                raise OSError("device disconnected")

            def close(self):
                pass

        source = SerialSource(SerialConfig(), port_factory=lambda _cfg: Broken())
        source.open()
        with pytest.raises(TelemetryError):
            source.read()
        assert source.state.value == "FAILED"

    def test_config_validation(self) -> None:
        with pytest.raises(ConfigurationError):
            SerialConfig(port="")
        with pytest.raises(ConfigurationError):
            SerialConfig(baudrate=0)
        with pytest.raises(ConfigurationError):
            SerialConfig(delimiter=b"")

    def test_close_clears_buffer(self) -> None:
        port = FakeSerialPort([])
        source = SerialSource(SerialConfig(), port_factory=lambda _cfg: port)
        source.open()
        source.close()
        assert port.closed
