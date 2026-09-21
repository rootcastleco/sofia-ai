# Telemetry Adapters & Protocols

Sofia Engine ingests data through a pluggable transport layer (`sofia_ai.telemetry`). The core runtime never imports transport libraries directly; all protocol clients are implemented as modular adapters implementing the `TelemetrySource` contract.

---

## 1. The `TelemetrySource` Contract

Every telemetry adapter inherits from `TelemetrySource` (`sofia_ai.telemetry.base`) and implements a standard lifecycle:

```python
from abc import ABC, abstractmethod
from sofia_ai.core.contracts import TelemetrySample

class TelemetrySource(ABC):
    @abstractmethod
    def open(self) -> None:
        """Establish network connections, open file handles, or configure ports."""

    @abstractmethod
    def read(self, *, max_records: int | None = None) -> list[TelemetrySample]:
        """Fetch up to max_records bounded samples without blocking indefinitely."""

    @abstractmethod
    def close(self) -> None:
        """Cleanly release sockets, serial descriptors, and file locks."""

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
```

---

## 2. Built-in Adapters (Standard Library / NumPy Only)

These adapters are always available without installing optional extras:

### 1. `CsvSource`
Reads tabular telemetry with column mapping, timestamp parsing, and NaN checks:
```python
from sofia_ai.telemetry import CsvSource

source = CsvSource(
    "examples/data/vibration.csv",
    device_id="pump-01",
    channel="vibration_x",
    unit="g",
    sample_rate=1000.0
)
with source:
    samples = source.read(max_records=1024)
```

### 2. `JsonlSource`
Streams newline-delimited JSON with strict schema validation:
```python
from sofia_ai.telemetry import JsonlSource

source = JsonlSource("examples/data/edge_offline.jsonl")
with source:
    samples = source.read(max_records=512)
```

### 3. `ReplaySource`
Replays pre-recorded telemetry files with configurable rate multipliers (e.g. `rate=2.0` for 2x real-time) or deterministic step-by-step advancement for reproducible benchmarking.

### 4. `SyntheticTelemetrySource`
Generates deterministic synthetic vibration, temperature, and speed signals using injected random generators (`numpy.random.Generator`) for automated testing and CI pipelines.

---

## 3. Industrial Protocol Adapters (Optional Extras)

Install protocol support via `pip install "sofia-engine[industrial]"`:

### 1. MQTT Adapter (`MqttSource`)
Designed for IoT gateways and broker-based IIoT topologies:
* **TLS Encryption**: Enforces TLS 1.3 with CA certificate verification.
* **Credentials**: Read from environment or secure keyring via `sofia_ai.security.secrets`.
* **Payload Limit**: Frames exceeding `max_payload_bytes` (default 1 MiB) are dropped to prevent memory exhaustion attacks (Threat T-06).

### 2. Modbus TCP & RTU (`ModbusSource`)
Direct PLC and sensor polling over Ethernet or RS-485:
* Supports Modbus Function Codes:
  - FC 01: Read Coils
  - FC 02: Read Discrete Inputs
  - FC 03: Read Holding Registers
  - FC 04: Read Input Registers
* **Endianness Support**: Configurable 16-bit, 32-bit big-endian, little-endian, and byte-swapped representations for float/integer conversion.

### 3. Serial / RS-485 (`SerialSource`)
Direct microcontroller or vibration transmitter communication:
* Configurable baud rate, parity, stop bits, and timeout.
* Frame delimiters and checksum/CRC validation before sample parsing.

---

## 4. Custom Adapter Implementation & Plugin Registry

To add a proprietary protocol (e.g. CAN bus, OPC-UA, or EtherCAT):

```python
from sofia_ai.telemetry import TelemetrySource, register_source
from sofia_ai.core.contracts import TelemetrySample, DataQuality

class CanBusSource(TelemetrySource):
    def __init__(self, interface: str = "can0", **kwargs):
        super().__init__(source_id=f"can_{interface}", **kwargs)
        self.interface = interface
        self._bus = None

    def open(self) -> None:
        # Initialize CAN interface
        pass

    def read(self, *, max_records: int | None = None) -> list[TelemetrySample]:
        # Read CAN frames and construct strongly-typed TelemetrySample objects
        return []

    def close(self) -> None:
        # Close CAN socket
        pass

# Register in Sofia's global plugin registry
register_source("can", CanBusSource)
```
