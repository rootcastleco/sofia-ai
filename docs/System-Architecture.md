# System Architecture

Sofia Engine is engineered as an offline-first, modular edge runtime for industrial telemetry, condition monitoring, digital signal processing (DSP), and safe technical automation. This document describes the runtime components, data flows, dependency constraints, and failure containment mechanisms.

---

## 1. High-Level Subsystems

The runtime consists of six strictly decoupled layers:

```
                                   SOFIA ENGINE
                                        |
                            +-----------------------+
                            |     Engine Runtime    |
                            |   (edge/runtime.py)   |
                            +-----------------------+
                                        |
        +-------------------------------+-------------------------------+
        |                               |                               |
  TELEMETRY LAYER               SIGNAL PIPELINE                  MODEL RUNTIME
  telemetry/                    signal/ + features/               inference/
        |                               |                               |
  TelemetrySource (ABC)          windowing                     ModelBackend (ABC)
   |- CsvSource                  filters                        |- ThresholdDetector
   |- JsonlSource                resample                       |- ZScoreDetector
   |- SyntheticSource            detrend                        |- MadDetector
   |- ReplaySource               envelope                       |- EwmaDetector
   |- MqttSource      [extra]    spectral                       |- CusumDetector
   |- ModbusSource    [extra]    -----                          |- OnnxBackend  [extra]
   |- SerialSource    [extra]    statistical features           |- TorchBackend [extra]
   |- registry (plugins)         spectral features              |- registry (plugins)
        |                        rotating-machinery ext              |
        |                               |                            |
        +-------------------------------+----------------------------+
                                        |
                                 DECISION ENGINE
                            diagnostics/ + decision/
                                        |
                     evidence -> severity -> confidence -> uncertainty
                                        |
                            +-----------+-----------+
                            |                       |
                      HealthEvent            CommandRequest
                            |                       |
                            |                  PolicyEngine
                            |                  (default DENY)
                            |                       |
                            |              +--------+--------+
                            |            DENY             APPROVE
                            |              |                 |
                            v              v                 v
        +-------------------+---------------+-----------------+----------------+
        |                   |               |                                  |
       API                 CLI             Observability                   Edge runtime
  (FastAPI extra)      cli/main.py    observability/                  edge/{buffer,
        |                   |         logging, metrics,               store_forward,
        |                   |            health                        reconnect}
        |                   |                                                    |
        +-------------------+----------------------------------------------------+
                                        |
                                 Embedded Export
                            embedded/ (C99 reference runtime,
                            fixed-point Q16.16, golden vectors)
```

---

## 2. Core Contracts & Data Model

All telemetry samples, signal windows, feature vectors, inference results, and command requests use strongly typed dataclasses defined in `sofia_ai.core.contracts`:

1. **`TelemetrySample`**:
   - `timestamp: float` (wall-clock event time in seconds)
   - `device_id: str` (unique asset identifier)
   - `channel: str` (sensor channel, e.g. `vibration_x`)
   - `value: float` (physical measurement, validated $\notin \{\text{NaN}, \pm\infty\}$)
   - `unit: str` (physical unit, e.g. `g`, `mm/s`, `RPM`)
   - `quality: DataQuality` (`GOOD`, `STALE`, `MISSING`, `INVALID`, `OUT_OF_RANGE`, `DUPLICATE`)
   - `sequence_number: int` (monotonic counter for replay detection)
   - `ingestion_time: float` (local monotonic processing timestamp)

2. **`SignalWindow`**:
   - Fixed-length array of contiguous samples aligned by index, carrying sample rate $f_s$, start timestamp, and quality mask.

3. **`FeatureVector`**:
   - Named, ordered vector of physical and spectral metrics (deterministic tuple ordering, never derived from dictionary key order).
   - Carries `extractor_id` and semantic `extractor_version`.

4. **`InferenceResult`**:
   - Carries `model_id`, `model_version`, anomaly classification outcome (`NORMAL`, `ANOMALY`), numeric `score`, `confidence` $\in [0, 1]$, and `uncertainty` $\in [0, 1]$.

5. **`HealthEvent`**:
   - Produced by the `DiagnosticEngine`. Maps evidence, severity (`NORMAL`, `LOW`, `MEDIUM`, `HIGH`, `CRITICAL`), confidence, and human-readable recommendations.

6. **`CommandRequest` / `CommandDecision`**:
   - Formal actuation intents evaluated by the `PolicyEngine`.

---

## 3. Module Boundaries & Forbidden Dependencies

Sofia enforces strict architectural isolation via automated AST checks in `tests/architecture/test_boundaries.py`:

| Module | Purpose | Allowed Imports | Forbidden Imports |
|---|---|---|---|
| `core/` | Contracts, units, time, quality, errors | `stdlib`, `numpy` | Any other Sofia module, any protocol, any ML framework |
| `signal/` | Pure DSP algorithms | `core`, `numpy` | `telemetry`, `inference`, `torch`, `onnxruntime` |
| `features/` | Feature extraction | `core`, `signal`, `numpy` | `inference`, `decision`, `telemetry` |
| `inference/` | Detectors & ML backends | `core`, `features`, `security` | `telemetry`, `decision`, `copilot` |
| `diagnostics/` | Evidence fusion & health scoring | `core`, `features`, `inference` | `decision`, `telemetry`, `copilot` |
| `decision/` | Safety gate (`PolicyEngine`) | `core` | `inference`, `copilot`, `torch` |
| `edge/` | Buffers, store-forward, runtime | all core modules | none |

### Absolute Invariants:
1. **Core isolation**: `core/` imports nothing from any other Sofia module.
2. **Zero framework pollution**: No core module imports `torch`, `onnxruntime`, `paho.mqtt`, `pymodbus`, `pyserial`, or `fastapi`.
3. **No insecure deserialization**: `pickle.load` / `pickle.loads` are forbidden across the codebase.
4. **No global seeding**: Neither `random.seed` nor `np.random.seed` are called globally; RNGs are injected.

---

## 4. Determinism & Timing Model

Industrial reproducibility requires exact determinism:
* **Pure DSP Functions**: Every signal routine satisfies $f(\mathbf{x}, \text{config}) \to \mathbf{y}$ with no internal clocks or hidden state.
* **Injected Time Sources**: Clocks are accessed strictly through `TimeSource` interfaces (e.g. `FixedTimeSource` for test suites).
* **Sample-Rate Time Axis**: All frequency and time-domain DSP coordinates derive from discrete sample indices $n$ and sampling rate $f_s$ ($t = n / f_s$), never from operating system wall-clock deltas.

---

## 5. Bounded Resource Envelopes

Sofia Engine declares explicit ceilings for all dynamic structures:

| Structure | Default Ceiling | Config Key | Overflow Behavior |
|---|---|---|---|
| **Ingest Buffer** | 4,096 samples | `buffer.max_samples` | Drop oldest sample, increment drop counter |
| **Window Length** | 1,024 samples | `window.length` | Bounded sliding window |
| **Store-and-Forward File** | 64 MiB | `store_forward.max_bytes` | File rotation, oldest log archived/evicted |
| **Reconnect Attempts** | 8 retries | `reconnect.max_attempts` | Exponential backoff capped at 30 s |
| **Payload Size** | 1 MiB | `transport.max_payload_bytes` | Reject frame immediately |
| **Metric History** | 10,000 samples | `metrics.max_samples` | Ring buffer FIFO eviction |

---

## 6. Failure Modes & Fault Containment

The engine declares three standard failure classifications:
1. **`REJECT`**: Malformed or non-finite inputs ($\text{NaN}, \pm\infty$). The sample is rejected, counted as `malformed`, and logged once per window.
2. **`DEGRADE`**: Incomplete or stale data. Processing continues, but data quality is tagged `STALE` or `ESTIMATED`, degrading diagnostic confidence.
3. **`FAIL`**: Component or transport failure (e.g., disconnected sensor bus). The error raises a typed `TelemetryError`, containment logic logs the event, and the edge runtime maintains pipeline continuity without crashing.
