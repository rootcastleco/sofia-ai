# Sofia Engine Runtime v3 — Architecture

> **Author:** Rootcastle Engineering & Innovation  
> **Status:** Approved Architectural Blueprint

---

## 1. High-Level Architectural Flow

```
+---------------------------------------------------------------------------------------+
| 1. Ingestion & Protocol Adapters (Optional / Isolated)                                |
|    MQTT / Modbus / Serial / CSV / Synthetic (Core has zero transport imports)          |
+---------------------------------------------------------------------------------------+
                                           |
                                           v
+---------------------------------------------------------------------------------------+
| 2. Validation & Quality Layer (Core: contracts.py, quality.py, units.py)             |
|    - Typed primitives: Sample, SignalFrame, SignalMetadata                            |
|    - Finite float verification (rejection of NaN/Inf)                                 |
|    - SignalQuality tagging (GOOD, DEGRADED, STALE, MISSING, INVALID)                  |
+---------------------------------------------------------------------------------------+
                                           |
                                           v
+---------------------------------------------------------------------------------------+
| 3. Bounded Signal Buffers (Core: ring_buffer.py)                                      |
|    - Deterministic memory ceilings (e.g. 4096 samples, Drop-Oldest)                   |
|    - Sample-index time axis (never derived from wall clock intervals)                  |
+---------------------------------------------------------------------------------------+
                                           |
                                           v
+---------------------------------------------------------------------------------------+
| 4. Scientific DSP Runtime (Core: sofia_ai.signal)                                     |
|    - Vibration: Parseval-conserving Welch PSD, Hilbert analytic envelope, kinematics  |
|    - Electrical: IEEE 519 harmonics, Fortescue 3-phase symmetrical components, power  |
|    - Acoustic: ASTM E1316 AE energy, ringdown, cavitation indexing                    |
|    - IMU: 3-axis acceleration magnitude, dynamic tilt (pitch/roll), derivative jerk   |
+---------------------------------------------------------------------------------------+
                                           |
                                           v
+---------------------------------------------------------------------------------------+
| 5. Versioned Feature Runtime (Core: sofia_ai.features)                                |
|    - Ordered, immutable, versioned FeatureVector containing Feature primitives        |
|    - Schema validation and schema compatibility checking                              |
+---------------------------------------------------------------------------------------+
                                           |
                    +----------------------+----------------------+
                    |                                             |
                    v                                             v
+---------------------------------------+   +---------------------------------------+
| 6a. Statistical & ML Backends         |   | 6b. Sofia Assembly VM & Self-Training |
|     - Robust MAD, EWMA, CUSUM         |   |     - SofiaAsmVM (Register bytecode)  |
|     - Optional ONNX/Torch (Isolated)  |   |     - Full backprop & SGD updates     |
+---------------------------------------+   +---------------------------------------+
                    |                                             |
                    +----------------------+----------------------+
                                           |
                                           v
+---------------------------------------------------------------------------------------+
| 7. Evidence Fusion & Diagnostics (Core: sofia_ai.diagnostics)                         |
|    - DiagnosticEvidence records with physical metrics and reference values            |
|    - Evidence-scaled confidence and uncertainty estimation                            |
|    - HealthEvent & HealthScore computation (0-100 with uncertainty band)              |
+---------------------------------------------------------------------------------------+
                                           |
                    +----------------------+----------------------+
                    |                                             |
                    v                                             v
+---------------------------------------+   +---------------------------------------+
| 8. Safety Gate & Policy Engine        |   | 9. Advisory AI Copilot                |
|    - DEFAULT DENY state machine       |   |    - NVIDIA NIM / OpenRouter / Local  |
|    - Interlocks & operator gates      |   |    - Structured context builder       |
|    - Replay defense (Nonce + TTL)     |   |    - STRICTLY ADVISORY (No actuation) |
|    - Audited CommandDecision          |   +---------------------------------------+
+---------------------------------------+
                    |
                    v
+---------------------------------------------------------------------------------------+
| 10. Actuator / Industrial Control Output (Behind strict physical safety interlocks)  |
+---------------------------------------------------------------------------------------+
```

---

## 2. Layering & Dependency Rules

1. **`sofia-core` Dependency Envelope:**
   - **Allowed:** `numpy`, standard Python libraries (`math`, `time`, `typing`, `dataclasses`, `enum`, `json`, `hashlib`, `urllib`).
   - **Forbidden:** `torch`, `onnxruntime`, `fastapi`, `paho-mqtt`, `pymodbus`, `pyserial`, `requests`.
2. **Adapter Boundaries:**
   - Protocols (`telemetry/mqtt.py`, `telemetry/modbus.py`) import `sofia_ai.core.contracts`. Core never imports protocol modules.
   - Machine Learning Backends (`inference/onnx_backend.py`, `inference/torch_backend.py`) are lazily loaded or injected; failure to import optional backends raises `BackendUnavailableError` without preventing core operation.
3. **Cross-Language Equivalents:**
   - **Python:** High-level scientific runtime, feature curation, automated training, CLI.
   - **TypeScript (`@rootcastle/sofia-engine`):** Edge gateway & Node.js runtime, zero dependencies, pure TypedArray math.
   - **C99 (`embedded/`):** Bare-metal microcontroller runtime, static allocations, fixed-point Q16.16 and single-precision float.
