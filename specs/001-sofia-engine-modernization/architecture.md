# SOFIA ENGINE — Architecture

## 1. System diagram

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
        API              CLI             Observability                   Edge runtime
   (FastAPI extra)   cli/main.py    observability/                  edge/{buffer,
        |                |          logging, metrics,               store_forward,
        |                |             health                        reconnect}
        |                |                                                    |
        +----------------+----------------------------------------------------+
                                        |
                                 Embedded export
                            embedded/ (C99 reference runtime,
                            fixed-point, golden vectors)
```

## 2. Module boundaries and dependency rules

`src/sofia_ai/` layout:

| Module | Responsibility | May import |
|---|---|---|
| `core/` | Contracts, units, time, quality, validation, errors, serialization, hashing | stdlib, numpy |
| `telemetry/` | Transport adapters + `TelemetrySource` ABC + registry | `core`, `observability`, `security` |
| `signal/` | Pure DSP: windowing, filters, spectral, resample, detrend, envelope | `core`, numpy |
| `features/` | Deterministic feature extraction over windows | `core`, `signal` |
| `inference/` | `ModelBackend` ABC, detectors, manifests, registry, optional backends | `core`, `features`, `security` |
| `diagnostics/` | Evidence, rules, health events, health scoring | `core`, `features`, `inference` |
| `decision/` | Command requests + policy engine (default DENY) | `core` |
| `edge/` | Bounded buffers, store-and-forward, reconnect, runtime, health | everything above |
| `learning/rl/` | Experimental RL (training/simulation only) | `core`, torch (optional) |
| `copilot/` | Optional engineering assistant; evidence consumers only | `core`, `diagnostics` |
| `observability/` | Structured logging, metrics, redaction | `core` |
| `security/` | Safe IO, signing, secrets | `core` |
| `config/` | Typed, validated, versioned configuration | `core` |
| `cli/` | Command-line surface | all |
| `experimental/quantum/` | Quantum-inspired simulation, quarantined | `core`, numpy |
| `compat/` | Deprecation shims for pre-2.0 API | all |

Hard rules (enforced by `tests/architecture/test_boundaries.py`):

1. `core/` imports nothing from any other Sofia module.
2. Nothing in `core/`, `signal/`, `features/`, `inference/`, `diagnostics/`, `decision/` imports
   `torch`, `onnxruntime`, `paho.mqtt`, `pymodbus`, `pyserial`, or `fastapi`.
3. `experimental/` is imported by nothing outside `experimental/` and `compat/`.
4. `copilot/` imports no actuation surface.
5. No module calls `pickle.load` / `pickle.loads`.
6. No module calls `random.seed` / `np.random.seed` at import or construction time.

## 3. Data flow

```
sensor/PLC/file  --TelemetrySample-->  bounded ingest buffer (N samples, drop-oldest)
                                              |
                                     window assembly
                                   (length L, hop H, aligned to sample index)
                                              |
                                    validation + quality marking
                                   (NaN/inf -> INVALID, gap -> MISSING,
                                    reorder -> flagged, dup -> DUPLICATE)
                                              |
                              +---------------+----------------+
                              |                                |
                     statistical features              spectral features
                        (time domain)                  (FFT/PSD/Welch)
                              |                                |
                              +---------------+----------------+
                                              |
                                       FeatureVector
                                       (named, ordered, versioned)
                                              |
                                      ModelBackend.infer()
                                              |
                                       InferenceResult
                                   (score, confidence, uncertainty)
                                              |
                                     DiagnosticEngine
                                   (rules -> evidence -> severity)
                                              |
                                        HealthEvent
                                              |
                            +-----------------+-----------------+
                            |                                   |
                      copilot summary                     policy engine
                     (engineer-facing,                    (command only,
                      non-authoritative)                   default DENY)
```

## 4. Determinism model

* DSP functions are pure: `f(ndarray, config) -> ndarray`, no clocks, no RNG.
* RNG ownership: components that need randomness receive an injected
  `numpy.random.Generator`. Nothing seeds globals.
* Time is injected via `TimeSource`. Tests use `FixedTimeSource`.
* Feature ordering is a fixed tuple, never derived from dict iteration.
* JSON serialization uses sorted keys and fixed float formatting.

## 5. Resource ceilings

| Component | Ceiling | Config key |
|---|---|---|
| Ingest buffer | samples (default 4096) | `buffer.max_samples` |
| Window length | samples (default 1024) | `window.length` |
| Store-and-forward file | bytes (default 64 MiB) | `store_forward.max_bytes` |
| Reconnect attempts | count (default 8) + backoff ceiling (30 s) | `reconnect.*` |
| Payload size | bytes (default 1 MiB) | `transport.max_payload_bytes` |
| Log rate | records/min/key (default 30) | `logging.rate_limit_per_min` |
| Metric history | samples (default 10 000) | `metrics.max_samples` |
| Quantum dimension | `2 ** qubits`, hard ceiling 2**12 | `experimental/quantum` |

## 6. Failure model

Every stage declares one of:

* `REJECT` — input invalid; drop, count `malformed`, log once per key per window.
* `DEGRADE` — partial input; process with reduced confidence, mark quality.
* `FAIL` — component error; typed exception, counted, pipeline continues.

The pipeline must never exit on a single sample. Only a fatal configuration error
(validated at load time) may prevent startup.

## 7. Embedded boundary

```
Python:  features/extractor.py  --FeatureContract-->  tools/gen_golden_vectors.py
                                                              |
                                                     golden_vectors.json
                                                              |
C99:     embedded/src/sofia_features.c  <-- compares against golden vectors
         embedded/src/sofia_fixed.h     (Q16.16 preprocessing, documented range)
```

The C runtime is standalone: no heap allocation after `sofia_features_init`, no recursion,
fixed-size buffers, all externally-supplied lengths validated. CMSIS-DSP mapping and
TFLite Micro integration are documented as guidance only and are **NOT VERIFIED** in this
repository (no toolchain or target hardware is present in CI).
