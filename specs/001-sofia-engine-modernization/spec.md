# SOFIA ENGINE — Modernization Specification

**Spec ID:** `001-sofia-engine-modernization`
**Status:** Implemented
**Repository:** `https://github.com/rootcastleco/sofia-rl`
**Owner:** Rootcastle Engineering & Innovation
**License:** Apache License 2.0 (unchanged; see `LICENSE`)
**Date:** 2026-09-21

---

## 1. Purpose

Turn `rootcastleco/sofia-rl` from an experimental AI repository into a modular, offline-first,
evidence-producing edge intelligence engine for industrial telemetry, machine health, embedded ML
and bounded technical automation.

Sofia is **not** a chatbot. It converts machine telemetry into structured engineering evidence
using deterministic signal processing, pluggable inference backends and explicit safety
boundaries.

## 2. Non-negotiables

| ID | Rule |
|---|---|
| SOFIA-NFR-001 | Deterministic: same input + same config + same seed ⇒ byte-identical output |
| SOFIA-NFR-002 | Offline-first: no network access is required for any core capability |
| SOFIA-NFR-003 | Bounded: every buffer, queue, window and retention structure has an explicit ceiling |
| SOFIA-NFR-004 | Honest: no published performance/accuracy claim without a reproducible artifact in-repo |
| SOFIA-NFR-005 | Safe: no code path converts inference output into machinery actuation without the policy layer |
| SOFIA-NFR-006 | Framework-independent: core must not import PyTorch, ONNX Runtime, or a protocol client |
| SOFIA-NFR-007 | Fail-safe: unrecoverable component failure degrades to `DEGRADED`, never to silent wrong output |
| SOFIA-NFR-008 | No functional-safety certification is claimed (IEC 61508 / ISO 13849 / IEC 62443) |

---

## 3. Functional requirements

### 3.1 Core contracts — `SOFIA-FR-*`

| ID | Requirement | Implementation | Test |
|---|---|---|---|
| SOFIA-FR-001 | `TelemetrySample` carries timestamp, device_id, source, channel, value, unit, quality, sequence_number, tags, ingestion_time | `core/contracts.py` | `tests/unit/test_contracts.py` |
| SOFIA-FR-002 | `TelemetryFrame` groups samples by (device, time) with bounded length | `core/contracts.py` | `tests/unit/test_contracts.py` |
| SOFIA-FR-003 | `SignalWindow` carries values, sample_rate, start_index, start_time, channel, unit, quality mask | `core/contracts.py` | `tests/unit/test_contracts.py` |
| SOFIA-FR-004 | `FeatureVector` carries ordered names, values, extractor id and version | `core/contracts.py` | `tests/unit/test_contracts.py` |
| SOFIA-FR-005 | `InferenceResult` exposes model_id, model_version, output, score, confidence, uncertainty, latency, evidence, timestamp | `inference/base.py` | `tests/unit/test_inference.py` |
| SOFIA-FR-006 | `DiagnosticEvidence` records source, metric, observed value, reference value, description, weight | `diagnostics/evidence.py` | `tests/unit/test_diagnostics.py` |
| SOFIA-FR-007 | `HealthEvent` exposes severity, event_type, device_id, channels, evidence, confidence, recommendation, diagnostic_source, rule_version, timestamp, event_id | `diagnostics/health.py` | `tests/unit/test_diagnostics.py` |
| SOFIA-FR-008 | `MachineState` enumerates operating states as an explicit state machine input | `core/contracts.py` | `tests/unit/test_contracts.py` |
| SOFIA-FR-009 | `CommandRequest` / `CommandDecision` model actuation intents and policy verdicts | `decision/commands.py` | `tests/unit/test_policy.py` |
| SOFIA-FR-010 | `ModelManifest` carries model id, semantic version, input/output schema, preprocessing version, checksum, created_at, compatibility, optional signature | `inference/manifest.py` | `tests/unit/test_manifest.py` |
| SOFIA-FR-011 | All contracts serialize to JSON round-trip identically | `core/serialization.py` | `tests/unit/test_serialization.py` |
| SOFIA-FR-012 | Non-finite values (`NaN`, `±inf`) are rejected at construction, not propagated | `core/validation.py` | `tests/unit/test_validation.py` |

### 3.2 Units — `SOFIA-UNIT-*`

| ID | Requirement | Implementation | Test |
|---|---|---|---|
| SOFIA-UNIT-001 | Every sample carries an explicit unit string; no implicit conversion | `core/units.py` | `tests/unit/test_units.py` |
| SOFIA-UNIT-002 | Unit registry covers acceleration (`g`, `m/s2`), velocity (`mm/s`, `in/s`), displacement (`um`, `mm`, `mil`), temperature (`C`, `F`, `K`), rotation (`RPM`, `Hz`, `rad/s`), electrical (`V`, `mV`, `A`, `mA`, `kW`, `W`), pressure (`Pa`, `kPa`, `bar`, `psi`), flow, position | `core/units.py` | `tests/unit/test_units.py` |
| SOFIA-UNIT-003 | Conversion is explicit, dimension-checked, and rejects cross-dimension conversion | `core/units.py` | `tests/unit/test_units.py` |
| SOFIA-UNIT-004 | Unit arithmetic: `Quantity` refuses `g + mm/s` | `core/units.py` | `tests/unit/test_units.py` |

### 3.3 Time — `SOFIA-TIME-*`

| ID | Requirement | Implementation | Test |
|---|---|---|---|
| SOFIA-TIME-001 | Event time (wall clock), ingestion time and monotonic processing time are distinct fields | `core/time.py` | `tests/unit/test_time.py` |
| SOFIA-TIME-002 | DSP derives its axis from sample_rate + sample index, never from wall-clock deltas | `signal/*` | `tests/unit/test_spectral.py` |
| SOFIA-TIME-003 | Clock anomaly detection: backwards jumps, duplicate timestamps, forward jumps beyond tolerance | `core/time.py` | `tests/unit/test_time.py` |
| SOFIA-TIME-004 | Injectable clock (`TimeSource`) so tests are deterministic | `core/time.py` | `tests/unit/test_time.py` |

### 3.4 Data quality — `SOFIA-DQ-*`

| ID | Requirement | Implementation | Test |
|---|---|---|---|
| SOFIA-DQ-001 | `DataQuality` enumerates GOOD, STALE, MISSING, INVALID, OUT_OF_RANGE, DUPLICATE, ESTIMATED, UNSYNCHRONIZED | `core/quality.py` | `tests/unit/test_quality.py` |
| SOFIA-DQ-002 | Quality is visible to every detector; detectors degrade confidence on non-GOOD input | `inference/detectors.py` | `tests/unit/test_detectors.py` |
| SOFIA-DQ-003 | Bad data must not produce a confident diagnosis — confidence is scaled by a quality factor | `diagnostics/engine.py` | `tests/unit/test_diagnostics.py` |

### 3.5 Telemetry — `SOFIA-TLM-*`

| ID | Requirement | Implementation | Test |
|---|---|---|---|
| SOFIA-TLM-001 | `TelemetrySource` abstract interface; core never imports a transport | `telemetry/base.py` | `tests/contract/test_telemetry_contract.py` |
| SOFIA-TLM-002 | File adapters: CSV, JSONL | `telemetry/file_sources.py` | `tests/unit/test_file_sources.py` |
| SOFIA-TLM-003 | Replay adapter with rate control and loop control | `telemetry/replay.py` | `tests/unit/test_replay.py` |
| SOFIA-TLM-004 | In-memory / synthetic adapter for tests and demos | `telemetry/synthetic.py` | `tests/unit/test_synthetic.py` |
| SOFIA-TLM-005 | MQTT adapter behind an optional extra (not in minimal install) | `telemetry/mqtt.py` | `tests/unit/test_mqtt_adapter.py` |
| SOFIA-TLM-006 | Modbus TCP/RTU adapter behind an optional extra | `telemetry/modbus.py` | `tests/unit/test_modbus_adapter.py` |
| SOFIA-TLM-007 | Serial adapter behind an optional extra | `telemetry/serial_adapter.py` | `tests/unit/test_serial_adapter.py` |
| SOFIA-TLM-008 | Adapter registry so new protocols integrate without editing core | `telemetry/registry.py` | `tests/unit/test_registry.py` |
| SOFIA-TLM-009 | Every adapter enforces payload size limits, timeouts, retry ceilings | `telemetry/base.py` | `tests/contract/test_telemetry_contract.py` |
| SOFIA-TLM-010 | Transport failure surfaces as a typed error and does not crash the pipeline | `telemetry/base.py` | `tests/integration/test_pipeline_faults.py` |

### 3.6 Signal processing — `SOFIA-SIG-*`

| ID | Requirement | Implementation | Test |
|---|---|---|---|
| SOFIA-SIG-001 | Time-domain features: mean, rms, peak, peak_to_peak, variance, std, crest_factor, skewness, kurtosis, zero_crossing_rate, shape_factor, impulse_factor, margin_factor | `features/statistical.py` | `tests/unit/test_statistical.py`, `tests/property/test_signal_properties.py` |
| SOFIA-SIG-002 | Spectral: FFT magnitude, one/two-sided PSD, Welch PSD, spectral centroid, spectral peaks, band energy, spectral entropy, spectral flatness | `signal/spectral.py`, `features/spectral.py` | `tests/unit/test_spectral.py` |
| SOFIA-SIG-003 | Filtering: moving average, Butterworth-style IIR via SOS-free cascade, median, high-pass by subtraction | `signal/filters.py` | `tests/unit/test_filters.py` |
| SOFIA-SIG-004 | Resampling: decimation and linear interpolation with explicit rate in/out | `signal/resample.py` | `tests/unit/test_resample.py` |
| SOFIA-SIG-005 | Windowing: fixed-length, hop-based, Hann/Hamming/Blackman/rectangular | `signal/windowing.py` | `tests/unit/test_windowing.py` |
| SOFIA-SIG-006 | Detrending: constant and linear | `signal/detrend.py` | `tests/unit/test_detrend.py` |
| SOFIA-SIG-007 | Envelope extraction via Hilbert transform | `signal/envelope.py` | `tests/unit/test_envelope.py` |
| SOFIA-SIG-008 | Rotating-machinery extension points: order analysis, harmonic extraction, sideband energy, speed-aware resampling, bearing-related spectral bands | `features/rotating.py` | `tests/unit/test_rotating.py` |
| SOFIA-SIG-009 | Every feature function is pure, deterministic, and validated against closed-form expectations | all | `tests/property/test_signal_properties.py` |

### 3.7 Inference — `SOFIA-INF-*`

| ID | Requirement | Implementation | Test |
|---|---|---|---|
| SOFIA-INF-001 | `ModelBackend` interface: `load`, `infer`, `metadata`, `close` | `inference/base.py` | `tests/contract/test_model_backend_contract.py` |
| SOFIA-INF-002 | Statistical backends: static threshold, rolling z-score, robust MAD, EWMA, IQR, CUSUM | `inference/detectors.py` | `tests/unit/test_detectors.py` |
| SOFIA-INF-003 | Backend registry + plugin loading | `inference/registry.py` | `tests/unit/test_inference_registry.py` |
| SOFIA-INF-004 | Manifest validated on load: schema, version compatibility, checksum | `inference/manifest.py` | `tests/unit/test_manifest.py` |
| SOFIA-INF-005 | Incompatible models are rejected loudly, never silently loaded | `inference/manifest.py` | `tests/negative/test_model_rejection.py` |
| SOFIA-INF-006 | ONNX Runtime backend behind optional extra | `inference/onnx_backend.py` | skipped when extra absent |
| SOFIA-INF-007 | PyTorch backend behind optional extra | `inference/torch_backend.py` | skipped when extra absent |
| SOFIA-INF-008 | Inference failures are typed and counted, never swallowed | `inference/base.py` | `tests/negative/test_inference_failures.py` |
| SOFIA-INF-009 | Latency is measured with `time.perf_counter` and reported per result | `inference/base.py` | `tests/unit/test_inference.py` |

### 3.8 Diagnostics — `SOFIA-DIAG-*`

| ID | Requirement | Implementation | Test |
|---|---|---|---|
| SOFIA-DIAG-001 | Rule-based diagnostic primitives with explicit rule versions | `diagnostics/rules.py` | `tests/unit/test_rules.py` |
| SOFIA-DIAG-002 | Diagnostic engine combines evidence into `HealthEvent`s with combined confidence | `diagnostics/engine.py` | `tests/unit/test_diagnostics.py` |
| SOFIA-DIAG-003 | Health scoring maps evidence to a 0–100 machine-health score with an uncertainty band | `diagnostics/health.py` | `tests/unit/test_health_score.py` |
| SOFIA-DIAG-004 | Severity is derived from evidence, never inferred by an LLM | `diagnostics/engine.py` | `tests/unit/test_diagnostics.py` |
| SOFIA-DIAG-005 | Sofia does not claim to diagnose specific machine faults without validated evidence | docs | `docs/diagnostics.md` |

### 3.9 Safety / decision — `SOFIA-SAFE-*`

| ID | Requirement | Implementation | Test |
|---|---|---|---|
| SOFIA-SAFE-001 | Command policy default is **DENY** | `decision/policy.py` | `tests/unit/test_policy.py` |
| SOFIA-SAFE-002 | Policy inputs: action allowlist, machine state, equipment mode, operator approval, rate limit, physical limits, interlock, confidence floor, model version, freshness, replay protection | `decision/policy.py` | `tests/unit/test_policy.py` |
| SOFIA-SAFE-003 | No bypass: every command must evaluate through the policy engine | `decision/policy.py` | `tests/unit/test_policy.py` |
| SOFIA-SAFE-004 | Replay protection: nonce/TTL window rejects previously seen command ids | `decision/policy.py` | `tests/unit/test_policy.py` |
| SOFIA-SAFE-005 | RL actuation is disabled by default; training/simulation only | `learning/rl/agent.py` | `tests/unit/test_rl_agent.py` |
| SOFIA-SAFE-006 | Documentation states Sofia is not a certified SIS/PLC safety system | `docs/safety.md`, `SECURITY.md` | docs |

### 3.10 Edge runtime — `SOFIA-EDGE-*`

| ID | Requirement | Implementation | Test |
|---|---|---|---|
| SOFIA-EDGE-001 | Bounded ring buffer with explicit overflow policy (drop-oldest / drop-newest / reject) | `edge/buffer.py` | `tests/unit/test_buffer.py` |
| SOFIA-EDGE-002 | Store-and-forward persistence with byte ceiling and rotation | `edge/store_forward.py` | `tests/unit/test_store_forward.py` |
| SOFIA-EDGE-003 | Reconnect handling with bounded retry and backoff ceiling | `edge/reconnect.py` | `tests/unit/test_reconnect.py` |
| SOFIA-EDGE-004 | Edge runtime: ingest → window → features → inference → events, all local | `edge/runtime.py` | `tests/integration/test_edge_runtime.py` |
| SOFIA-EDGE-005 | Connectivity loss does not crash the pipeline | `edge/runtime.py` | `tests/integration/test_pipeline_faults.py` |
| SOFIA-EDGE-006 | Health/readiness interface for gateway deployment | `edge/health.py` | `tests/unit/test_edge_health.py` |
| SOFIA-EDGE-007 | Three deployment classes documented; claims per class marked VERIFIED / NOT VERIFIED | `docs/edge-deployment.md`, `docs/embedded.md` | docs |

### 3.11 Observability — `SOFIA-OBS-*`

| ID | Requirement | Implementation | Test |
|---|---|---|---|
| SOFIA-OBS-001 | Structured JSON logging with fixed field set | `observability/logging.py` | `tests/unit/test_logging.py` |
| SOFIA-OBS-002 | Log rate limiting per message key | `observability/logging.py` | `tests/unit/test_logging.py` |
| SOFIA-OBS-003 | Redaction of secret-like keys from logs and events | `observability/redaction.py` | `tests/unit/test_redaction.py` |
| SOFIA-OBS-004 | Metrics: samples received/dropped, malformed, queue depth, window latency, inference latency/failures, model version, anomaly events, reconnect attempts, protocol failures, policy denies, buffer saturation | `observability/metrics.py` | `tests/unit/test_metrics.py` |
| SOFIA-OBS-005 | Correlation and device identifiers attached to records | `observability/logging.py` | `tests/unit/test_logging.py` |
| SOFIA-OBS-006 | Metrics expose percentile histograms (p50/p95/p99) | `observability/metrics.py` | `tests/unit/test_metrics.py` |

### 3.12 Security — `SOFIA-SEC-*`

| ID | Requirement | Implementation | Test |
|---|---|---|---|
| SOFIA-SEC-001 | No `pickle` on any load path | `security/safeio.py` | `tests/security/test_no_pickle.py` |
| SOFIA-SEC-002 | Model artifacts carry checksum; mismatch rejects load | `inference/manifest.py` | `tests/negative/test_model_rejection.py` |
| SOFIA-SEC-003 | Optional Ed25519 signature verification hook for manifests | `security/signing.py` | `tests/security/test_signing.py` |
| SOFIA-SEC-004 | Path traversal rejected on all file inputs | `security/safeio.py` | `tests/security/test_safe_paths.py` |
| SOFIA-SEC-005 | Secrets read from environment; never committed | `security/secrets.py` | `tests/security/test_secrets.py` |
| SOFIA-SEC-006 | Payload size ceilings on all adapters | `telemetry/base.py` | `tests/contract/test_telemetry_contract.py` |
| SOFIA-SEC-007 | Plugin loading is explicit and allow-listed | `inference/registry.py` | `tests/security/test_plugin_loading.py` |
| SOFIA-SEC-008 | Threat model covering the 21 enumerated threats | `threat-model.md` | docs |
| SOFIA-SEC-009 | Typed errors; no bare `except Exception: pass` | `core/errors.py` | `tests/security/test_no_bare_except.py` |

### 3.13 Embedded — `SOFIA-EMB-*`

| ID | Requirement | Implementation | Test |
|---|---|---|---|
| SOFIA-EMB-001 | Documented export boundary: feature contract → generated C | `embedded/README.md` | docs |
| SOFIA-EMB-002 | Reference C99 runtime implementing the same feature set, no heap after init | `embedded/src/sofia_features.c/.h` | `embedded/tests/test_sofia_features.c` |
| SOFIA-EMB-003 | Fixed-point (Q-format) preprocessing path with documented range | `embedded/src/sofia_fixed.h` | C test |
| SOFIA-EMB-004 | CMSIS-DSP adapter guidance documented | `docs/embedded.md` | docs |
| SOFIA-EMB-005 | Golden-vector test: Python features vs C features within tolerance | `tools/gen_golden_vectors.py`, `embedded/tests/` | `tests/embedded/test_golden_vectors.py` |
| SOFIA-EMB-006 | Every MCU claim labeled VERIFIED / PARTIALLY VERIFIED / NOT VERIFIED | `docs/embedded.md` | docs |
| SOFIA-EMB-007 | Bounded loops, no recursion, no allocation after init in C | `embedded/src/*` | C test + review |

### 3.14 Packaging / CLI — `SOFIA-PKG-*`

| ID | Requirement | Implementation | Test |
|---|---|---|---|
| SOFIA-PKG-001 | Root `pyproject.toml`, PEP 517/518 build | `pyproject.toml` | `tests/unit/test_packaging.py` |
| SOFIA-PKG-002 | License metadata = Apache-2.0, matching `LICENSE` | `pyproject.toml` | `tests/unit/test_packaging.py` |
| SOFIA-PKG-003 | URLs point at `rootcastleco/sofia-rl`; author = Rootcastle Engineering & Innovation | `pyproject.toml` | `tests/unit/test_packaging.py` |
| SOFIA-PKG-004 | Optional dependency groups: `mqtt`, `modbus`, `serial`, `onnx`, `torch`, `industrial`, `dev`, `docs` | `pyproject.toml` | `tests/unit/test_packaging.py` |
| SOFIA-PKG-005 | `sofia --help` works | `cli/main.py` | `tests/unit/test_cli.py` |
| SOFIA-PKG-006 | `python -m sofia_ai --help` works | `sofia_ai/__main__.py` | `tests/unit/test_cli.py` |
| SOFIA-PKG-007 | Supported Python matrix derived from upstream EOL data | `.github/workflows/ci.yml` | CI |
| SOFIA-PKG-008 | `pip install -e ".[dev]"` then `pytest` from clean clone | docs | CI |

### 3.15 Backward compatibility — `SOFIA-COMPAT-*`

| ID | Requirement | Implementation | Test |
|---|---|---|---|
| SOFIA-COMPAT-001 | `from sofia_ai import SofiaModel, QuantumNeuralEngine, NLPProcessor, QuantumConfig` still works | `sofia_ai/__init__.py` (lazy) | `tests/compat/test_legacy_imports.py` |
| SOFIA-COMPAT-002 | Legacy imports emit `DeprecationWarning` | `sofia_ai/compat.py` | `tests/compat/test_legacy_imports.py` |
| SOFIA-COMPAT-003 | `QuantumConfig`, `NLPConfig`, `ModelConfig` remain constructible with legacy kwargs | `config/legacy.py` | `tests/compat/test_legacy_config.py` |
| SOFIA-COMPAT-004 | `SofiaRLAgent` API preserved (state_size, action_size, act, step, learn, save/load) | `learning/rl/agent.py` | `tests/unit/test_rl_agent.py` |
| SOFIA-COMPAT-005 | Migration guide with version boundary | `migration.md`, `CHANGELOG.md` | docs |

### 3.16 Experimental quantum — `SOFIA-QEXP-*`

| ID | Requirement | Implementation | Test |
|---|---|---|---|
| SOFIA-QEXP-001 | Component moved to `sofia_ai.experimental.quantum`, API preserved | `experimental/quantum/engine.py` | `tests/compat/test_legacy_imports.py` |
| SOFIA-QEXP-002 | Documented as **quantum-inspired simulation**, no quantum hardware | `experimental/quantum/README.md` | docs |
| SOFIA-QEXP-003 | Resource envelope documented and enforced (`2**n` growth, hard ceiling) | `experimental/quantum/engine.py` | `tests/unit/test_quantum_envelope.py` |
| SOFIA-QEXP-004 | Benchmarked against a classical baseline (random-projection / PCA / linear) | `benchmarks/bench_quantum_baseline.py` | `tests/unit/test_quantum_benchmark.py` |
| SOFIA-QEXP-005 | Result and absence of measured advantage stated explicitly in docs | `docs/experimental-quantum.md` | docs |

### 3.17 Copilot — `SOFIA-COP-*`

| ID | Requirement | Implementation | Test |
|---|---|---|---|
| SOFIA-COP-001 | Copilot layer is outside the deterministic pipeline | `copilot/` | `tests/unit/test_copilot.py` |
| SOFIA-COP-002 | Copilot consumes structured evidence only; cannot emit commands | `copilot/base.py` | `tests/unit/test_copilot.py` |
| SOFIA-COP-003 | Provider adapters, no vendor lock-in; offline no-op provider is default | `copilot/providers.py` | `tests/unit/test_copilot.py` |
| SOFIA-COP-004 | Core works with no LLM installed | `copilot/offline.py` | `tests/integration/test_offline_pipeline.py` |

---

## 4. Out of scope

* Certified functional safety (SIL/PL) — explicitly disclaimed.
* Real quantum hardware execution.
* Vendor-specific PLC programming environments.
* Claiming diagnostic accuracy on any public dataset (no dataset is bundled).
* Publishing to PyPI under a new name (requires owner authorization).
