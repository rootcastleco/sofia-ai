# SOFIA ENGINE — Test Strategy

## 1. Layers

| Layer | Directory | Purpose | Gate |
|---|---|---|---|
| Unit | `tests/unit/` | Pure functions, contracts, config, detectors, policy, buffers, RL | CI, fast |
| Contract | `tests/contract/` | Every `TelemetrySource` and every `ModelBackend` obeys its ABC | CI |
| Integration | `tests/integration/` | End-to-end pipelines, offline operation | CI |
| Negative | `tests/negative/` | Corrupt/NaN/inf/huge/wrong-schema/replay inputs | CI |
| Property | `tests/property/` | Hypothesis invariants on DSP and unit math | CI |
| Architecture | `tests/architecture/` | Module boundary and forbidden-dependency rules | CI |
| Security | `tests/security/` | No pickle, no bare except, redaction, path confinement, secrets | CI |
| Compatibility | `tests/compat/` | Legacy `from sofia_ai import ...` surface | CI |
| Embedded | `tests/embedded/` | Python↔C golden vector equivalence | CI when a C compiler is present |

## 2. Coverage targets

| Scope | Target | Mechanism |
|---|---|---|
| Repository | ≥ 85 % line coverage | `pyproject.toml` `[tool.coverage.report] fail_under = 85` |
| `core/`, `decision/`, `inference/manifest.py`, `core/validation.py`, `core/units.py` | ≥ 95 % branch coverage | separate `coverage` run in CI, reported |
| New code | branch coverage measured, not just line | `--cov-branch` |

Coverage is necessary, not sufficient: PRs must include branch/failure-path tests
(`CONTRIBUTING.md` states this).

## 3. Determinism requirements

* No test may depend on wall-clock time. `FixedTimeSource` is injected.
* No test may call `random.seed` or `np.random.seed`. RNGs are injected `numpy.random.Generator`s.
* No test may depend on dict ordering.
* Property tests declare `deadline=None` where DSP cost is variable.
* `tests/conftest.py` installs a global autouse fixture that fails any test that leaves a
  non-deterministic global mutated.

## 4. Negative-test matrix

| Input class | Expected behavior | Test |
|---|---|---|
| `NaN` value | `ValidationError`, quality INVALID, counted `malformed` | `tests/negative/test_bad_values.py` |
| `inf` value | same | same |
| Value outside channel range | quality `OUT_OF_RANGE`, detector confidence scaled | same |
| Oversized payload | `PayloadTooLargeError`, source closed | `tests/negative/test_payload_limits.py` |
| Truncated CSV row | row dropped, counted, stream continues | `tests/negative/test_malformed_frames.py` |
| Truncated Modbus frame | `TelemetryProtocolError` | same |
| Wrong model input schema | `ModelSchemaError`, no inference | `tests/negative/test_model_rejection.py` |
| Corrupted manifest JSON | `ManifestError` | same |
| Checksum mismatch | `ModelIntegrityError`, load refused | same |
| Duplicate timestamps | quality `DUPLICATE` | `tests/negative/test_time_anomalies.py` |
| Out-of-order samples | flagged, reorder-safe window assembly | same |
| Backwards clock jump | counted `clock_anomaly` | same |
| Empty window | `InsufficientDataError`, no event | `tests/negative/test_empty.py` |
| Exhausted queue | drop-oldest, saturation metric incremented | `tests/negative/test_buffer_saturation.py` |
| Stale telemetry | quality `STALE`, confidence scaled | `tests/negative/test_stale.py` |
| Inference raises | `InferenceError`, counted, pipeline continues | `tests/negative/test_inference_failures.py` |
| Corrupted store-and-forward file | rotation, not crash | `tests/negative/test_state_corruption.py` |
| Path traversal in `--model` | `UnsafePathError` | `tests/security/test_safe_paths.py` |

## 5. Fault injection

| Fault | Mechanism | Test |
|---|---|---|
| Connection loss | source raises mid-stream | `tests/integration/test_pipeline_faults.py` |
| Reconnect | bounded retry with injected clock | `tests/unit/test_reconnect.py` |
| Incomplete window | producer stops mid-window | `tests/integration/test_pipeline_faults.py` |
| Inference failure | backend raises | `tests/negative/test_inference_failures.py` |
| Corrupted state | store-and-forward file truncated | `tests/negative/test_state_corruption.py` |
| Disk pressure | byte ceiling reached | `tests/unit/test_store_forward.py` |

## 6. Property tests (Hypothesis)

| Property | Test |
|---|---|
| RMS of constant signal equals the constant, for all finite constants | `test_signal_properties.py` |
| RMS ≥ |mean| | same |
| Crest factor = peak / RMS ≥ 1 | same |
| Zero-crossing rate ∈ [0, 1] | same |
| Kurtosis of a Gaussian sample ≈ 3 (tolerance) | same |
| Parseval: sum(|x|²)·dt ≈ ∫PSD | `test_spectral_properties.py` |
| PSD is non-negative and even-symmetric | same |
| Spectral centroid lies within [0, fs/2] | same |
| Unit conversion round-trip x→y→x is identity within float tolerance | `test_unit_properties.py` |
| Cross-dimension conversion always raises | same |
| Bounded buffer never exceeds capacity and never loses newest ordering | `test_buffer_properties.py` |
| JSON round-trip of every contract preserves equality | `test_serialization_properties.py` |

## 7. What is deliberately not tested

* Third-party broker/PLC behavior (no broker in CI; fakes used instead).
* Numerical equality across numpy/BLAS versions beyond stated tolerances.
* MCU execution (no hardware in CI) — golden vectors only.
* LLM provider output (non-deterministic by nature); only the offline no-op provider is tested.

## 8. Rule

Legitimate failing tests are never deleted to obtain green CI. A test may only be removed with a
CHANGELOG entry explaining why the behavior it asserted is no longer required.
