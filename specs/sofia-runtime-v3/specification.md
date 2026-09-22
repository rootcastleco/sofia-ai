# Sofia Engine Runtime v3 — Specification

> **Spec Version:** 3.0.0-draft  
> **Target Release:** Sofia Engine 3.0.0a1  
> **Status:** Approved for Implementation  
> **Author:** Rootcastle Engineering & Innovation

---

## 1. Core Runtime Contracts & Primitives (`SOFIA-CORE-*`)

| ID | Requirement | Success Criteria |
|---|---|---|
| `SOFIA-CORE-001` | **Typed Primitive: `Sample`** | Single measurement with `timestamp` (UTC ISO-8601/float), `device_id`, `channel`, `value` (finite float), `unit` (canonical string), `quality` (`SignalQuality`), and optional `metadata`. |
| `SOFIA-CORE-002` | **Typed Primitive: `SignalFrame`** | Bounded collection of `Sample` instances with strict maximum sample ceiling, uniform or tagged timestamps, and immutability. |
| `SOFIA-CORE-003` | **Typed Primitive: `SignalMetadata`** | Ingestion metadata containing `sample_rate`, `channel_name`, `physical_unit`, `sensor_id`, `calibration_id`, and `scale_factor`. |
| `SOFIA-CORE-004` | **Typed Primitive: `SignalQuality`** | Enum: `GOOD`, `DEGRADED`, `STALE`, `OUT_OF_BOUNDS`, `SATURATED`, `MISSING`, `INVALID`, `ESTIMATED`. |
| `SOFIA-CORE-005` | **Typed Primitive: `Feature`** | Single named feature with `name`, `value`, `unit`, `uncertainty` (float $\ge 0$), `quality` (`SignalQuality`), `algorithm`, and `algorithm_version`. |
| `SOFIA-CORE-006` | **Typed Primitive: `FeatureVector`** | Ordered, immutable, versioned (`schema_version`) collection of `Feature` objects with fast array conversion (`as_array()`), dictionary lookup, and schema compatibility validation. |
| `SOFIA-CORE-007` | **Typed Primitive: `InferenceRequest`** | Payload containing `model_id`, `model_version`, `features` (`FeatureVector`), `context` (mapping), and `request_id`. |
| `SOFIA-CORE-008` | **Typed Primitive: `InferenceResult`** | Structured output containing `model_id`, `outcome` (str/int), `score` (float), `confidence` ($[0, 1]$), `uncertainty` ($[0, 1]$), `evidence` (tuple of `DiagnosticEvidence`), and `latency_ms`. |
| `SOFIA-CORE-009` | **Typed Primitive: `DiagnosticEvidence`** | Verifiable physical evidence containing `source`, `metric`, `observed_value`, `reference_value`, `unit`, `weight`, and `description`. |
| `SOFIA-CORE-010` | **Typed Primitive: `HealthEvent`** | Domain event containing `event_id` (deterministic sha256), `severity` (`INFO`, `NOTICE`, `WARNING`, `CRITICAL`), `device_id`, `evidence`, `confidence`, `recommendation`, and timestamp. |
| `SOFIA-CORE-011` | **Typed Primitive: `HealthScore`** | Composite score in $[0, 100]$ accompanied by dynamic `uncertainty_band` ($\pm \Delta$) and explicit contributor breakdown. |
| `SOFIA-CORE-012` | **Typed Primitive: `CommandRequest`** | Actuation proposal with `command_id`, `target_device`, `action`, `parameters`, `evidence_ids`, `nonce`, `ttl_seconds`, and `requester`. |
| `SOFIA-CORE-013` | **Typed Primitive: `CommandDecision`** | Policy outcome: `verdict` (`APPROVE`, `DENY`, `MANUAL_REVIEW`), `reason_code`, `policy_version`, and audit signature. |
| `SOFIA-CORE-014` | **Typed Primitive: `RuntimeFault`** | Structured fault reporting containing `fault_code`, `subsystem`, `severity`, `message`, `recoverable` (bool), and timestamp. |
| `SOFIA-CORE-015` | **Dependency Isolation** | `sofia-core` must have zero runtime dependencies beyond `numpy>=1.24` and Python standard library. Architecture tests must fail if any prohibited package is imported. |

---

## 2. Scientific DSP & Mathematical Correctness (`SOFIA-DSP-*`)

| ID | Requirement | Success Criteria |
|---|---|---|
| `SOFIA-DSP-001` | **Parseval Energy Conservation** | For any signal $x[n]$, total time-domain energy $\sum \|x[n]\|^2 \Delta t$ equals total frequency-domain energy $\int S_{xx}(f) df$ within $0.1\%$ relative tolerance across all window types (Hann, Hamming, Blackman, Flat-top). |
| `SOFIA-DSP-002` | **Window Normalization & Coherent Gain** | Window coherent gain ($S_1 = \sum w[n]$) and noise power gain ($S_2 = \sum w[n]^2$) are explicitly computed and applied during spectral amplitude and PSD estimation. |
| `SOFIA-DSP-003` | **Single-Tone Frequency & Amplitude Accuracy** | For pure tone $A \sin(2\pi f_0 t)$, recovered peak frequency error $\le \frac{f_s}{2N}$ (or $\le 0.01\%$ with parabolic interpolation), and recovered amplitude error $\le 0.5\%$. |
| `SOFIA-DSP-004` | **Analytic Signal & Hilbert Envelope** | Envelope of $x(t) = [1 + m \cos(2\pi f_m t)] \cos(2\pi f_c t)$ accurately demodulates modulation depth $m$ within $1.0\%$ relative error. |
| `SOFIA-DSP-005` | **Electrical Power Quality (IEEE 519 / IEC 61000-4-30 Compatible)** | Active power ($P$), reactive power ($Q$), apparent power ($S$), power factor ($PF$), and THD up to 50th harmonic computed with explicit formulation. |
| `SOFIA-DSP-006` | **Fortescue Symmetrical Components** | 3-phase unbalanced voltage yields exact positive ($V_1$), negative ($V_2$), and zero ($V_0$) sequence components, and Voltage Unbalance Factor ($VUF = \|V_2\| / \|V_1\| \times 100\%$). |
| `SOFIA-DSP-007` | **Acoustic Emission & Cavitation Indexing** | AE energy, ringdown counts, rise time, and high-frequency spectral ratio (Cavitation Index $C_p$) computed deterministically. |
| `SOFIA-DSP-008` | **IMU Kinematics & Jerk** | Acceleration magnitude $\|\mathbf{a}\|$, pitch, roll, and numerical derivative jerk ($d\mathbf{a}/dt$) computed with bounded differences. |
| `SOFIA-DSP-009` | **Golden Vector Fixtures** | Golden test vectors in `tests/golden/dsp_golden.json` generated analytically and validated across Python, TypeScript, and C99. |
| `SOFIA-DSP-010` | **Non-Finite & Edge Case Handling** | DSP functions reject `NaN`, `Inf`, empty arrays, and signals below minimum required length with explicit `SignalValidationError`. |

---

## 3. Sofia Assembly Virtual Machine (`SOFIA-VM-*`)

| ID | Requirement | Success Criteria |
|---|---|---|
| `SOFIA-VM-001` | **Formal VM Specification** | Deterministic register-based VM with 16 registers (`R0`-`R7`, `ACC`, `LR`, `ERR`, `PC`, `SP`, `FLAGS`, `TEMP1`, `TEMP2`), fixed memory buffer, and 32-bit/64-bit instruction encoding. |
| `SOFIA-VM-002` | **Structured Execution Result: `VMExecutionResult`** | VM execution returns `VMExecutionResult(status, cycles, fault, pc, registers)` where status is an enum (`HALTED`, `CYCLE_LIMIT`, `INVALID_OPCODE`, `INVALID_REGISTER`, `MEMORY_FAULT`, `NUMERIC_FAULT`). |
| `SOFIA-VM-003` | **Memory Safety & Bounds Verification** | Memory access outside allocated buffer bounds immediately halts VM with `MEMORY_FAULT`. Unchecked indexing is prohibited. |
| `SOFIA-VM-004` | **Cycle Accounting & Infinite Loop Defense** | VM terminates with `CYCLE_LIMIT` when executed instruction count exceeds configured `max_cycles`. |
| `SOFIA-VM-005` | **Instruction Set Completeness** | Supports arithmetic (`ADD`, `SUB`, `MUL`, `DIV`, `FMA`), vector primitives (`VEC_DOT`, `VEC_FMA`, `VEC_SUB`), activations (`ACT_RELU`, `GRAD_RELU`), and optimization (`UPDATE_SGD`, `COMPUTE_MSE`). |
| `SOFIA-VM-006` | **Deterministic Execution** | Identical initial memory and bytecode must produce bit-identical registers and memory across repeated runs. |

---

## 4. In-Situ Neural Self-Training (`SOFIA-ML-*`)

| ID | Requirement | Success Criteria |
|---|---|---|
| `SOFIA-ML-001` | **Full Backpropagation Implementation** | `AssemblyNeuralNetwork` implements complete mathematical backpropagation: input $\to$ $W_1, B_1 \to \text{ReLU} \to W_2, B_2 \to \hat{Y} \to \text{Loss} \to \nabla W_2, \nabla B_2 \to \nabla W_1, \nabla B_1 \to \text{SGD Update}$. |
| `SOFIA-ML-002` | **Gradient Buffer Isolation** | Gradient buffers are explicitly zeroed or overwritten per training step, preventing accidental accumulation. |
| `SOFIA-ML-003` | **Numerical Gradient Checking** | Analytical gradients match finite-difference numerical gradients ($\frac{f(\theta+\epsilon)-f(\theta-\epsilon)}{2\epsilon}$) with relative error $< 10^{-5}$ on test fixtures. |
| `SOFIA-ML-004` | **Deterministic Convergence** | Solves standard benchmark tasks (e.g., linear regression, non-linear function fitting) with monotonic loss reduction and exact reproduction given a fixed seed. |
| `SOFIA-ML-005` | **Safe Model Manifest: `sofia.model.v1`** | Model artifacts saved in JSON + deterministic NPZ weights with SHA-256 parameter hashing. Untrusted `pickle` deserialization is prohibited. |

---

## 5. Embedded C99 Runtime (`SOFIA-EMB-*`)

| ID | Requirement | Success Criteria |
|---|---|---|
| `SOFIA-EMB-001` | **Strict C99 Compliance** | Embedded code compiles cleanly with `-Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror` on GCC and Clang. |
| `SOFIA-EMB-002` | **Zero Dynamic Allocation Post-Init** | No calls to `malloc`, `calloc`, or `free` after runtime initialization. All buffers are fixed-capacity or caller-allocated arenas. |
| `SOFIA-EMB-003` | **Compile-Time Resource Ceilings** | Explicit macros: `SOFIA_MAX_FEATURES`, `SOFIA_MAX_MODEL_PARAMETERS`, `SOFIA_MAX_SIGNAL_WINDOW`. |
| `SOFIA-EMB-004` | **Golden Vector Conformance** | C99 test runner executes against `tests/golden/` fixtures and verifies outputs against Python baseline within declared epsilon ($10^{-4}$). |
| `SOFIA-EMB-005` | **Footprint Measurement** | Automated build script measures and records exact flash (text/rodata) and RAM (data/bss) binary sizes. |

---

## 6. Safety Gate & Policy Engine (`SOFIA-SAFE-*`)

| ID | Requirement | Success Criteria |
|---|---|---|
| `SOFIA-SAFE-001` | **Default DENY State Machine** | Any `CommandRequest` not matching an explicit allowlisted rule, or failing interlocks, is rejected with `CommandDecision(verdict=DENY)`. |
| `SOFIA-SAFE-002` | **Replay Protection (Nonce + TTL)** | Requests with duplicate nonces or expired timestamps ($t - t_{\text{req}} > \text{TTL}$) are immediately denied. |
| `SOFIA-SAFE-003` | **LLM Actuation Firewall** | Generative models and copilot providers have zero code paths or interfaces to execute commands or modify policy tables. |
| `SOFIA-SAFE-004` | **Audit Trail Logging** | Every command evaluation logs a structured audit record containing `request_id`, `decision`, `reason_code`, `policy_version`, and cryptographic evidence hash. |

---

## 7. Security & Threat Model (`SOFIA-SEC-*`)

| ID | Requirement | Success Criteria |
|---|---|---|
| `SOFIA-SEC-001` | **Threat Model Coverage** | Threat model addresses 21 edge threats including poisoned telemetry, malformed packets, model tampering, and denial-of-service. |
| `SOFIA-SEC-002` | **Zero Pickle Deserialization** | Models and configurations use JSON/NPZ with strict schema validation. `pickle.loads` is banned. |
| `SOFIA-SEC-003` | **Secret Scrubbing in Logs** | API keys, credentials, and sensitive tokens are masked/scrubbed before writing to logs. |

---

## 8. Provider Adapters & Honest Fine-Tuning (`SOFIA-API-*`)

| ID | Requirement | Success Criteria |
|---|---|---|
| `SOFIA-API-001` | **Provider Capability Detection** | Provider adapters declare exact supported features (`file_upload`, `fine_tuning`, `job_cancel`). Unsupported methods raise `FeatureNotSupportedError`. |
| `SOFIA-API-002` | **Honest Dry-Run Mode** | Dry-run mode produces `{"simulated": true, "status": "DRY_RUN", "metrics": null}`. Zero fabricated loss or accuracy values. |
| `SOFIA-API-003` | **Real Upload Semantics** | File upload workflow follows remote provider API contracts: dataset $\to$ remote file upload $\to$ file ID $\to$ job creation $\to$ status poll $\to$ model ID. |
