# Sofia Engine Runtime v3 — Engineering Constitution

> **Status:** Active Normative Standard  
> **Authority:** Rootcastle Engineering & Innovation  
> **Scope:** Sofia Engine Core, Runtimes, Ingestion, DSP, ML/VM, Diagnostics, Safety, and Language Bindings (Python, TypeScript, C99).

---

## 1. Fundamental Principle: Evidence Beats Claims

1. **Reality Over Assertion:**
   No capability, speedup, accuracy, compliance, or robustness claim shall be made in code, documentation, benchmarks, or metadata without verifiable, reproducible automated evidence in this repository.
2. **Implementation is Authoritative:**
   If documentation and code disagree, the executable implementation is authoritative until reconciled.
3. **Unverified Classification:**
   Any performance attribute, hardware compatibility, or standard alignment that cannot be independently proven via automated test or measurement in this repository must be explicitly marked: `NOT VERIFIED`.
4. **Zero Fabrication:**
   Under no circumstances shall benchmark numbers, latency metrics, accuracy percentages, test results, fine-tuning checkpoints, or compliance claims be simulated, hardcoded, or fabricated. Dry-run modes must explicitly declare `"simulated": true` and `"metrics": null`.

---

## 2. Architectural Invariants

### Invariant 1: Air-Gapped & Offline-First Core
The deterministic scientific runtime (`sofia-core`) must execute completely offline with zero network, cloud, or external service dependencies.
* **Permitted Core Dependencies:** NumPy (`numpy>=1.24`) and Python Standard Library.
* **Prohibited Core Dependencies:** PyTorch, ONNX Runtime, FastAPI, Requests, Pydantic, urllib external network calls, MQTT brokers, OpenAI/NVIDIA/OpenRouter SDKs.
* **Integration Boundary:** All network transports, cloud LLMs, and external hardware bridges must reside in isolated adapter packages/modules that depend on `sofia-core`, never the reverse.

### Invariant 2: Bounded Resource Ceilings
Every buffer, queue, window, cache, loop, and retention structure in the runtime execution path must have an explicit compile-time or initialization-time upper bound. Unbounded memory allocations, unbounded recursion, and infinite retry loops without backoff are strictly prohibited.

### Invariant 3: Strict Determinism
Given identical input signals, identical configuration, and identical pseudo-random seeds, the runtime must produce byte-identical results across runs. All time-axes in digital signal processing must derive strictly from sample counts and sample rates, never from non-deterministic wall-clock intervals.

### Invariant 4: Default DENY Safety Boundary
Actuation and machine control decisions must strictly follow the **Default DENY** security model. No inference model, neural network, DSP threshold, quantum kernel, or generative LLM may directly actuate physical machinery. All proposed actions must pass through an auditable, deterministic `PolicyEngine` with explicit interlocks, operator gates, replay protection (Nonce + TTL), and cryptographic validation.

### Invariant 5: LLM Advisory Isolation
Generative models and LLM copilots are strictly advisory subsystems outside the deterministic control core. They consume structured, validated evidence records and produce human-readable diagnostic summaries. They have zero direct execution privileges on machinery or policy configuration.

### Invariant 6: Strict Cross-Language Conformance
The Python, TypeScript, and C99 runtimes must produce numerical outputs matching within declared epsilon tolerances against golden fixtures in `tests/golden/`. No cross-language divergence in DSP, feature extraction, or assembly VM execution is permitted.

---

## 3. Code Engineering Rules

1. **Typed Runtime Primitives:** Engineering values must never be passed as raw, unstructured dictionaries across subsystem boundaries.
2. **Explicit Physical Units:** Every physical quantity must declare its dimensional unit (`m/s2`, `mm/s`, `um`, `Hz`, `RPM`, `V`, `A`, `C`, `bar`, `Pa`). Implicit unit conversions are forbidden.
3. **Finite Floating-Point Arithmetic:** `NaN` and `±Inf` must be rejected at input ingestion and construction boundaries.
4. **Embedded C99 Rules:**
   - Zero heap allocation (`malloc`, `calloc`) after initialization.
   - Fixed-size arrays, bounded loops, explicit integer widths (`int32_t`, `uint64_t`, etc.).
   - Strict compiler flags: `-Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror`.
5. **No Silent Fallbacks:** Catch-all exception swallowing (`except Exception: pass`) is prohibited. Failures must be typed, observable, logged, and safe.
