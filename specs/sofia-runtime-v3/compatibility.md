# Sofia Engine Runtime v3 — Compatibility & Migration Policy

> **Author:** Rootcastle Engineering & Innovation  
> **Status:** Active Normative Standard

---

## 1. Compatibility Policy

1. **No Casual Breakage:**
   Sofia 2.x public contracts and APIs must remain functional throughout the Sofia 3.x series. Existing customer code importing `sofia_ai.core.contracts`, `sofia_ai.features`, `sofia_ai.diagnostics`, and `sofia_ai.decision` must continue to execute without breaking changes.
2. **Planned Compatibility Windows:**
   - **Sofia 1.x Legacy Names:** (`SofiaModel`, `QuantumNeuralEngine`, `NLPProcessor`, `QuantumConfig`) resolve via `sofia_ai.compat` and emit `DeprecationWarning`. Maintained until Sofia 4.0.
   - **Sofia 2.x Contracts:** (`TelemetrySample`, `SignalWindow`, `FeatureVector`, `InferenceResult`) continue to exist as first-class or aliased primitives in `sofia_ai.core.contracts`.
   - **Sofia 3.x Primitives:** (`Sample`, `SignalFrame`, `SignalMetadata`, `Feature`, `VMExecutionResult`) provide stricter typing, uncertainty fields, and schema versioning, interoperating transparently with 2.x contracts.

---

## 2. API Mapping & Migration Table

| 2.x Contract / Module | 3.x Evolutionary Contract / Module | Compatibility Status | Notes |
|---|---|---|---|
| `TelemetrySample` | `Sample` / `TelemetrySample` | Fully Supported | `Sample` is an alias/subclass of `TelemetrySample` with explicit physical unit validation. |
| `SignalWindow` | `SignalWindow` / `SignalFrame` | Fully Supported | `SignalFrame` provides immutable framing; `SignalWindow` remains available for sliding buffers. |
| `FeatureVector` | `FeatureVector` (versioned) | Fully Supported | Added `schema_version`, `uncertainty` fields; existing `.as_array()` and `.as_dict()` preserved. |
| `InferenceResult` | `InferenceResult` | Fully Supported | Added explicit `uncertainty` and `evidence` fields; existing properties preserved. |
| `SofiaAsmVM` | `SofiaAsmVM` | Hardened | Added `VMExecutionResult`, cycle accounting, memory bounds checks. Existing bytecode programs remain valid. |
| `AssemblyNeuralNetwork` | `AssemblyNeuralNetwork` | Enhanced | Full mathematical backpropagation implemented; existing `.forward()` and `.train_step()` signatures preserved. |
| `AutoFineTuner` | `AutoFineTuner` | Refactored | Real provider capability detection, remote file upload semantics, honest dry-run. |

---

## 3. Rollback & Transition Strategy

If an edge deployment encounters an issue with a 3.x feature:
1. Every new capability is protected behind feature flags or separate import namespaces.
2. The core deterministic pipeline remains pure Python/NumPy with zero stateful side-effects.
3. Checkpoint files generated in 2.x format remain readable by 3.x model loaders via schema auto-detection.
