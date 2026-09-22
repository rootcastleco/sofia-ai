# Sofia Engine Runtime v3 — Implementation Tasks

> **Author:** Rootcastle Engineering & Innovation  
> **Status:** Active Execution Tracker

---

## Phase 0: Repository Audit & Specification (Done)
- [x] Create dedicated branch `feature/sofia-runtime-v3`.
- [x] Audit current codebase, specs, tests, packaging, metadata.
- [x] Author `constitution.md`, `specification.md`, `architecture.md`, `threat-model.md`, `performance-budgets.md`, `compatibility.md`, `tasks.md`, `verification-matrix.md`.

---

## Phase 1: Core Runtime Contracts & Architecture Boundaries
- [x] Task 1.1: Implement runtime primitives in `src/sofia_ai/core/contracts.py`: `Sample`, `SignalFrame`, `SignalMetadata`, `SignalQuality`, `Feature`, `InferenceRequest`, `RuntimeFault`.
- [x] Task 1.2: Add explicit physical unit checks and validation rules to new primitives.
- [x] Task 1.3: Author architecture boundary test `tests/architecture/test_boundaries.py` enforcing zero prohibited imports in `sofia-core`.
- [x] Task 1.4: Unit tests for all new core contracts in `tests/unit/test_contracts_v3.py`.

---

## Phase 2: Scientific DSP Correctness & Golden Vectors
- [x] Task 2.1: Verify & enforce Parseval energy conservation in `src/sofia_ai/signal/spectral.py`.
- [x] Task 2.2: Implement window coherent gain ($S_1$) and noise power gain ($S_2$) corrections.
- [x] Task 2.3: Generate analytic golden vectors in `tests/golden/dsp_golden.json` (single tone, multi-tone, AM envelope, 3-phase unbalance, white noise, impulse).
- [x] Task 2.4: Implement golden vector test suite `tests/unit/test_dsp_golden.py`.
- [x] Task 2.5: Ensure non-finite and edge-case signal validation.

---

## Phase 3: Versioned Feature Runtime, Diagnostics & Uncertainty
- [x] Task 3.1: Upgrade `FeatureVector` in `src/sofia_ai/core/contracts.py` and `src/sofia_ai/features/` with schema versioning (`schema_version`) and compatibility checking.
- [x] Task 3.2: Implement `DiagnosticEvidence` and update `DiagnosticEngine` to scale confidence by data quality and SNR.
- [x] Task 3.3: Implement `HealthScore` with dynamic uncertainty bands ($\pm \Delta$).
- [x] Task 3.4: Write unit tests for feature versioning and uncertainty in `tests/unit/test_diagnostics_v3.py`.

---

## Phase 4: Sofia Assembly VM Hardening
- [x] Task 4.1: Define `VMExecutionResult`, `VMStatus`, and `VMFault` in `src/sofia_ai/learning/asm/vm.py`.
- [x] Task 4.2: Implement memory bounds checking and cycle accounting (`max_cycles`).
- [x] Task 4.3: Implement strict instruction decoding and reject invalid opcodes/registers.
- [x] Task 4.4: Write comprehensive VM fault and fuzz tests in `tests/unit/test_asm_vm_hardening.py`.

---

## Phase 5: Mathematically Correct In-Situ Neural Self-Training
- [x] Task 5.1: Implement complete analytical backpropagation in `src/sofia_ai/learning/asm/neural.py` (Layer 2 $\to$ Layer 1, $W_1, B_1, W_2, B_2$).
- [x] Task 5.2: Explicitly zero gradient buffers before each training step.
- [x] Task 5.3: Implement numerical gradient checking vs finite differences (relative error $< 10^{-5}$) in `tests/unit/test_neural_gradients.py`.
- [x] Task 5.4: Test deterministic convergence on regression and XOR problems.
- [x] Task 5.5: Implement safe model manifest `sofia.model.v1` with SHA-256 weight checksums in `src/sofia_ai/learning/manifest.py`.

---

## Phase 6: Portable C99 Runtime & Microcontroller Target
- [x] Task 6.1: Audit `embedded/` C code, ensure strict `-Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror` clean compilation.
- [x] Task 6.2: Enforce zero dynamic allocation after initialization and compile-time ceilings (`SOFIA_MAX_FEATURES`, etc.).
- [x] Task 6.3: Implement C99 test runner against golden vectors and test embedded conformance in `tests/unit/test_embedded_conformance.py`.
- [x] Task 6.4: Verify Q16.16 fixed point emulation.

---

## Phase 7: Cross-Language Conformance (Python / TypeScript / C99)
- [x] Task 7.1: Synchronize TypeScript SDK (`packages/sofia-engine/src/`) with new contracts and VM hardening.
- [x] Task 7.2: Create cross-language conformance test runner verifying Python, TypeScript, and C outputs against golden vectors (`packages/sofia-engine/src/conformance.test.ts`).

---

## Phase 8: Provider Adapter Redesign & Honest Fine-Tuning
- [x] Task 8.1: Implement provider capability detection (`file_upload`, `fine_tuning`, `job_cancel`) in `src/sofia_ai/learning/finetune/`.
- [x] Task 8.2: Implement realistic remote file upload semantics.
- [x] Task 8.3: Enforce honest dry-run: `{"simulated": true, "status": "DRY_RUN", "metrics": null}`.
- [x] Task 8.4: Write unit tests for provider adapters and dry-run honesty in `tests/unit/test_finetune_v3.py`.

---

## Phase 9: Security & PolicyEngine Hardening
- [x] Task 9.1: Enforce Nonce + TTL replay protection in `src/sofia_ai/decision/policy.py`.
- [x] Task 9.2: Verify default DENY state machine and LLM actuation firewall.
- [x] Task 9.3: Implement automated secret scrubbing in logging formatters and mappings.
- [x] Task 9.4: Write security and policy tests in `tests/unit/test_security_v3.py`.

---

## Phase 10: Observability & Performance Benchmarks
- [x] Task 10.1: Implement structured telemetry counters, latency trackers, and fault metrics in `src/sofia_ai/observability/`.
- [x] Task 10.2: Implement automated benchmark runner `benchmarks/run_benchmarks.py` measuring throughput, latency, and memory.
- [x] Task 10.3: Record baseline benchmark results in `benchmarks/results/baseline_benchmark.json`.

---

## Phase 11: Documentation & Metadata Credibility Pass
- [x] Task 11.1: Audit `README.md` and documentation: remove unverified claims and hype, ensure all statements reflect actual code and tests.
- [x] Task 11.2: Audit repository URLs: update `rootcastleco/sofia-rl` to `rootcastleco/sofia-ai`.
- [x] Task 11.3: Update version metadata to `3.0.0a1` across `pyproject.toml`, `src/sofia_ai/__init__.py`, and `packages/sofia-engine/package.json`.

---

## Phase 12: Full Convergence Audit & Release Preparation
- [x] Task 12.1: Run full test suite (`pytest`, `mypy --strict`, `ruff`, npm tests, C tests).
- [x] Task 12.2: Generate final verification matrix and detailed engineering report.
