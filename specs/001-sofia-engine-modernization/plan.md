# SOFIA ENGINE — Implementation Plan

Strategy: **branch-by-abstraction**. The legacy package is preserved behind compatibility shims
while a new `src/sofia_ai/` package is built alongside it. Each phase leaves the repository
importable and testable.

## Phase 0 — Baseline
1. Audit the working tree; write `audit.md`. **DONE**
2. Vendor nothing; keep `LICENSE` (Apache-2.0) byte-identical. **DONE**

## Phase 1 — Packaging + core contracts
* Root `pyproject.toml` (PEP 517/518), Apache-2.0 metadata, Rootcastle attribution,
  `rootcastleco/sofia-rl` URLs, optional extras, dev group.
* `src/sofia_ai/core/`: errors, units, time, quality, validation, contracts, serialization, hashing.
* `src/sofia_ai/config/`: typed loader.
* Decision: `src/` layout so the installed package cannot accidentally import the legacy tree.

## Phase 2 — Telemetry + signal pipeline
* `telemetry/base.py` ABC + registry; CSV/JSONL/synthetic/replay sources; MQTT/Modbus/serial behind
  extras with lazy imports.
* `signal/`: windowing, filters, spectral, resample, detrend, envelope.
* `features/`: statistical, spectral, rotating-machinery extension points, extractor.

## Phase 3 — Inference + anomaly detection
* `inference/base.py` `ModelBackend`; `inference/manifest.py`; detectors
  (threshold, z-score, MAD, EWMA, IQR, CUSUM); registry; optional ONNX and PyTorch backends.

## Phase 4 — Diagnostics + health events
* `diagnostics/`: evidence, rules, engine, health scoring.
* `decision/`: `CommandRequest`, `CommandDecision`, policy engine (default DENY), replay guard.

## Phase 5 — RL + experimental quantum migration
* `learning/rl/`: networks, buffers, agent — API preserved, RNG owned, deterministic eval mode,
  checkpoint validation, actuation disabled by default.
* `experimental/quantum/`: engine preserved, resource ceiling enforced, classical baseline
  benchmark, honest documentation.

## Phase 6 — Edge runtime
* `edge/`: bounded ring buffer, store-and-forward, reconnect with backoff ceiling, runtime, health.

## Phase 7 — Embedded export boundary
* `embedded/`: C99 reference feature runtime, Q16.16 fixed-point header, golden vectors,
  build/test script. Marked per-target VERIFIED / NOT VERIFIED.

## Phase 8 — Observability + security
* `observability/`: structured JSON logging with rate limiting, metrics with percentiles, redaction.
* `security/`: safe IO with path confinement, Ed25519 signing hook, env-only secrets.

## Phase 9 — CLI + examples
* `cli/main.py`: `info`, `doctor`, `inspect`, `analyze`, `replay`, `benchmark`, `models`, `devices`,
  `config`. No actuation commands.
* `examples/`: five runnable demos + generated data.

## Phase 10 — Docs + branding + discoverability
* README, ARCHITECTURE, CONTRIBUTING, SECURITY, CODE_OF_CONDUCT, ROADMAP, CHANGELOG, SUPPORT,
  CITATION.cff, NOTICE, `docs/` (MkDocs), Rootcastle SVG assets, CI workflows, issue templates,
  Dependabot, CodeQL.

## Sequencing constraints
* No phase may leave `pytest` red.
* `compat/` shims are added in the same phase that moves a legacy module, never later.
* No benchmark number is written into any README until `benchmarks/` produces it.

## Rejected alternatives
| Option | Why rejected |
|---|---|
| In-place refactor of `sofia_ai/` | Legacy tree has no `__init__.py` in subpackages and no tests worth protecting in place; a parallel `src/` tree gives a clean boundary and lets shims be tested |
| Delete `agent.py` and the quantum module | Mission requires audit-first and backward compatibility; both are preserved with honest labeling |
| MkDocs Material theme | Upstream announced maintenance mode with EOL 2026-11-05; core MkDocs + custom CSS is maintainable |
| Adopt `pydantic` for contracts | Adds a heavy dependency to the minimal install; hand-written dataclass validation keeps the base install numpy-only |
| Add `scipy` for DSP | Would make the minimal install heavy; required transforms are implemented directly with numpy and validated against closed-form expectations |
