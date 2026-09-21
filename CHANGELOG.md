# Changelog

All notable changes to **Sofia Engine** are documented here. Format based on
[Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Added

- **`sofia-engine` 2.0 era rebuild.** A from-scratch implementation of the
  Sofia platform: telemetry ingestion (CSV/JSONL/memory/replay/synthetic +
  MQTT/Modbus/serial), signal processing, statistical & spectral & rotating
  feature extraction, anomaly detection, diagnostics, a safety-first command
  `PolicyEngine` (default DENY), bounded edge buffers with store-and-forward,
  reinforcement learning (dueling-DQN, torch extra), an experimental bounded
  quantum simulator with a classical baseline, a copilot kept outside the
  control path, a `sofia` CLI and a benchmarking suite.
- **Embedded C reference** of the DSP feature extractors with golden-vector
  conformance tests.
- **Test pyramid**: unit, contract, integration, property (Hypothesis),
  architecture boundary, security sweep, and negative tests. Coverage facade is
  configured to enforce ≥ 85%.
- **Specifications**: full audit, specification, architecture, threat model,
  test strategy and task tracking with `SOFIA-*` requirement IDs.

### Changed

- Legacy `sofia_ai` package moved to `legacy/sofia_ai_v1/` (unchanged).
  Root-level `agent.py` is now a deprecation shim over `sofia_ai.learning.rl`.
- Runtime dependency surface reduced to NumPy; all protocols became extras.
- Determinism: global RNG seeding removed; generators and time sources are
  injected.

### Fixed

- One-sided PSD folding now respects the transform parity (even vs odd
  transform lengths), so Parseval's relation holds exactly for both cases.
- Band-energy estimates use the correct bin-width normalization for arbitrary
  frequency axes.

### Security

- No pickle on any load path; no eval/exec/shell=True in library code
  (enforced by tests).
- Command policy has no bypass and defaults to deny.

## [1.0.0] — legacy

The original `sofia_ai` research prototype. Preserved at
`legacy/sofia_ai_v1/` for reference and compatibility. Not supported.