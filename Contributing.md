# Contributing to Sofia Engine

Thank you for considering contributing to **Sofia Engine**.

This project is a from-scratch rebuild of the original `sofia-rl` codebase. The
legacy implementation lives under `legacy/sofia_ai_v1/` and is preserved for
reference and compatibility. All new development happens in `src/sofia_ai/`.

## Ground rules

- **Evidence beats hype.** Every published capability must have executable
  tests or benchmarks in this repository. No made-up numbers, no unverifiable
  accuracy claims. If a feature is not yet proven, say so in code and docs.
- **Safety first.** The command/decision layer defaults to DENY. There is no
  bypass. Changes that loosen a safety invariant will not be merged.
- **Bounded resources.** Buffers, queues, caches and the experimental quantum
  simulator must keep explicit ceilings. Memory must already be finite.
- **Deterministic by default.** The core must not seed or mutate global RNGs.
  Pass in a generator, or inject a `TimeSource`.
- **Offline-first.** Runtime dependencies are limited to `numpy`. Optional
  integrations (MQTT, Modbus, serial, ONNX, torch, API) are extras and must
  never be imported by the core at import time.

## Repository layout

- `src/sofia_ai/` — the package
- `tests/` — unit / contract / integration / property / architecture /
  security / compat tests
- `embedded/` — C reference implementation of the DSP feature extractors
- `examples/` — runnable end-to-end demos
- `specs/001-sofia-engine-modernization/` — the original audit, specification,
  architecture, threat model, and task record for this rebuild
- `legacy/sofia_ai_v1/` — preserved legacy implementation

## Development workflow

1. Fork the repository and create a feature branch.
2. Make small, reviewable changes. Follow the existing style (`ruff format`,
   `ruff check`, `mypy --strict`).
3. Write or update tests for any behavioural change.
4. Run the full verification locally (see below) and make sure everything is
   green.
5. Open a Pull Request with a clear description and link to related issues
   or requirements (reference `SOFIA-*` IDs from the spec where relevant).
6. For breaking changes or design proposals, open an issue first.

## Local verification

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev]"
$env:PYTHONPATH = "src"    # PowerShell; on POSIX: export PYTHONPATH=src

# tests
.venv/Scripts/python -m pytest            # full suite
.venv/Scripts/python -m pytest --cov      # with coverage (>= 85% required)

# static analysis
.venv/Scripts/ruff check .
.venv/Scripts/ruff format --check .
.venv/Scripts/mypy src/sofia_ai

# package build
.venv/Scripts/python -m build

# examples
.venv/Scripts/python examples/01_basic_analysis.py
```

Optional integrations are verified with their extra installed:

```bash
.venv/Scripts/python -m pip install -e ".[industrial,onnx,torch]"
```

Live-broker/hardware tests are skipped automatically when no broker or serial
port is available (`requires_broker` marker). Embedded host tests
(`requires_cc`) require a C compiler and are skipped otherwise.

## Code of conduct

See `CODE_OF_CONDUCT.md`.