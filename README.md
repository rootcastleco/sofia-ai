# Sofia Engine

Offline-first edge intelligence for industrial telemetry, machine health,
embedded ML and safe technical automation.

Sofia Engine is a from-scratch, engineering-grade rebuild of the original
`sofia-rl` codebase. It keeps the Apache-2.0 license, preserves first-major
-version API compatibility with the legacy package, and replaces the prototype
with a tested, documented, resource-bounded platform that runs with **NumPy as
its only runtime dependency**.

The legacy implementation is preserved unmodified under
[`legacy/sofia_ai_v1/`](legacy/sofia_ai_v1/).

---

## Why Sofia Engine

Industrial decision support demands properties that research prototypes
usually lack:

- **Evidence beats hype** — every module ships with executable tests. We do
  not publish accuracy or performance numbers we cannot reproduce from this
  repository.
- **Safety first** — the command/decision layer defaults to **DENY**. There is
  no bypass path.
- **Bounded resources** — buffers, queues, caches and even the experimental
  quantum simulator have explicit ceilings. Memory usage is always finite.
- **Deterministic by default** — the core uses injected RNGs and time sources,
  never global seeds, so runs are repeatable.
- **Offline-first** — core functionality has no network, broker, GPU or
  database dependency. Integrations (MQTT, Modbus, serial, ONNX, torch, API)
  are optional extras that are never imported by the core.

## What it does

| Area | Modules |
| --- | --- |
| Telemetry ingestion | CSV / JSONL / in-memory / replay / synthetic sources, MQTT, Modbus, serial (extras) |
| Signal processing | Windowing, filtering, resampling, detrending, envelope, spectral density (correct Parseval folding) |
| Feature extraction | Statistical, spectral and rotating-machine features |
| Machine health | Anomaly detectors (threshold, z-score, rolling std, IQR, velocity, nonlinearity), ONNX & torch backends |
| Diagnostics | Rule-based diagnosis with confidence, guided by data quality |
| Safe decisions | `PolicyEngine` with explicit rules, operator approval, physical interlocks; default DENY |
| Edge resilience | Bounded ring buffers, store-and-forward with byte ceilings, reconnect logic, health monitoring |
| Reinforcement learning | Replay buffers, dueling-DQN; training/simulation only (extra, lazy torch) |
| Copilot | Technical decision-support assistant, isolated from the control path |
| Quantum research | Experimental bounded simulator with an honest classical baseline (extra) |
| Security | Safe deserialization (no pickle), secrets handling, payload signing |
| Tooling | `sofia` CLI + a benchmarking suite with p50/p95/p99 and environment capture |
| Embedded reference | C implementation of the DSP feature extractors with golden-vector tests |

## Install

Requires Python 3.11+.

```bash
# minimal runtime (numpy only)
pip install sofia-engine

# with protocol adapters
pip install "sofia-engine[industrial]"

# everything (also pulls torch, onnxruntime)
pip install "sofia-engine[dev]"
```

From source (development):

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev]"
```

> On Windows the `python` launcher can be shadowed by the Microsoft Store
> alias; use the venv interpreter directly (`.venv\Scripts\python.exe`).

## Quick start

### Python

```python
from sofia_ai.telemetry.synthetic import SyntheticTelemetrySource
from sofia_ai.signal import power_spectral_density
from sofia_ai.telemetry import stream_to_array

telemetry = stream_to_array(SyntheticTelemetrySource(seed=42), limit=1024)
values = telemetry["vibration"]["vibration_x"]

psd = power_spectral_density(values, sample_rate=2560.0)
print(f"RMS vibration: {telemetry['vibration'].rms:.3f} units")
print(f"Dominant frequency: {psd.frequencies[psd.values.argmax()]:.1f} Hz")
```

### Command line

```bash
sofia info                          # package, extras, platform
sofia doctor                        # environment + optional-integration status
sofia analyze examples/data/vibration.csv --column vibration_x
sofia benchmark                     # run the built-in benchmark suite
sofia inspect examples/data/manifest.json
sofia replay examples/data/edge_offline.jsonl
```

Run the shipped demos:

```bash
.venv/Scripts/python examples/01_basic_analysis.py
.venv/Scripts/python examples/02_batch_health.py
.venv/Scripts/python examples/03_decision_gate.py
.venv/Scripts/python examples/04_edge_offline.py
.venv/Scripts/python examples/05_copilot_and_rl.py
```

## Testing

```
tests/unit         behavioural unit tests
tests/contract     public-interface contract tests
tests/integration  end-to-end pipelines
tests/property     Hypothesis property tests (Parseval, buffer invariants, ...)
tests/architecture module-boundary & forbidden-dependency enforcement
tests/security     source sweeps (no pickle, no eval, no bare-except swallowing, ...)
tests/negative     failure-mode tests
```

```bash
$env:PYTHONPATH = "src"        # PowerShell — the package isn't installed yet
.venv/Scripts/python -m pytest                # full suite
.venv/Scripts/python -m pytest --cov          # coverage (>= 85% enforced)
```

Optional-integration tests skip automatically when torch / a broker / a serial
port / a C compiler is unavailable (`torch`, `requires_broker`, `requires_cc`
markers).

## Embedded reference

[`embedded/`](embedded/) contains a C implementation of the core DSP feature
extractors with a python-generated golden-vector suite, so host and embedded
behavior stay in lock-step. Build it where a C toolchain is available:

```bash
cd embedded && make test
```

## Specifications

The full engineering record — audit, specification, architecture, threat model,
test strategy and task tracking — lives in
[`specs/001-sofia-engine-modernization/`](specs/001-sofia-engine-modernization/).
Requirement IDs (`SOFIA-*`) are traceable from spec to implementation to tests.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Ground rules in one line: evidence
beats hype, safety first, bounded resources, deterministic by default,
offline-first.

## Security

See [SECURITY.md](SECURITY.md) for the threat model and security policy. Please
report vulnerabilities privately; never file a public issue for a security bug.

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).