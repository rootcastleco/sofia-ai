# CLI & Developer Operations

Sofia Engine ships with an engineering command-line interface (`sofia`) and a rigorous verification toolchain to support edge commissioning, automated diagnostics, and reproducible development.

---

## 1. CLI Reference (`sofia`)

The CLI is installed as a console script via `pip install sofia-engine`. It can also be invoked directly as a Python module: `python -m sofia_ai`.

### `sofia doctor`
Inspects the local runtime environment, Python version, hardware architecture, and checks the status of optional integration libraries (MQTT, Modbus, PyTorch, ONNX Runtime):
```bash
$ sofia doctor
Sofia Engine Doctor (v2.0.0)
[OK] Python 3.12.10 (win32)
[OK] Core numerical runtime: NumPy 2.2.3
[OK] Telemetry adapters: CSV, JSONL, Memory, Replay, Synthetic
[INFO] Industrial protocol extras: NOT INSTALLED (paho-mqtt, pymodbus, pyserial)
[INFO] Deep learning extras: NOT INSTALLED (onnxruntime, torch)
[OK] Security posture: no-pickle enforced, safe deserialization active
```

### `sofia info`
Displays detailed package metadata, license, build commit, and deployment tier capabilities.

### `sofia analyze`
Performs immediate offline signal analysis on a raw telemetry file, extracting time-domain statistics and spectral peaks:
```bash
sofia analyze examples/data/vibration.csv --column vibration_x --sample-rate 1000.0 --shaft-hz 25.0
```

### `sofia benchmark`
Executes internal latency and throughput benchmarks, reporting percentile histograms ($p50$, $p95$, $p99$):
```bash
sofia benchmark --iterations 500 --window-size 1024
```

### `sofia replay`
Replays a recorded JSONL telemetry stream with deterministic timestamp synchronization:
```bash
sofia replay examples/data/edge_offline.jsonl --rate 1.0
```

---

## 2. Development Setup

Sofia requires **Python 3.11+**.

```bash
# 1. Clone repository
git clone https://github.com/rootcastleco/sofia-rl.git
cd sofia-rl

# 2. Create virtual environment
python -m venv .venv

# On Linux/macOS:
source .venv/bin/activate
# On Windows (PowerShell):
.venv\Scripts\Activate.ps1

# 3. Install in editable mode with development dependencies
pip install -e ".[dev]"
```

---

## 3. Code Quality & Static Analysis

All code in Sofia Engine adheres to strict typing, formatting, and security linting rules configured in [`pyproject.toml`](https://github.com/rootcastleco/sofia-rl/blob/main/pyproject.toml):

```bash
# 1. Linting & Formatting (Ruff)
ruff check src/ tests/ examples/
ruff format --check src/ tests/ examples/

# 2. Strict Static Type Checking (Mypy)
mypy src/

# 3. Dependency & Security Auditing
pip-audit
```

---

## 4. Test Suite Execution

The repository maintains an extensive test matrix across multiple test categories:

```bash
# Set PYTHONPATH to src for local runs
$env:PYTHONPATH = "src"

# Run complete test suite with branch coverage enforcement (>= 85%)
pytest --cov=sofia_ai --cov-report=term-missing

# Run isolated test suites
pytest tests/unit               # Mathematical and algorithmic unit tests
pytest tests/contract           # TelemetrySource and ModelBackend contract verification
pytest tests/integration        # Full pipeline fault containment tests
pytest tests/property           # Hypothesis property tests (Parseval, buffer invariants)
pytest tests/architecture       # Architectural boundaries & forbidden import scans
pytest tests/security           # Source scans for pickle, eval, and hardcoded secrets
pytest tests/negative           # Corrupt frames, out-of-bounds inputs, and model rejections
```
