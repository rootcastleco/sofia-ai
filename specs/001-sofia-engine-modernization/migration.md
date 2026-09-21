# SOFIA ENGINE — Migration Guide

Target version boundary: **2.0.0**. Everything in the 1.x surface is preserved behind
`sofia_ai.compat` with a `DeprecationWarning`. Removal is planned for 3.0.0 at the earliest.

---

## 1. Import mapping

| Pre-2.0 import | 2.0 import | Status |
|---|---|---|
| `from sofia_ai import SofiaModel` | `from sofia_ai.compat import SofiaModel` | Deprecated (still works from `sofia_ai`) |
| `from sofia_ai import QuantumNeuralEngine` | `from sofia_ai.experimental.quantum import QuantumNeuralEngine` | Deprecated shim |
| `from sofia_ai import NLPProcessor` | `from sofia_ai.copilot.legacy_nlp import NLPProcessor` | Deprecated shim |
| `from sofia_ai import QuantumConfig` | `from sofia_ai.config.legacy import QuantumConfig` | Preserved, validation corrected |
| `from sofia_ai.utils.config import NLPConfig` | `from sofia_ai.config.legacy import NLPConfig` | Preserved |
| `from sofia_ai.utils.config import ModelConfig` | `from sofia_ai.config.legacy import ModelConfig` | Preserved |
| `from sofia_ai.core.quantum_engine import QuantumState` | `from sofia_ai.experimental.quantum import QuantumState` | Deprecated shim |
| `from sofia_ai.core.nlp_processor import Token, ProcessedText` | `from sofia_ai.copilot.legacy_nlp import Token, ProcessedText` | Deprecated shim |
| `agent.py` root module | `from sofia_ai.learning.rl import SofiaRLAgent, QNetwork, ReplayBuffer` | Deprecated shim at `agent.py` |

`from sofia_ai import SofiaModel, QuantumNeuralEngine, NLPProcessor, QuantumConfig` continues to
work and is covered by `tests/compat/test_legacy_imports.py`.

## 2. Behavioral changes (with rationale)

| Change | Rationale | Version | Migration |
|---|---|---|---|
| `QuantumConfig(qubits=…)` now enforces a hard ceiling of 12 | `2**n` memory growth made qubits ≥ 16 unconstructible (S2-01) | 2.0.0 | Use ≤ 12, or use the documented classical baselines instead |
| `QuantumNeuralEngine.optimize` no longer raises | The original `np.outer` gradient had incompatible shape (S1-05) | 2.0.0 | No action; signature preserved |
| `NLPProcessor` no longer allocates a 307 MB matrix at construction | Resource ceiling (S2-04) | 2.0.0 | Set `vocab_size`/`embedding_dim` explicitly; allocation is now lazy |
| `SofiaModel.save`/`load` no longer use `pickle` | Unsafe deserialization (S2-02) | 2.0.0 | Re-save models with `save()`; old `.pkl` files are not loadable by design |
| `ReplayBuffer`/`SofiaRLAgent` no longer seed global RNG | Global side effect (S2-03) | 2.0.0 | Pass `seed=`; use the returned `agent.rng` for reproducible sampling |
| `ModelConfig.nlp` must be `NLPConfig`, not `NLPProcessor` | Type confusion in the legacy test (S1-04) | 2.0.0 | `ModelConfig(nlp=NLPConfig(vocab_size=…, embedding_dim=…))` |
| Root `setup.py` → root `pyproject.toml` | Packaging was non-functional (S1-02, S1-03) | 2.0.0 | `pip install -e ".[dev]"` |
| CLI entry point `sofia` now resolves | Previously pointed at a nonexistent module (S1-02) | 2.0.0 | `sofia --help` |
| License metadata = Apache-2.0 | Matched to `LICENSE` (S3-01) | 2.0.0 | No action |
| Quantum component moved to `experimental/` | No measured advantage; quarantine (S3-04) | 2.0.0 | Update import path or keep using the shim |
| NLP component moved to `copilot/legacy_nlp/` | Rule-based, not the documented transformer system (S1-06) | 2.0.0 | Update import path or keep using the shim |

## 3. Regression envelope

These behaviors are asserted by `tests/compat/` and must not change before 3.0.0:

1. `SofiaModel().process(text)` returns an object with `.text`, `.intent`, `.confidence`,
   `.sentiment`, `.entities`, `.processing_time_ms`, `.metadata`.
2. `SofiaModel().chat("Hello!")` returns a greeting string.
3. `NLPProcessor().extract_intent("Hello, how are you?") == ("greeting", c)` with `c > 0`.
4. `NLPProcessor().analyze_sentiment("This is excellent")` → `positive > negative`.
5. `QuantumNeuralEngine(num_qubits=4).dimension == 16`; `create_superposition()` has unit norm.
6. `QuantumConfig(qubits=8, entanglement_depth=3).to_dict()["qubits"] == 8`.
7. `ModelConfig().save(p)` / `ModelConfig.load(p)` round-trip.
8. `SofiaRLAgent(state_size=4, action_size=2).act(np.zeros(4))` returns an int in `[0, 2)`.
9. `ReplayBuffer(capacity=5)` holds at most 5 items.

## 4. Deprecation mechanics

`sofia_ai/compat.py` emits `DeprecationWarning` with a fixed message format:

```
SofiaModel is deprecated since Sofia 2.0.0 and will be removed in 3.0.0.
Use sofia_ai.copilot.legacy_nlp / sofia_ai.diagnostics instead.
```

Warnings are raised on access via module `__getattr__`, so merely importing `sofia_ai` does not
warn — only using a deprecated name does.

## 5. Data migration

No on-disk format from 1.x is carried forward:

* `.pkl` model files: **not loadable**. Rationale: they are an arbitrary-code-execution vector and
  there is no schema to validate. Re-save using the 2.0 `save()` (JSON manifest + NPZ artifact).
* JSON config files written by `ModelConfig.save()` remain loadable by `config/legacy.py`.

## 6. Checklist for downstream users

1. Replace `pip install -r requirements.txt` with `pip install -e ".[dev]"`.
2. Replace `python -m pytest sofia_ai/tests` with `pytest`.
3. Replace `from sofia_ai import ...` with the 2.0 paths in the table above.
4. Add `torch` via the `torch` extra if you use `learning.rl`; it is no longer implied.
5. Remove any reliance on `agent.py`.
