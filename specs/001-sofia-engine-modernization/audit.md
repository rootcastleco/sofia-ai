# SOFIA ENGINE — Current-State Audit (Pre-Modernization Baseline)

Repository: `https://github.com/rootcastleco/sofia-rl`
Audit date: 2026-09-21
Auditor role: principal systems architect
Repository VCS state at audit time: **no commits on `master`** (`git log` empty). All findings below are derived from the working tree, not from history.

---

## 1. Inventory

| Path | Bytes | Purpose | Verdict |
|---|---:|---|---|
| `agent.py` | 6548 | PyTorch DQN (`SofiaRLAgent`, `QNetwork`, `ReplayBuffer`) | Orphaned, unreferenced |
| `LICENSE` | 11357 | Apache License 2.0 (full text) | Authoritative license |
| `README.md` | 2268 | Root project README | Marketing + factual errors |
| `Contributing.md` | 513 | Contribution guidance | Malformed (wrapped in ```` ```markdown ```` fence) |
| `.gitignore` | 622 | Ignore rules | Malformed (wrapped in ```` ``` ```` fence) |
| `sofia_ai/__init__.py` | 562 | Package init | Eager imports of heavy modules |
| `sofia_ai/README.md` | 3451 | Package README | Unbacked benchmark table |
| `sofia_ai/requirements.txt` | 14 | `numpy>=1.24.0` only | Incomplete |
| `sofia_ai/setup.py` | 1758 | Nested setuptools script | Wrong location, wrong metadata |
| `sofia_ai/core/nlp_processor.py` | 11769 | Rule-based NLP | Rule-based, not "transformer" |
| `sofia_ai/core/quantum_engine.py` | 8142 | Dense-matrix "quantum" simulation | Exponential memory |
| `sofia_ai/models/sofia_model.py` | 11697 | `SofiaModel` facade | Pickle-based persistence |
| `sofia_ai/utils/config.py` | 8653 | Dataclass configs | Mixed concerns |
| `sofia_ai/tests/test_sofia.py` | 6407 | 17 tests | Contains a type-confusion bug |

No CI configuration, no `pyproject.toml`, no `docs/`, no `examples/`, no `benchmarks/`, no
`SECURITY.md`, no `CITATION.cff`, no `NOTICE`, no issue templates, no Dependabot config.

---

## 2. Findings

Severity key: **S1** blocking/incorrect, **S2** security or resource hazard, **S3** credibility/quality.

### S1-01 — Root `agent.py` is disconnected from the package
`agent.py` imports `torch` and defines a DQN agent. Nothing in `sofia_ai/` imports it; nothing in
the repository imports `agent.py`. It is not installed by any packaging metadata. Two unrelated
products (NLP chatbot + RL agent) share a repository with no integration point.

### S1-02 — `setup.py` declares a CLI entry point that does not exist
```python
entry_points={"console_scripts": ["sofia=sofia_ai.cli:main"]}
```
There is no `sofia_ai/cli.py` and no `main` function anywhere in the tree. Any wheel built from
this metadata installs a console script that raises `ModuleNotFoundError` on invocation.

### S1-03 — Packaging is not root-level and does not declare the package
`setup.py` lives at `sofia_ai/setup.py`. Run from that directory, `find_packages()` resolves
`core`, `models`, `utils`, `tests` — none of which contain `__init__.py`. The distribution would
not contain an importable `sofia_ai`. It also reads `README.md` and `requirements.txt` by relative
path that only resolve inside that directory.

### S1-04 — Test mixes a runtime object with a configuration field
`sofia_ai/tests/test_sofia.py:114`
```python
config = ModelConfig(quantum=QuantumConfig(...), nlp=NLPProcessor(vocab_size=10000, embedding_dim=256))
```
`ModelConfig.nlp` is typed `NLPConfig`. A `NLPProcessor` is passed. The test only passes because
`SofiaModel.__init__` reads `config.nlp.vocab_size` / `.embedding_dim` / `.languages`, and
`NLPProcessor` happens to expose attributes of the same name. It is an accidental duck-type pass,
not a valid configuration.

### S1-05 — `QuantumNeuralEngine.optimize` is dimensionally inconsistent (raises)
`quantum_engine.py:233`
```python
gradient = np.outer(loss_gradient, weight.T)     # weight.T shape (D,D) -> flattened D*D
self.weights[i] -= learning_rate * gradient      # weights[i] shape (D,D) -> broadcast error
```
`np.outer` flattens both operands, producing shape `(D, D*D)`. The in-place subtraction against a
`(D, D)` array raises `ValueError`. The optimization path is unreachable/dead code.

### S1-06 — Vocabulary is never populated; all embeddings are identical
`nlp_processor.py:80-88,142-151`. `_build_vocabulary()` returns only four special tokens.
`tokenize()` never inserts observed words into `self.vocabulary`. `_get_embedding()` therefore
returns the `<UNK>` row for every real token. The `embeddings` array produced by `process()` is a
stack of identical rows and carries zero lexical information. Every downstream consumer
(`get_contextual_representation`, `SofiaModel.process` → quantum features) operates on noise.

### S1-07 — Documented API does not exist
Root `README.md:41-48` instructs:
```python
from sofia_core import SofiaQuantumNLP
sofia = SofiaQuantumNLP(model_size="large")
```
No module `sofia_core` exists; no class `SofiaQuantumNLP` exists; `SofiaModel` has no
`model_size` parameter. The documented quickstart cannot execute.

### S2-01 — Exponential memory growth in the quantum component
`quantum_engine.py:54` `self.dimension = 2 ** num_qubits`.
`_initialize_weights()` allocates `entanglement_depth` matrices of shape `(D, D)` and runs
`np.linalg.svd` on each. Memory and time therefore scale as `O(depth · 4^n)`:

| `num_qubits` | `D` | float64 per matrix | ×depth 3 | SVD feasibility |
|---:|---:|---:|---:|---|
| 8 (default) | 256 | 0.5 MB | 1.6 MB | feasible |
| 12 | 4096 | 134 MB | 403 MB | marginal |
| 16 | 65 536 | 34 GB | 103 GB | infeasible |
| 20 | 1 048 576 | 8.8 TB | 26 TB | infeasible |

`QuantumConfig.__post_init__` explicitly permits `qubits` up to **20**, i.e. the validated
configuration space includes configurations that cannot be constructed on any machine. This is
unsuitable for edge/embedded execution by construction, not merely by tuning.

### S2-02 — Unsafe deserialization in `SofiaModel.load`
`sofia_model.py:298-323` calls `pickle.load()` on a caller-supplied path with no signature,
checksum, format version, or allow-list. Loading an attacker-controlled `.pkl` yields arbitrary
code execution. There is no manifest, no schema validation, and no compatibility gate.

### S2-03 — Global RNG mutation as a constructor side effect
`agent.py:36-41` and `:79-81`. `ReplayBuffer.__init__` and `SofiaRLAgent.__init__` call
`random.seed()` / `np.random.seed()` on the *global* generators. Constructing an agent silently
re-seeds every other consumer of the process-wide RNG. There is no owned `np.random.Generator`.

### S2-04 — Large eager allocation at construction
`nlp_processor.py:88` allocates `np.random.randn(50000, 768)` ≈ **307 MB** (float64) per
`NLPProcessor` instance, during `__init__`. There is no lazy path, no memory ceiling, and no
dtype control. `SofiaModel()` therefore allocates >300 MB before doing any work.

### S2-05 — Unbounded, unvalidated inputs
No component validates for `NaN`/`inf`, bounds, payload size, or timestamp ordering. There are no
bounded queues anywhere in the repository. No timeout exists on any operation. Failure modes
(raw `IndexError`, `ValueError`, silent `nan` propagation) are unspecified.

### S3-01 — License inconsistency
`LICENSE` is Apache License 2.0. Root `README.md:53` and `sofia_ai/README.md:95` both state MIT.
`setup.py:29` declares `License :: OSI Approved :: MIT License`. Three sources, two licenses.

### S3-02 — Unbacked benchmark table
`sofia_ai/README.md:80-87` publishes accuracy/latency figures (98.7 %, 12 ms, 97.9 %, 15 ms,
99.1 %, 10 ms, 96.8 %, 25 ms). No benchmark code, no dataset, no harness, and no hardware
description exist anywhere in the repository. The measured system also contains no classifier, so
the "accuracy" column cannot even in principle be produced by this code.

### S3-03 — Unbacked capability claims
"Transformer-based Models", "multi-head attention", "Memory Networks", "Real-time Processing",
"Multi-language Support", "quantum computational efficiency", "unprecedented levels of language
understanding". No transformer, attention, memory network, latency test, or non-English language
path exists in the code. `languages=['en']` is a stored list only.

### S3-04 — No quantum advantage evidence
The "quantum" component is a dense real-valued matrix simulation. `forward()` computes
`normalize(W_d … W_1 normalize(x))`, then `|·|²`, i.e. a composition of random orthogonal maps
followed by a nonlinearity. No benchmark compares it against a classical baseline. The quantum
narrative ("entanglement", "interference", "measurement") is applied to operations that do not
have those semantics: `apply_entanglement` is a matrix–vector product, `measure` is
`np.random.choice`, and `get_quantum_features` computes `|·|²` of an already-normalized
probability vector, which is not an entropy of anything well defined.

### S3-05 — Identity/attribution is placeholder
`setup.py:23` `url="https://github.com/your-org/sofia-ai"`; `author="Sofia AI Team"`;
`contact@sofia-ai.org`; a Discord invite. None of these correspond to
**Rootcastle Engineering & Innovation** or to `rootcastleco/sofia-rl`.

### S3-06 — Dependency set inconsistent with code
`requirements.txt` declares `numpy` only. `agent.py` imports `torch`. No declared dependency
covers `torch`, `pytest`, or the tools named in `Contributing.md` (Black, isort, pre-commit).

### S3-07 — No test/lint/type/CI infrastructure
No `pyproject.toml`, no `pytest.ini`/config, no coverage configuration, no formatter
configuration, no type-check configuration, no workflows. Coverage is unmeasured.

### S3-08 — Malformed metadata files
`.gitignore` and `Contributing.md` are both wrapped in Markdown code fences, so the leading
```` ``` ```` line is treated as content (`.gitignore` therefore ignores a literal ```` ``` ````
pattern and, on some toolchains, nothing else behaves as intended).

### S3-09 — Subpackages are not packages
`sofia_ai/core`, `sofia_ai/models`, `sofia_ai/utils`, `sofia_ai/tests` contain no `__init__.py`.
Imports only work because of Python 3 namespace packages. `find_packages()` cannot discover them.

---

## 3. Verified vs. unverified

All findings above were verified by reading the working tree. No claim in this audit rests on the
instruction text; where the instruction text and the tree disagree, the tree was used.

The single item from the instruction list that could **not** be confirmed as stated: the claim that
`SofiaRLAgent` uses "Double DQN". It does (`local.argmax` for action selection, `target.gather`
for evaluation, `agent.py:143-145`), so that part of the existing code is correct and is preserved.

---

## 4. Preservation set (regression envelope)

The following existing behaviors are retained, not deleted:

| Symbol | Original location | Disposition |
|---|---|---|
| `SofiaModel` | `sofia_ai/models/sofia_model.py` | Preserved as deprecated legacy facade |
| `QuantumNeuralEngine`, `QuantumState` | `sofia_ai/core/quantum_engine.py` | Migrated to `sofia_ai.experimental.quantum`, API preserved |
| `NLPProcessor`, `Token`, `ProcessedText` | `sofia_ai/core/nlp_processor.py` | Migrated to `sofia_ai.copilot.legacy_nlp`, API preserved |
| `QuantumConfig`, `NLPConfig`, `ModelConfig` | `sofia_ai/utils/config.py` | Preserved, validation corrected, re-exported |
| `SofiaRLAgent`, `QNetwork`, `ReplayBuffer` | `agent.py` | Migrated to `sofia_ai.learning.rl`, API preserved |
| `from sofia_ai import SofiaModel, QuantumNeuralEngine, NLPProcessor, QuantumConfig` | `sofia_ai/__init__.py` | Preserved (lazy, deprecation-warned) |

See `migration.md` for the full mapping and `convergence.md` for the audit of the final state.
