# SOFIA ENGINE — Task Record

Status legend: `DONE` implemented and verified in this repository · `DOCS` documentation-only ·
`BLOCKED` requires owner authorization or external infrastructure.

## Phase 0 — Baseline
| Task | Requirement | Status |
|---|---|---|
| Audit working tree, enumerate defects | — | DONE |
| Preserve `LICENSE` byte-identical | SOFIA-NFR-004 | DONE |

## Phase 1 — Packaging + core
| Task | Requirement | Status |
|---|---|---|
| Root `pyproject.toml` with PEP 517/518 backend | SOFIA-PKG-001 | DONE |
| Apache-2.0 license metadata + classifier | SOFIA-PKG-002 | DONE |
| Rootcastle URLs and attribution | SOFIA-PKG-003 | DONE |
| Optional extras: mqtt/modbus/serial/onnx/torch/industrial/dev/docs | SOFIA-PKG-004 | DONE |
| `core/errors.py` typed error hierarchy | SOFIA-SEC-009 | DONE |
| `core/units.py` unit registry + dimension checking | SOFIA-UNIT-001..004 | DONE |
| `core/time.py` TimeSource, clock anomaly detection | SOFIA-TIME-001..004 | DONE |
| `core/quality.py` DataQuality enum + policy | SOFIA-DQ-001 | DONE |
| `core/validation.py` finite/range/identifier checks | SOFIA-FR-012 | DONE |
| `core/contracts.py` 8 domain contracts | SOFIA-FR-001..009 | DONE |
| `core/serialization.py` JSON round-trip | SOFIA-FR-011 | DONE |
| `config/` typed validated loader with env overrides | §39 | DONE |

## Phase 2 — Telemetry + signal
| Task | Requirement | Status |
|---|---|---|
| `telemetry/base.py` `TelemetrySource` ABC + limits | SOFIA-TLM-001,009,010 | DONE |
| CSV / JSONL / synthetic / replay sources | SOFIA-TLM-002..004 | DONE |
| MQTT source behind extra | SOFIA-TLM-005 | DONE |
| Modbus TCP/RTU source behind extra | SOFIA-TLM-006 | DONE |
| Serial source behind extra | SOFIA-TLM-007 | DONE |
| Adapter registry | SOFIA-TLM-008 | DONE |
| `signal/` windowing, filters, resample, detrend, envelope | SOFIA-SIG-003..007 | DONE |
| `signal/spectral.py` FFT/PSD/Welch/centroid/peaks | SOFIA-SIG-002 | DONE |
| `features/statistical.py` 13 time-domain features | SOFIA-SIG-001 | DONE |
| `features/spectral.py` spectral features + band energy | SOFIA-SIG-002 | DONE |
| `features/rotating.py` order/harmonic/sideband extensions | SOFIA-SIG-008 | DONE |
| `features/extractor.py` deterministic ordered extractor | SOFIA-SIG-009 | DONE |

## Phase 3 — Inference
| Task | Requirement | Status |
|---|---|---|
| `inference/base.py` `ModelBackend` + `InferenceResult` | SOFIA-INF-001,005,009 | DONE |
| `inference/detectors.py` 6 statistical detectors | SOFIA-INF-002 | DONE |
| `inference/manifest.py` + checksum + compatibility gate | SOFIA-INF-004,005 | DONE |
| `inference/registry.py` explicit registration | SOFIA-INF-003, SOFIA-SEC-007 | DONE |
| ONNX backend behind extra | SOFIA-INF-006 | DONE |
| PyTorch backend behind extra | SOFIA-INF-007 | DONE |
| Typed inference failure handling | SOFIA-INF-008 | DONE |

## Phase 4 — Diagnostics + safety
| Task | Requirement | Status |
|---|---|---|
| `diagnostics/evidence.py` | SOFIA-FR-006 | DONE |
| `diagnostics/rules.py` versioned rules | SOFIA-DIAG-001 | DONE |
| `diagnostics/engine.py` evidence → HealthEvent | SOFIA-DIAG-002,004 | DONE |
| `diagnostics/health.py` health score + uncertainty band | SOFIA-DIAG-003 | DONE |
| `decision/commands.py` | SOFIA-FR-009 | DONE |
| `decision/policy.py` default-DENY engine | SOFIA-SAFE-001..003 | DONE |
| `decision/replay.py` nonce/TTL replay guard | SOFIA-SAFE-004 | DONE |

## Phase 5 — RL + quantum
| Task | Requirement | Status |
|---|---|---|
| `learning/rl/networks.py` | SOFIA-COMPAT-004 | DONE |
| `learning/rl/buffers.py` bounded buffer, owned RNG | SOFIA-SAFE-005, S2-03 | DONE |
| `learning/rl/agent.py` API preserved, deterministic eval | SOFIA-COMPAT-004 | DONE |
| `experimental/quantum/engine.py` preserved + ceiling | SOFIA-QEXP-001,003 | DONE |
| Classical baseline benchmark | SOFIA-QEXP-004 | DONE |
| Honest "no measured advantage" documentation | SOFIA-QEXP-005 | DONE |

## Phase 6 — Edge
| Task | Requirement | Status |
|---|---|---|
| `edge/buffer.py` bounded ring buffer | SOFIA-EDGE-001 | DONE |
| `edge/store_forward.py` byte-ceiling persistence | SOFIA-EDGE-002 | DONE |
| `edge/reconnect.py` bounded retry/backoff | SOFIA-EDGE-003 | DONE |
| `edge/runtime.py` full local pipeline | SOFIA-EDGE-004,005 | DONE |
| `edge/health.py` readiness/liveness | SOFIA-EDGE-006 | DONE |

## Phase 7 — Embedded
| Task | Requirement | Status |
|---|---|---|
| `embedded/src/sofia_features.c/.h` C99 runtime | SOFIA-EMB-002,007 | DONE (host-verified) |
| `embedded/src/sofia_fixed.h` Q16.16 | SOFIA-EMB-003 | DONE (host-verified) |
| Golden vector generator + comparison | SOFIA-EMB-005 | DONE |
| CMSIS-DSP / TFLite Micro guidance | SOFIA-EMB-004 | DOCS (NOT VERIFIED) |

## Phase 8 — Observability + security
| Task | Requirement | Status |
|---|---|---|
| `observability/logging.py` structured + rate-limited | SOFIA-OBS-001,002,005 | DONE |
| `observability/metrics.py` counters, gauges, percentiles | SOFIA-OBS-004,006 | DONE |
| `observability/redaction.py` | SOFIA-OBS-003, SOFIA-SEC-015 | DONE |
| `security/safeio.py` path confinement, JSON-only loaders | SOFIA-SEC-001,004 | DONE |
| `security/signing.py` Ed25519 hook | SOFIA-SEC-003 | DONE |
| `security/secrets.py` env-only | SOFIA-SEC-005 | DONE |

## Phase 9 — CLI + examples
| Task | Requirement | Status |
|---|---|---|
| `cli/main.py` with 9 subcommands | SOFIA-PKG-005 | DONE |
| `sofia_ai/__main__.py` | SOFIA-PKG-006 | DONE |
| 5 runnable examples + data generator | §20 | DONE |
| `benchmarks/` suite with p50/p95/p99 | §23 | DONE |

## Phase 10 — Docs + branding + CI
| Task | Requirement | Status |
|---|---|---|
| README rebuild, claims match behavior | §28, §48 | DONE |
| ARCHITECTURE / CONTRIBUTING / SECURITY / CoC / ROADMAP / CHANGELOG / SUPPORT | §26 | DONE |
| CITATION.cff, NOTICE | §26 | DONE |
| `docs/` MkDocs site + tutorials | §32, §33 | DONE |
| Rootcastle SVG assets | §27 | DONE |
| GitHub Actions: ci, codeql, docs, release, benchmarks | §24, §25 | DONE |
| Issue templates, PR template, Dependabot | §34 | DONE |
| GitHub topics/description script (requires auth) | §31 | BLOCKED — requires owner `gh` authorization |

## Blocked items
| Item | Blocker |
|---|---|
| Applying repository description/topics via GitHub API | Requires authenticated `gh`/API write access; a local script `tools/set_repo_metadata.sh` is provided for the owner to run |
| Publishing a package to PyPI under a new name | Requires owner authorization; explicitly out of scope |
| MCU toolchain build (STM32/ESP32) | No cross-toolchain or hardware in this environment; marked NOT VERIFIED |
| Real MQTT/Modbus broker integration test | No broker in this environment; adapters are unit-tested with injected fakes |
