# Security Policy & Threat Model

Industrial IoT edge nodes operate in adversarial or exposed network environments (factory floors, substation gateways, remote pump stations). Sofia Engine is built around a comprehensive **threat model** and a defense-in-depth security architecture.

---

## 1. Trust Boundaries

Sofia defines six explicit trust boundaries (documented in [`specs/001-sofia-engine-modernization/threat-model.md`](https://github.com/rootcastleco/sofia-rl/blob/main/specs/001-sofia-engine-modernization/threat-model.md)):

```
[ Field Network / PLCs / Sensors ]
              |
====== TB-1: Transport Boundary ======
              |
[ Telemetry Adapters (MQTT, Modbus, Serial) ]
              |
      [ Bounded Ring Buffers ]
              |
[ DSP & Feature Extraction Core ]
              |
====== TB-2: Model Supply Boundary ======
              |
[ Model Manifests & Inference Backends ]
              |
[ Diagnostic Engine & Evidence Scoring ]
              |
====== TB-4: Operator Boundary ======
              |
[ CLI / API / Copilot ]
              |
====== TB-5: Actuation Boundary ======
              |
[ PolicyEngine (Default DENY) ]
              |
[ Physical Actuators / PLCs ]
```

---

## 2. Core Security Controls

### 1. Absolute Ban on Insecure Deserialization (`SOFIA-SEC-001`)
Python's native `pickle` module allows arbitrary code execution during deserialization. Sofia Engine enforces:
* **Zero `pickle` across the entire codebase**: Verified via automated AST scans in `tests/security/test_no_pickle.py`.
* Telemetry, features, and configurations are serialized exclusively via strict JSON, JSONL, or NumPy `.npz` binary formats.

### 2. Model Supply Chain Integrity (`SOFIA-SEC-002` & `003`)
All machine learning models and anomaly detector configurations require a validated `ModelManifest`:
* **Schema Validation**: Explicit typing of input feature names, tensor shapes, and preprocessing version.
* **Cryptographic Checksum**: SHA-256 hash of the artifact file must match the manifest. Mismatched artifacts are rejected loudly.
* **Digital Signatures**: Optional Ed25519 asymmetric signature verification for air-gapped deployments.

### 3. Replay & Injection Defense (`SOFIA-SEC-004` & `006`)
* **Monotonic Sequence Numbers**: Telemetry samples carry sequential IDs; duplicates and backwards timestamps are flagged `DUPLICATE` or `STALE`.
* **Payload Ceilings**: Ingest buffers and network adapters reject frames exceeding declared byte limits (default 1 MiB) to prevent memory exhaustion attacks.
* **Strict Schema Parsing**: CSV and JSONL parsers reject newline or delimiter injection attempts.

### 4. Default "DENY" Control Path (`SOFIA-SAFE-001`)
The `PolicyEngine` ensures that inference models cannot directly trigger equipment movements. Every command requires operator approval, equipment operational state verification, rate limiting, and physical interlock clearance.

---

## 3. Threat Model Enumeration (Selected Highlights)

| Threat ID | Description | Impact | Implemented Mitigation |
|---|---|---|---|
| **T-01** | Telemetry Value Poisoning | Skew anomaly detectors to mask faults or trigger false trips. | Robust estimators (MAD, median), physical range limits, and quality marking `OUT_OF_RANGE`. |
| **T-02** | Malformed Protocol Frames | Buffer overflows or parser crash in Modbus/Serial streams. | Strict frame length validation, typed protocol exceptions, drop-and-count strategy. |
| **T-03** | Protocol Injection | Control character injection into CSV/JSONL streams. | Fixed column mappings, `json.loads` only, rejection of unescaped newlines. |
| **T-04** | Telemetry Replay Attack | Replaying old healthy sensor streams to disguise failing hardware. | Monotonic sequence validation, TTL freshness window, clock anomaly detection. |
| **T-06** | Compromised MQTT Broker | Malicious broker floods topic with oversized payloads. | Strict TLS encryption, payload size limits (1 MiB), memory-bounded ring buffers. |
| **T-11** | Model Artifact Tampering | Replacing model weights with adversarial backdoor. | SHA-256 hash enforcement in manifest, optional Ed25519 signature checks. |
| **T-17** | Actuation Bypass | Attacker triggers PLC movements bypassing safety rules. | Universal `PolicyEngine` gate; no bypass path exists in library code. |

---

## 4. Vulnerability Disclosure Policy

If you discover a security vulnerability in Sofia Engine:
* **Do NOT open a public GitHub issue.**
* Open a private security advisory on GitHub: **Security $\to$ Report a vulnerability**.
* Or contact the maintainers directly via **Rootcastle Engineering & Innovation** at `security@rootcastle.com` (or as outlined in [`SECURITY.md`](https://github.com/rootcastleco/sofia-rl/blob/main/SECURITY.md)).

Rootcastle acknowledges reports within 5 business days and provides coordinated remediation timelines.
