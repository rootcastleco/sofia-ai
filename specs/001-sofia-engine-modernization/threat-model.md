# SOFIA ENGINE — Threat Model

**Scope:** `src/sofia_ai/**`, `embedded/**`, CLI, configuration, packaging and CI.
**Method:** asset → threat → control → residual risk, per component.
**Status:** controls implemented where marked `IMPLEMENTED`; otherwise `DOCUMENTED`.

---

## 0. Trust boundaries

| Boundary | Inside | Outside | Crossing mechanism |
|---|---|---|---|
| TB-1 Transport | Telemetry adapters | Field network, brokers, PLCs | MQTT/Modbus/serial/file bytes |
| TB-2 Model supply | Model loader, backends | Model files, registries, vendors | manifest + artifact + checksum |
| TB-3 Configuration | Config loader | YAML/JSON files, environment | loader with schema validation |
| TB-4 Operator | CLI, API, copilot | Human / upstream system | argv, HTTP, prompts |
| TB-5 Actuation | Policy engine | Machine / actuator driver | `CommandDecision` only |
| TB-6 Build/CI | Repository workflows | GitHub Actions marketplace, PyPI | pinned actions, hashes |

---

## 1. Threats and controls

### T-01 Malicious telemetry values
*Threat:* attacker publishes extreme or crafted values to steer detectors, trigger false
maintenance events, or mask a real fault.
*Controls:* `SOFIA-SEC` range validation; per-channel `min`/`max` in config; robust detectors
(MAD, median-based) resistant to point outliers; quality marking `OUT_OF_RANGE`; evidence
records the observed value so an analyst can see the input. **IMPLEMENTED**

### T-02 Malformed binary frames
*Threat:* truncated or misaligned Modbus/serial/CAN frames cause index errors or misinterpretation.
*Controls:* explicit frame length checks before indexing; typed `TelemetryProtocolError`; parser
bounded by declared frame size; malformed frames counted and dropped, never partially parsed.
**IMPLEMENTED** (`telemetry/base.py`, `telemetry/modbus.py`, `telemetry/serial_adapter.py`)

### T-03 Protocol injection
*Threat:* control characters / separators injected into CSV or JSONL to forge extra samples or
fields, or into serial line protocols to forge frames.
*Controls:* strict field count validation; no `eval`/shell interpretation of any field; CSV parsed
with fixed column mapping; JSONL parsed with `json.loads` only; serial frames length- and
checksum-bounded; newline characters rejected in identifier fields by `core/validation.py`.
**IMPLEMENTED**

### T-04 Replay attacks (telemetry)
*Threat:* attacker re-sends a previously valid sample stream to make a machine look healthy.
*Controls:* sequence-number tracking with monotonic expectation; duplicate detection emits
`DUPLICATE` quality; freshness window (`max_age_s`) marks `STALE`; clock-anomaly detection flags
backwards timestamps. **IMPLEMENTED** (`core/time.py`, `edge/runtime.py`)

### T-05 Spoofed devices
*Threat:* unknown device_id injects data attributed to real equipment.
*Controls:* optional device allow-list in configuration (`devices.allowlist`); unknown devices are
rejected and counted, not silently admitted; adapter-level authentication hooks (TLS/client certs
for MQTT) documented. **IMPLEMENTED** (allowlist) / **DOCUMENTED** (broker auth)

### T-06 Compromised MQTT publisher
*Threat:* an attacker with broker credentials floods or poisons the topic tree.
*Controls:* TLS configuration supported (`tls.enabled`, `tls.ca_cert`); credentials read from
environment only (`security/secrets.py`), never from committed config; payload size ceiling;
per-topic rate limiting guidance; QoS and bounded in-flight queue. **IMPLEMENTED** (size limits,
env secrets) / **DOCUMENTED** (broker hardening)

### T-07 Denial of service / resource exhaustion
*Threat:* high-rate ingest or huge payloads exhaust memory or CPU.
*Controls:* bounded ring buffer with drop-oldest policy; payload byte ceiling; per-source rate
ceiling; bounded window count; bounded metric history; bounded log rate; explicit
`max_in_flight` on transports. **IMPLEMENTED** (`edge/buffer.py`, `telemetry/base.py`,
`observability/*`)

### T-08 Queue exhaustion
*Threat:* consumer slower than producer; unbounded growth.
*Controls:* `BoundedRingBuffer` drops oldest and increments `buffer.dropped` and
`buffer.saturation` metrics; saturation is exported for alerting. **IMPLEMENTED**

### T-09 Malicious configuration
*Threat:* crafted config disables validation, raises limits, or points writers at sensitive paths.
*Controls:* typed configuration with schema and range validation at load; path confinement via
`security/safeio.py` (`resolve_within`); no config key enables arbitrary code; environment
overrides restricted to a declared allow-list. **IMPLEMENTED**

### T-10 Malicious model files
*Threat:* crafted artifact triggers parser bugs or code execution.
*Controls:* no `pickle` anywhere (`tests/security/test_no_pickle.py` enforces this); manifests are
JSON; artifacts are checksummed; `OnnxBackend` loads via ONNX Runtime rather than Python
deserialization; `TorchBackend` documents that `torch.load` is not used on untrusted inputs.
**IMPLEMENTED**

### T-11 Unsafe Python deserialization
*Threat:* `pickle`/`yaml.unsafe_load`/`eval` on untrusted input.
*Controls:* `security/safeio.py` provides the only loaders, all JSON-based; a repository test greps
the source tree for `pickle.load`, `yaml.unsafe_load`, `eval(`, `exec(`, `marshal.load`,
`subprocess` with `shell=True`. **IMPLEMENTED**

### T-12 Dependency compromise
*Threat:* upstream package or CI action compromised.
*Controls:* optional extras keep the minimal dependency surface at **numpy only**; Dependabot
enabled; `pip-audit` job in CI; pinned GitHub Actions by major version with `permissions:`
least-privilege; SBOM generation on release. **IMPLEMENTED**

### T-13 Poisoned models
*Threat:* model trained to produce desired-but-wrong outputs.
*Controls:* manifests record `training_metadata` and `provenance`; checksum pins the artifact;
optional Ed25519 signature; model version recorded on every `InferenceResult` and `HealthEvent`
so a poisoned version is traceable. **IMPLEMENTED** / **DOCUMENTED** (organizational review)

### T-14 Model replacement
*Threat:* attacker swaps a model file in place.
*Controls:* checksum verification on every load; mismatch raises `ModelIntegrityError` and refuses
to load; recommended read-only model directory and signed manifests. **IMPLEMENTED**

### T-15 Leaked credentials
*Threat:* broker/API secrets committed or logged.
*Controls:* `security/secrets.py` reads from environment only; `observability/redaction.py`
redacts `password`, `token`, `secret`, `key`, `authorization`, `passwd`, `api_key`, `bearer`;
a test asserts redaction coverage; `.gitignore` excludes `.env*`; secret scanning guidance in
`SECURITY.md`. **IMPLEMENTED**

### T-16 Log injection
*Threat:* crafted device_id or channel containing newlines forges log records.
*Controls:* structured JSON logging encodes via `json.dumps`, so embedded newlines/quotes cannot
break record framing; identifier validation rejects control characters. **IMPLEMENTED**

### T-17 Arbitrary plugin execution
*Threat:* plugin name resolves to attacker-controlled module.
*Controls:* registry is explicit — backends and sources are registered by the application, not
discovered by scanning; no `importlib` path from untrusted data; optional entry-point loading is
allow-listed by prefix. **IMPLEMENTED**

### T-18 Path traversal
*Threat:* `../../` in a model, config, or data path escapes its root.
*Controls:* `security/safeio.py:resolve_within(root, candidate)` resolves and requires the result
to be inside `root`; symlink resolution applied; used by CLI and model loader.
**IMPLEMENTED**

### T-19 Unsafe command execution
*Threat:* command request triggers shell/actuator with attacker-controlled parameters.
*Controls:* `CommandRequest` parameters are a typed mapping validated against the action
declaration; no shell invocation anywhere; policy engine default DENY; no CLI command performs
actuation. **IMPLEMENTED**

### T-20 Command replay
*Threat:* a previously approved command is resubmitted.
*Controls:* `ReplayGuard` with bounded nonce set and TTL; expired or seen command ids are denied;
freshness check (`max_age_s`) in policy. **IMPLEMENTED**

### T-21 Remote API compromise
*Threat:* the optional HTTP API is exposed and abused.
*Controls:* API is an optional extra, disabled by default; bind host defaults to `127.0.0.1`;
read-only by default; command endpoints require explicit opt-in and pass through the policy
engine; auth hook interface provided. **IMPLEMENTED** / **DOCUMENTED**

---

## 2. Residual risks (accepted, documented)

| # | Residual risk | Rationale / mitigation |
|---|---|---|
| R-1 | Statistical detectors cannot distinguish a real process change from a sophisticated slow-ramp attack | Requires process-level authentication (TLS/client certs) and physical sensor trust, outside software scope |
| R-2 | ONNX Runtime / PyTorch are large native dependencies with their own CVE surface | Placed behind extras; `pip-audit` in CI; minimal install is numpy-only |
| R-3 | MQTT/Modbus security depends on broker/PLC configuration | Documented in `SECURITY.md`; Sofia cannot enforce broker ACLs |
| R-4 | Model provenance is only as strong as the signing key management | Ed25519 hook provided; key management is organizational |
| R-5 | Embedded C runtime has no hardware-in-loop verification in this repo | Marked NOT VERIFIED; golden vectors only cover numerical equivalence |
| R-6 | No functional-safety certification | Explicitly disclaimed; Sofia is not a SIS/PLC safety system |
| R-7 | Time-of-check/time-of-use on model files on a writable filesystem | Recommend read-only mounts, documented |

---

## 3. Explicit prohibition

**Never load arbitrary untrusted pickle content.** There is no code path in Sofia Engine that
calls `pickle.load`, `pickle.loads`, `yaml.unsafe_load`, `eval`, `exec`, `marshal.load`, or
`subprocess(..., shell=True)`. `tests/security/test_no_pickle.py` and
`tests/security/test_no_bare_except.py` fail the build if any is introduced.
