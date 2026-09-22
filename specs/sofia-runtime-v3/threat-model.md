# Sofia Engine Runtime v3 — Threat Model

> **Author:** Rootcastle Engineering & Innovation  
> **Classification:** Security Engineering Standard  
> **Status:** Active

---

## 1. Trust Boundaries

```
+---------------------------------------------------------------------------------------+
| UNTRUSTED ZONE                                                                        |
| - External Sensor Networks, Field Buses (MQTT, Modbus RTU/TCP, RS-485 Serial)        |
| - Telemetry data files (CSV, JSONL, binary streams)                                  |
| - External Model Files & Checkpoints                                                  |
| - External LLM APIs & Cloud Responses (OpenAI, NVIDIA NIM, OpenRouter)                |
+---------------------------------------------------------------------------------------+
                                           |
                              [Strict Input Validation]
                              [Schema & Integrity Check]
                                           v
+---------------------------------------------------------------------------------------+
| TRUSTED RUNTIME ZONE                                                                  |
| - Validated Typed Contracts (Sample, SignalFrame, FeatureVector)                      |
| - Deterministic DSP Pipeline & Bounded Ring Buffers                                  |
| - SofiaAsmVM (Memory-bounded, cycle-bounded bytecode execution)                       |
| - Diagnostic Engine & Evidence Accumulator                                           |
+---------------------------------------------------------------------------------------+
                                           |
                              [Default DENY Gate]
                              [Physical Interlocks & Nonce/TTL]
                                           v
+---------------------------------------------------------------------------------------+
| SAFETY-CRITICAL CONTROL ZONE                                                          |
| - PolicyEngine                                                                        |
| - Actuator & PLC Interfaces                                                           |
| - Hardware Emergency Stop (E-Stop) Interlocks                                         |
+---------------------------------------------------------------------------------------+
```

---

## 2. The 21 Edge Threats & Mitigations

| Threat ID | Threat Name | Vector | Impact | Mitigation Strategy |
|---|---|---|---|---|
| `THREAT-01` | Malicious / Spoofed Telemetry | Injected sensor values over bus | False anomaly or masked failure | Strict schema validation, rate-of-change ceilings, sensor calibration check. |
| `THREAT-02` | Malformed / Oversized Packets | Packet flood or corrupted frame | Buffer overflow, heap exhaustion | Bounded ring buffers, max frame byte ceilings, drop-oldest policy. |
| `THREAT-03` | Replay Attacks on Commands | Intercepted valid `CommandRequest` | Unintended machinery actuation | Mandatory Nonce + TTL verification; replayed nonces are rejected immediately. |
| `THREAT-04` | Model Artifact Tampering | Substituted model files | Arbitrary code execution | Prohibition of `pickle`; models stored as JSON manifest + NPZ with SHA-256 validation. |
| `THREAT-05` | Poisoned Training Datasets | Corrupted telemetry in fine-tuning | Degraded diagnostic accuracy | Dataset curation validation, token bounds, outlier rejection before export. |
| `THREAT-06` | API-Key Leakage in Logs | Logging request headers/env | Compromise of cloud credentials | Automated secret masking/scrubbing in logging formatters. |
| `THREAT-07` | Prompt Injection via Copilot | Adversarial text in telemetry metadata | Attempted policy modification | Copilot output is strictly advisory; zero execution privilege on `PolicyEngine`. |
| `THREAT-08` | Compromised LLM Provider | Malicious text payload from API | Misleading operator advice | Schema-validated structured evidence; copilot can only cite given facts. |
| `THREAT-09` | Malicious VM Bytecode | Hand-crafted invalid instructions | Memory corruption, VM crash | Instruction opcode validation, strict memory bounds checks, explicit cycle limits. |
| `THREAT-10` | Path Traversal in Loaders | `../../etc/passwd` in model path | Unauthorized file access | Path resolution check: must reside inside configured models directory. |
| `THREAT-11` | Command Injection via CLI | Shell metacharacters in CLI args | Host command execution | `subprocess` with `shell=False`, argument list passing, strict identifier validation. |
| `THREAT-12` | Unsafe Deserialization | Corrupted JSON or pickled state | Memory fault or RCE | Strict typed dataclass deserializers with field-level type checking. |
| `THREAT-13` | Dependency Compromise | Vulnerability in 3rd party package | System compromise | Zero third-party runtime dependencies in `sofia-core` (NumPy only). |
| `THREAT-14` | Resource Exhaustion (CPU) | Infinite loops in DSP or VM | Edge gateway freeze | Fixed loop bounds, FFT length ceilings ($N \le 65536$), VM `max_cycles` limit. |
| `THREAT-15` | Memory Exhaustion (RAM) | Unbounded queue accumulation | Out-of-memory crash | Fixed buffer capacities with drop-oldest policy; zero unbounded queues. |
| `THREAT-16` | Clock Skew / Nonce Reuse | Desynchronized edge clock | Replay window bypass | Monotonic processing clock + timestamp jump detection ($> 5$s flagged). |
| `THREAT-17` | Actuator Bypass | Direct code call to actuator | Unauthorized actuation | Actuation primitives reside only behind `PolicyEngine` with default DENY. |
| `THREAT-18` | Eavesdropping on Field Bus | Sniffing plaintext bus | Telemetry leakage | TLS recommendation for MQTT; data obfuscation/hashing for sensitive channels. |
| `THREAT-19` | Calibration Spoofing | Modified sensor scale/offset | Inaccurate physical calculation | Signed calibration certificates, calibration expiration dates. |
| `THREAT-20` | Storage / Flash Corruption | Power loss during write | Corrupted checkpoint | Atomic file writes (`tempfile` + `os.replace`), checksum verification. |
| `THREAT-21` | Unauthorized Policy Mutation | Runtime tampering of policy rules | Permissive actuation | Policy configuration is immutable once loaded; changes require operator restart. |
