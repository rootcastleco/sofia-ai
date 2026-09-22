# Security Policy

## Reporting a vulnerability

Please do **not** file a public GitHub issue for a security vulnerability.
Report it privately to the maintainers instead:

- Open a private advisory: GitHub → *Security* → *Report a vulnerability*
- Or email the Rootcastle Engineering & Innovation team via the repository
  maintainers with the subject `[sofia-ai security]`.

Please include: module and version affected, a description of the weakness,
and — if you have one — a minimal reproduction. Coordinated disclosure is
appreciated: we aim to acknowledge reports within 5 business days.

## Scope

In scope: the `src/sofia_ai/` package, the embedded reference implementation,
and their documented behavior. The legacy tree under `legacy/sofia_ai_v1/` is
preserved for reference only and is not a supported runtime; it is not under
active hardening but should still be disclosed through the same channel.

## Supported versions

| Version | Supported |
| --- | --- |
| 3.x (sofia-engine) | Yes |
| 2.x (sofia-engine) | Maintenance only |
| 1.x (legacy sofia_ai) | No — reference only |

## Design posture

- **Default DENY.** The command/decision layer (`PolicyEngine`) rejects
  commands by default: empty allowlist, operator approval required, physical
  interlocks must be clear, and a deny posture refuses actions when the machine
  state is unknown. There is no bypass path, including for internal callers.
- **No pickle.** Deserialization paths accept JSON, JSONL and NPZ only. The
  `security` suite enforces that `pickle`/`shelve`/`marshal`/`yaml.unsafe_load`
  never appear in library code.
- **No eval / exec / shell=True.** Enforced by source scans.
- **Bounded resources.** All buffers, queues and caches carry explicit
  ceilings; the experimental quantum simulator enforces a hard memory ceiling.
- **Deterministic by default.** No global RNG seeding in library code; RNGs and
  clocks are injected, so outputs are reproducible and audit-friendly.
- **No secrets in code or logs.** The secrets helper manages credentials via
  environment or keyring; nothing is written to logs.

## Verification

Run the security and architecture test suites locally:

```bash
$env:PYTHONPATH = "src"
.venv/Scripts/python -m pytest tests/security tests/architecture
```

Both must pass before a release is cut. The threat model lives in
`specs/001-sofia-engine-modernization/threat-model.md`.