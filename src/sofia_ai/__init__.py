"""SOFIA ENGINE — open-source edge intelligence for machines, telemetry,
embedded systems, industrial diagnostics and technical automation.

Sofia converts machine telemetry into structured engineering evidence locally,
using deterministic signal processing, pluggable inference backends and explicit
safety boundaries.

Design commitments:

* **offline-first** — no network access is required for any core capability;
* **bounded** — every buffer, queue and retention structure has a declared ceiling;
* **framework-independent** — the core imports neither PyTorch, ONNX Runtime nor
  any protocol client;
* **honest** — no performance or accuracy claim is published without a
  reproducible artifact in this repository;
* **safe** — no inference result reaches an actuator without passing the policy
  engine, whose default posture is DENY.

Sofia is not a certified SIS/PLC safety system.

Quick start::

    from sofia_ai.telemetry import SyntheticVibrationSource, SyntheticMachineProfile
    from sofia_ai.features import extract_from_array
    from sofia_ai.inference import create_backend, build_manifest_for

    source = SyntheticVibrationSource(SyntheticMachineProfile(), n_samples=4096)
    source.open()
    samples = source.read()
    features = extract_from_array([s.value for s in samples], 1000.0)

    backend = create_backend("mad")
    backend.load(build_manifest_for(backend, model_id="demo"))
    result = backend.infer(features)
    print(result.outcome, result.confidence, result.uncertainty)

Pre-2.0 imports (``SofiaModel``, ``QuantumNeuralEngine``, ``NLPProcessor``,
``QuantumConfig``) still resolve through :mod:`sofia_ai.compat` and emit a
``DeprecationWarning``.
"""

from __future__ import annotations

__version__ = "2.2.0"
__author__ = "Rootcastle Engineering & Innovation"
__license__ = "Apache-2.0"

#: Machine-readable capability manifest. Used by ``sofia info`` and by the docs.
CAPABILITIES: dict[str, object] = {
    "version": __version__,
    "license": __license__,
    "organization": "Rootcastle Engineering & Innovation",
    "scientific_ai": True,
    "deployment_classes": {
        "A_workstation_server": "VERIFIED",
        "B_edge_linux": "PARTIALLY VERIFIED",
        "C_microcontroller": "PARTIALLY VERIFIED",
    },
    "telemetry_adapters": [
        "csv", "jsonl", "memory", "replay", "synthetic", "mqtt", "modbus", "serial",
    ],
    "inference_backends": [
        "threshold", "zscore", "mad", "ewma", "iqr", "cusum", "callable", "onnx", "torch",
    ],
    "dsp_domains": [
        "vibration", "electrical", "acoustic", "imu_kinematics",
    ],
    "learning_subsystems": [
        "asm_neural_self_training", "auto_finetune", "reinforcement_learning",
    ],
    "quantum_emulation": True,
    "offline_first": True,
    "functional_safety_certified": False,
}

__all__ = ["CAPABILITIES", "__author__", "__license__", "__version__", "compat"]


def __getattr__(name: str) -> object:
    """Resolve deprecated 1.x names lazily, with a ``DeprecationWarning``."""
    if name in ("SofiaModel", "QuantumNeuralEngine", "NLPProcessor", "QuantumConfig"):
        from . import compat

        return getattr(compat, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | {"SofiaModel", "QuantumNeuralEngine", "NLPProcessor",
                                    "QuantumConfig"})
