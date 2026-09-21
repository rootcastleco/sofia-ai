"""PyTorch backend (optional extra: ``pip install sofia-engine[torch]``).

Torch is a **research and training** backend in Sofia, not the default edge
inference path — ONNX Runtime is preferred for deployment. This module exists so
models developed with :mod:`sofia_ai.learning.rl` can be executed through the same
``ModelBackend`` interface as everything else.

Security note: ``torch.load`` is never called on untrusted input. Only state dicts
loaded from a checksum-verified path are accepted, and ``weights_only=True`` is
requested when the installed torch version supports it.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any

import numpy as np

from ..core.contracts import FeatureVector
from ..core.errors import InferenceError, ModelError
from .base import BackendMetadata, InferenceOutcome, InferenceResult, ModelBackend
from .manifest import ModelManifest

__all__ = ["TorchBackend"]


class TorchBackend(ModelBackend):
    """Execute a PyTorch ``nn.Module`` over a :class:`FeatureVector`.

    Args:
        module_factory: Callable returning an ``nn.Module``. Required, because
            Sofia will not reconstruct a model class from data.
        state_dict_loader: Injectable loader for tests. Defaults to ``torch.load``
            with ``weights_only=True`` when supported.
    """

    def __init__(
        self,
        *,
        module_factory: Callable[[ModelManifest], Any] | None = None,
        state_dict_loader: Callable[..., Any] | None = None,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self._module_factory = module_factory
        self._loader = state_dict_loader
        self.device = device
        self._module: Any = None
        self._torch: Any = None
        self._threshold: float | None = None

    def _load(self, manifest: ModelManifest, artifact: Any) -> None:
        if self._module_factory is None and artifact is None:
            raise ModelError(
                "TorchBackend requires either a module_factory or a module artifact",
                details={"model": manifest.identity},
            )
        self._import_torch()  # fail fast with a helpful message if torch is absent
        if artifact is not None:
            module = artifact
        else:
            assert self._module_factory is not None
            module = self._module_factory(manifest)
        if manifest.artifact_path:
            manifest.verify_artifact()
            state = self._load_state_dict(manifest.artifact_path)
            try:
                module.load_state_dict(state)
            except (RuntimeError, ValueError, TypeError) as exc:
                raise ModelError(
                    f"state dict does not match the module for {manifest.identity}: {exc}",
                    details={"model": manifest.identity},
                ) from exc
        module.to(self.device)
        module.eval()
        self._module = module
        meta = dict(manifest.training_metadata or {})
        self._threshold = float(meta["anomaly_threshold"]) if "anomaly_threshold" in meta else None

    def _import_torch(self) -> Any:
        if self._torch is not None:
            return self._torch
        try:
            import torch
        except ImportError as exc:
            raise ModelError(
                "torch is not installed. Install the 'torch' extra: "
                "pip install sofia-engine[torch]",
                details={"extra": "torch"},
            ) from exc
        self._torch = torch
        return torch

    def _load_state_dict(self, path: str) -> Any:
        torch = self._import_torch()
        if self._loader is not None:
            return self._loader(path)
        try:
            signature = inspect.signature(torch.load)
            if "weights_only" in signature.parameters:
                return torch.load(path, map_location=self.device, weights_only=True)
        except (OSError, ValueError, TypeError) as exc:  # pragma: no cover
            raise ModelError(f"cannot load {path}: {exc}", details={"path": path}) from exc
        return torch.load(path, map_location=self.device)

    def _infer(self, features: FeatureVector) -> InferenceResult:
        assert self.manifest is not None
        torch = self._import_torch()
        self.manifest.check_input(features.names)
        names = self.manifest.input_schema.features or features.names
        values = features.select(tuple(names)).as_array()
        try:
            tensor = torch.from_numpy(np.asarray(values, dtype=np.float32)).unsqueeze(0)
            tensor = tensor.to(self.device)
            with torch.no_grad():
                output = self._module(tensor)
            score = float(np.asarray(output.detach().cpu().numpy(), dtype=np.float64).ravel()[0])
        except Exception as exc:
            raise InferenceError(
                f"torch inference failed: {exc}",
                details={"model": self.manifest.identity},
            ) from exc
        if not np.isfinite(score):
            raise InferenceError("torch produced a non-finite score",
                                 details={"score": repr(score)})
        if self._threshold is None:
            return self._result(
                InferenceOutcome.UNKNOWN,
                score=score,
                confidence=0.0,
                evidence={"reason": "no anomaly_threshold in manifest"},
                features=features,
                metadata={"backend": "torch", "device": self.device},
            )
        anomalous = score >= self._threshold
        margin = abs(score - self._threshold) / max(abs(self._threshold), 1e-9)
        return self._result(
            InferenceOutcome.ANOMALOUS if anomalous else InferenceOutcome.NORMAL,
            score=score,
            confidence=float(np.clip(margin / (1.0 + margin), 0.0, 1.0)),
            evidence={"threshold": self._threshold},
            features=features,
            metadata={"backend": "torch", "device": self.device},
        )

    def metadata(self) -> BackendMetadata:
        version = "not-installed"
        try:
            import torch

            version = str(torch.__version__)
        except ImportError:
            version = "not-installed"
        return BackendMetadata(
            backend_id="TorchBackend",
            kind="torch",
            version=version,
            supports_batch=True,
            requires_extra="torch",
        )

    def close(self) -> None:
        self._module = None
        super().close()
