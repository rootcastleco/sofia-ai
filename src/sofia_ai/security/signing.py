"""Manifest signature verification (optional).

Provides an Ed25519 verification hook for model manifests and model artifacts
(threats T-13, T-14). Verification is *optional* by design: Sofia works offline
without any key material, but a deployment that loads third-party models should
enable it.

Requires ``cryptography`` or PyNaCl when used. If neither is installed and
verification is requested, a :class:`SignatureError` is raised — never a silent
skip, which would be a hidden fallback.
"""

from __future__ import annotations

import base64
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final

from ..core.errors import SignatureError

__all__ = [
    "ALGORITHM",
    "NullVerifier",
    "SignatureVerifier",
    "load_public_key_b64",
    "verify_detached",
]

ALGORITHM: Final[str] = "ed25519"


@dataclass(frozen=True, slots=True)
class SignatureVerifier:
    """Verifies detached Ed25519 signatures over manifest bytes."""

    public_key_b64: str
    algorithm: str = ALGORITHM

    def __post_init__(self) -> None:
        if self.algorithm != ALGORITHM:
            raise SignatureError(
                f"unsupported signature algorithm {self.algorithm!r}",
                details={"algorithm": self.algorithm},
            )
        if not self.public_key_b64.strip():
            raise SignatureError("public key must not be empty", details={})

    def verify(self, payload: bytes, signature_b64: str) -> bool:
        """Return True if the signature verifies. Raises on unusable key material."""
        return verify_detached(payload, signature_b64, self.public_key_b64)


class NullVerifier:
    """Explicit no-op verifier for offline development.

    Using this type is an intentional, greppable decision. It never silently
    replaces a configured verifier.
    """

    __slots__ = ()

    def verify(self, _payload: bytes, _signature_b64: str) -> bool:
        return True


def load_public_key_b64(path: str) -> str:
    """Load a base64 (or PEM) Ed25519 public key from disk."""
    from pathlib import Path

    text = Path(path).read_text(encoding="utf-8").strip()
    if "BEGIN PUBLIC KEY" in text:
        try:
            from cryptography.hazmat.primitives.serialization import load_pem_public_key
        except ImportError as exc:
            raise SignatureError(
                "PEM keys require the 'cryptography' package", details={"path": path}
            ) from exc
        key = load_pem_public_key(text.encode("utf-8"))
        try:
            from cryptography.hazmat.primitives.serialization import (
                Encoding,
                PublicFormat,
            )
        except ImportError as exc:  # pragma: no cover - same package
            raise SignatureError("cryptography is unavailable", details={}) from exc
        raw = key.public_bytes(Encoding.Raw, PublicFormat.Raw)
        return base64.b64encode(raw).decode("ascii")
    return text


def verify_detached(payload: bytes, signature_b64: str, public_key_b64: str) -> bool:
    """Verify an Ed25519 detached signature.

    Raises:
        SignatureError: if the crypto backend is missing or the material is malformed.
    """
    if not signature_b64 or not public_key_b64:
        raise SignatureError("signature and public key must both be provided", details={})
    try:
        key_bytes = base64.b64decode(public_key_b64, validate=True)
        sig_bytes = base64.b64decode(signature_b64, validate=True)
    except (ValueError, TypeError) as exc:
        raise SignatureError("signature material is not valid base64", details={}) from exc

    try:
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    except ImportError as exc:
        raise SignatureError(
            "signature verification requires the 'cryptography' package",
            details={"backend": "cryptography"},
        ) from exc

    if len(key_bytes) != 32:
        raise SignatureError(
            f"Ed25519 public key must be 32 bytes, got {len(key_bytes)}",
            details={"length": len(key_bytes)},
        )
    try:
        Ed25519PublicKey.from_public_bytes(key_bytes).verify(sig_bytes, payload)
    except InvalidSignature:
        return False
    except ValueError as exc:
        raise SignatureError(f"signature verification failed: {exc}", details={}) from exc
    return True


def manifest_signing_payload(manifest: Mapping[str, Any]) -> bytes:
    """Canonical bytes covered by a manifest signature.

    The ``signature`` field itself is excluded so the signature can be stored
    inside the manifest it protects.
    """
    fields = {k: v for k, v in manifest.items() if k != "signature"}
    canonical = json.dumps(fields, sort_keys=True, separators=(",", ":"),
                           ensure_ascii=True, allow_nan=False)
    return canonical.encode("utf-8")
