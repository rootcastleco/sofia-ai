"""Integrity primitives: checksums and content addressing.

Used to make model artifacts tamper-evident (threat T-14) and to give every
feature vector a deterministic identity. SHA-256 is used throughout; no MD5 or
SHA-1 is exposed anywhere in Sofia.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Final

__all__ = [
    "CHUNK_SIZE",
    "sha256_bytes",
    "sha256_file",
    "sha256_json",
    "verify_checksum",
]

CHUNK_SIZE: Final[int] = 1 << 20  # 1 MiB streaming read


def sha256_bytes(payload: bytes) -> str:
    """Hex SHA-256 of a byte string."""
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path, *, chunk_size: int = CHUNK_SIZE) -> str:
    """Streaming SHA-256 of a file; memory use is independent of file size."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(obj: object) -> str:
    """SHA-256 of the canonical JSON encoding of ``obj``."""
    from .serialization import encode

    return hashlib.sha256(encode(obj).encode("utf-8")).hexdigest()


def verify_checksum(path: str | Path, expected: str) -> bool:
    """Constant-work comparison of a file checksum against ``expected``.

    Comparison is done on the digest strings; both sides are normalized to
    lowercase hex before comparison.
    """
    if not isinstance(expected, str) or len(expected) != 64:
        return False
    actual = sha256_file(path)
    return _constant_time_equals(actual.lower(), expected.lower())


def _constant_time_equals(a: str, b: str) -> bool:
    if len(a) != len(b):
        return False
    diff = 0
    for x, y in zip(a, b, strict=True):
        diff |= ord(x) ^ ord(y)
    return diff == 0
