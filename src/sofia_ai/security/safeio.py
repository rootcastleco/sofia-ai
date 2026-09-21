"""Path confinement and safe (non-pickle) deserialization.

Two controls live here:

* :func:`resolve_within` — the single path-traversal guard used by the CLI, the
  model loader and the store-and-forward layer (threat T-18).
* JSON/NPZ loaders — the only deserialization entry points Sofia provides. There
  is no function in this module that executes arbitrary code from data.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

from ..core.errors import UnsafePathError, ValidationError

__all__ = [
    "MAX_PATH_LENGTH",
    "REPLACEMENT",
    "atomic_write_bytes",
    "ensure_directory",
    "resolve_within",
    "safe_read_json",
    "safe_write_json",
]

MAX_PATH_LENGTH: Final[int] = 4096
REPLACEMENT: Final[str] = "***REDACTED***"


def resolve_within(root: str | os.PathLike[str], candidate: str | os.PathLike[str]) -> Path:
    """Resolve ``candidate`` and require it to stay inside ``root``.

    Both symlinks and ``..`` segments are resolved before the containment check,
    so a symlink pointing outside ``root`` is rejected.

    Raises:
        UnsafePathError: if the resolved path escapes ``root``.
    """
    root_path = Path(os.path.realpath(root))
    raw = str(candidate)
    if len(raw) > MAX_PATH_LENGTH:
        raise UnsafePathError(
            f"path exceeds {MAX_PATH_LENGTH} characters",
            details={"length": len(raw)},
        )
    target = Path(os.path.realpath(os.path.join(root_path, raw))) \
        if not os.path.isabs(raw) else Path(os.path.realpath(raw))
    try:
        target.relative_to(root_path)
    except ValueError as exc:
        raise UnsafePathError(
            f"path {raw!r} escapes the permitted root {str(root_path)!r}",
            details={"candidate": raw, "root": str(root_path)},
        ) from exc
    return target


def ensure_directory(path: str | os.PathLike[str]) -> Path:
    """Create a directory if absent and return its real path."""
    resolved = Path(os.path.realpath(path))
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def safe_read_json(path: str | os.PathLike[str], *, root: str | os.PathLike[str] | None = None,
                   max_bytes: int = 8 << 20) -> Mapping[str, Any]:
    """Read a JSON object from disk with a size ceiling and an optional root confinement.

    Raises:
        UnsafePathError: if ``root`` is given and the path escapes it.
        ValidationError: if the file is too large or not a JSON object.
    """
    target = resolve_within(root, path) if root is not None else Path(os.path.realpath(path))
    size = target.stat().st_size
    if size > max_bytes:
        raise ValidationError(
            f"file {target} is {size} bytes, above the {max_bytes} byte limit",
            details={"path": str(target), "size": size},
        )
    with target.open("r", encoding="utf-8") as handle:
        try:
            payload = json.load(handle)
        except ValueError as exc:
            raise ValidationError(
                f"file {target} is not valid JSON", details={"path": str(target)}
            ) from exc
    if not isinstance(payload, dict):
        raise ValidationError(
            f"file {target} must contain a JSON object", details={"path": str(target)}
        )
    return payload


def safe_write_json(path: str | os.PathLike[str], payload: Mapping[str, Any]) -> Path:
    """Write JSON deterministically (sorted keys, no NaN)."""
    text = json.dumps(payload, sort_keys=True, indent=2, allow_nan=False)
    return atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_bytes(path: str | os.PathLike[str], data: bytes) -> Path:
    """Write bytes atomically via a temporary file plus ``os.replace``.

    Prevents a partially-written model or state file from being read after a
    crash or power loss.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    with tmp.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, target)
    return target
