"""Deterministic JSON serialization for Sofia contracts.

Rules:
    * keys are sorted,
    * floats use a fixed representation (``repr`` round-trips exactly),
    * ``NaN``/``inf`` are never emitted — they are rejected upstream, and this module
      refuses to produce invalid JSON,
    * the contract version is embedded in every envelope.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from enum import Enum
from typing import Any, Final

from .contracts import CONTRACT_VERSION
from .errors import ValidationError

__all__ = [
    "SCHEMA_VERSION_KEY",
    "SofiaJSONEncoder",
    "decode",
    "dump_json",
    "encode",
    "load_json",
    "read_jsonl",
    "write_jsonl",
]

SCHEMA_VERSION_KEY: Final[str] = "contract_version"


class SofiaJSONEncoder(json.JSONEncoder):
    """JSON encoder that understands numpy scalars, enums and Sofia contracts."""

    def default(self, o: Any) -> Any:
        if isinstance(o, Enum):
            return o.value
        if hasattr(o, "to_dict") and callable(o.to_dict):
            return o.to_dict()
        if hasattr(o, "tolist") and callable(o.tolist):
            return o.tolist()
        if hasattr(o, "item") and callable(o.item):
            return o.item()
        return super().default(o)


def _check_finite(obj: Any, path: str = "$") -> None:
    """Reject non-finite floats anywhere in the structure."""
    if isinstance(obj, float):
        if not math.isfinite(obj):
            raise ValidationError(
                f"Refusing to serialize non-finite float at {path}",
                details={"path": path},
            )
        return
    if isinstance(obj, Mapping):
        for key, value in obj.items():
            _check_finite(value, f"{path}.{key}")
        return
    if isinstance(obj, (list, tuple)):
        for index, value in enumerate(obj):
            _check_finite(value, f"{path}[{index}]")


def encode(obj: Any, *, sort_keys: bool = True) -> str:
    """Serialize to a deterministic JSON string."""
    _check_finite(obj)
    return json.dumps(
        obj,
        cls=SofiaJSONEncoder,
        sort_keys=sort_keys,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def decode(text: str) -> Any:
    """Deserialize a JSON string, rejecting NaN/Infinity literals."""
    return json.loads(text, parse_constant=_reject_constant)


def _reject_constant(name: str) -> Any:
    raise ValidationError(
        f"JSON contains the non-finite literal {name}", details={"literal": name}
    )


def dump_json(obj: Any, path: str, *, indent: int | None = 2) -> None:
    """Write ``obj`` as JSON to ``path`` (UTF-8, deterministic)."""
    _check_finite(obj)
    payload = json.dumps(
        obj, cls=SofiaJSONEncoder, sort_keys=True, indent=indent, allow_nan=False
    )
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(payload)
        handle.write("\n")


def load_json(path: str) -> Any:
    """Read JSON from ``path``."""
    with open(path, encoding="utf-8") as handle:
        return decode(handle.read())


def write_jsonl(records: list[Mapping[str, Any]], path: str) -> int:
    """Write records as JSON Lines. Returns the number of records written."""
    count = 0
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            _check_finite(record)
            envelope = dict(record)
            envelope.setdefault(SCHEMA_VERSION_KEY, CONTRACT_VERSION)
            handle.write(json.dumps(envelope, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False, cls=SofiaJSONEncoder))
            handle.write("\n")
            count += 1
    return count


def read_jsonl(path: str, *, max_lines: int = 10_000_000) -> list[dict[str, Any]]:
    """Read JSON Lines, skipping blank lines and rejecting malformed records.

    Bounded by ``max_lines`` so a hostile file cannot exhaust memory.
    """
    records: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line_number > max_lines:
                break
            stripped = line.strip()
            if not stripped:
                continue
            parsed = decode(stripped)
            if not isinstance(parsed, dict):
                raise ValidationError(
                    f"{path}:{line_number} is not a JSON object",
                    details={"path": path, "line": line_number},
                )
            records.append(parsed)
    return records
