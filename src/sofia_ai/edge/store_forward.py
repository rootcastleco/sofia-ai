"""Bounded store-and-forward persistence for offline operation.

When connectivity is lost, samples are persisted locally up to a byte ceiling and
replayed when connectivity returns. The store never grows without bound: when the
ceiling is reached, the oldest file is rotated out and the loss is counted.

Files are written atomically and read back with strict validation, so a truncated
or corrupted file is discarded rather than crashing the pipeline (threat: corrupted
state).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from ..core.contracts import TelemetrySample
from ..core.errors import StorageError, ValidationError
from ..core.validation import validate_positive_int
from ..security.safeio import atomic_write_bytes, ensure_directory, resolve_within

__all__ = ["StoreForward", "StoreStats"]

FILE_PREFIX: Final[str] = "sofia-sf-"
FILE_SUFFIX: Final[str] = ".jsonl"


@dataclass(slots=True)
class StoreStats:
    """Store-and-forward counters."""

    writes: int = 0
    batches: int = 0
    bytes_written: int = 0
    drops: int = 0
    rotations: int = 0
    corrupt_files: int = 0
    replayed: int = 0


@dataclass(slots=True)
class StoreForward:
    """Bounded, rotating local persistence.

    Args:
        directory: Storage directory. Confined: every path is resolved within it.
        max_bytes: Total byte ceiling across all files.
        max_files: Maximum number of retained files.
        flush_every: Records buffered in memory before a file is written.
    """

    directory: str = "sofia_store"
    max_bytes: int = 64 * 1024 * 1024
    max_files: int = 8
    flush_every: int = 128
    stats: StoreStats = field(default_factory=StoreStats)

    _root: Path = field(init=False, repr=False)
    _pending: list[dict[str, Any]] = field(default_factory=list, init=False, repr=False)
    _sequence: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        self.max_bytes = validate_positive_int(self.max_bytes, "max_bytes", maximum=1 << 40)
        self.max_files = validate_positive_int(self.max_files, "max_files", maximum=1024)
        self.flush_every = validate_positive_int(self.flush_every, "flush_every",
                                                 maximum=1 << 20)
        self._root = ensure_directory(self.directory)

    # -- write path --------------------------------------------------------

    def append(self, sample: TelemetrySample) -> bool:
        """Buffer a sample for persistence. Returns False if dropped."""
        self._pending.append(sample.to_dict())
        self.stats.writes += 1
        if len(self._pending) >= self.flush_every:
            self.flush()
        return True

    def extend(self, samples: list[TelemetrySample]) -> int:
        for sample in samples:
            self.append(sample)
        return len(samples)

    def flush(self) -> int:
        """Write buffered records to a file and enforce the byte ceiling."""
        if not self._pending:
            return 0
        payload = "".join(json.dumps(r, sort_keys=True, allow_nan=False) + "\n"
                          for r in self._pending)
        data = payload.encode("utf-8")
        self._sequence += 1
        name = f"{FILE_PREFIX}{int(time.time() * 1000):013d}-{self._sequence:06d}{FILE_SUFFIX}"
        target = resolve_within(self._root, name)
        try:
            atomic_write_bytes(target, data)
        except OSError as exc:
            raise StorageError(
                f"store-and-forward write failed: {exc}", details={"path": str(target)}
            ) from exc
        written = len(self._pending)
        self._pending.clear()
        self.stats.batches += 1
        self.stats.bytes_written += len(data)
        self._enforce_limits()
        return written

    def _enforce_limits(self) -> None:
        files = self._files()
        total = sum(f.stat().st_size for f in files)
        while files and (total > self.max_bytes or len(files) > self.max_files):
            oldest = files[0]
            size = oldest.stat().st_size
            try:
                oldest.unlink()
            except OSError as exc:  # pragma: no cover - filesystem dependent
                raise StorageError(
                    f"store-and-forward rotation failed: {exc}",
                    details={"path": str(oldest)},
                ) from exc
            self.stats.rotations += 1
            self.stats.drops += size
            total -= size
            files = self._files()

    def _files(self) -> list[Path]:
        return sorted(self._root.glob(f"{FILE_PREFIX}*{FILE_SUFFIX}"), key=lambda p: p.name)

    # -- read path ---------------------------------------------------------

    def read_all(self) -> list[TelemetrySample]:
        """Read and remove every persisted record, skipping corrupt files."""
        samples: list[TelemetrySample] = []
        for path in self._files():
            try:
                samples.extend(self._read_file(path))
            except (ValueError, OSError, KeyError, ValidationError):
                self.stats.corrupt_files += 1
                continue
            finally:
                path.unlink(missing_ok=True)
        return samples

    def _read_file(self, path: Path) -> list[TelemetrySample]:
        out: list[TelemetrySample] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    record = json.loads(stripped)
                except ValueError as exc:
                    raise ValueError(f"{path}:{line_number} is not valid JSON") from exc
                if not isinstance(record, dict):
                    raise TypeError(f"{path}:{line_number} is not an object")
                out.append(TelemetrySample.from_dict(record))
        return out

    # -- introspection -----------------------------------------------------

    @property
    def pending(self) -> int:
        return len(self._pending)

    @property
    def file_count(self) -> int:
        return len(self._files())

    @property
    def bytes_used(self) -> int:
        return sum(f.stat().st_size for f in self._files())

    @property
    def utilization(self) -> float:
        return self.bytes_used / self.max_bytes

    def describe(self) -> dict[str, Any]:
        return {
            "directory": str(self._root),
            "pending": self.pending,
            "files": self.file_count,
            "bytes_used": self.bytes_used,
            "max_bytes": self.max_bytes,
            "utilization": self.utilization,
            "stats": {
                "writes": self.stats.writes,
                "batches": self.stats.batches,
                "bytes_written": self.stats.bytes_written,
                "drops": self.stats.drops,
                "rotations": self.stats.rotations,
                "corrupt_files": self.stats.corrupt_files,
                "replayed": self.stats.replayed,
            },
        }

    def clear(self) -> None:
        """Remove all persisted files and buffered records."""
        for path in self._files():
            path.unlink(missing_ok=True)
        self._pending.clear()
