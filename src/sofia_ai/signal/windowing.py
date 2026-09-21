"""Windowing: fixed-length, hop-based segmentation with deterministic alignment.

Windows are addressed by **sample index**, never by wall-clock time. A window that
cannot be filled is not emitted.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import numpy as np

from ..core.contracts import SignalWindow, TelemetrySample
from ..core.errors import InsufficientDataError, SignalError
from ..core.validation import validate_positive_int

__all__ = [
    "WindowSpec",
    "apply_window_function",
    "blackman",
    "frame_signal",
    "hamming",
    "hann",
    "rectangular",
    "sliding_windows",
    "windows_from_samples",
]


@dataclass(frozen=True, slots=True)
class WindowSpec:
    """Window geometry."""

    length: int = 1024
    hop: int = 512

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "length", validate_positive_int(self.length, "length", maximum=1 << 20)
        )
        object.__setattr__(self, "hop", validate_positive_int(self.hop, "hop", maximum=1 << 20))
        if self.hop > self.length:
            raise SignalError(
                "hop must not exceed length", details={"length": self.length, "hop": self.hop}
            )

    def count_for(self, n_samples: int) -> int:
        """Number of complete windows available in ``n_samples`` samples."""
        if n_samples < self.length:
            return 0
        return 1 + (n_samples - self.length) // self.hop


def frame_signal(values: np.ndarray, spec: WindowSpec) -> np.ndarray:
    """Segment a 1-D signal into a 2-D array of overlapping frames.

    Args:
        values: 1-D input.
        spec: Window geometry.

    Returns:
        Array of shape ``(n_windows, length)``.

    Raises:
        InsufficientDataError: if fewer than ``length`` samples are available.
        SignalError: if the input is not 1-D.
    """
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise SignalError("frame_signal expects a 1-D array", details={"ndim": arr.ndim})
    if arr.size < spec.length:
        raise InsufficientDataError(
            f"need at least {spec.length} samples, got {arr.size}",
            details={"need": spec.length, "got": int(arr.size)},
        )
    n_windows = spec.count_for(arr.size)
    strides = (arr.strides[0] * spec.hop, arr.strides[0])
    framed = np.lib.stride_tricks.as_strided(
        arr, shape=(n_windows, spec.length), strides=strides, writeable=False
    )
    return np.ascontiguousarray(framed, dtype=np.float64)


def sliding_windows(
    values: np.ndarray,
    *,
    length: int,
    hop: int | None = None,
    sample_rate: float,
    device_id: str,
    channel: str,
    unit: str,
    start_index: int = 0,
    start_time: float | None = None,
) -> Iterator[SignalWindow]:
    """Yield :class:`SignalWindow` objects over a signal."""
    spec = WindowSpec(length=length, hop=hop or length)
    frames = frame_signal(values, spec)
    for i in range(frames.shape[0]):
        index = start_index + i * spec.hop
        yield SignalWindow(
            values=frames[i].copy(),
            sample_rate=sample_rate,
            device_id=device_id,
            channel=channel,
            unit=unit,
            start_index=index,
            start_time=None if start_time is None else start_time + index / sample_rate,
        )


def windows_from_samples(
    samples: Sequence[TelemetrySample],
    *,
    length: int,
    hop: int | None = None,
    sample_rate: float,
    unit: str,
    channel: str,
    device_id: str,
) -> list[SignalWindow]:
    """Build windows from a list of :class:`TelemetrySample`-like objects."""
    values = np.asarray([float(s.value) for s in samples], dtype=np.float64)
    if values.size < length:
        return []
    start_time = float(samples[0].timestamp) if samples else None
    return list(
        sliding_windows(
            values,
            length=length,
            hop=hop,
            sample_rate=sample_rate,
            device_id=device_id,
            channel=channel,
            unit=unit,
            start_time=start_time,
        )
    )


def rectangular(n: int) -> np.ndarray:
    """Rectangular (boxcar) window of length ``n``."""
    validate_positive_int(n, "n", maximum=1 << 20)
    return np.ones(n, dtype=np.float64)


def hann(n: int, *, periodic: bool = True) -> np.ndarray:
    """Hann window. ``periodic=True`` is the correct choice for spectral analysis."""
    validate_positive_int(n, "n", maximum=1 << 20)
    denom = n if periodic else n - 1
    if denom <= 0:
        return np.ones(n, dtype=np.float64)
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n, dtype=np.float64) / denom)


def hamming(n: int, *, periodic: bool = True) -> np.ndarray:
    """Hamming window."""
    validate_positive_int(n, "n", maximum=1 << 20)
    denom = n if periodic else n - 1
    if denom <= 0:
        return np.ones(n, dtype=np.float64)
    return 0.54 - 0.46 * np.cos(2.0 * np.pi * np.arange(n, dtype=np.float64) / denom)


def blackman(n: int, *, periodic: bool = True) -> np.ndarray:
    """Blackman window."""
    validate_positive_int(n, "n", maximum=1 << 20)
    denom = n if periodic else n - 1
    if denom <= 0:
        return np.ones(n, dtype=np.float64)
    k = np.arange(n, dtype=np.float64)
    return (
        0.42
        - 0.5 * np.cos(2.0 * np.pi * k / denom)
        + 0.08 * np.cos(4.0 * np.pi * k / denom)
    )


_WINDOW_FUNCTIONS = {
    "rectangular": rectangular,
    "hann": hann,
    "hamming": hamming,
    "blackman": blackman,
}


def apply_window_function(values: np.ndarray, name: str = "hann") -> np.ndarray:
    """Multiply a signal by a named window function.

    Raises:
        SignalError: if ``name`` is not a known window.
    """
    func = _WINDOW_FUNCTIONS.get(name)
    if func is None:
        raise SignalError(
            f"unknown window {name!r}. Known: {sorted(_WINDOW_FUNCTIONS)}",
            details={"window": name},
        )
    arr = np.asarray(values, dtype=np.float64)
    return np.asarray(arr * func(arr.size), dtype=np.float64)
