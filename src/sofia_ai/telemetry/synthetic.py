"""Deterministic synthetic telemetry generation.

Used by tests, benchmarks and the offline demos. Every generator takes an
injected :class:`numpy.random.Generator`; nothing here seeds a global RNG.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np

from ..core.contracts import TelemetrySample
from ..core.errors import ValidationError
from ..core.time import TimeSource
from ..core.validation import validate_positive_float, validate_positive_int
from .base import SourceLimits, TelemetrySource

__all__ = [
    "MultiChannelSyntheticSource",
    "SyntheticMachineProfile",
    "SyntheticVibrationSource",
    "inject_faults",
    "rotating_machine_signal",
]

DEFAULT_UNIT: Final[str] = "g"


def rotating_machine_signal(
    n_samples: int,
    sample_rate: float,
    *,
    shaft_hz: float = 25.0,
    harmonics: Sequence[float] = (1.0, 2.0, 3.0),
    amplitudes: Sequence[float] = (1.0, 0.4, 0.15),
    noise_std: float = 0.05,
    bearing_hz: float | None = None,
    bearing_amplitude: float = 0.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate a synthetic rotating-machine acceleration signal.

    The signal is a sum of shaft harmonics with optional bearing tone and Gaussian
    noise. It is deterministic given ``rng`` (or fully deterministic when
    ``rng is None``, in which case noise is omitted).

    Args:
        n_samples: Number of samples.
        sample_rate: Sample rate in Hz.
        shaft_hz: Shaft rotation frequency in Hz.
        harmonics: Harmonic orders of the shaft frequency.
        amplitudes: Amplitude per harmonic, in g.
        noise_std: Gaussian noise standard deviation in g.
        bearing_hz: Optional non-synchronous bearing tone frequency.
        bearing_amplitude: Bearing tone amplitude in g.
        rng: Optional owned random generator.

    Returns:
        Float64 array of length ``n_samples``.
    """
    n_samples = validate_positive_int(n_samples, "n_samples", maximum=10_000_000)
    sample_rate = validate_positive_float(sample_rate, "sample_rate")
    if len(harmonics) != len(amplitudes):
        raise ValidationError(
            "harmonics and amplitudes must have equal length",
            details={"harmonics": len(harmonics), "amplitudes": len(amplitudes)},
        )
    if shaft_hz <= 0 or shaft_hz >= sample_rate / 2:
        raise ValidationError(
            f"shaft_hz must be in (0, {sample_rate / 2})",
            details={"shaft_hz": shaft_hz, "nyquist": sample_rate / 2},
        )

    t = np.arange(n_samples, dtype=np.float64) / sample_rate
    signal = np.zeros(n_samples, dtype=np.float64)
    for order, amplitude in zip(harmonics, amplitudes, strict=True):
        signal += float(amplitude) * np.sin(2.0 * math.pi * shaft_hz * float(order) * t)
    if bearing_hz is not None and bearing_amplitude > 0:
        signal += float(bearing_amplitude) * np.sin(2.0 * math.pi * float(bearing_hz) * t)
    if rng is not None and noise_std > 0:
        signal += rng.normal(0.0, float(noise_std), size=n_samples)
    return signal


def inject_faults(
    signal: np.ndarray,
    *,
    start_fraction: float = 0.6,
    amplitude_gain: float = 3.0,
    offset: float = 0.0,
) -> np.ndarray:
    """Apply a deterministic late-window degradation to a signal.

    Args:
        signal: Input array.
        start_fraction: Fraction of the record after which degradation begins.
        amplitude_gain: Multiplicative gain applied after the onset.
        offset: Additive DC offset applied after the onset.
    """
    arr = np.asarray(signal, dtype=np.float64)
    if not 0.0 <= start_fraction <= 1.0:
        raise ValidationError(
            "start_fraction must be within [0, 1]", details={"value": start_fraction}
        )
    out = arr.copy()
    onset = int(len(out) * start_fraction)
    out[onset:] = out[onset:] * float(amplitude_gain) + float(offset)
    return out


@dataclass(slots=True)
class SyntheticMachineProfile:
    """A named machine profile for reproducible synthetic telemetry."""

    device_id: str = "sim-pump-01"
    channel: str = "vibration_x"
    unit: str = DEFAULT_UNIT
    sample_rate: float = 1000.0
    shaft_hz: float = 25.0
    harmonics: tuple[float, ...] = (1.0, 2.0, 3.0)
    amplitudes: tuple[float, ...] = (1.0, 0.4, 0.15)
    noise_std: float = 0.05
    bearing_hz: float | None = None
    bearing_amplitude: float = 0.0
    seed: int = 1234


class SyntheticVibrationSource(TelemetrySource):
    """Emits a synthetic vibration waveform as ``TelemetrySample`` objects."""

    def __init__(
        self,
        profile: SyntheticMachineProfile | None = None,
        *,
        n_samples: int = 4096,
        degrade: bool = False,
        start_time: float = 1_700_000_000.0,
        source_id: str = "synthetic",
        limits: SourceLimits | None = None,
        batch_size: int = 256,
        clock: TimeSource | None = None,
    ) -> None:
        super().__init__(source_id=source_id, limits=limits)
        self.profile = profile or SyntheticMachineProfile()
        self.n_samples = validate_positive_int(n_samples, "n_samples", maximum=10_000_000)
        self.degrade = degrade
        self.start_time = start_time
        self.batch_size = max(1, int(batch_size))
        self.clock = clock
        self._rng = np.random.default_rng(self.profile.seed)
        self._values: np.ndarray | None = None
        self._cursor = 0

    def _open(self) -> None:
        self._rng = np.random.default_rng(self.profile.seed)
        signal = rotating_machine_signal(
            self.n_samples,
            self.profile.sample_rate,
            shaft_hz=self.profile.shaft_hz,
            harmonics=self.profile.harmonics,
            amplitudes=self.profile.amplitudes,
            noise_std=self.profile.noise_std,
            bearing_hz=self.profile.bearing_hz,
            bearing_amplitude=self.profile.bearing_amplitude,
            rng=self._rng,
        )
        if self.degrade:
            signal = inject_faults(signal, start_fraction=0.6, amplitude_gain=3.0)
        self._values = signal
        self._cursor = 0

    def read(self, *, max_records: int | None = None) -> list[TelemetrySample]:
        self._require_open()
        assert self._values is not None
        count = self.batch_size if max_records is None else int(max_records)
        if count <= 0:
            return []
        end = min(self._cursor + count, self.n_samples)
        if self._cursor >= end:
            return []
        out: list[TelemetrySample] = []
        for index in range(self._cursor, end):
            out.append(
                TelemetrySample(
                    timestamp=self.start_time + index / self.profile.sample_rate,
                    device_id=self.profile.device_id,
                    channel=self.profile.channel,
                    value=float(self._values[index]),
                    unit=self.profile.unit,
                    source=self.source_id,
                    sequence_number=index,
                    ingestion_time=None if self.clock is None else self.clock.wall(),
                )
            )
        self._cursor = end
        self.stats.records_read += len(out)
        self.stats.records_emitted += len(out)
        return out

    def values(self) -> np.ndarray:
        """The full generated waveform (opens the source if needed)."""
        if self._values is None:
            self.open()
        assert self._values is not None
        return self._values


class MultiChannelSyntheticSource(TelemetrySource):
    """Multiple correlated channels from one synthetic machine.

    Channels share the shaft frequency so multivariate detectors have real
    structure to work with rather than independent noise.
    """

    def __init__(
        self,
        profiles: tuple[SyntheticMachineProfile, ...] | None = None,
        *,
        n_samples: int = 4096,
        start_time: float = 1_700_000_000.0,
        degrade_fraction: float = 0.0,
        source_id: str = "synthetic-multi",
        limits: SourceLimits | None = None,
    ) -> None:
        super().__init__(source_id=source_id, limits=limits)
        self.profiles = profiles or (
            SyntheticMachineProfile(channel="vibration_x"),
            SyntheticMachineProfile(channel="vibration_y", amplitudes=(0.8, 0.3, 0.1)),
            SyntheticMachineProfile(channel="temperature", unit="C"),
        )
        self.n_samples = validate_positive_int(n_samples, "n_samples", maximum=10_000_000)
        self.start_time = start_time
        self.degrade_fraction = degrade_fraction
        self._sources: list[SyntheticVibrationSource] = []
        self._opened = False

    def _open(self) -> None:
        self._sources = [
            SyntheticVibrationSource(
                profile=p,
                n_samples=self.n_samples,
                degrade=self.degrade_fraction > 0.0,
                start_time=self.start_time,
                source_id=self.source_id,
                limits=self.limits,
                batch_size=self.n_samples,
            )
            for p in self.profiles
        ]
        for src in self._sources:
            src.open()
        self._opened = True

    def read(self, *, max_records: int | None = None) -> list[TelemetrySample]:
        self._require_open()
        out: list[TelemetrySample] = []
        for index, src in enumerate(self._sources):
            profile = self.profiles[index]
            if profile.unit in ("C", "F", "K"):
                values = self._thermal_waveform(profile)
                for i, value in enumerate(values[: self.n_samples]):
                    out.append(
                        TelemetrySample(
                            timestamp=self.start_time + i / profile.sample_rate,
                            device_id=profile.device_id,
                            channel=profile.channel,
                            value=float(value),
                            unit=profile.unit,
                            source=self.source_id,
                            sequence_number=i,
                        )
                    )
                continue
            out.extend(src.read(max_records=self.n_samples))
        out.sort(key=lambda s: (s.timestamp, s.channel))
        self.stats.records_read += len(out)
        self.stats.records_emitted += len(out)
        return out

    def _thermal_waveform(self, profile: SyntheticMachineProfile) -> np.ndarray:
        t = np.arange(self.n_samples, dtype=np.float64) / profile.sample_rate
        base = 45.0 + 5.0 * np.sin(2.0 * math.pi * 0.05 * t)
        if self.degrade_fraction > 0.0:
            onset = int(self.n_samples * 0.6)
            ramp = np.arange(self.n_samples, dtype=np.float64)
            base = base + np.where(
                np.arange(self.n_samples) >= onset,
                (ramp - onset) / max(1, self.n_samples - onset) * 25.0,
                0.0,
            )
        return base

    def frames(self) -> Iterator[list[TelemetrySample]]:
        """Yield per-timestamp channel groups."""
        samples = self.read()
        grouped: dict[float, list[TelemetrySample]] = {}
        for sample in samples:
            grouped.setdefault(sample.timestamp, []).append(sample)
        for key in sorted(grouped):
            yield grouped[key]
