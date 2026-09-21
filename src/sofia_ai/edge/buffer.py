"""Bounded ring buffer with an explicit overflow policy.

Every ingest path in Sofia goes through this buffer. There is no unbounded queue
anywhere in the pipeline: if a consumer stalls, the buffer drops according to a
declared policy and records the loss in metrics rather than growing forever.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Generic, TypeVar

from ..core.errors import BufferOverflowError
from ..core.validation import validate_positive_int

__all__ = ["BoundedRingBuffer", "ChannelBuffer", "OverflowPolicy"]

T = TypeVar("T")


class OverflowPolicy(StrEnum):
    """What to do when the buffer is full."""

    DROP_OLDEST = "drop_oldest"
    """Evict the oldest item. Keeps the freshest data; loses history."""

    DROP_NEWEST = "drop_newest"
    """Reject the incoming item. Preserves history; loses the newest sample."""

    REJECT = "reject"
    """Raise :class:`BufferOverflowError`. Use where loss is unacceptable."""


@dataclass(slots=True)
class BoundedRingBuffer(Generic[T]):
    """Fixed-capacity FIFO with counters.

    Args:
        capacity: Maximum retained items.
        policy: Overflow behaviour.
    """

    capacity: int = 4096
    policy: OverflowPolicy = OverflowPolicy.DROP_OLDEST
    _items: deque[T] = field(init=False, repr=False)
    dropped: int = field(default=0, init=False)
    written: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        from collections import deque

        self.capacity = validate_positive_int(self.capacity, "capacity", maximum=10_000_000)
        self.policy = (self.policy if isinstance(self.policy, OverflowPolicy)
                       else OverflowPolicy(str(self.policy)))
        self._items = deque(maxlen=self.capacity)

    def push(self, item: T) -> bool:
        """Append an item. Returns True if retained.

        Raises:
            BufferOverflowError: when full and policy is ``REJECT``.
        """
        if len(self._items) >= self.capacity:
            if self.policy is OverflowPolicy.REJECT:
                raise BufferOverflowError(
                    f"buffer at capacity {self.capacity} and policy is REJECT",
                    details={"capacity": self.capacity},
                )
            if self.policy is OverflowPolicy.DROP_NEWEST:
                self.dropped += 1
                return False
            self.dropped += 1
        self._items.append(item)
        self.written += 1
        return True

    def extend(self, items: list[T]) -> int:
        """Append a batch. Returns the number retained."""
        return sum(1 for item in items if self.push(item))

    def pop(self) -> T:
        """Remove and return the oldest item."""
        if not self._items:
            raise BufferOverflowError("pop from an empty buffer", details={})
        return self._items.popleft()

    def drain(self, max_items: int | None = None) -> list[T]:
        """Remove and return up to ``max_items`` items in FIFO order."""
        count = len(self._items) if max_items is None else int(max_items)
        if count <= 0:
            return []
        out: list[T] = []
        for _ in range(min(count, len(self._items))):
            out.append(self._items.popleft())
        return out

    def peek(self, count: int = 1) -> list[T]:
        """Return up to ``count`` items without removing them."""
        return list(self._items)[: max(0, int(count))]

    def clear(self) -> None:
        self._items.clear()

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[T]:
        return iter(list(self._items))

    @property
    def is_full(self) -> bool:
        return len(self._items) >= self.capacity

    @property
    def saturation(self) -> float:
        """Fill fraction in ``[0, 1]``. Reported as the ``buffer_saturation`` metric."""
        return len(self._items) / self.capacity

    def stats(self) -> dict[str, int | float | str]:
        return {
            "capacity": self.capacity,
            "size": len(self._items),
            "written": self.written,
            "dropped": self.dropped,
            "saturation": self.saturation,
            "policy": self.policy.value,
        }


@dataclass(slots=True)
class ChannelBuffer(Generic[T]):
    """Per-channel buffers sharing one capacity and policy.

    Bounded in *both* dimensions: ``max_channels`` caps the number of channels and
    ``capacity`` caps each channel, so a hostile source cannot create unbounded
    channel cardinality.
    """

    capacity: int = 1024
    max_channels: int = 64
    policy: OverflowPolicy = OverflowPolicy.DROP_OLDEST
    _channels: dict[str, BoundedRingBuffer[T]] = field(init=False, repr=False)
    rejected_channels: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.capacity = validate_positive_int(self.capacity, "capacity", maximum=10_000_000)
        self.max_channels = validate_positive_int(self.max_channels, "max_channels",
                                                  maximum=10_000)
        self._channels = {}

    def push(self, channel: str, item: T) -> bool:
        """Append to a channel's buffer, creating it if allowed."""
        buffer = self._channels.get(channel)
        if buffer is None:
            if len(self._channels) >= self.max_channels:
                self.rejected_channels += 1
                return False
            buffer = BoundedRingBuffer(capacity=self.capacity, policy=self.policy)
            self._channels[channel] = buffer
        return buffer.push(item)

    def drain(self, channel: str, max_items: int | None = None) -> list[T]:
        buffer = self._channels.get(channel)
        return [] if buffer is None else buffer.drain(max_items)

    def channels(self) -> tuple[str, ...]:
        return tuple(self._channels)

    def total_size(self) -> int:
        return sum(len(b) for b in self._channels.values())

    def total_dropped(self) -> int:
        return sum(b.dropped for b in self._channels.values())

    def saturation(self) -> float:
        if not self._channels:
            return 0.0
        return max(b.saturation for b in self._channels.values())
