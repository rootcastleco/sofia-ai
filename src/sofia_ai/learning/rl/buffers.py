"""Bounded replay buffers with owned randomness.

Differences from the pre-2.0 ``ReplayBuffer``:

* the RNG is **owned** (``numpy.random.Generator``), never the global one,
* ``sample`` is deterministic given the seed and the call sequence,
* capacity is enforced and overflow is counted,
* sampling from an empty or under-full buffer raises a typed error rather than
  raising a cryptic library error.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Generic, TypeVar

import numpy as np

from ...core.errors import InsufficientDataError
from ...core.validation import validate_positive_int

__all__ = ["Experience", "PrioritizedReplayBuffer", "ReplayBuffer"]

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class Experience:
    """One RL transition. Typed so a wrong shape is caught at construction."""

    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool

    def __post_init__(self) -> None:
        if self.state.shape != self.next_state.shape:
            raise InsufficientDataError(
                "state and next_state must have the same shape",
                details={"state": list(self.state.shape),
                         "next": list(self.next_state.shape)},
            )
        if not np.all(np.isfinite(self.state)) or not np.all(np.isfinite(self.next_state)):
            raise InsufficientDataError("experience contains non-finite values", details={})
        if not np.isfinite(float(self.reward)):
            raise InsufficientDataError("reward must be finite", details={})


class ReplayBuffer(Generic[T]):
    """Bounded ring buffer with uniform sampling.

    Args:
        capacity: Maximum stored items. Oldest are evicted first.
        seed: Seed for this buffer's **own** generator. The global RNG is untouched.
    """

    def __init__(self, capacity: int, seed: int = 0) -> None:
        self.capacity = validate_positive_int(capacity, "capacity", maximum=10_000_000)
        self.seed = int(seed)
        self._rng = np.random.default_rng(self.seed)
        self._memory: deque[T] = deque(maxlen=self.capacity)
        self.evicted = 0

    def push(self, item: T) -> None:
        """Append an item, evicting the oldest when full."""
        if len(self._memory) >= self.capacity:
            self.evicted += 1
        self._memory.append(item)

    def extend(self, items: Sequence[T]) -> None:
        for item in items:
            self.push(item)

    def sample(self, batch_size: int) -> list[T]:
        """Uniformly sample ``batch_size`` items without replacement."""
        n = validate_positive_int(batch_size, "batch_size", maximum=self.capacity)
        if n > len(self._memory):
            raise InsufficientDataError(
                f"cannot sample {n} items from a buffer holding {len(self._memory)}",
                details={"requested": n, "available": len(self._memory)},
            )
        indices = self._rng.choice(len(self._memory), size=n, replace=False)
        items = list(self._memory)
        return [items[int(i)] for i in indices]

    def clear(self) -> None:
        self._memory.clear()

    def reseed(self, seed: int) -> None:
        """Reset this buffer's generator. Does not touch the global RNG."""
        self.seed = int(seed)
        self._rng = np.random.default_rng(self.seed)

    def __len__(self) -> int:
        return len(self._memory)

    @property
    def is_full(self) -> bool:
        return len(self._memory) >= self.capacity

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"ReplayBuffer({len(self._memory)}/{self.capacity}, seed={self.seed})"


class PrioritizedReplayBuffer(ReplayBuffer[Experience]):
    """Proportional prioritized replay.

    Priorities are bounded and stored alongside the experiences; the buffer stays
    bounded in every dimension.
    """

    def __init__(self, capacity: int, seed: int = 0, *, alpha: float = 0.6,
                 epsilon: float = 1e-6) -> None:
        super().__init__(capacity, seed)
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self._priorities: deque[float] = deque(maxlen=self.capacity)

    def push_with_priority(self, experience: Experience, priority: float) -> None:
        """Push an experience with an explicit non-negative priority."""
        if not np.isfinite(priority) or priority < 0:
            raise InsufficientDataError("priority must be finite and non-negative",
                                        details={"priority": priority})
        self._priorities.append(float(priority) + self.epsilon)
        self.push(experience)

    def sample_prioritized(self, batch_size: int) -> tuple[list[Experience], np.ndarray]:
        """Sample proportional to ``priority ** alpha``.

        Returns:
            ``(experiences, indices)`` so the caller can update priorities.
        """
        n = validate_positive_int(batch_size, "batch_size", maximum=self.capacity)
        if n > len(self._priorities):
            raise InsufficientDataError(
                f"cannot sample {n} items from {len(self._priorities)}",
                details={"requested": n, "available": len(self._priorities)},
            )
        weights = np.asarray(self._priorities, dtype=np.float64) ** self.alpha
        total = float(np.sum(weights))
        probabilities = np.full(len(weights), 1.0 / len(weights)) if total <= 0 else weights / total
        indices = self._rng.choice(len(probabilities), size=n, replace=False, p=probabilities)
        items = list(self._memory)
        return [items[int(i)] for i in indices], indices

    def update_priority(self, index: int, priority: float) -> None:
        """Update one priority in place."""
        if not 0 <= index < len(self._priorities):
            raise InsufficientDataError("priority index out of range",
                                        details={"index": index})
        priorities = list(self._priorities)
        priorities[index] = float(priority) + self.epsilon
        self._priorities = deque(priorities, maxlen=self.capacity)

