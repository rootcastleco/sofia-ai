"""Linear detrending."""

from __future__ import annotations

import numpy as np

from ..core.errors import InsufficientDataError

__all__ = ["detrend", "detrend_linear", "polynomial_trend"]


def polynomial_trend(values: np.ndarray, degree: int = 1) -> np.ndarray:
    """Least-squares polynomial trend of the signal.

    Args:
        values: 1-D input.
        degree: Polynomial degree (1 = linear).

    Returns:
        The fitted trend, same length as the input.
    """
    x = np.asarray(values, dtype=np.float64)
    if x.size == 0:
        return x.copy()
    if x.size < degree + 1:
        raise InsufficientDataError(
            f"need at least {degree + 1} samples for a degree-{degree} fit",
            details={"size": int(x.size), "degree": degree},
        )
    index = np.arange(x.size, dtype=np.float64)
    coefficients = np.polyfit(index, x, degree)
    return np.polyval(coefficients, index)


def detrend_linear(values: np.ndarray) -> np.ndarray:
    """Remove the least-squares linear trend."""
    x = np.asarray(values, dtype=np.float64)
    if x.size == 0:
        return x.copy()
    return np.asarray(x - polynomial_trend(x, degree=1), dtype=np.float64)


def detrend(values: np.ndarray, mode: str = "linear") -> np.ndarray:
    """Detrend with a named mode: ``"linear"``, ``"constant"`` or ``"none"``."""
    from ..core.errors import SignalError

    if mode == "linear":
        return detrend_linear(values)
    if mode == "constant":
        from .filters import detrend_constant

        return detrend_constant(values)
    if mode == "none":
        return np.asarray(values, dtype=np.float64).copy()
    raise SignalError(f"unknown detrend mode {mode!r}", details={"mode": mode})
