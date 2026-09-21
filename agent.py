"""Deprecated root-level import shim.

The historical ``agent.py`` lived at the repository root and imported ``torch``
at module import time. It has been replaced by
:mod:`sofia_ai.learning.rl`, which exposes the same public names without a hard
``torch`` import.

Importing :mod:`sofia_ai` first is required; this shim only works when the
package is installed or ``src`` is on ``sys.path`` (e.g. ``PYTHONPATH=src``).

.. deprecated:: 2.0
   Use ``sofia_ai.learning.rl`` instead.
"""

from __future__ import annotations

import warnings as _warnings

from sofia_ai.learning.rl import (
    SofiaRLAgent,
    QNetwork,
    ReplayBuffer,
)

_warnings.warn(
    "Importing 'agent' from the repository root is deprecated and will be "
    "removed in a future release; use 'sofia_ai.learning.rl' instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["SofiaRLAgent", "QNetwork", "ReplayBuffer"]