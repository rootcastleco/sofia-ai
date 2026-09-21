"""Q-network definitions for the experimental RL module.

Torch is imported lazily: the module can be imported (and its non-torch paths
tested) without the ``torch`` extra installed. Importing :class:`QNetwork` without
torch raises a typed error explaining which extra to install — never a bare
``ModuleNotFoundError``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ...core.errors import ModelError

__all__ = ["DuelingQNetwork", "QNetwork", "build_mlp", "torch_available"]


def torch_available() -> bool:
    """Whether PyTorch is importable in this environment."""
    import importlib.util

    return importlib.util.find_spec("torch") is not None


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise ModelError(
            "torch is not installed. Install the 'torch' extra: "
            "pip install sofia-engine[torch]",
            details={"extra": "torch"},
        ) from exc
    return torch


def build_mlp(input_size: int, output_size: int,
              hidden_sizes: Sequence[int] = (128, 128)) -> Any:
    """Build a simple MLP using torch. Validates shape arguments first."""
    torch = _torch()
    nn = torch.nn
    if input_size <= 0 or output_size <= 0:
        raise ModelError("network sizes must be positive",
                         details={"input": input_size, "output": output_size})
    layers: list[Any] = []
    in_size = int(input_size)
    for size in hidden_sizes:
        if int(size) <= 0:
            raise ModelError("hidden layer sizes must be positive", details={"size": size})
        layers.append(nn.Linear(in_size, int(size)))
        layers.append(nn.ReLU(inplace=True))
        in_size = int(size)
    layers.append(nn.Linear(in_size, int(output_size)))
    return nn.Sequential(*layers)


class QNetwork:
    """MLP Q-network. Thin wrapper so the agent stays framework-agnostic at the API level."""

    def __init__(self, state_size: int, action_size: int,
                 hidden_sizes: Sequence[int] = (128, 128)) -> None:
        torch = _torch()
        self.torch = torch
        self.state_size = int(state_size)
        self.action_size = int(action_size)
        self.hidden_sizes = tuple(int(h) for h in hidden_sizes)
        self.net = build_mlp(self.state_size, self.action_size, self.hidden_sizes)

    def forward(self, x: Any) -> Any:
        """Forward pass."""
        return self.net(x)

    def __call__(self, x: Any) -> Any:
        return self.forward(x)

    def parameters(self) -> Any:
        return self.net.parameters()

    def state_dict(self) -> Any:
        return self.net.state_dict()

    def load_state_dict(self, state: Any) -> None:
        self.net.load_state_dict(state)

    def to(self, device: str) -> QNetwork:
        self.net.to(device)
        return self

    def eval(self) -> QNetwork:
        self.net.eval()
        return self

    def train(self) -> QNetwork:
        self.net.train()
        return self


class DuelingQNetwork(QNetwork):
    """Dueling architecture: separate value and advantage streams.

    Useful when many actions have similar value — the value stream learns faster
    and the advantage stream only needs to learn the differences.
    """

    def __init__(self, state_size: int, action_size: int,
                 hidden_sizes: Sequence[int] = (128, 128)) -> None:
        torch = _torch()
        nn = torch.nn
        super().__init__(state_size, action_size, hidden_sizes)
        if not hidden_sizes:
            raise ModelError("DuelingQNetwork requires at least one hidden layer", details={})
        last_hidden = int(hidden_sizes[-1])
        self.feature = build_mlp(state_size, last_hidden, tuple(int(h)
                                                                for h in hidden_sizes[:-1]))
        self.value_head = nn.Linear(last_hidden, 1)
        self.advantage_head = nn.Linear(last_hidden, int(action_size))

    def forward(self, x: Any) -> Any:
        features = self.feature(x)
        value = self.value_head(features)
        advantage = self.advantage_head(features)
        return value + advantage - advantage.mean(dim=-1, keepdim=True)
