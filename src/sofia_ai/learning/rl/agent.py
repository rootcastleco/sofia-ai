"""DQN agent behind an explicit experimental boundary.

Preserves the pre-2.0 ``SofiaRLAgent`` API (``state_size``, ``action_size``,
``act``, ``step``, ``learn``, ``save_checkpoint``, ``load_checkpoint``) while
fixing the defects found in the audit:

* **RNG ownership** — the agent owns a ``numpy.random.Generator``; the global
  ``random``/``numpy.random`` state is never seeded (was S2-03),
* **deterministic evaluation** — :meth:`act` honours ``training=False`` /
  :meth:`eval_mode` and never explores during evaluation,
* **bounded replay** — :class:`ReplayBuffer` with a hard capacity,
* **checkpoint validation** — version, shape and key checks before loading,
* **device handling** — explicit, with a CPU fallback and no silent CUDA surprise,
* **no actuation** — the agent is **training/simulation only**. Any action it
  proposes must pass through :class:`sofia_ai.decision.policy.PolicyEngine`.

Online RL must never automatically control industrial machinery.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Final

import numpy as np

from ...core.errors import ModelError, ValidationError
from ...core.validation import validate_positive_float, validate_positive_int
from .buffers import Experience, ReplayBuffer
from .networks import QNetwork, _torch

__all__ = ["CHECKPOINT_VERSION", "RLConfig", "SofiaRLAgent"]

CHECKPOINT_VERSION: Final[str] = "2.0"


@dataclass(frozen=True, slots=True)
class RLConfig:
    """Hyperparameters. All validated."""

    buffer_size: int = 10_000
    batch_size: int = 64
    gamma: float = 0.99
    tau: float = 1e-3
    lr: float = 5e-4
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995
    hidden_sizes: tuple[int, ...] = (128, 128)
    grad_clip: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "buffer_size",
            validate_positive_int(self.buffer_size, "buffer_size", maximum=10_000_000),
        )
        object.__setattr__(
            self, "batch_size",
            validate_positive_int(self.batch_size, "batch_size", maximum=self.buffer_size),
        )
        object.__setattr__(self, "gamma", validate_positive_float(self.gamma, "gamma",
                                                                  maximum=1.0))
        object.__setattr__(self, "tau", validate_positive_float(self.tau, "tau", maximum=1.0))
        object.__setattr__(self, "lr", validate_positive_float(self.lr, "lr", maximum=1.0))
        object.__setattr__(self, "epsilon_start",
                           _probability(self.epsilon_start, "epsilon_start"))
        object.__setattr__(self, "epsilon_end", _probability(self.epsilon_end, "epsilon_end"))
        object.__setattr__(self, "epsilon_decay",
                           validate_positive_float(self.epsilon_decay, "epsilon_decay",
                                                   maximum=1.0))
        if self.epsilon_end > self.epsilon_start:
            raise ValidationError("epsilon_end must not exceed epsilon_start", details={})
        object.__setattr__(self, "grad_clip",
                           validate_positive_float(self.grad_clip, "grad_clip", maximum=1e6))


def _probability(value: float, name: str) -> float:
    from ...core.validation import validate_probability

    return validate_probability(value, name)


@dataclass(slots=True)
class SofiaRLAgent:
    """Double-DQN agent for simulation. Not a controller.

    Args:
        state_size: Observation dimension.
        action_size: Action count.
        config: Hyperparameters.
        device: Torch device. ``None`` selects CUDA if available, else CPU.
        seed: Seed for this agent's owned RNGs.
    """

    state_size: int
    action_size: int
    config: RLConfig = field(default_factory=RLConfig)
    device: str | None = None
    seed: int = 0

    _initialised: bool = field(default=False, init=False, repr=False)

    rng: np.random.Generator = field(init=False)
    qnetwork_local: Any = field(init=False)
    qnetwork_target: Any = field(init=False)
    optimizer: Any = field(init=False)
    memory: ReplayBuffer[Any] = field(init=False)
    epsilon: float = field(init=False)
    training: bool = field(init=False)
    steps: int = field(init=False)
    episodes: int = field(init=False)
    last_loss: float | None = field(init=False)

    def __post_init__(self) -> None:
        self.state_size = validate_positive_int(self.state_size, "state_size", maximum=1 << 20)
        self.action_size = validate_positive_int(self.action_size, "action_size", maximum=1 << 16)
        torch = _torch()

        # Owned randomness. The global RNG is deliberately left alone.
        self.rng = np.random.default_rng(int(self.seed))
        torch.manual_seed(int(self.seed))

        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.qnetwork_local = QNetwork(self.state_size, self.action_size,
                                       self.config.hidden_sizes).to(self.device)
        self.qnetwork_target = QNetwork(self.state_size, self.action_size,
                                        self.config.hidden_sizes).to(self.device)
        # Start with an exact copy: the target must not begin as an untrained net.
        self.hard_update()
        self.optimizer = torch.optim.Adam(self.qnetwork_local.parameters(), lr=self.config.lr)
        self.memory = ReplayBuffer(self.config.buffer_size, seed=int(self.seed))
        self.epsilon = float(self.config.epsilon_start)
        self.training = True
        self.steps = 0
        self.episodes = 0
        self.last_loss: float | None = None
        self._initialised = True

    # -- modes -------------------------------------------------------------

    def eval_mode(self) -> SofiaRLAgent:
        """Switch to deterministic evaluation: no exploration, no learning."""
        self.training = False
        self.qnetwork_local.eval()
        self.qnetwork_target.eval()
        return self

    def train_mode(self) -> SofiaRLAgent:
        """Switch back to training."""
        self.training = True
        self.qnetwork_local.train()
        return self

    # -- acting ------------------------------------------------------------

    def act(self, state: np.ndarray | Sequence[float], *, training: bool | None = None) -> int:
        """Select an action.

        Args:
            state: Observation, shape ``(state_size,)``.
            training: Override the agent mode. In evaluation mode exploration is
                disabled and the action is the argmax of the local network.

        Returns:
            Action index in ``[0, action_size)``.
        """
        explore = self.training if training is None else training
        array = np.asarray(state, dtype=np.float32).ravel()
        if array.size != self.state_size:
            raise ValidationError(
                f"state has {array.size} values, expected {self.state_size}",
                details={"got": int(array.size), "expected": self.state_size},
            )
        if not np.all(np.isfinite(array)):
            raise ValidationError("state contains non-finite values", details={})

        if explore and float(self.rng.random()) < self.epsilon:
            return int(self.rng.integers(0, self.action_size))

        torch = _torch()
        tensor = torch.from_numpy(array).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.qnetwork_local(tensor)
        return int(q_values.argmax(dim=1).item())

    def step(self, state: Any, action: int, reward: float, next_state: Any,
             done: bool) -> float | None:
        """Store a transition and learn when the buffer has enough samples.

        Returns:
            The loss if a learning step ran, else ``None``.
        """
        self.memory.push(Experience(
            state=np.asarray(state, dtype=np.float32).ravel(),
            action=int(action),
            reward=float(reward),
            next_state=np.asarray(next_state, dtype=np.float32).ravel(),
            done=bool(done),
        ))
        if len(self.memory) >= self.config.batch_size:
            return self.learn(self.memory.sample(self.config.batch_size))
        return None

    # -- learning ----------------------------------------------------------

    def learn(self, experiences: Sequence[Experience]) -> float:
        """One Double-DQN gradient step.

        Local network selects the next action; target network evaluates it.
        """
        if not experiences:
            raise ValidationError("learn() requires at least one experience", details={})
        if not self.training:
            raise ModelError(
                "agent is in evaluation mode; call train_mode() before learning",
                details={},
            )
        torch = _torch()
        nn = torch.nn

        states = torch.from_numpy(
            np.asarray([e.state for e in experiences], dtype=np.float32)
        ).to(self.device)
        actions = torch.from_numpy(
            np.asarray([[e.action] for e in experiences], dtype=np.int64)
        ).to(self.device)
        rewards = torch.from_numpy(
            np.asarray([[e.reward] for e in experiences], dtype=np.float32)
        ).to(self.device)
        next_states = torch.from_numpy(
            np.asarray([e.next_state for e in experiences], dtype=np.float32)
        ).to(self.device)
        dones = torch.from_numpy(
            np.asarray([[float(e.done)] for e in experiences], dtype=np.float32)
        ).to(self.device)

        if states.shape[1] != self.state_size:
            raise ValidationError(
                f"experience state width {states.shape[1]} != {self.state_size}",
                details={"got": int(states.shape[1]), "expected": self.state_size},
            )

        current_q = self.qnetwork_local(states).gather(1, actions)
        with torch.no_grad():
            next_actions = self.qnetwork_local(next_states).argmax(dim=1, keepdim=True)
            q_targets_next = self.qnetwork_target(next_states).gather(1, next_actions)
        q_targets = rewards + (self.config.gamma * q_targets_next * (1.0 - dones))

        loss = nn.MSELoss()(current_q, q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        if self.config.grad_clip > 0:
            nn.utils.clip_grad_norm_(self.qnetwork_local.parameters(), self.config.grad_clip)
        self.optimizer.step()
        self.soft_update()

        self.steps += 1
        self.last_loss = float(loss.item())
        return self.last_loss

    def soft_update(self) -> None:
        """Polyak average the target network toward the local network."""
        self._polyak(self.config.tau)

    def hard_update(self) -> None:
        """Copy the local network into the target network exactly."""
        self._polyak(1.0)

    def _polyak(self, tau: float) -> None:
        torch = _torch()
        with torch.no_grad():
            for target_param, local_param in zip(
                self.qnetwork_target.parameters(), self.qnetwork_local.parameters(),
                strict=True,
            ):
                target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def decay_epsilon(self) -> float:
        """Multiplicatively decay epsilon, floored at ``epsilon_end``."""
        self.epsilon = max(self.config.epsilon_end, self.epsilon * self.config.epsilon_decay)
        return self.epsilon

    # -- persistence -------------------------------------------------------

    def save_checkpoint(self, filepath: str) -> str:
        """Save a versioned checkpoint. Weights only; no executable payload."""
        torch = _torch()
        payload = {
            "checkpoint_version": CHECKPOINT_VERSION,
            "created_at": time.time(),
            "state_size": self.state_size,
            "action_size": self.action_size,
            "seed": int(self.seed),
            "config": {
                "buffer_size": self.config.buffer_size,
                "batch_size": self.config.batch_size,
                "gamma": self.config.gamma,
                "tau": self.config.tau,
                "lr": self.config.lr,
                "epsilon_start": self.config.epsilon_start,
                "epsilon_end": self.config.epsilon_end,
                "epsilon_decay": self.config.epsilon_decay,
                "hidden_sizes": list(self.config.hidden_sizes),
            },
            "state_dict": self.qnetwork_local.state_dict(),
            "epsilon": self.epsilon,
            "steps": self.steps,
            "memory_size": len(self.memory),
        }
        torch.save(payload, filepath)
        return filepath

    @classmethod
    def load_checkpoint(cls, filepath: str, *, device: str | None = None,
                        map_location: str | None = None) -> SofiaRLAgent:
        """Load a checkpoint with validation.

        Raises:
            ModelError: on version mismatch, shape mismatch, or missing keys.
        """
        torch = _torch()
        location = map_location or device or "cpu"
        try:
            checkpoint = torch.load(filepath, map_location=location, weights_only=False)
        except TypeError:  # older torch without weights_only
            checkpoint = torch.load(filepath, map_location=location)
        if not isinstance(checkpoint, dict):
            raise ModelError("checkpoint is not a mapping", details={"path": filepath})

        version = str(checkpoint.get("checkpoint_version", "1.0"))
        if version.split(".", maxsplit=1)[0] != CHECKPOINT_VERSION.split(".")[0]:
            raise ModelError(
                f"checkpoint version {version!r} is incompatible with "
                f"{CHECKPOINT_VERSION!r}",
                details={"version": version},
            )
        for key in ("state_size", "action_size", "state_dict", "config"):
            if key not in checkpoint:
                raise ModelError(
                    f"checkpoint is missing required key {key!r}",
                    details={"missing": key},
                )

        cfg = dict(checkpoint["config"])
        agent = cls(
            state_size=int(checkpoint["state_size"]),
            action_size=int(checkpoint["action_size"]),
            config=RLConfig(
                buffer_size=int(cfg.get("buffer_size", 10_000)),
                batch_size=int(cfg.get("batch_size", 64)),
                gamma=float(cfg.get("gamma", 0.99)),
                tau=float(cfg.get("tau", 1e-3)),
                lr=float(cfg.get("lr", 5e-4)),
                epsilon_start=float(cfg.get("epsilon_start", 1.0)),
                epsilon_end=float(cfg.get("epsilon_end", 0.01)),
                epsilon_decay=float(cfg.get("epsilon_decay", 0.995)),
                hidden_sizes=tuple(int(h) for h in cfg.get("hidden_sizes", (128, 128))),
            ),
            device=device or location,
            seed=int(checkpoint.get("seed", 0)),
        )
        try:
            agent.qnetwork_local.load_state_dict(checkpoint["state_dict"])
        except (RuntimeError, ValueError, TypeError) as exc:
            raise ModelError(
                f"checkpoint state dict does not match the network: {exc}",
                details={"path": filepath},
            ) from exc
        agent.hard_update()
        agent.epsilon = float(checkpoint.get("epsilon", agent.epsilon))
        agent.steps = int(checkpoint.get("steps", 0))
        return agent

    @staticmethod
    def validate_checkpoint(filepath: str) -> dict[str, Any]:
        """Inspect a checkpoint without constructing an agent.

        Returns a summary dict. Raises :class:`ModelError` if unusable.
        """
        torch = _torch()
        try:
            checkpoint = torch.load(filepath, map_location="cpu", weights_only=False)
        except TypeError:  # pragma: no cover - older torch
            checkpoint = torch.load(filepath, map_location="cpu")
        if not isinstance(checkpoint, dict):
            raise ModelError("checkpoint is not a mapping", details={"path": filepath})
        version = str(checkpoint.get("checkpoint_version", "unknown"))
        if version.split(".", maxsplit=1)[0] != CHECKPOINT_VERSION.split(".")[0]:
            raise ModelError(
                f"checkpoint version {version!r} is incompatible with {CHECKPOINT_VERSION!r}",
                details={"version": version},
            )
        return {
            "checkpoint_version": version,
            "state_size": checkpoint.get("state_size"),
            "action_size": checkpoint.get("action_size"),
            "steps": checkpoint.get("steps"),
            "has_state_dict": "state_dict" in checkpoint,
        }

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return (
            f"SofiaRLAgent(state_size={self.state_size}, action_size={self.action_size}, "
            f"device={self.device}, epsilon={self.epsilon:.3f}, steps={self.steps})"
        )

