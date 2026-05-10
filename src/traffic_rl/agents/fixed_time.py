from __future__ import annotations

from pathlib import Path

import numpy as np

from traffic_rl.agents.base import RLAgent


class FixedTimeAgent(RLAgent):
    """Baseline agent that cycles through phases on a fixed timer.

    Mimics real-world fixed-time traffic signal control: each phase holds for
    ``cycle_steps`` decision intervals before advancing to the next.

    At the default ``cycle_steps=6`` and ``decision_interval=5s``, each phase
    holds for 30 seconds — a common real-world pre-timed signal setting. A
    trained RL agent that cannot beat this baseline has no practical value.
    """

    def __init__(self, action_size: int, cycle_steps: int = 6) -> None:
        if action_size < 1:
            raise ValueError(f"action_size must be >= 1, got {action_size}")
        if cycle_steps < 1:
            raise ValueError(f"cycle_steps must be >= 1, got {cycle_steps}")
        self.action_size = action_size
        self.cycle_steps = cycle_steps
        self._step = 0

    def act(self, state_vector: np.ndarray, train: bool = False) -> int:
        phase = (self._step // self.cycle_steps) % self.action_size
        self._step += 1
        return phase

    def observe(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        pass  # stateless — no learning

    def save(self, path: str | Path) -> None:
        pass  # nothing to persist

    def load(self, path: str | Path) -> None:
        pass  # nothing to restore
