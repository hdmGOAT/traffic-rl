from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class RLAgent(ABC):
    @abstractmethod
    def act(self, state_vector: np.ndarray, train: bool = True) -> int:
        raise NotImplementedError

    @abstractmethod
    def observe(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        raise NotImplementedError

    def save(self, path: str | Path) -> None:
        raise NotImplementedError(f"save() is not implemented for {self.__class__.__name__}")

    def load(self, path: str | Path) -> None:
        raise NotImplementedError(f"load() is not implemented for {self.__class__.__name__}")
