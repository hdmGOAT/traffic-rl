from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol

from traffic_rl.types import Observation


class TrafficEnv(ABC):
    @abstractmethod
    def reset(self) -> Observation:
        raise NotImplementedError

    @abstractmethod
    def step(self, action: int) -> tuple[Observation, float, bool, dict]:
        raise NotImplementedError

    @property
    @abstractmethod
    def action_size(self) -> int:
        raise NotImplementedError

    def set_replay_file(self, replay_file: str) -> None:
        return None


class SupportsSeed(Protocol):
    def seed(self, seed: int) -> None:
        ...
