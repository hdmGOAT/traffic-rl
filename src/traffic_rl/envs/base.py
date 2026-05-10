from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol

from traffic_rl.types import Observation


# Abstract base class that every traffic environment must implement.
# Using ABC forces subclasses (CityFlowTrafficEnv, MockTrafficEnv) to define
# the three methods below — if they don't, Python raises an error at import time.
class TrafficEnv(ABC):

    @abstractmethod
    def reset(self) -> Observation:
        """Reset the simulation to its initial state and return the first observation.

        Called at the start of every episode. The agent uses the returned
        Observation to pick its first action.
        """
        raise NotImplementedError

    @abstractmethod
    def step(self, action: int) -> tuple[Observation, float, bool, dict]:
        """Advance the simulation by one decision interval and return results.

        Args:
            action: The phase index the agent wants to activate (0-indexed).

        Returns:
            observation: The new state of the intersection after the step.
            reward:      Scalar signal telling the agent how good this step was.
            done:        True when the episode has reached its time horizon.
            info:        Dictionary of diagnostic metrics (queue length, throughput,
                         travel time, etc.) used for evaluation but not for training.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def action_size(self) -> int:
        """Number of distinct traffic-light phases the agent can choose from."""
        raise NotImplementedError

    def set_replay_file(self, replay_file: str) -> None:
        """Tell the environment where to write a CityFlow replay log.

        Only meaningful for the CityFlow backend; the mock backend ignores it.
        Defined here with a no-op default so callers don't need to branch on type.
        """
        return None


# Protocol (structural type hint) for environments that support explicit seeding.
# Not all environments need this, so it's a separate interface rather than part of TrafficEnv.
class SupportsSeed(Protocol):
    def seed(self, seed: int) -> None:
        ...
