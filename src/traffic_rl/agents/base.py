from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


# Every agent — whether DQN, tabular Q-learning, or fixed-time — must implement
# these three methods. Using ABC prevents accidentally instantiating an incomplete agent.
class RLAgent(ABC):

    @abstractmethod
    def act(self, state_vector: np.ndarray, train: bool = True) -> int:
        """Choose an action (phase index) given the current state.

        Args:
            state_vector: Flattened observation from Observation.as_vector().
            train:        When True, the agent may explore (e.g. epsilon-greedy).
                          When False (evaluation), always pick the best known action.

        Returns:
            Integer index of the chosen traffic-light phase.
        """
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
        """Record the outcome of the last action and update the agent's knowledge.

        This is where learning happens — replay buffer updates, Q-value updates,
        epsilon decay, target network syncs, etc.

        Args:
            state:      The state before the action was taken.
            action:     The phase index that was chosen.
            reward:     The reward signal received from the environment.
            next_state: The state after the action was applied.
            done:       True if this was the last step of the episode.
        """
        raise NotImplementedError

    def save(self, path: str | Path) -> None:
        """Persist the agent's learned weights/tables to disk.

        Default raises NotImplementedError. Agents with learnable parameters
        must override this. Stateless agents (e.g. FixedTimeAgent) can no-op.
        """
        raise NotImplementedError(f"save() is not implemented for {self.__class__.__name__}")

    def load(self, path: str | Path) -> None:
        """Restore previously saved weights/tables from disk.

        Default raises NotImplementedError. Agents with learnable parameters
        must override this. Stateless agents (e.g. FixedTimeAgent) can no-op.
        """
        raise NotImplementedError(f"load() is not implemented for {self.__class__.__name__}")

    # Optional runtime control hooks for stopping/starting learning.
    def freeze(self) -> None:
        """Request the agent stop updating its parameters going forward.

        Default implementation is a no-op; learnable agents should override
        to disable learning (e.g. skip observe/optimise and stop epsilon decay).
        """
        return

    def unfreeze(self) -> None:
        """Re-enable learning after a previous freeze.

        Default is a no-op; learnable agents may override to re-enable updates.
        """
        return
