from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

from traffic_rl.agents.base import RLAgent


class TabularQAgent(RLAgent):
    """Classic Q-learning using a lookup table instead of a neural network.

    The state space is discretised (each continuous value floored to an integer)
    so that (state, action) pairs can be stored as dictionary entries.
    Practical only for small state spaces — breaks down when the number of
    lanes/phases grows, because unseen state combinations start with Q=0.
    """

    def __init__(
        self,
        action_size: int,
        gamma: float,           # Discount factor: how much future rewards are worth vs immediate ones.
        learning_rate: float,   # Step size for Q-value updates (alpha in the Bellman equation).
        epsilon_start: float,   # Initial exploration rate (1.0 = fully random at the start).
        epsilon_end: float,     # Minimum exploration rate the agent decays down to.
        epsilon_decay: float,   # Multiplicative decay applied to epsilon after each step.
        seed: int = 7,
    ) -> None:
        self.action_size = action_size
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        # Q-table: maps a discretised state tuple → array of Q-values, one per action.
        # defaultdict auto-initialises unseen states to all zeros (optimistic initialisation).
        self.q_table: dict[tuple[int, ...], np.ndarray] = defaultdict(
            lambda: np.zeros(self.action_size, dtype=np.float32)
        )
        self.rng = np.random.default_rng(seed)

    def _to_key(self, state_vector: np.ndarray) -> tuple[int, ...]:
        """Discretise a continuous state vector into a hashable tuple.

        Each float value is floored to the nearest integer, reducing the
        continuous state space to a finite set of buckets that can be
        used as dictionary keys.
        """
        bucketized = np.floor(state_vector).astype(int)
        return tuple(int(value) for value in bucketized)

    def act(self, state_vector: np.ndarray, train: bool = True) -> int:
        """Epsilon-greedy action selection.

        During training, explore randomly with probability epsilon;
        otherwise pick the action with the highest Q-value for this state.
        During evaluation (train=False), always exploit (no random actions).
        """
        key = self._to_key(state_vector)
        if train and self.rng.random() < self.epsilon:
            # Explore: pick a random phase.
            return int(self.rng.integers(0, self.action_size))
        # Exploit: pick the phase with the highest estimated Q-value.
        return int(np.argmax(self.q_table[key]))

    def observe(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Apply the Q-learning update rule (Bellman equation).

        new_Q = old_Q + lr * (reward + gamma * max_Q(next_state) - old_Q)

        If the episode is done, there is no future state so the target is
        just the immediate reward (gamma term is zeroed out).
        Epsilon is decayed after every update to shift from exploration to exploitation.
        """
        state_key = self._to_key(state)
        next_key = self._to_key(next_state)

        current_q = self.q_table[state_key][action]
        target = reward
        if not done:
            # Add discounted value of the best action in the next state.
            target += self.gamma * float(np.max(self.q_table[next_key]))

        # Move the Q-value a fraction (learning_rate) towards the target.
        self.q_table[state_key][action] = current_q + self.learning_rate * (target - current_q)

        # Decay exploration rate — never below epsilon_end.
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def save(self, path: str | Path) -> None:
        """Save the Q-table and current epsilon to a compressed .npz file."""
        checkpoint = Path(path)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)

        keys = list(self.q_table.keys())
        values = [self.q_table[key] for key in keys]
        # NumPy can't serialise Python tuples directly — dtype=object allows it.
        np.savez_compressed(
            checkpoint,
            keys=np.array(keys, dtype=object),
            values=np.array(values, dtype=object),
            epsilon=np.array([self.epsilon], dtype=np.float32),
        )

    def load(self, path: str | Path) -> None:
        """Restore a previously saved Q-table and epsilon from disk."""
        checkpoint = Path(path)
        if not checkpoint.exists():
            raise FileNotFoundError(f"Agent checkpoint not found: {checkpoint}")

        # allow_pickle=True is needed because keys/values were stored as dtype=object.
        data = np.load(checkpoint, allow_pickle=True)
        self.q_table = defaultdict(lambda: np.zeros(self.action_size, dtype=np.float32))
        keys = data["keys"]
        values = data["values"]
        for key, value in zip(keys, values):
            tuple_key = tuple(int(item) for item in key)
            self.q_table[tuple_key] = np.array(value, dtype=np.float32)
        self.epsilon = float(data["epsilon"][0])
