from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

from traffic_rl.agents.base import RLAgent


class TabularQAgent(RLAgent):
    def __init__(
        self,
        action_size: int,
        gamma: float,
        learning_rate: float,
        epsilon_start: float,
        epsilon_end: float,
        epsilon_decay: float,
        seed: int = 7,
    ) -> None:
        self.action_size = action_size
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.q_table: dict[tuple[int, ...], np.ndarray] = defaultdict(
            lambda: np.zeros(self.action_size, dtype=np.float32)
        )
        self.rng = np.random.default_rng(seed)

    def _to_key(self, state_vector: np.ndarray) -> tuple[int, ...]:
        bucketized = np.floor(state_vector).astype(int)
        return tuple(int(value) for value in bucketized)

    def act(self, state_vector: np.ndarray, train: bool = True) -> int:
        key = self._to_key(state_vector)
        if train and self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, self.action_size))
        return int(np.argmax(self.q_table[key]))

    def observe(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        state_key = self._to_key(state)
        next_key = self._to_key(next_state)

        current_q = self.q_table[state_key][action]
        target = reward
        if not done:
            target += self.gamma * float(np.max(self.q_table[next_key]))

        self.q_table[state_key][action] = current_q + self.learning_rate * (target - current_q)
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def save(self, path: str | Path) -> None:
        checkpoint = Path(path)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)

        keys = list(self.q_table.keys())
        values = [self.q_table[key] for key in keys]
        np.savez_compressed(
            checkpoint,
            keys=np.array(keys, dtype=object),
            values=np.array(values, dtype=object),
            epsilon=np.array([self.epsilon], dtype=np.float32),
        )

    def load(self, path: str | Path) -> None:
        checkpoint = Path(path)
        if not checkpoint.exists():
            raise FileNotFoundError(f"Agent checkpoint not found: {checkpoint}")

        data = np.load(checkpoint, allow_pickle=True)
        self.q_table = defaultdict(lambda: np.zeros(self.action_size, dtype=np.float32))
        keys = data["keys"]
        values = data["values"]
        for key, value in zip(keys, values):
            tuple_key = tuple(int(item) for item in key)
            self.q_table[tuple_key] = np.array(value, dtype=np.float32)
        self.epsilon = float(data["epsilon"][0])
