"""Dueling DQN agent: separates value and advantage streams in the network architecture."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from traffic_rl.agents.base import RLAgent


@dataclass(slots=True)
class _Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class _ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self._buffer: deque[_Transition] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self._buffer)

    def add(self, transition: _Transition) -> None:
        self._buffer.append(transition)

    def sample(self, batch_size: int, rng: np.random.Generator) -> list[_Transition]:
        indices = rng.choice(len(self._buffer), size=batch_size, replace=False)
        return [self._buffer[int(index)] for index in indices]


class _DuelingQNetwork:
    """Q-network with separate value and advantage streams."""

    def __init__(self, state_size: int, hidden_dim: int, action_size: int, rng: np.random.Generator) -> None:
        scale_1 = np.sqrt(2.0 / max(1, state_size))
        scale_2 = np.sqrt(2.0 / max(1, hidden_dim))
        scale_3 = np.sqrt(2.0 / max(1, hidden_dim))
        
        # Shared hidden layer
        self.w1 = rng.normal(0.0, scale_1, size=(state_size, hidden_dim)).astype(np.float32)
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)
        
        # Value stream
        self.w_value = rng.normal(0.0, scale_2, size=(hidden_dim, 1)).astype(np.float32)
        self.b_value = np.zeros(1, dtype=np.float32)
        
        # Advantage stream
        self.w_adv = rng.normal(0.0, scale_3, size=(hidden_dim, action_size)).astype(np.float32)
        self.b_adv = np.zeros(action_size, dtype=np.float32)

    def forward(self, states: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        z1 = states @ self.w1 + self.b1
        a1 = np.maximum(z1, 0.0)
        
        value = a1 @ self.w_value + self.b_value
        advantage = a1 @ self.w_adv + self.b_adv
        
        # Combine: Q(s,a) = V(s) + A(s,a) - mean(A(s,:))
        mean_adv = np.mean(advantage, axis=1, keepdims=True)
        q_values = value + (advantage - mean_adv)
        
        return z1, a1, value, advantage, q_values

    def predict(self, states: np.ndarray) -> np.ndarray:
        _, _, _, _, q_values = self.forward(states)
        return q_values

    def copy_from(self, other: "_DuelingQNetwork") -> None:
        self.w1 = other.w1.copy()
        self.b1 = other.b1.copy()
        self.w_value = other.w_value.copy()
        self.b_value = other.b_value.copy()
        self.w_adv = other.w_adv.copy()
        self.b_adv = other.b_adv.copy()


class DuelingDQNAgent(RLAgent):
    """Dueling DQN: separates value and advantage streams for better learning."""

    def __init__(
        self,
        action_size: int,
        gamma: float,
        learning_rate: float,
        epsilon_start: float,
        epsilon_end: float,
        epsilon_decay: float,
        hidden_dim: int,
        batch_size: int,
        replay_capacity: int,
        learning_starts: int,
        target_update_interval: int,
        train_frequency: int,
        seed: int = 7,
    ) -> None:
        self.action_size = action_size
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.learning_starts = learning_starts
        self.target_update_interval = max(1, target_update_interval)
        self.train_frequency = max(1, train_frequency)

        self.rng = np.random.default_rng(seed)
        self.replay_buffer = _ReplayBuffer(replay_capacity)
        self.online_net: _DuelingQNetwork | None = None
        self.target_net: _DuelingQNetwork | None = None
        self.update_step = 0

    def _ensure_initialized(self, state_vector: np.ndarray) -> None:
        if self.online_net is not None and self.target_net is not None:
            return
        state_size = int(state_vector.shape[0])
        self.online_net = _DuelingQNetwork(state_size, self.hidden_dim, self.action_size, self.rng)
        self.target_net = _DuelingQNetwork(state_size, self.hidden_dim, self.action_size, self.rng)
        self.target_net.copy_from(self.online_net)

    def act(self, state_vector: np.ndarray, train: bool = True) -> int:
        self._ensure_initialized(state_vector)

        if train and self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, self.action_size))

        state_batch = state_vector.astype(np.float32).reshape(1, -1)
        q_values = self.online_net.predict(state_batch)
        return int(np.argmax(q_values[0]))

    def observe(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self._ensure_initialized(state)

        transition = _Transition(
            state=state.astype(np.float32).copy(),
            action=int(action),
            reward=float(reward),
            next_state=next_state.astype(np.float32).copy(),
            done=bool(done),
        )
        self.replay_buffer.add(transition)
        self.update_step += 1

        if len(self.replay_buffer) >= self.learning_starts and self.update_step % self.train_frequency == 0:
            self._optimize()

        if self.update_step % self.target_update_interval == 0:
            self.target_net.copy_from(self.online_net)

        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def _optimize(self) -> None:
        if len(self.replay_buffer) < self.batch_size:
            return

        batch = self.replay_buffer.sample(self.batch_size, self.rng)
        states = np.vstack([transition.state for transition in batch]).astype(np.float32)
        actions = np.array([transition.action for transition in batch], dtype=np.int64)
        rewards = np.array([transition.reward for transition in batch], dtype=np.float32)
        next_states = np.vstack([transition.next_state for transition in batch]).astype(np.float32)
        dones = np.array([transition.done for transition in batch], dtype=np.float32)

        z1, a1, value, advantage, q_values = self.online_net.forward(states)
        next_q_values = self.target_net.predict(next_states)

        max_next_q = np.max(next_q_values, axis=1)
        targets = rewards + (1.0 - dones) * self.gamma * max_next_q

        batch_indices = np.arange(self.batch_size)
        q_selected = q_values[batch_indices, actions]

        error = (q_selected - targets).astype(np.float32)
        grad_q = np.zeros_like(q_values, dtype=np.float32)
        grad_q[batch_indices, actions] = (2.0 / self.batch_size) * error

        # Backprop through dueling architecture
        # Q(s,a) = V(s) + A(s,a) - mean(A(s,:))
        # grad_V = sum_a(grad_Q[:, a])
        # grad_A[:, a] = grad_Q[:, a] - mean_a(grad_Q[:, a])
        num_actions = self.action_size
        grad_value_scalar = np.sum(grad_q, axis=1, keepdims=True)  # (batch_size, 1)
        grad_adv = grad_q - np.mean(grad_q, axis=1, keepdims=True)  # (batch_size, num_actions)

        grad_w_value = a1.T @ grad_value_scalar  # (hidden_dim, 1)
        grad_b_value = np.sum(grad_value_scalar, axis=0)  # (1,)

        grad_w_adv = a1.T @ grad_adv  # (hidden_dim, num_actions)
        grad_b_adv = np.sum(grad_adv, axis=0)  # (num_actions,)

        # Gradient w.r.t. hidden layer
        grad_a1 = (grad_value_scalar @ self.online_net.w_value.T) + (grad_adv @ self.online_net.w_adv.T)
        grad_z1 = grad_a1 * (z1 > 0.0)

        grad_w1 = states.T @ grad_z1
        grad_b1 = np.sum(grad_z1, axis=0)

        self.online_net.w1 -= self.learning_rate * grad_w1
        self.online_net.b1 -= self.learning_rate * grad_b1
        self.online_net.w_value -= self.learning_rate * grad_w_value
        self.online_net.b_value -= self.learning_rate * grad_b_value
        self.online_net.w_adv -= self.learning_rate * grad_w_adv
        self.online_net.b_adv -= self.learning_rate * grad_b_adv

    def save(self, path: str | Path) -> None:
        if self.online_net is None or self.target_net is None:
            return

        checkpoint = Path(path)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            checkpoint,
            online_w1=self.online_net.w1,
            online_b1=self.online_net.b1,
            online_w_value=self.online_net.w_value,
            online_b_value=self.online_net.b_value,
            online_w_adv=self.online_net.w_adv,
            online_b_adv=self.online_net.b_adv,
            target_w1=self.target_net.w1,
            target_b1=self.target_net.b1,
            target_w_value=self.target_net.w_value,
            target_b_value=self.target_net.b_value,
            target_w_adv=self.target_net.w_adv,
            target_b_adv=self.target_net.b_adv,
            epsilon=np.array([self.epsilon], dtype=np.float32),
            update_step=np.array([self.update_step], dtype=np.int64),
        )

    def load(self, path: str | Path) -> None:
        checkpoint = Path(path)
        if not checkpoint.exists():
            raise FileNotFoundError(f"Agent checkpoint not found: {checkpoint}")

        data = np.load(checkpoint)
        state_size = int(data["online_w1"].shape[0])
        self._ensure_initialized(np.zeros(state_size, dtype=np.float32))

        self.online_net.w1 = data["online_w1"].astype(np.float32)
        self.online_net.b1 = data["online_b1"].astype(np.float32)
        self.online_net.w_value = data["online_w_value"].astype(np.float32)
        self.online_net.b_value = data["online_b_value"].astype(np.float32)
        self.online_net.w_adv = data["online_w_adv"].astype(np.float32)
        self.online_net.b_adv = data["online_b_adv"].astype(np.float32)
        self.target_net.w1 = data["target_w1"].astype(np.float32)
        self.target_net.b1 = data["target_b1"].astype(np.float32)
        self.target_net.w_value = data["target_w_value"].astype(np.float32)
        self.target_net.b_value = data["target_b_value"].astype(np.float32)
        self.target_net.w_adv = data["target_w_adv"].astype(np.float32)
        self.target_net.b_adv = data["target_b_adv"].astype(np.float32)
        self.epsilon = float(data["epsilon"][0])
        self.update_step = int(data["update_step"][0])
