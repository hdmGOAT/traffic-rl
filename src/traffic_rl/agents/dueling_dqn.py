"""Dueling DQN agent.

Dueling DQN uses a split network architecture:
  - One stream estimates V(s): how good is this state in general?
  - Another stream estimates A(s,a): how much better is action a compared to average?
  - Combined: Q(s,a) = V(s) + A(s,a) - mean_a(A(s,a))

Subtracting the mean advantage keeps the decomposition identifiable
(prevents the network from finding degenerate V/A splits that cancel out).

The main benefit: the value stream can learn which states are good/bad without
needing to evaluate every action, which helps in states where action choice
doesn't matter much (e.g. a clear intersection). This leads to better policy
evaluation and faster convergence on many RL benchmarks.
"""
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
    """Fixed-capacity circular buffer for experience replay."""

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
    """Q-network with a shared trunk that splits into value and advantage streams.

    Architecture:
        input → shared hidden layer (ReLU)
                    ↓               ↓
            value stream        advantage stream
            (hidden → 1)        (hidden → action_size)
                    ↓               ↓
                Q(s,a) = V(s) + A(s,a) - mean_a(A(s,:))
    """

    def __init__(self, state_size: int, hidden_dim: int, action_size: int, rng: np.random.Generator) -> None:
        scale_1 = np.sqrt(2.0 / max(1, state_size))
        scale_2 = np.sqrt(2.0 / max(1, hidden_dim))
        scale_3 = np.sqrt(2.0 / max(1, hidden_dim))

        # Shared trunk: all actions pass through this layer first.
        self.w1 = rng.normal(0.0, scale_1, size=(state_size, hidden_dim)).astype(np.float32)
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)

        # Value stream: single output — how good is this state regardless of action?
        self.w_value = rng.normal(0.0, scale_2, size=(hidden_dim, 1)).astype(np.float32)
        self.b_value = np.zeros(1, dtype=np.float32)

        # Advantage stream: one output per action — relative benefit of each action.
        self.w_adv = rng.normal(0.0, scale_3, size=(hidden_dim, action_size)).astype(np.float32)
        self.b_adv = np.zeros(action_size, dtype=np.float32)

    def forward(self, states: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Forward pass returning all intermediate values needed for backprop.

        Returns:
            z1:        Pre-ReLU shared hidden layer.
            a1:        Post-ReLU shared hidden layer.
            value:     State value estimates V(s), shape (batch, 1).
            advantage: Action advantage estimates A(s,a), shape (batch, action_size).
            q_values:  Combined Q-values Q(s,a), shape (batch, action_size).
        """
        z1 = states @ self.w1 + self.b1
        a1 = np.maximum(z1, 0.0)   # ReLU activation.

        value     = a1 @ self.w_value + self.b_value   # (batch, 1)
        advantage = a1 @ self.w_adv  + self.b_adv      # (batch, action_size)

        # Combine streams: subtract mean advantage to make V and A identifiable.
        mean_adv = np.mean(advantage, axis=1, keepdims=True)
        q_values = value + (advantage - mean_adv)

        return z1, a1, value, advantage, q_values

    def predict(self, states: np.ndarray) -> np.ndarray:
        """Forward pass returning only Q-values (used where intermediates aren't needed)."""
        _, _, _, _, q_values = self.forward(states)
        return q_values

    def copy_from(self, other: "_DuelingQNetwork") -> None:
        """Copy all weights from another network (used to sync target ← online)."""
        self.w1      = other.w1.copy()
        self.b1      = other.b1.copy()
        self.w_value = other.w_value.copy()
        self.b_value = other.b_value.copy()
        self.w_adv   = other.w_adv.copy()
        self.b_adv   = other.b_adv.copy()


class DuelingDQNAgent(RLAgent):
    """Dueling DQN: split value/advantage streams for better state-value estimation."""

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
        # Lazily initialised on first observation (state size unknown at construction).
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
        """Epsilon-greedy action selection using the dueling Q-network."""
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
        """Store experience, conditionally optimise, sync target network, decay epsilon."""
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
        """Dueling DQN gradient update.

        The loss is the same MSE as DQN, but backprop must flow through
        the split architecture. The gradient of Q w.r.t. the value stream
        is the sum over all actions; for the advantage stream it's the
        per-action gradient minus the mean (mirror of the forward-pass formula).
        """
        if len(self.replay_buffer) < self.batch_size:
            return

        batch = self.replay_buffer.sample(self.batch_size, self.rng)
        states      = np.vstack([t.state      for t in batch]).astype(np.float32)
        actions     = np.array( [t.action     for t in batch], dtype=np.int64)
        rewards     = np.array( [t.reward     for t in batch], dtype=np.float32)
        next_states = np.vstack([t.next_state for t in batch]).astype(np.float32)
        dones       = np.array( [t.done       for t in batch], dtype=np.float32)

        # Forward pass through online network — save intermediates for backprop.
        z1, a1, value, advantage, q_values = self.online_net.forward(states)

        # Use target network for stable next-state Q estimates.
        next_q_values = self.target_net.predict(next_states)
        max_next_q    = np.max(next_q_values, axis=1)
        targets       = rewards + (1.0 - dones) * self.gamma * max_next_q

        batch_indices = np.arange(self.batch_size)
        q_selected    = q_values[batch_indices, actions]

        error  = (q_selected - targets).astype(np.float32)
        grad_q = np.zeros_like(q_values, dtype=np.float32)
        grad_q[batch_indices, actions] = (2.0 / self.batch_size) * error

        # Backprop through the dueling combination: Q = V + A - mean(A)
        # ∂Q/∂V  = sum over all actions of ∂Q/∂Q_i (since V is shared).
        grad_value_scalar = np.sum(grad_q, axis=1, keepdims=True)  # (batch, 1)
        # ∂Q/∂A_i = ∂Q/∂Q_i − mean_j(∂Q/∂Q_j)  (chain rule through A - mean(A)).
        grad_adv = grad_q - np.mean(grad_q, axis=1, keepdims=True)  # (batch, action_size)

        # Backprop through value stream weights.
        grad_w_value = a1.T @ grad_value_scalar   # (hidden, 1)
        grad_b_value = np.sum(grad_value_scalar, axis=0)

        # Backprop through advantage stream weights.
        grad_w_adv = a1.T @ grad_adv              # (hidden, action_size)
        grad_b_adv = np.sum(grad_adv, axis=0)

        # Backprop through shared hidden layer (gradients from both streams combine here).
        grad_a1 = (grad_value_scalar @ self.online_net.w_value.T) + (grad_adv @ self.online_net.w_adv.T)
        grad_z1 = grad_a1 * (z1 > 0.0)   # ReLU gradient gate.
        grad_w1 = states.T @ grad_z1
        grad_b1 = np.sum(grad_z1, axis=0)

        # SGD update for all weight groups.
        self.online_net.w1      -= self.learning_rate * grad_w1
        self.online_net.b1      -= self.learning_rate * grad_b1
        self.online_net.w_value -= self.learning_rate * grad_w_value
        self.online_net.b_value -= self.learning_rate * grad_b_value
        self.online_net.w_adv   -= self.learning_rate * grad_w_adv
        self.online_net.b_adv   -= self.learning_rate * grad_b_adv

    def save(self, path: str | Path) -> None:
        if self.online_net is None or self.target_net is None:
            return
        checkpoint = Path(path)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            checkpoint,
            online_w1=self.online_net.w1,       online_b1=self.online_net.b1,
            online_w_value=self.online_net.w_value, online_b_value=self.online_net.b_value,
            online_w_adv=self.online_net.w_adv,  online_b_adv=self.online_net.b_adv,
            target_w1=self.target_net.w1,        target_b1=self.target_net.b1,
            target_w_value=self.target_net.w_value, target_b_value=self.target_net.b_value,
            target_w_adv=self.target_net.w_adv,  target_b_adv=self.target_net.b_adv,
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
        self.online_net.w1      = data["online_w1"].astype(np.float32)
        self.online_net.b1      = data["online_b1"].astype(np.float32)
        self.online_net.w_value = data["online_w_value"].astype(np.float32)
        self.online_net.b_value = data["online_b_value"].astype(np.float32)
        self.online_net.w_adv   = data["online_w_adv"].astype(np.float32)
        self.online_net.b_adv   = data["online_b_adv"].astype(np.float32)
        self.target_net.w1      = data["target_w1"].astype(np.float32)
        self.target_net.b1      = data["target_b1"].astype(np.float32)
        self.target_net.w_value = data["target_w_value"].astype(np.float32)
        self.target_net.b_value = data["target_b_value"].astype(np.float32)
        self.target_net.w_adv   = data["target_w_adv"].astype(np.float32)
        self.target_net.b_adv   = data["target_b_adv"].astype(np.float32)
        self.epsilon    = float(data["epsilon"][0])
        self.update_step = int(data["update_step"][0])
