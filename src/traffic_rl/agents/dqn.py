from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from traffic_rl.agents.base import RLAgent


# A single experience tuple: the agent's "memory" of one decision step.
# Stored in the replay buffer and replayed later during training.
@dataclass(slots=True)
class _Transition:
    state: np.ndarray       # The intersection state before the action.
    action: int             # The phase index that was chosen.
    reward: float           # The reward received for that action.
    next_state: np.ndarray  # The intersection state after the action.
    done: bool              # Whether this was the last step of the episode.


class _ReplayBuffer:
    """Fixed-size circular buffer that stores past experiences for random sampling.

    Random sampling breaks the temporal correlation between consecutive steps,
    which would otherwise cause the neural network to overfit to recent patterns
    and become unstable. This is one of the two key ideas that make DQN work.
    """

    def __init__(self, capacity: int) -> None:
        # deque with maxlen automatically drops the oldest entry when full.
        self._buffer: deque[_Transition] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self._buffer)

    def add(self, transition: _Transition) -> None:
        """Store a new experience, evicting the oldest one if at capacity."""
        self._buffer.append(transition)

    def sample(self, batch_size: int, rng: np.random.Generator) -> list[_Transition]:
        """Randomly draw a mini-batch of past experiences without replacement."""
        indices = rng.choice(len(self._buffer), size=batch_size, replace=False)
        return [self._buffer[int(index)] for index in indices]


class _QNetwork:
    """Two-layer fully-connected neural network implemented from scratch in NumPy.

    Architecture: input → hidden (ReLU) → output (Q-values, one per action).
    No deep-learning framework is used — weights are updated manually via backprop.
    """

    def __init__(self, state_size: int, hidden_dim: int, action_size: int, rng: np.random.Generator) -> None:
        # He initialisation: scale weights by sqrt(2 / fan_in) to prevent gradients
        # vanishing or exploding through the ReLU activation layer.
        scale_1 = np.sqrt(2.0 / max(1, state_size))
        scale_2 = np.sqrt(2.0 / max(1, hidden_dim))
        # Layer 1 weights and biases: maps state → hidden representation.
        self.w1 = rng.normal(0.0, scale_1, size=(state_size, hidden_dim)).astype(np.float32)
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)
        # Layer 2 weights and biases: maps hidden representation → Q-values per action.
        self.w2 = rng.normal(0.0, scale_2, size=(hidden_dim, action_size)).astype(np.float32)
        self.b2 = np.zeros(action_size, dtype=np.float32)

    def forward(self, states: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run a forward pass and return intermediate activations for backprop.

        Returns:
            z1: Pre-activation hidden layer values (needed for ReLU gradient).
            a1: Post-activation hidden layer values (needed for weight gradients).
            q_values: Final Q-value estimates, shape (batch_size, action_size).
        """
        z1 = states @ self.w1 + self.b1       # Linear transform: input → hidden pre-activation.
        a1 = np.maximum(z1, 0.0)              # ReLU: zero out negative activations.
        q_values = a1 @ self.w2 + self.b2     # Linear transform: hidden → Q-values.
        return z1, a1, q_values

    def predict(self, states: np.ndarray) -> np.ndarray:
        """Run a forward pass and return only Q-values (no intermediate values needed)."""
        _, _, q_values = self.forward(states)
        return q_values

    def copy_from(self, other: "_QNetwork") -> None:
        """Copy all weights from another network into this one (used for target network sync)."""
        self.w1 = other.w1.copy()
        self.b1 = other.b1.copy()
        self.w2 = other.w2.copy()
        self.b2 = other.b2.copy()


class DQNAgent(RLAgent):
    """Deep Q-Network agent.

    Uses two neural networks (online and target) plus a replay buffer to learn
    a Q-function mapping (state, action) → expected cumulative reward.
    The online network is trained every few steps; the target network is a
    lagged copy updated periodically to stabilise training.
    """

    def __init__(
        self,
        action_size: int,
        gamma: float,                  # Discount factor for future rewards (0=myopic, 1=far-sighted).
        learning_rate: float,          # Gradient descent step size for weight updates.
        epsilon_start: float,          # Starting exploration rate (1.0 = fully random).
        epsilon_end: float,            # Minimum exploration rate the agent decays down to.
        epsilon_decay: float,          # Multiplicative factor applied to epsilon each step.
        hidden_dim: int,               # Number of neurons in the single hidden layer.
        batch_size: int,               # How many experiences to sample per training update.
        replay_capacity: int,          # Maximum number of experiences stored in the buffer.
        learning_starts: int,          # Steps to wait before starting training (fill the buffer first).
        target_update_interval: int,   # How many steps between target network copies.
        train_frequency: int,          # Train every N steps (1 = every step).
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
        # Networks are created lazily on the first call to act() because the
        # state size isn't known until we see the first observation.
        self.online_net: _QNetwork | None = None
        self.target_net: _QNetwork | None = None
        self.update_step = 0  # Total observe() calls — drives training and target sync schedules.

    def _ensure_initialized(self, state_vector: np.ndarray) -> None:
        """Lazily create both networks once we know the state vector size."""
        if self.online_net is not None and self.target_net is not None:
            return
        state_size = int(state_vector.shape[0])
        self.online_net = _QNetwork(state_size, self.hidden_dim, self.action_size, self.rng)
        self.target_net = _QNetwork(state_size, self.hidden_dim, self.action_size, self.rng)
        # Target starts as an exact copy of online so they begin identical.
        self.target_net.copy_from(self.online_net)

    def act(self, state_vector: np.ndarray, train: bool = True) -> int:
        """Choose a traffic-light phase using epsilon-greedy exploration.

        During training: explore randomly with probability epsilon, otherwise
        query the online network for the best action.
        During evaluation (train=False): always use the network (no randomness).
        """
        self._ensure_initialized(state_vector)

        if train and self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, self.action_size))

        # Reshape to (1, state_size) so the network processes it as a 1-item batch.
        state_batch = state_vector.astype(np.float32).reshape(1, -1)
        q_values = self.online_net.predict(state_batch)
        # Pick the action (phase) with the highest predicted Q-value.
        return int(np.argmax(q_values[0]))

    def observe(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Store the experience, trigger training if conditions are met, and decay epsilon."""
        self._ensure_initialized(state)

        # Pack the experience and push it into the replay buffer.
        transition = _Transition(
            state=state.astype(np.float32).copy(),
            action=int(action),
            reward=float(reward),
            next_state=next_state.astype(np.float32).copy(),
            done=bool(done),
        )
        self.replay_buffer.add(transition)
        self.update_step += 1

        # Start training only after the buffer has collected enough diverse experiences.
        # Train every train_frequency steps to balance compute vs learning speed.
        if len(self.replay_buffer) >= self.learning_starts and self.update_step % self.train_frequency == 0:
            self._optimize()

        # Periodically copy online → target to give the target a stable learning signal.
        if self.update_step % self.target_update_interval == 0:
            self.target_net.copy_from(self.online_net)

        # Reduce exploration over time: the agent gradually shifts from random to learned behaviour.
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def _optimize(self) -> None:
        """Run one gradient descent step on the online network using a random mini-batch.

        Implements the DQN loss:
            L = mean( (Q_online(s,a) - target)^2 )
        where:
            target = r  +  gamma * max_a( Q_target(s') )   if not done
            target = r                                       if done

        Gradients are computed by hand (no autograd) and applied with vanilla SGD.
        """
        if len(self.replay_buffer) < self.batch_size:
            return

        batch = self.replay_buffer.sample(self.batch_size, self.rng)

        # Stack individual experience fields into batch arrays for vectorised operations.
        states      = np.vstack([t.state      for t in batch]).astype(np.float32)
        actions     = np.array( [t.action     for t in batch], dtype=np.int64)
        rewards     = np.array( [t.reward     for t in batch], dtype=np.float32)
        next_states = np.vstack([t.next_state for t in batch]).astype(np.float32)
        dones       = np.array( [t.done       for t in batch], dtype=np.float32)

        # Forward pass through online network — save intermediates for backprop.
        z1, a1, q_values = self.online_net.forward(states)

        # Use target network for next-state estimates (stabilises training).
        next_q_values = self.target_net.predict(next_states)

        # Bellman target: immediate reward + discounted best future value.
        # Multiplying by (1 - done) zeros out the future term on terminal steps.
        max_next_q = np.max(next_q_values, axis=1)
        targets = rewards + (1.0 - dones) * self.gamma * max_next_q

        batch_indices = np.arange(self.batch_size)
        # Extract the Q-value the network predicted for the action that was actually taken.
        q_selected = q_values[batch_indices, actions]

        # Error = predicted Q − target Q. MSE loss → gradient = 2/N * error.
        error = (q_selected - targets).astype(np.float32)
        # Build a gradient matrix shaped like q_values, non-zero only at the taken action.
        grad_q = np.zeros_like(q_values, dtype=np.float32)
        grad_q[batch_indices, actions] = (2.0 / self.batch_size) * error

        # Backprop through layer 2 (hidden → output).
        grad_w2 = a1.T @ grad_q
        grad_b2 = np.sum(grad_q, axis=0)

        # Backprop through ReLU: gradient is zero wherever the pre-activation was negative.
        grad_a1 = grad_q @ self.online_net.w2.T
        grad_z1 = grad_a1 * (z1 > 0.0)

        # Backprop through layer 1 (input → hidden).
        grad_w1 = states.T @ grad_z1
        grad_b1 = np.sum(grad_z1, axis=0)

        # Vanilla SGD weight update: move weights in the opposite direction of the gradient.
        self.online_net.w1 -= self.learning_rate * grad_w1
        self.online_net.b1 -= self.learning_rate * grad_b1
        self.online_net.w2 -= self.learning_rate * grad_w2
        self.online_net.b2 -= self.learning_rate * grad_b2

    def save(self, path: str | Path) -> None:
        """Write all network weights and training state to a compressed .npz file."""
        if self.online_net is None or self.target_net is None:
            return  # Nothing learned yet — skip silently.

        checkpoint = Path(path)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            checkpoint,
            online_w1=self.online_net.w1,
            online_b1=self.online_net.b1,
            online_w2=self.online_net.w2,
            online_b2=self.online_net.b2,
            target_w1=self.target_net.w1,
            target_b1=self.target_net.b1,
            target_w2=self.target_net.w2,
            target_b2=self.target_net.b2,
            # Also save epsilon and update_step so training can resume mid-run.
            epsilon=np.array([self.epsilon], dtype=np.float32),
            update_step=np.array([self.update_step], dtype=np.int64),
        )

    def load(self, path: str | Path) -> None:
        """Restore weights and training state from a previously saved checkpoint."""
        checkpoint = Path(path)
        if not checkpoint.exists():
            raise FileNotFoundError(f"Agent checkpoint not found: {checkpoint}")

        data = np.load(checkpoint)
        # Infer state_size from the saved weight shape so we don't need to pass it explicitly.
        state_size = int(data["online_w1"].shape[0])
        self._ensure_initialized(np.zeros(state_size, dtype=np.float32))

        self.online_net.w1 = data["online_w1"].astype(np.float32)
        self.online_net.b1 = data["online_b1"].astype(np.float32)
        self.online_net.w2 = data["online_w2"].astype(np.float32)
        self.online_net.b2 = data["online_b2"].astype(np.float32)
        self.target_net.w1 = data["target_w1"].astype(np.float32)
        self.target_net.b1 = data["target_b1"].astype(np.float32)
        self.target_net.w2 = data["target_w2"].astype(np.float32)
        self.target_net.b2 = data["target_b2"].astype(np.float32)
        self.epsilon = float(data["epsilon"][0])
        self.update_step = int(data["update_step"][0])
