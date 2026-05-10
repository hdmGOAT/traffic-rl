from __future__ import annotations

from traffic_rl.agents.base import RLAgent
from traffic_rl.agents.dqn import DQNAgent
from traffic_rl.agents.double_dqn import DoubleDQNAgent
from traffic_rl.agents.dueling_dqn import DuelingDQNAgent
from traffic_rl.agents.fixed_time import FixedTimeAgent
from traffic_rl.agents.tabular_q import TabularQAgent
from traffic_rl.config import AppConfig


def build_agent(cfg: AppConfig, action_size: int) -> RLAgent:
    """Instantiate the correct agent based on the config's agent_type setting.

    Supported types:
        'dqn'         — Deep Q-Network with replay buffer and target network.
        'double_dqn'  — DQN variant that reduces Q-value overestimation.
        'dueling_dqn' — DQN variant with split value/advantage network streams.
        'tabular_q'   — Classic Q-learning with a lookup table (no neural network).
        'q_learning'  — Alias for tabular_q.
        'fixed_time'  — Stateless baseline that cycles phases on a fixed timer.

    The fixed_time agent is special: it ignores all training hyperparameters
    and only needs action_size to cycle phases correctly.
    """
    agent_type = cfg.training.agent_type.lower()

    if agent_type == "dqn":
        return DQNAgent(
            action_size=action_size,
            gamma=cfg.training.gamma,
            learning_rate=cfg.training.learning_rate,
            epsilon_start=cfg.training.epsilon_start,
            epsilon_end=cfg.training.epsilon_end,
            epsilon_decay=cfg.training.epsilon_decay,
            hidden_dim=cfg.training.hidden_dim,
            batch_size=cfg.training.batch_size,
            replay_capacity=cfg.training.replay_capacity,
            learning_starts=cfg.training.learning_starts,
            target_update_interval=cfg.training.target_update_interval,
            train_frequency=cfg.training.train_frequency,
            seed=cfg.seed,
        )

    if agent_type == "double_dqn":
        return DoubleDQNAgent(
            action_size=action_size,
            gamma=cfg.training.gamma,
            learning_rate=cfg.training.learning_rate,
            epsilon_start=cfg.training.epsilon_start,
            epsilon_end=cfg.training.epsilon_end,
            epsilon_decay=cfg.training.epsilon_decay,
            hidden_dim=cfg.training.hidden_dim,
            batch_size=cfg.training.batch_size,
            replay_capacity=cfg.training.replay_capacity,
            learning_starts=cfg.training.learning_starts,
            target_update_interval=cfg.training.target_update_interval,
            train_frequency=cfg.training.train_frequency,
            seed=cfg.seed,
        )

    if agent_type == "dueling_dqn":
        return DuelingDQNAgent(
            action_size=action_size,
            gamma=cfg.training.gamma,
            learning_rate=cfg.training.learning_rate,
            epsilon_start=cfg.training.epsilon_start,
            epsilon_end=cfg.training.epsilon_end,
            epsilon_decay=cfg.training.epsilon_decay,
            hidden_dim=cfg.training.hidden_dim,
            batch_size=cfg.training.batch_size,
            replay_capacity=cfg.training.replay_capacity,
            learning_starts=cfg.training.learning_starts,
            target_update_interval=cfg.training.target_update_interval,
            train_frequency=cfg.training.train_frequency,
            seed=cfg.seed,
        )

    if agent_type in {"tabular_q", "q_learning"}:
        return TabularQAgent(
            action_size=action_size,
            gamma=cfg.training.gamma,
            learning_rate=cfg.training.learning_rate,
            epsilon_start=cfg.training.epsilon_start,
            epsilon_end=cfg.training.epsilon_end,
            epsilon_decay=cfg.training.epsilon_decay,
            seed=cfg.seed,
        )

    if agent_type == "fixed_time":
        # Cycle steps default = 6: at decision_interval=5s that's 30s per phase,
        # matching a common real-world pre-timed controller setting.
        return FixedTimeAgent(action_size=action_size)

    raise ValueError(f"Unsupported agent_type '{cfg.training.agent_type}'.")
