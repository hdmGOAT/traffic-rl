from __future__ import annotations

from traffic_rl.agents.base import RLAgent
from traffic_rl.agents.dqn import DQNAgent
from traffic_rl.agents.tabular_q import TabularQAgent
from traffic_rl.config import AppConfig


def build_agent(cfg: AppConfig, action_size: int) -> RLAgent:
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
    raise ValueError(f"Unsupported agent_type '{cfg.training.agent_type}'.")
