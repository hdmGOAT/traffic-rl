from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class EnvironmentConfig:
    backend: str = "mock"
    intersection_id: str = "intersection_0"
    num_lanes: int = 4
    num_phases: int = 2
    min_green_time: int = 5
    decision_interval: int = 5
    episode_horizon_seconds: int = 3600
    cityflow_config_path: str | None = None
    cityflow_thread_num: int = 1


@dataclass(slots=True)
class RewardConfig:
    type: str = "queue_length"


@dataclass(slots=True)
class TrainingConfig:
    episodes: int = 20
    max_steps: int = 120
    agent_type: str = "dqn"
    gamma: float = 0.95
    learning_rate: float = 0.1
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.995
    hidden_dim: int = 64
    batch_size: int = 32
    replay_capacity: int = 5000
    learning_starts: int = 200
    target_update_interval: int = 100
    train_frequency: int = 1


@dataclass(slots=True)
class AppConfig:
    seed: int = 7
    output_dir: str = "outputs"
    env: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


def _deep_get(dct: dict[str, Any], key: str, default: Any) -> Any:
    value = dct.get(key, default)
    return default if value is None else value


def load_config(config_path: str | Path) -> AppConfig:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    env_raw = _deep_get(raw, "env", {})
    reward_raw = _deep_get(raw, "reward", {})
    train_raw = _deep_get(raw, "training", {})

    cfg = AppConfig(
        seed=int(_deep_get(raw, "seed", 7)),
        output_dir=str(_deep_get(raw, "output_dir", "outputs")),
        env=EnvironmentConfig(
            backend=str(_deep_get(env_raw, "backend", "mock")),
            intersection_id=str(_deep_get(env_raw, "intersection_id", "intersection_0")),
            num_lanes=int(_deep_get(env_raw, "num_lanes", 4)),
            num_phases=int(_deep_get(env_raw, "num_phases", 2)),
            min_green_time=int(_deep_get(env_raw, "min_green_time", 5)),
            decision_interval=int(_deep_get(env_raw, "decision_interval", 5)),
            episode_horizon_seconds=int(_deep_get(env_raw, "episode_horizon_seconds", 3600)),
            cityflow_config_path=_deep_get(env_raw, "cityflow_config_path", None),
            cityflow_thread_num=int(_deep_get(env_raw, "cityflow_thread_num", 1)),
        ),
        reward=RewardConfig(type=str(_deep_get(reward_raw, "type", "queue_length"))),
        training=TrainingConfig(
            episodes=int(_deep_get(train_raw, "episodes", 20)),
            max_steps=int(_deep_get(train_raw, "max_steps", 120)),
            agent_type=str(_deep_get(train_raw, "agent_type", "dqn")),
            gamma=float(_deep_get(train_raw, "gamma", 0.95)),
            learning_rate=float(_deep_get(train_raw, "learning_rate", 0.1)),
            epsilon_start=float(_deep_get(train_raw, "epsilon_start", 1.0)),
            epsilon_end=float(_deep_get(train_raw, "epsilon_end", 0.05)),
            epsilon_decay=float(_deep_get(train_raw, "epsilon_decay", 0.995)),
            hidden_dim=int(_deep_get(train_raw, "hidden_dim", 64)),
            batch_size=int(_deep_get(train_raw, "batch_size", 32)),
            replay_capacity=int(_deep_get(train_raw, "replay_capacity", 5000)),
            learning_starts=int(_deep_get(train_raw, "learning_starts", 200)),
            target_update_interval=int(_deep_get(train_raw, "target_update_interval", 100)),
            train_frequency=int(_deep_get(train_raw, "train_frequency", 1)),
        ),
    )

    if cfg.env.cityflow_config_path:
        cityflow_path = Path(cfg.env.cityflow_config_path)
        if not cityflow_path.is_absolute():
            cfg.env.cityflow_config_path = str((path.parent / cityflow_path).resolve())

    return cfg
