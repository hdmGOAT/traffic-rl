from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class EnvironmentConfig:
    """All settings that control how the traffic simulation behaves."""

    # Which simulation backend to use: 'mock' (fast, no dependencies) or
    # 'cityflow' (realistic microsimulation, requires CityFlow to be installed).
    backend: str = "mock"

    # ID of the specific intersection the agent controls.
    # Must match an intersection ID in the CityFlow road network file.
    intersection_id: str = "intersection_0"

    # Number of incoming lanes the agent observes. Determines the observation vector size.
    num_lanes: int = 4

    # Number of traffic-light phases the agent can choose from.
    # Also determines the size of the action space.
    num_phases: int = 2

    # Minimum number of decision steps a phase must stay green before the agent
    # can switch to another. Prevents unrealistically rapid phase flickering.
    min_green_time: int = 5

    # How many simulation seconds pass between each agent decision.
    # Larger values = fewer decisions per episode, coarser control.
    decision_interval: int = 5

    # Total simulated seconds per episode (e.g. 3600 = one simulated hour).
    episode_horizon_seconds: int = 3600

    # Path to the CityFlow JSON engine config file.
    # Only required when backend='cityflow'; ignored for 'mock'.
    cityflow_config_path: str | None = None

    # Number of CPU threads CityFlow uses for simulation (higher = faster but more CPU).
    cityflow_thread_num: int = 1


@dataclass(slots=True)
class RewardConfig:
    """Settings that control how the training reward signal is computed."""

    # Currently only 'queue_length' is supported: reward = -(sum of waiting vehicles).
    type: str = "queue_length"


@dataclass(slots=True)
class TrainingConfig:
    """Hyperparameters for the RL training loop."""

    # Total number of training episodes to run.
    episodes: int = 20

    # Maximum decision steps per episode (safety cap — normally episode_horizon handles this).
    max_steps: int = 120

    # Which agent to train: 'dqn', 'double_dqn', 'dueling_dqn', 'tabular_q', 'fixed_time'.
    agent_type: str = "dqn"

    # Discount factor: how much future rewards are worth relative to immediate ones.
    # 0 = fully myopic (only cares about now), 1 = fully far-sighted (treats all future equally).
    gamma: float = 0.95

    # Learning rate for gradient descent weight updates.
    learning_rate: float = 0.1

    # Starting exploration rate. 1.0 = agent acts randomly at first.
    epsilon_start: float = 1.0

    # Minimum exploration rate. The agent always keeps at least this much randomness.
    epsilon_end: float = 0.05

    # How fast epsilon decays. At 0.995, epsilon halves roughly every 138 steps.
    epsilon_decay: float = 0.995

    # Number of neurons in the neural network's hidden layer.
    hidden_dim: int = 64

    # Number of experiences sampled from the replay buffer each training step.
    batch_size: int = 32

    # Maximum number of experiences stored in the replay buffer.
    # Older experiences are evicted when the buffer is full.
    replay_capacity: int = 5000

    # Steps to collect before training begins (fills the buffer with diverse experiences first).
    learning_starts: int = 200

    # How many steps between each target network sync (online → target copy).
    # Larger = more stable but slower to reflect learned improvements.
    target_update_interval: int = 100

    # Train every N steps. 1 = train after every single step.
    train_frequency: int = 1

    # If true, when epsilon reaches epsilon_end the trainer will freeze learning
    # for the remainder of the run so later episodes act as evaluation.
    freeze_on_epsilon_end: bool = True


@dataclass(slots=True)
class AppConfig:
    """Top-level config that groups all sub-configs together."""

    # Random seed used across the environment, agent, and numpy to ensure reproducibility.
    seed: int = 7

    # Directory where checkpoints, training summaries, and reports are written.
    output_dir: str = "outputs"

    env: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


def _deep_get(dct: dict[str, Any], key: str, default: Any) -> Any:
    """Read a key from a dict, treating explicit None the same as missing.

    PyYAML parses unset keys as None, so we must distinguish 'key is absent'
    from 'key is present but set to a non-None value'.
    """
    value = dct.get(key, default)
    return default if value is None else value


def load_config(config_path: str | Path) -> AppConfig:
    """Read a YAML config file from disk and return a fully populated AppConfig.

    Any key not present in the YAML falls back to the dataclass default.
    If cityflow_config_path is a relative path, it is resolved relative to
    the directory containing the YAML file so configs stay portable.
    """
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    # Pull each top-level section, defaulting to an empty dict if absent.
    env_raw   = _deep_get(raw, "env",      {})
    reward_raw= _deep_get(raw, "reward",   {})
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
            freeze_on_epsilon_end=bool(_deep_get(train_raw, "freeze_on_epsilon_end", True)),
        ),
    )

    # Resolve relative cityflow_config_path so it works regardless of cwd.
    if cfg.env.cityflow_config_path:
        cityflow_path = Path(cfg.env.cityflow_config_path)
        if not cityflow_path.is_absolute():
            cfg.env.cityflow_config_path = str((path.parent / cityflow_path).resolve())

    return cfg
