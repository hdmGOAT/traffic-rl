from pathlib import Path

import numpy as np

from traffic_rl.agents.dqn import DQNAgent
from traffic_rl.agents.factory import build_agent
from traffic_rl.config import load_config


def test_factory_builds_dqn() -> None:
    cfg = load_config(Path("configs/default.yaml"))
    cfg.training.agent_type = "dqn"
    agent = build_agent(cfg, action_size=2)
    assert isinstance(agent, DQNAgent)


def test_dqn_observe_updates_epsilon() -> None:
    agent = DQNAgent(
        action_size=2,
        gamma=0.95,
        learning_rate=0.001,
        epsilon_start=1.0,
        epsilon_end=0.1,
        epsilon_decay=0.9,
        hidden_dim=16,
        batch_size=4,
        replay_capacity=100,
        learning_starts=4,
        target_update_interval=2,
        train_frequency=1,
        seed=7,
    )

    state = np.array([1.0, 0.0, 2.0, 1.0], dtype=np.float32)
    next_state = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)

    initial_epsilon = agent.epsilon
    for _ in range(6):
        action = agent.act(state, train=True)
        agent.observe(state, action, -1.0, next_state, False)

    assert agent.epsilon < initial_epsilon


def test_dqn_save_and_load_roundtrip(tmp_path: Path) -> None:
    agent = DQNAgent(
        action_size=2,
        gamma=0.95,
        learning_rate=0.001,
        epsilon_start=1.0,
        epsilon_end=0.1,
        epsilon_decay=0.9,
        hidden_dim=8,
        batch_size=4,
        replay_capacity=64,
        learning_starts=4,
        target_update_interval=2,
        train_frequency=1,
        seed=7,
    )

    state = np.array([1.0, 0.0, 2.0, 1.0], dtype=np.float32)
    next_state = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    for _ in range(6):
        action = agent.act(state, train=True)
        agent.observe(state, action, -1.0, next_state, False)

    checkpoint = tmp_path / "agent_checkpoint.npz"
    agent.save(checkpoint)

    loaded = DQNAgent(
        action_size=2,
        gamma=0.95,
        learning_rate=0.001,
        epsilon_start=1.0,
        epsilon_end=0.1,
        epsilon_decay=0.9,
        hidden_dim=8,
        batch_size=4,
        replay_capacity=64,
        learning_starts=4,
        target_update_interval=2,
        train_frequency=1,
        seed=99,
    )
    loaded.load(checkpoint)

    test_state = np.array([2.0, 1.0, 0.0, 3.0], dtype=np.float32)
    assert loaded.act(test_state, train=False) == agent.act(test_state, train=False)
