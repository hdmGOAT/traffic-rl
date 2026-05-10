from pathlib import Path

import numpy as np
import pytest

from traffic_rl.agents.dqn import DQNAgent
from traffic_rl.agents.fixed_time import FixedTimeAgent
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


# ---------------------------------------------------------------------------
# FixedTimeAgent tests
# ---------------------------------------------------------------------------

def test_fixed_time_cycles_phases() -> None:
    agent = FixedTimeAgent(action_size=4, cycle_steps=3)
    # Steps 0,1,2 → phase 0; steps 3,4,5 → phase 1; etc.
    expected = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 0, 0, 0]
    state = np.zeros(4, dtype=np.float32)
    actions = [agent.act(state) for _ in expected]
    assert actions == expected


def test_fixed_time_single_phase() -> None:
    agent = FixedTimeAgent(action_size=1, cycle_steps=5)
    state = np.zeros(4, dtype=np.float32)
    for _ in range(20):
        assert agent.act(state) == 0


def test_fixed_time_observe_is_noop() -> None:
    agent = FixedTimeAgent(action_size=2, cycle_steps=6)
    state = np.zeros(4, dtype=np.float32)
    # observe must not raise and must not change phase counter
    agent.observe(state, 0, -1.0, state, False)
    assert agent._step == 0


def test_fixed_time_save_load_are_noops(tmp_path: Path) -> None:
    agent = FixedTimeAgent(action_size=2, cycle_steps=6)
    path = tmp_path / "fixed_time.npz"
    agent.save(path)   # must not raise
    agent.load(path)   # must not raise


def test_fixed_time_rejects_invalid_args() -> None:
    with pytest.raises(ValueError, match="action_size"):
        FixedTimeAgent(action_size=0)
    with pytest.raises(ValueError, match="cycle_steps"):
        FixedTimeAgent(action_size=2, cycle_steps=0)


def test_factory_builds_fixed_time() -> None:
    cfg = load_config(Path("configs/default.yaml"))
    cfg.training.agent_type = "fixed_time"
    agent = build_agent(cfg, action_size=8)
    assert isinstance(agent, FixedTimeAgent)
    assert agent.action_size == 8
