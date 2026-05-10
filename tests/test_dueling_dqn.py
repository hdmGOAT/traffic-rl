"""Unit tests for Dueling DQN agent."""
import numpy as np

from traffic_rl.agents.dueling_dqn import DuelingDQNAgent


def test_dueling_dqn_initialization():
    """Test agent initialization."""
    agent = DuelingDQNAgent(
        action_size=4,
        gamma=0.99,
        learning_rate=0.001,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        hidden_dim=64,
        batch_size=32,
        replay_capacity=10000,
        learning_starts=1000,
        target_update_interval=1000,
        train_frequency=4,
        seed=42,
    )
    assert agent.action_size == 4
    assert agent.epsilon == 1.0


def test_dueling_dqn_act():
    """Test act method."""
    agent = DuelingDQNAgent(
        action_size=4,
        gamma=0.99,
        learning_rate=0.001,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        hidden_dim=64,
        batch_size=32,
        replay_capacity=10000,
        learning_starts=1000,
        target_update_interval=1000,
        train_frequency=4,
        seed=42,
    )
    state = np.random.randn(10).astype(np.float32)
    action = agent.act(state, train=True)
    assert isinstance(action, (int, np.integer))
    assert 0 <= action < 4


def test_dueling_dqn_observe():
    """Test observe method and learning."""
    agent = DuelingDQNAgent(
        action_size=4,
        gamma=0.99,
        learning_rate=0.001,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        hidden_dim=64,
        batch_size=32,
        replay_capacity=10000,
        learning_starts=100,
        target_update_interval=50,
        train_frequency=1,
        seed=42,
    )

    # Collect some transitions
    for _ in range(150):
        state = np.random.randn(10).astype(np.float32)
        action = agent.act(state, train=True)
        next_state = np.random.randn(10).astype(np.float32)
        reward = np.random.randn()
        done = False
        agent.observe(state, action, reward, next_state, done)

    # Check epsilon decayed
    assert agent.epsilon < 1.0
    assert agent.update_step > 100


def test_dueling_dqn_save_load(tmp_path):
    """Test save and load functionality."""
    agent1 = DuelingDQNAgent(
        action_size=4,
        gamma=0.99,
        learning_rate=0.001,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        hidden_dim=64,
        batch_size=32,
        replay_capacity=10000,
        learning_starts=100,
        target_update_interval=50,
        train_frequency=1,
        seed=42,
    )

    # Collect some transitions to modify weights
    for _ in range(120):
        state = np.random.randn(10).astype(np.float32)
        action = agent1.act(state, train=True)
        next_state = np.random.randn(10).astype(np.float32)
        reward = np.random.randn()
        done = False
        agent1.observe(state, action, reward, next_state, done)

    # Save
    checkpoint_path = tmp_path / "agent.npz"
    agent1.save(checkpoint_path)

    # Load into new agent
    agent2 = DuelingDQNAgent(
        action_size=4,
        gamma=0.99,
        learning_rate=0.001,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        hidden_dim=64,
        batch_size=32,
        replay_capacity=10000,
        learning_starts=100,
        target_update_interval=50,
        train_frequency=1,
        seed=42,
    )
    agent2.load(checkpoint_path)

    # Verify same weights
    assert np.allclose(agent1.online_net.w1, agent2.online_net.w1)
    assert np.allclose(agent1.online_net.w_value, agent2.online_net.w_value)
    assert np.allclose(agent1.online_net.w_adv, agent2.online_net.w_adv)
    assert np.isclose(agent1.epsilon, agent2.epsilon, rtol=1e-5)
