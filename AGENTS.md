# RL Agent Implementations & Comparison

This directory contains four reinforcement learning agents for traffic signal control. You can easily compare their performance to determine which works best for your use case.

## Agents Implemented

### 1. **Tabular Q-Learning** (`tabular_q`)
- **Type**: Value-based, Tabular
- **Pros**: Simplest, fastest, guaranteed convergence for small domains
- **Cons**: Cannot handle large state spaces
- **Best for**: Baseline comparison, small discrete problems
- **File**: [src/traffic_rl/agents/tabular_q.py](src/traffic_rl/agents/tabular_q.py)

### 2. **DQN** (`dqn`)
- **Type**: Value-based, Deep Neural Network
- **Key Innovation**: Replay buffer + Target network to stabilize learning
- **Pros**: Handles large state spaces, relatively stable training
- **Cons**: Suffers from Q-value overestimation
- **Best for**: Standard deep RL baseline
- **File**: [src/traffic_rl/agents/dqn.py](src/traffic_rl/agents/dqn.py)

### 3. **Double DQN** (`double_dqn`)
- **Type**: Value-based, Deep Neural Network + Double Q-learning
- **Key Innovation**: Uses online network to select actions, target network to evaluate them
- **Reduces**: Q-value overestimation bias (major DQN limitation)
- **Pros**: More stable and reliable than DQN
- **Cons**: Slightly more computation than DQN
- **Best for**: When you need better learning stability than standard DQN
- **File**: [src/traffic_rl/agents/double_dqn.py](src/traffic_rl/agents/double_dqn.py)
- **Research**: [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461)

### 4. **Dueling DQN** (`dueling_dqn`)
- **Type**: Value-based, Deep Neural Network + Dueling architecture
- **Key Innovation**: Splits network into separate Value V(s) and Advantage A(s,a) streams
- **Insight**: V(s) learns state value independent of actions, A(s,a) learns relative action advantages
- **Formula**: Q(s,a) = V(s) + (A(s,a) - mean(A(s,:)))
- **Pros**: Better feature learning, faster convergence in some domains
- **Cons**: More complex architecture
- **Best for**: When state value and action advantages are clearly separable
- **File**: [src/traffic_rl/agents/dueling_dqn.py](src/traffic_rl/agents/dueling_dqn.py)
- **Research**: [Dueling Network Architectures for Deep Reinforcement Learning](https://arxiv.org/abs/1511.06581)

## Usage

### Train a Single Agent

```python
from traffic_rl.agents.factory import build_agent
from traffic_rl.config import AppConfig

# Create config
cfg = AppConfig()
cfg.training.agent_type = "double_dqn"  # or "dqn", "dueling_dqn", "tabular_q"
cfg.training.learning_rate = 0.001
cfg.training.gamma = 0.99

# Build agent
agent = build_agent(cfg, action_size=4)

# Interact with environment
for episode in range(num_episodes):
    state = env.reset()
    for step in range(max_steps):
        action = agent.act(state, train=True)
        next_state, reward, done, _ = env.step(action)
        agent.observe(state, action, reward, next_state, done)
        state = next_state
        if done:
            break
    
    # Checkpoint
    agent.save(f"checkpoints/agent_ep_{episode}.npz")
```

### Compare All Four Agents

Run the comparison benchmark to see which agent performs best:

```bash
python -m traffic_rl.cli.compare_agents
```

This will:
1. Train each agent on a synthetic environment
2. Measure average reward, total steps, and wall-clock time
3. Output results and save to `outputs/agent_comparison/benchmark_results.json`

**Example Output:**
```
======================================================================
SUMMARY
======================================================================
Agent                  Avg Reward   Total Reward    Epsilon   Time (s)
----------------------------------------------------------------------
tabular_q                 -0.5727        -2.8635     0.0100      0.288
dqn                       -1.1215        -5.6077     0.0100      0.619
double_dqn                -1.9075        -9.5374     0.0100      0.641
dueling_dqn               -4.5218       -22.6089     0.0100      0.971
----------------------------------------------------------------------
🏆 Best Agent: TABULAR_Q (Avg Reward: -0.5727)
```

## Configuration Parameters

All agents accept common hyperparameters:

- `gamma` (float): Discount factor (0.99 typical)
- `learning_rate` (float): Step size for weight updates (0.001 typical)
- `epsilon_start` (float): Initial exploration rate (1.0)
- `epsilon_end` (float): Final exploration rate (0.01)
- `epsilon_decay` (float): Decay factor per step (0.995 typical)
- `seed` (int): Random seed for reproducibility

Deep agents (DQN, Double DQN, Dueling DQN) also have:
- `hidden_dim` (int): Hidden layer size (64 typical)
- `batch_size` (int): Batch size for learning (32 typical)
- `replay_capacity` (int): Size of replay buffer (10000 typical)
- `learning_starts` (int): Steps before training begins (500 typical)
- `target_update_interval` (int): Steps between target network updates (500 typical)
- `train_frequency` (int): Train every N steps (4 typical)

## Testing

Unit tests for each agent:

```bash
# Test Double DQN
python -m pytest tests/test_double_dqn.py -v

# Test Dueling DQN  
python -m pytest tests/test_dueling_dqn.py -v

# Test all agents
python -m pytest tests/test_*dqn.py tests/test_agents.py -v
```

## Which Agent Should I Use?

| Use Case | Recommended Agent |
|----------|-------------------|
| Small discrete state space | Tabular Q-Learning |
| Quick baseline | DQN |
| Production (stable learning) | Double DQN |
| Complex state space | Dueling DQN |
| Unsure | Start with Double DQN |

## Next Steps

1. **Run the comparison** to see which agent works best on your environment
2. **Train the best agent** longer to convergence
3. **Hyperparameter tuning**: Adjust learning rate, epsilon decay, network size
4. **Combine approaches**: Consider combining Double DQN + Dueling architectures

## References

- **DQN**: [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236)
- **Double DQN**: [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461)
- **Dueling DQN**: [Dueling Network Architectures for Deep Reinforcement Learning](https://arxiv.org/abs/1511.06581)
