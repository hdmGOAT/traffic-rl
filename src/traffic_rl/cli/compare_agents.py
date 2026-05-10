#!/usr/bin/env python
"""
Benchmark script to compare all four RL agents on the same environment.

Comparison:
1. Tabular Q-learning - simplest baseline
2. DQN - deep Q-learning with replay buffer
3. Double DQN - reduces overestimation bias
4. Dueling DQN - separates value and advantage streams
"""
import json
from pathlib import Path
from dataclasses import dataclass, asdict

import numpy as np
from traffic_rl.agents.factory import build_agent
from traffic_rl.config import AppConfig


@dataclass
class BenchmarkResult:
    """Store benchmark results for an agent."""

    agent_name: str
    total_reward: float
    avg_reward: float
    final_epsilon: float
    total_steps: int
    wall_time_seconds: float


def create_agent_config(agent_type: str) -> AppConfig:
    """Create a config for the specified agent type."""
    cfg = AppConfig()
    cfg.training.agent_type = agent_type
    cfg.training.gamma = 0.99
    cfg.training.learning_rate = 0.001
    cfg.training.epsilon_start = 1.0
    cfg.training.epsilon_end = 0.01
    cfg.training.epsilon_decay = 0.995
    cfg.training.hidden_dim = 64
    cfg.training.batch_size = 32
    cfg.training.replay_capacity = 10000
    cfg.training.learning_starts = 500
    cfg.training.target_update_interval = 500
    cfg.training.train_frequency = 4
    cfg.seed = 42

    return cfg


def simple_environment_step(agent, state: np.ndarray, step: int) -> tuple[np.ndarray, float, bool]:
    """Simple synthetic environment for testing agents."""
    action = agent.act(state, train=True)

    # Reward designed to test agent learning
    # Moving towards higher state values is rewarded
    target = np.sin(step / 50.0) * 5
    error = np.linalg.norm(state - target)
    reward = 1.0 - (error / 10.0)  # Normalized reward in [-1, 2]

    # Generate next state with some dynamics
    next_state = 0.9 * state + 0.1 * np.random.randn(10).astype(np.float32)
    next_state = np.clip(next_state, -5, 5).astype(np.float32)

    done = step >= 999  # Episode length 1000

    return next_state, reward, done


def benchmark_agent(agent_type: str, num_episodes: int = 5, steps_per_episode: int = 1000) -> BenchmarkResult:
    """Run benchmark for a single agent type."""
    import time

    cfg = create_agent_config(agent_type)
    agent = build_agent(cfg, action_size=4)

    total_reward = 0.0
    total_steps = 0
    episode_rewards = []

    start_time = time.time()

    for episode in range(num_episodes):
        state = np.random.randn(10).astype(np.float32)
        episode_reward = 0.0

        for step in range(steps_per_episode):
            next_state, reward, done = simple_environment_step(agent, state, step)
            agent.observe(state, agent.act(state, train=False), reward, next_state, done)

            episode_reward += reward
            total_reward += reward
            total_steps += 1

            state = next_state
            if done:
                break

        episode_rewards.append(episode_reward)

    wall_time = time.time() - start_time

    result = BenchmarkResult(
        agent_name=agent_type,
        total_reward=total_reward,
        avg_reward=np.mean(episode_rewards),
        final_epsilon=agent.epsilon,
        total_steps=total_steps,
        wall_time_seconds=wall_time,
    )

    return result


def run_comparison(output_dir: str = "outputs/agent_comparison"):
    """Run benchmark comparison of all four agents."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    agent_types = ["tabular_q", "dqn", "double_dqn", "dueling_dqn"]

    print("=" * 70)
    print("RL Agent Comparison Benchmark")
    print("=" * 70)
    print(f"Agents: {', '.join(agent_types)}")
    print(f"Episodes: 5, Steps per episode: 1000")
    print()

    results = {}
    for agent_type in agent_types:
        print(f"Benchmarking {agent_type.upper():20s} ...", end=" ", flush=True)
        try:
            result = benchmark_agent(agent_type)
            results[agent_type] = result
            print(f"✓ Reward: {result.avg_reward:7.3f}, Time: {result.wall_time_seconds:6.3f}s")
        except Exception as e:
            print(f"✗ Error: {e}")

    # Print summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Agent':<20} {'Avg Reward':>12} {'Total Reward':>14} {'Epsilon':>10} {'Time (s)':>10}")
    print("-" * 70)

    best_reward = float('-inf')
    best_agent = None

    for agent_type in agent_types:
        if agent_type in results:
            r = results[agent_type]
            print(
                f"{r.agent_name:<20} {r.avg_reward:>12.4f} {r.total_reward:>14.4f} "
                f"{r.final_epsilon:>10.4f} {r.wall_time_seconds:>10.3f}"
            )
            if r.avg_reward > best_reward:
                best_reward = r.avg_reward
                best_agent = r.agent_name

    print("-" * 70)
    if best_agent:
        print(f"🏆 Best Agent: {best_agent.upper()} (Avg Reward: {best_reward:.4f})")
    print()

    # Save results to JSON
    results_json = output_path / "benchmark_results.json"
    results_data = {name: asdict(result) for name, result in results.items()}
    with open(results_json, "w") as f:
        json.dump(results_data, f, indent=2)
    print(f"Results saved to: {results_json}")


if __name__ == "__main__":
    run_comparison()
