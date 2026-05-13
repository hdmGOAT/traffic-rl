from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import json
import numpy as np
from tqdm.auto import tqdm

from traffic_rl.agents.factory import build_agent
from traffic_rl.config import AppConfig
from traffic_rl.envs.factory import build_env


@dataclass(slots=True)
class TrainingSummary:
    """Aggregated results written to disk after a training run completes."""

    episodes: int           # Total episodes trained.
    episode_rewards: list[float]  # Per-episode cumulative reward (useful for plotting convergence).
    average_reward: float   # Mean reward across all episodes — headline training metric.


def run_training(cfg: AppConfig) -> TrainingSummary:
    """Run the full RL training loop and return a summary.

    Each episode:
      1. Reset the environment to its initial state.
      2. Step through the episode: agent acts → environment steps → agent learns.
      3. Record the total reward for the episode.
    After all episodes, save the agent checkpoint and write a JSON summary.
    """
    env   = build_env(cfg)
    agent = build_agent(cfg, env.action_size)

    rewards: list[float] = []
    progress = tqdm(range(cfg.training.episodes), desc="Training episodes", unit="ep")

    frozen = False
    for episode_idx in progress:
        obs   = env.reset()
        state = obs.as_vector()      # Flatten observation into a vector the network can consume.
        episode_reward = 0.0

        for _step in range(cfg.training.max_steps):
            # Agent chooses a phase. If we've frozen learning, act in evaluation mode.
            action = agent.act(state, train=not frozen)

            # Environment applies the phase and returns the next state + reward.
            next_obs, reward, done, _ = env.step(action)
            next_state = next_obs.as_vector()

            # Only observe (learn) while not frozen.
            if not frozen:
                agent.observe(state, action, reward, next_state, done)

            episode_reward += reward
            state = next_state

            if done:
                break  # Episode time horizon reached — start next episode.

        # After the episode, check whether we should freeze learning when epsilon bottoms out.
        # This helps get stable evaluation metrics once exploration stops.
        if (
            getattr(cfg.training, "freeze_on_epsilon_end", False)
            and not frozen
            and hasattr(agent, "epsilon")
            and getattr(agent, "epsilon") <= cfg.training.epsilon_end
        ):
            frozen = True
            # Request agent to stop learning and persist the checkpoint.
            if hasattr(agent, "freeze"):
                agent.freeze()
            _save_agent_checkpoint(cfg, agent)
            print(f"[train] Freezing learning at episode {episode_idx + 1}; epsilon reached {agent.epsilon:.4f}")

        rewards.append(float(episode_reward))
        running_avg = float(np.mean(rewards))

        # Update the progress bar with this episode's stats.
        progress.set_postfix(
            reward=f"{episode_reward:.1f}",
            avg=f"{running_avg:.1f}",
            trend=_reward_bar(episode_reward),
        )
        print(
            f"[train] ep {episode_idx + 1:03d}/{cfg.training.episodes:03d} "
            f"reward={episode_reward:8.1f} avg={running_avg:8.1f} "
            f"{_reward_bar(episode_reward)}"
        )

    summary = TrainingSummary(
        episodes=cfg.training.episodes,
        episode_rewards=rewards,
        average_reward=float(np.mean(rewards) if rewards else 0.0),
    )
    _write_summary(cfg, summary)
    _save_agent_checkpoint(cfg, agent)
    return summary


def _write_summary(cfg: AppConfig, summary: TrainingSummary) -> None:
    """Serialise the training summary to JSON in the configured output directory."""
    output_dir  = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "training_summary.json"
    output_file.write_text(json.dumps(asdict(summary), indent=2), encoding="utf-8")


def _save_agent_checkpoint(cfg: AppConfig, agent: object) -> None:
    """Save the trained agent's weights to a backend-specific .npz file."""
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Include the backend name in the filename so cityflow and mock checkpoints don't collide.
    checkpoint_path = output_dir / f"agent_checkpoint_{cfg.env.backend.lower()}.npz"

    if hasattr(agent, "save"):
        agent.save(checkpoint_path)


def _reward_bar(reward: float, width: int = 16) -> str:
    """Render a simple ASCII bar chart showing how good this episode's reward was.

    The reward scale is clamped to [-12000, 0]. A fully filled bar means
    reward ≈ 0 (no queues); an empty bar means reward ≈ -12000 (severe congestion).
    """
    clamped = float(np.clip(reward, -12000.0, 0.0))
    # Map [-12000, 0] → [0, width].
    filled = int(round((clamped + 12000.0) / 12000.0 * width))
    filled = max(0, min(width, filled))
    return f"[{'#' * filled}{'-' * (width - filled)}]"
