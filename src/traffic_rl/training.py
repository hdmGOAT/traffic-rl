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
    episodes: int
    episode_rewards: list[float]
    average_reward: float


def run_training(cfg: AppConfig) -> TrainingSummary:
    env = build_env(cfg)
    agent = build_agent(cfg, env.action_size)

    rewards: list[float] = []
    progress = tqdm(range(cfg.training.episodes), desc="Training episodes", unit="ep")
    for episode_idx in progress:
        obs = env.reset()
        state = obs.as_vector()
        episode_reward = 0.0

        for _step in range(cfg.training.max_steps):
            action = agent.act(state, train=True)
            next_obs, reward, done, _ = env.step(action)
            next_state = next_obs.as_vector()

            agent.observe(state, action, reward, next_state, done)
            episode_reward += reward
            state = next_state

            if done:
                break

        rewards.append(float(episode_reward))
        running_avg = float(np.mean(rewards))
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
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "training_summary.json"
    output_file.write_text(json.dumps(asdict(summary), indent=2), encoding="utf-8")


def _save_agent_checkpoint(cfg: AppConfig, agent: object) -> None:
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / f"agent_checkpoint_{cfg.env.backend.lower()}.npz"

    if hasattr(agent, "save"):
        agent.save(checkpoint_path)


def _reward_bar(reward: float, width: int = 16) -> str:
    # Less-negative reward means better traffic control.
    clamped = float(np.clip(reward, -12000.0, 0.0))
    filled = int(round((clamped + 12000.0) / 12000.0 * width))
    filled = max(0, min(width, filled))
    return f"[{'#' * filled}{'-' * (width - filled)}]"
