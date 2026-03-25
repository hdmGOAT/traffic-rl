from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from traffic_rl.agents.factory import build_agent
from traffic_rl.config import AppConfig
from traffic_rl.envs.factory import build_env


@dataclass(slots=True)
class EvaluationSummary:
    episodes: int
    average_reward: float
    average_queue: float
    average_throughput: float
    episode_rewards: list[float]


def run_evaluation(
    cfg: AppConfig,
    episodes: int = 5,
    checkpoint_path: str | None = None,
    replay_file: str | None = None,
    load_checkpoint: bool = True,
    show_progress: bool = True,
    progress_desc: str = "Evaluation episodes",
) -> EvaluationSummary:
    env = build_env(cfg)
    agent = build_agent(cfg, env.action_size)

    if load_checkpoint:
        resolved_checkpoint = _resolve_checkpoint(cfg, checkpoint_path)
        if resolved_checkpoint is not None:
            agent.load(resolved_checkpoint)

    if replay_file:
        env.set_replay_file(replay_file)

    rewards: list[float] = []
    avg_queues: list[float] = []
    throughputs: list[float] = []

    for _ in tqdm(range(episodes), desc=progress_desc, unit="ep", disable=not show_progress):
        obs = env.reset()
        state = obs.as_vector()
        episode_reward = 0.0

        for _step in range(cfg.training.max_steps):
            action = agent.act(state, train=False)
            next_obs, reward, done, info = env.step(action)
            state = next_obs.as_vector()
            episode_reward += reward
            avg_queues.append(float(info.get("avg_queue", 0.0)))
            throughputs.append(float(info.get("throughput", 0.0)))

            if done:
                break

        rewards.append(episode_reward)

    return EvaluationSummary(
        episodes=episodes,
        average_reward=float(np.mean(rewards) if rewards else 0.0),
        average_queue=float(np.mean(avg_queues) if avg_queues else 0.0),
        average_throughput=float(np.mean(throughputs) if throughputs else 0.0),
        episode_rewards=[float(value) for value in rewards],
    )


def _resolve_checkpoint(cfg: AppConfig, checkpoint_path: str | None) -> str | None:
    if checkpoint_path is None:
        backend_path = Path(cfg.output_dir) / f"agent_checkpoint_{cfg.env.backend.lower()}.npz"
        if backend_path.exists():
            return str(backend_path)
        legacy_path = Path(cfg.output_dir) / "agent_checkpoint.npz"
        return str(legacy_path) if legacy_path.exists() else None

    path = Path(checkpoint_path)
    if not path.is_absolute():
        path = Path.cwd() / path
    return str(path.resolve())
