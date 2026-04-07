from __future__ import annotations

from dataclasses import dataclass
import json
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


def generate_chart_from_replay(
    replay_path: str | Path,
    chart_path: str | Path,
    title: str = "Vehicle count",
) -> Path:
    replay = Path(replay_path)
    if not replay.exists():
        raise FileNotFoundError(f"Replay file was not found: {replay}")

    chart = Path(chart_path)
    chart.parent.mkdir(parents=True, exist_ok=True)

    lines = replay.read_text(encoding="utf-8").splitlines()
    counts: list[str] = []
    for index, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        if ";" not in line:
            raise ValueError(f"Invalid replay format at line {index}: missing ';' separator.")
        car_logs, _ = line.split(";", maxsplit=1)
        vehicles = [entry for entry in car_logs.split(",") if entry.strip()]
        counts.append(str(len(vehicles)))

    chart.write_text("\n".join([title, *counts]) + "\n", encoding="utf-8")
    return chart


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


def resolve_cityflow_file_path(cfg: AppConfig, file_path: str) -> Path:
    path = Path(file_path)
    if path.is_absolute():
        return path.resolve()

    if not cfg.env.cityflow_config_path:
        return (Path.cwd() / path).resolve()

    engine_path = Path(cfg.env.cityflow_config_path)
    engine_cfg = json.loads(engine_path.read_text(encoding="utf-8"))
    engine_dir = Path(str(engine_cfg.get("dir", "."))).resolve()
    return (engine_dir / path).resolve()


def resolve_replay_file_path(cfg: AppConfig, replay_file: str | None) -> Path | None:
    if replay_file:
        return resolve_cityflow_file_path(cfg, replay_file)

    if not cfg.env.cityflow_config_path:
        return None

    engine_path = Path(cfg.env.cityflow_config_path)
    engine_cfg = json.loads(engine_path.read_text(encoding="utf-8"))
    replay_log = engine_cfg.get("replayLogFile")
    if not isinstance(replay_log, str) or not replay_log:
        return None
    return resolve_cityflow_file_path(cfg, replay_log)
