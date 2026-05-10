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
    """Aggregated metrics from running an agent through one or more evaluation episodes.

    Unlike TrainingSummary (which only tracks reward), this captures all the
    metrics that matter for real-world comparison: queue length, throughput, and
    — crucially — average travel time (the gold-standard traffic engineering metric).
    """

    episodes: int                  # Number of episodes evaluated.
    average_reward: float          # Mean cumulative reward (the training proxy signal).
    average_queue: float           # Mean vehicles waiting per lane per step (lower = better).
    average_throughput: float      # Mean moving vehicles in the network per step (higher = better).
    average_travel_time: float     # Mean seconds each vehicle spent in the network (lower = better).
                                   # Only meaningful with the CityFlow backend; 0.0 for mock.
    episode_rewards: list[float]   # Per-episode rewards for statistical comparison.


def generate_chart_from_replay(
    replay_path: str | Path,
    chart_path: str | Path,
    title: str = "Vehicle count",
) -> Path:
    """Parse a CityFlow replay log and write a simple vehicle-count chart file.

    Each line in a CityFlow replay log lists vehicles active in that simulation
    second, separated by a semicolon from road-state data. This function counts
    vehicles per line and writes those counts as a plain text file.
    """
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
        # The part before the semicolon is a comma-separated list of vehicle entries.
        car_logs, _ = line.split(";", maxsplit=1)
        vehicles = [entry for entry in car_logs.split(",") if entry.strip()]
        counts.append(str(len(vehicles)))

    # First line is the chart title; subsequent lines are per-tick vehicle counts.
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
    """Run an agent through evaluation episodes (no training, no exploration).

    The agent always acts greedily (train=False). Metrics are collected from
    the info dict returned by env.step() and averaged across all steps.

    Args:
        cfg:              App config (determines backend, agent type, etc.).
        episodes:         Number of episodes to run.
        checkpoint_path:  Path to a trained agent checkpoint. If None, uses the
                          default backend-specific path in cfg.output_dir.
        replay_file:      If provided, tells CityFlow to record a replay log here.
        load_checkpoint:  Set to False to evaluate a fresh (untrained) agent.
        show_progress:    Whether to show a tqdm progress bar.
        progress_desc:    Label for the progress bar.
    """
    env   = build_env(cfg)
    agent = build_agent(cfg, env.action_size)

    if load_checkpoint:
        resolved_checkpoint = _resolve_checkpoint(cfg, checkpoint_path)
        if resolved_checkpoint is not None:
            agent.load(resolved_checkpoint)

    if replay_file:
        env.set_replay_file(replay_file)

    # Accumulators — collected every step across all episodes.
    rewards: list[float]       = []
    avg_queues: list[float]    = []
    throughputs: list[float]   = []
    travel_times: list[float]  = []

    for _ in tqdm(range(episodes), desc=progress_desc, unit="ep", disable=not show_progress):
        obs   = env.reset()
        state = obs.as_vector()
        episode_reward = 0.0

        for _step in range(cfg.training.max_steps):
            action = agent.act(state, train=False)   # Always exploit, never explore.
            next_obs, reward, done, info = env.step(action)
            state = next_obs.as_vector()
            episode_reward += reward

            # Collect per-step metrics from the environment's info dict.
            avg_queues.append(float(info.get("avg_queue", 0.0)))
            throughputs.append(float(info.get("throughput", 0.0)))
            # avg_travel_time accumulates inside CityFlow as vehicles complete trips.
            travel_times.append(float(info.get("avg_travel_time", 0.0)))

            if done:
                break

        rewards.append(episode_reward)

    return EvaluationSummary(
        episodes=episodes,
        average_reward=float(np.mean(rewards) if rewards else 0.0),
        average_queue=float(np.mean(avg_queues) if avg_queues else 0.0),
        average_throughput=float(np.mean(throughputs) if throughputs else 0.0),
        average_travel_time=float(np.mean(travel_times) if travel_times else 0.0),
        episode_rewards=[float(value) for value in rewards],
    )


def _resolve_checkpoint(cfg: AppConfig, checkpoint_path: str | None) -> str | None:
    """Find the checkpoint file to load, with a backend-specific default fallback."""
    if checkpoint_path is None:
        # Look for a backend-specific checkpoint first, then the legacy name.
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
    """Resolve a file path relative to the CityFlow engine config's 'dir' setting.

    CityFlow stores its data files (road network, replay logs, etc.) relative to
    the 'dir' key in the engine JSON. This helper makes those paths absolute.
    """
    path = Path(file_path)
    if path.is_absolute():
        return path.resolve()

    if not cfg.env.cityflow_config_path:
        return (Path.cwd() / path).resolve()

    engine_path = Path(cfg.env.cityflow_config_path)
    engine_cfg  = json.loads(engine_path.read_text(encoding="utf-8"))
    engine_dir  = Path(str(engine_cfg.get("dir", "."))).resolve()
    return (engine_dir / path).resolve()


def resolve_replay_file_path(cfg: AppConfig, replay_file: str | None) -> Path | None:
    """Return the absolute path to the CityFlow replay log, or None if not configured."""
    if replay_file:
        return resolve_cityflow_file_path(cfg, replay_file)

    if not cfg.env.cityflow_config_path:
        return None

    # Read the replay path from the engine config's 'replayLogFile' key.
    engine_path = Path(cfg.env.cityflow_config_path)
    engine_cfg  = json.loads(engine_path.read_text(encoding="utf-8"))
    replay_log  = engine_cfg.get("replayLogFile")
    if not isinstance(replay_log, str) or not replay_log:
        return None
    return resolve_cityflow_file_path(cfg, replay_log)
