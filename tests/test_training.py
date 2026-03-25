from pathlib import Path

from traffic_rl.config import load_config
from traffic_rl.training import run_training


def test_training_runs_and_returns_summary() -> None:
    cfg = load_config(Path("configs/default.yaml"))
    cfg.training.episodes = 2
    cfg.training.max_steps = 10
    summary = run_training(cfg)

    assert summary.episodes == 2
    assert len(summary.episode_rewards) == 2
