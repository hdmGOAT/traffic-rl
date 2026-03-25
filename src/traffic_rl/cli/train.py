from __future__ import annotations

import argparse
from pathlib import Path

from traffic_rl.config import load_config
from traffic_rl.training import run_training


def main() -> None:
    parser = argparse.ArgumentParser(description="Train RL traffic signal controller.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    summary = run_training(cfg)
    checkpoint_path = Path(cfg.output_dir) / f"agent_checkpoint_{cfg.env.backend.lower()}.npz"

    print("Training complete")
    print(f"Episodes: {summary.episodes}")
    print(f"Average reward: {summary.average_reward:.3f}")
    print(f"Checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
