from __future__ import annotations

import argparse

from traffic_rl.config import load_config
from traffic_rl.evaluation import run_evaluation


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RL traffic signal controller.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config file")
    parser.add_argument("--episodes", type=int, default=5, help="Number of eval episodes")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to trained agent checkpoint. If omitted, uses outputs/agent_checkpoint.npz when available.",
    )
    parser.add_argument(
        "--replay-file",
        default=None,
        help="CityFlow replay file path relative to engine config 'dir' (e.g., replay/eval_trained.txt).",
    )
    parser.add_argument(
        "--no-checkpoint",
        action="store_true",
        help="Disable checkpoint loading and evaluate a fresh (untrained) agent.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    summary = run_evaluation(
        cfg,
        episodes=args.episodes,
        checkpoint_path=args.checkpoint,
        replay_file=args.replay_file,
        load_checkpoint=not args.no_checkpoint,
    )

    print("Evaluation complete")
    print(f"Episodes: {summary.episodes}")
    print(f"Average reward: {summary.average_reward:.3f}")
    print(f"Average queue: {summary.average_queue:.3f}")
    print(f"Average throughput: {summary.average_throughput:.3f}")


if __name__ == "__main__":
    main()