from __future__ import annotations

import argparse

from traffic_rl.config import load_config
from traffic_rl.evaluation import (
    generate_chart_from_replay,
    resolve_cityflow_file_path,
    resolve_replay_file_path,
    run_evaluation,
)


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
    parser.add_argument(
        "--chart-file",
        default=None,
        help=(
            "Optional chart output file path. Relative paths are resolved under "
            "CityFlow engine config 'dir'."
        ),
    )
    parser.add_argument(
        "--chart-title",
        default="Vehicle count",
        help="Chart title (first line in chart file).",
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

    if args.chart_file is not None:
        replay_path = resolve_replay_file_path(cfg, args.replay_file)
        if replay_path is None:
            raise ValueError("Cannot resolve replay file path for chart generation.")

        chart_path = resolve_cityflow_file_path(cfg, args.chart_file)
        generated = generate_chart_from_replay(replay_path, chart_path, title=args.chart_title)
        print(f"Chart file: {generated}")

    print("Evaluation complete")
    print(f"Episodes: {summary.episodes}")
    print(f"Average reward: {summary.average_reward:.3f}")
    print(f"Average queue: {summary.average_queue:.3f}")
    print(f"Average throughput: {summary.average_throughput:.3f}")


if __name__ == "__main__":
    main()