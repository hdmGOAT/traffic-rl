from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from traffic_rl.analysis import compare_reward_distributions
from traffic_rl.config import load_config
from traffic_rl.evaluation import run_evaluation
from traffic_rl.visualization import write_rl_working_report_html


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate an HTML visualization showing whether RL is working."
    )
    parser.add_argument("--config", default="configs/cityflow.more_cycles.yaml", help="Path to YAML config")
    parser.add_argument("--episodes", type=int, default=10, help="Evaluation episodes per seed")
    parser.add_argument("--seeds", type=int, default=3, help="Number of seeds to evaluate")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Checkpoint path (defaults to backend-specific path in output_dir)",
    )
    parser.add_argument(
        "--html-file",
        default="outputs/rl_working_report.html",
        help="Where to write the HTML report",
    )
    parser.add_argument(
        "--json-file",
        default="outputs/rl_working_report.json",
        help="Where to write the JSON report",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    trained_rewards: list[float] = []
    untrained_rewards: list[float] = []
    trained_queues: list[float] = []
    untrained_queues: list[float] = []
    trained_throughputs: list[float] = []
    untrained_throughputs: list[float] = []

    for i in tqdm(range(args.seeds), desc="Visualize seeds", unit="seed"):
        run_seed = int(cfg.seed + i)

        trained_cfg = copy.deepcopy(cfg)
        trained_cfg.seed = run_seed
        trained = run_evaluation(
            trained_cfg,
            episodes=args.episodes,
            checkpoint_path=args.checkpoint,
            replay_file=None,
            load_checkpoint=True,
            show_progress=False,
        )
        trained_rewards.extend(trained.episode_rewards)
        trained_queues.append(trained.average_queue)
        trained_throughputs.append(trained.average_throughput)

        untrained_cfg = copy.deepcopy(cfg)
        untrained_cfg.seed = run_seed
        untrained = run_evaluation(
            untrained_cfg,
            episodes=args.episodes,
            checkpoint_path=None,
            replay_file=None,
            load_checkpoint=False,
            show_progress=False,
        )
        untrained_rewards.extend(untrained.episode_rewards)
        untrained_queues.append(untrained.average_queue)
        untrained_throughputs.append(untrained.average_throughput)

    stats = compare_reward_distributions(
        np.asarray(trained_rewards, dtype=np.float64),
        np.asarray(untrained_rewards, dtype=np.float64),
        seed=cfg.seed,
    )

    report = {
        "title": "Is RL Working? Trained vs Untrained",
        "config": str(Path(args.config).resolve()),
        "episodes_per_seed": args.episodes,
        "num_seeds": args.seeds,
        "num_samples_per_policy": len(trained_rewards),
        "trained_mean_reward": stats.trained_mean,
        "untrained_mean_reward": stats.untrained_mean,
        "mean_diff": stats.mean_diff,
        "ci95": [stats.ci_low, stats.ci_high],
        "p_value": stats.p_value,
        "cohen_d": stats.cohen_d,
        "is_significant_0_05": bool(stats.p_value < 0.05),
        "trained_mean_queue": float(np.mean(trained_queues) if trained_queues else 0.0),
        "untrained_mean_queue": float(np.mean(untrained_queues) if untrained_queues else 0.0),
        "trained_mean_throughput": float(np.mean(trained_throughputs) if trained_throughputs else 0.0),
        "untrained_mean_throughput": float(np.mean(untrained_throughputs) if untrained_throughputs else 0.0),
        "trained_rewards": [float(value) for value in trained_rewards],
        "untrained_rewards": [float(value) for value in untrained_rewards],
    }

    json_path = Path(args.json_file)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    html_path = write_rl_working_report_html(report, args.html_file)

    print("Visualization report complete")
    print(f"JSON report: {json_path}")
    print(f"HTML report: {html_path}")
    print(
        f"Reward diff (trained-untrained): {report['mean_diff']:.3f}, "
        f"p={report['p_value']:.5f}, significant={report['is_significant_0_05']}"
    )


if __name__ == "__main__":
    main()
