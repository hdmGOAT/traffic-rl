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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Statistically compare trained vs fixed-time and untrained policy performance."
    )
    parser.add_argument("--config", default="configs/cityflow.more_cycles.yaml", help="Path to YAML config")
    parser.add_argument("--episodes", type=int, default=20, help="Evaluation episodes per seed")
    parser.add_argument("--seeds", type=int, default=5, help="Number of seeds to evaluate")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Checkpoint path (defaults to backend-specific path in output_dir)",
    )
    parser.add_argument(
        "--report-file",
        default="outputs/compare_report.json",
        help="Where to write JSON comparison report",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    trained_rewards: list[float] = []
    untrained_rewards: list[float] = []
    fixed_time_rewards: list[float] = []

    trained_queues: list[float] = []
    untrained_queues: list[float] = []
    fixed_time_queues: list[float] = []

    trained_travel_times: list[float] = []
    untrained_travel_times: list[float] = []
    fixed_time_travel_times: list[float] = []

    trained_throughputs: list[float] = []
    untrained_throughputs: list[float] = []

    for i in tqdm(range(args.seeds), desc="Compare seeds", unit="seed"):
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

        fixed_cfg = copy.deepcopy(cfg)
        fixed_cfg.seed = run_seed
        fixed_cfg.training.agent_type = "fixed_time"
        fixed_time = run_evaluation(
            fixed_cfg,
            episodes=args.episodes,
            checkpoint_path=None,
            replay_file=None,
            load_checkpoint=False,
            show_progress=False,
        )

        trained_rewards.extend(trained.episode_rewards)
        untrained_rewards.extend(untrained.episode_rewards)
        fixed_time_rewards.extend(fixed_time.episode_rewards)

        trained_queues.append(trained.average_queue)
        untrained_queues.append(untrained.average_queue)
        fixed_time_queues.append(fixed_time.average_queue)

        trained_travel_times.append(trained.average_travel_time)
        untrained_travel_times.append(untrained.average_travel_time)
        fixed_time_travel_times.append(fixed_time.average_travel_time)

        trained_throughputs.append(trained.average_throughput)
        untrained_throughputs.append(untrained.average_throughput)

    stats = compare_reward_distributions(
        np.asarray(trained_rewards, dtype=np.float64),
        np.asarray(untrained_rewards, dtype=np.float64),
        seed=cfg.seed,
    )

    stats_vs_fixed = compare_reward_distributions(
        np.asarray(trained_rewards, dtype=np.float64),
        np.asarray(fixed_time_rewards, dtype=np.float64),
        seed=cfg.seed,
    )

    report = {
        "config": str(Path(args.config).resolve()),
        "episodes_per_seed": args.episodes,
        "num_seeds": args.seeds,
        "num_samples_per_policy": len(trained_rewards),
        # --- reward stats (trained vs random/untrained) ---
        "trained_mean_reward": stats.trained_mean,
        "untrained_mean_reward": stats.untrained_mean,
        "mean_diff": stats.mean_diff,
        "ci95": [stats.ci_low, stats.ci_high],
        "p_value": stats.p_value,
        "cohen_d": stats.cohen_d,
        "is_significant_0_05": bool(stats.p_value < 0.05),
        # --- reward stats (trained vs fixed-time) ---
        "fixed_time_mean_reward": float(np.mean(fixed_time_rewards)),
        "trained_vs_fixed_mean_diff": stats_vs_fixed.mean_diff,
        "trained_vs_fixed_ci95": [stats_vs_fixed.ci_low, stats_vs_fixed.ci_high],
        "trained_vs_fixed_p_value": stats_vs_fixed.p_value,
        "trained_vs_fixed_cohen_d": stats_vs_fixed.cohen_d,
        "trained_beats_fixed_time": bool(stats_vs_fixed.p_value < 0.05 and stats_vs_fixed.mean_diff > 0),
        # --- queue length ---
        "trained_mean_queue": float(np.mean(trained_queues)),
        "untrained_mean_queue": float(np.mean(untrained_queues)),
        "fixed_time_mean_queue": float(np.mean(fixed_time_queues)),
        # --- travel time ---
        "trained_mean_travel_time": float(np.mean(trained_travel_times)),
        "untrained_mean_travel_time": float(np.mean(untrained_travel_times)),
        "fixed_time_mean_travel_time": float(np.mean(fixed_time_travel_times)),
        # --- throughput ---
        "trained_mean_throughput": float(np.mean(trained_throughputs)),
        "untrained_mean_throughput": float(np.mean(untrained_throughputs)),
        # raw rewards for visualization
        "trained_rewards": trained_rewards,
        "untrained_rewards": untrained_rewards,
        "fixed_time_rewards": fixed_time_rewards,
    }

    report_path = Path(args.report_file)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("Comparison complete")
    print(f"Samples per policy:          {report['num_samples_per_policy']}")
    print()
    print("--- Trained vs Random (untrained) ---")
    print(f"Trained mean reward:         {report['trained_mean_reward']:.3f}")
    print(f"Untrained mean reward:       {report['untrained_mean_reward']:.3f}")
    print(f"Mean diff:                   {report['mean_diff']:.3f}")
    print(f"95% CI:                      [{report['ci95'][0]:.3f}, {report['ci95'][1]:.3f}]")
    print(f"p-value:                     {report['p_value']:.5f}")
    print(f"Cohen's d:                   {report['cohen_d']:.3f}")
    print(f"Significant (alpha=0.05):    {report['is_significant_0_05']}")
    print()
    print("--- Trained vs Fixed-Time baseline ---")
    print(f"Fixed-time mean reward:      {report['fixed_time_mean_reward']:.3f}")
    print(f"Mean diff:                   {report['trained_vs_fixed_mean_diff']:.3f}")
    print(f"95% CI:                      [{report['trained_vs_fixed_ci95'][0]:.3f}, {report['trained_vs_fixed_ci95'][1]:.3f}]")
    print(f"p-value:                     {report['trained_vs_fixed_p_value']:.5f}")
    print(f"Cohen's d:                   {report['trained_vs_fixed_cohen_d']:.3f}")
    print(f"Trained beats fixed-time:    {report['trained_beats_fixed_time']}")
    print()
    print("--- Queue length (lower is better) ---")
    print(f"Trained:                     {report['trained_mean_queue']:.3f}")
    print(f"Fixed-time:                  {report['fixed_time_mean_queue']:.3f}")
    print(f"Random:                      {report['untrained_mean_queue']:.3f}")
    print()
    print("--- Avg travel time in s (lower is better) ---")
    print(f"Trained:                     {report['trained_mean_travel_time']:.3f}")
    print(f"Fixed-time:                  {report['fixed_time_mean_travel_time']:.3f}")
    print(f"Random:                      {report['untrained_mean_travel_time']:.3f}")
    print()
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
