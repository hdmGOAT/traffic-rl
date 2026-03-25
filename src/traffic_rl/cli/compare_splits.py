from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm
import yaml

from traffic_rl.analysis import compare_reward_distributions
from traffic_rl.config import load_config
from traffic_rl.evaluation import run_evaluation


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run trained-vs-untrained significance analysis over PEMS train/val/test demand splits."
    )
    parser.add_argument("--config", default="configs/cityflow.more_cycles.yaml", help="Base YAML config path")
    parser.add_argument(
        "--flow-train",
        default="outputs/pems04/flow_train.json",
        help="CityFlow flow file for train split",
    )
    parser.add_argument(
        "--flow-val",
        default="outputs/pems04/flow_val.json",
        help="CityFlow flow file for val split",
    )
    parser.add_argument(
        "--flow-test",
        default="outputs/pems04/flow_test.json",
        help="CityFlow flow file for test split",
    )
    parser.add_argument("--episodes", type=int, default=10, help="Evaluation episodes per seed")
    parser.add_argument("--seeds", type=int, default=3, help="Number of seeds")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Checkpoint path (defaults to backend-specific checkpoint if omitted)",
    )
    parser.add_argument(
        "--report-dir",
        default="outputs/compare_splits",
        help="Directory for per-split and aggregate reports",
    )
    args = parser.parse_args()

    base_config_path = Path(args.config).resolve()
    report_dir = Path(args.report_dir).resolve()
    report_dir.mkdir(parents=True, exist_ok=True)

    split_paths = {
        "train": Path(args.flow_train).resolve(),
        "val": Path(args.flow_val).resolve(),
        "test": Path(args.flow_test).resolve(),
    }

    reports: dict[str, dict] = {}
    for split_name, flow_path in tqdm(split_paths.items(), desc="Comparing splits", unit="split"):
        report = _run_split(
            base_config_path=base_config_path,
            flow_path=flow_path,
            split_name=split_name,
            episodes=args.episodes,
            seeds=args.seeds,
            checkpoint=args.checkpoint,
            report_dir=report_dir,
        )
        reports[split_name] = report

    aggregate = {
        "base_config": str(base_config_path),
        "episodes_per_seed": args.episodes,
        "num_seeds": args.seeds,
        "checkpoint": args.checkpoint,
        "splits": reports,
    }
    aggregate_path = report_dir / "aggregate_report.json"
    aggregate_path.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")

    print("Split comparison complete")
    for split_name in ("train", "val", "test"):
        report = reports[split_name]
        print(
            f"{split_name}: diff={report['mean_diff']:.3f}, p={report['p_value']:.5f}, "
            f"significant={report['is_significant_0_05']}"
        )
    print(f"Aggregate report: {aggregate_path}")


def _run_split(
    *,
    base_config_path: Path,
    flow_path: Path,
    split_name: str,
    episodes: int,
    seeds: int,
    checkpoint: str | None,
    report_dir: Path,
) -> dict:
    if not flow_path.exists():
        raise FileNotFoundError(f"Flow file does not exist: {flow_path}")

    base_raw = yaml.safe_load(base_config_path.read_text(encoding="utf-8")) or {}
    cityflow_config_path = _resolve_from_yaml(base_config_path, str(base_raw["env"]["cityflow_config_path"]))

    engine_raw = json.loads(Path(cityflow_config_path).read_text(encoding="utf-8"))
    engine_dir = Path(engine_raw["dir"]).resolve()
    engine_raw["flowFile"] = _to_engine_relative(engine_dir, flow_path)

    engine_out = report_dir / f"cityflow_engine_{split_name}.json"
    engine_out.write_text(json.dumps(engine_raw, indent=2), encoding="utf-8")

    split_cfg_raw = copy.deepcopy(base_raw)
    split_cfg_raw["env"]["cityflow_config_path"] = str(engine_out)
    split_cfg_path = report_dir / f"cityflow_{split_name}.yaml"
    split_cfg_path.write_text(yaml.safe_dump(split_cfg_raw, sort_keys=False), encoding="utf-8")

    cfg = load_config(split_cfg_path)
    trained_rewards: list[float] = []
    untrained_rewards: list[float] = []

    for i in tqdm(range(seeds), desc=f"{split_name} seeds", unit="seed", leave=False):
        run_seed = int(cfg.seed + i)

        trained_cfg = copy.deepcopy(cfg)
        trained_cfg.seed = run_seed
        trained = run_evaluation(
            trained_cfg,
            episodes=episodes,
            checkpoint_path=checkpoint,
            replay_file=None,
            load_checkpoint=True,
            show_progress=True,
            progress_desc=f"{split_name} seed={run_seed} trained",
        )

        untrained_cfg = copy.deepcopy(cfg)
        untrained_cfg.seed = run_seed
        untrained = run_evaluation(
            untrained_cfg,
            episodes=episodes,
            checkpoint_path=None,
            replay_file=None,
            load_checkpoint=False,
            show_progress=True,
            progress_desc=f"{split_name} seed={run_seed} untrained",
        )

        trained_rewards.extend(trained.episode_rewards)
        untrained_rewards.extend(untrained.episode_rewards)

    stats = compare_reward_distributions(
        np.asarray(trained_rewards, dtype=np.float64),
        np.asarray(untrained_rewards, dtype=np.float64),
        seed=cfg.seed,
    )

    report = {
        "split": split_name,
        "flow_file": str(flow_path),
        "episodes_per_seed": episodes,
        "num_seeds": seeds,
        "num_samples_per_policy": len(trained_rewards),
        "trained_mean_reward": stats.trained_mean,
        "untrained_mean_reward": stats.untrained_mean,
        "mean_diff": stats.mean_diff,
        "ci95": [stats.ci_low, stats.ci_high],
        "p_value": stats.p_value,
        "cohen_d": stats.cohen_d,
        "is_significant_0_05": bool(stats.p_value < 0.05),
        "engine_config": str(engine_out),
        "split_config": str(split_cfg_path),
    }

    report_path = report_dir / f"report_{split_name}.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _resolve_from_yaml(config_path: Path, value: str) -> str:
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str((config_path.parent / path).resolve())


def _to_engine_relative(engine_dir: Path, flow_path: Path) -> str:
    return os.path.relpath(flow_path.resolve(), engine_dir)


if __name__ == "__main__":
    main()
