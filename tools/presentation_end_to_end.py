from __future__ import annotations

import copy
from dataclasses import asdict
from pathlib import Path

import numpy as np

from presentation_pipeline_helpers import (
    REPO_ROOT,
    StepPrinter,
    apply_shared_hyperparams,
    build_demands,
    compare_reward_distributions,
    create_split_configs,
    inspect_pems_tensor,
    load_config,
    load_pems_demand_config,
    make_pipeline_state,
    print_input_to_output_demo,
    print_postprocessed_preview,
    run_evaluation,
    run_training,
    save_report,
)


def _bar(value: float, scale: float = 3.0, width: int = 30) -> str:
    """Render a compact ASCII bar for reward trend printing.

    Positive values are shown with '+' and negative values with '-'.
    The scale controls how aggressively values are compressed into bar width.
    """
    clipped = max(-width, min(width, int(round(value / scale))))
    if clipped >= 0:
        return "+" * clipped
    return "-" * abs(clipped)


def run_presentation_flow() -> Path:
    """Run the full pipeline in a presentation-friendly, linear sequence.

    Design goal:
    Keep important ML calls (`run_training`, `run_evaluation`) visible directly
    in this file, while utility details live in the helper module.
    """
    printer = StepPrinter()

    # ---------------------------------------------------------------------
    # 1) Initialize global runtime state (paths, controls, hyperparameters)
    # ---------------------------------------------------------------------
    printer.header("Initialize Pipeline")
    state = make_pipeline_state()
    printer.kv("working_directory", Path.cwd())
    printer.kv("repo_root", REPO_ROOT)
    printer.kv("cityflow_available", state.has_cityflow)
    printer.kv("controls", asdict(state.controls))

    # ---------------------------------------------------------------------
    # 2) Load source demand config used to generate split-specific flow files
    # ---------------------------------------------------------------------
    printer.header("Load Demand Config")
    pems_cfg = load_pems_demand_config(state.paths.pems_config_path)
    printer.kv("pems_config_path", state.paths.pems_config_path)
    printer.kv("pems_npz_path", pems_cfg.pems_npz_path)

    # ---------------------------------------------------------------------
    # 3) Data preparation and preview
    # - inspect_pems_tensor: sanity checks on raw tensor distribution
    # - build_demands: creates train/val/test flow json files
    # - print_* helpers: readable preview for presentation audiences
    # ---------------------------------------------------------------------
    tensor = inspect_pems_tensor(printer, pems_cfg)
    demand_outputs, prep_summary = build_demands(printer, state, pems_cfg)
    print_postprocessed_preview(printer, demand_outputs)
    print_input_to_output_demo(printer, tensor, pems_cfg, demand_outputs)
    split_mode, split_cfg_paths = create_split_configs(printer, state, demand_outputs)

    # ---------------------------------------------------------------------
    # 4) Training (explicit call kept here on purpose)
    # ---------------------------------------------------------------------
    printer.header("Train Agent on Train Split")

    # Start from train split runtime config.
    train_cfg = load_config(split_cfg_paths["train"])

    # Apply one shared hyperparameter bundle so train/eval stay comparable.
    apply_shared_hyperparams(train_cfg, state.controls, state.hyperparams)

    # Set run-specific values for this script.
    train_cfg.seed = state.hyperparams.seed
    train_cfg.training.episodes = state.controls.train_episodes
    train_cfg.training.max_steps = state.controls.train_max_steps
    train_cfg.output_dir = str(state.paths.output_root)

    # Core training call.
    printer.info("Calling run_training(...)")
    train_summary = run_training(train_cfg)
    printer.info("Training finished")
    printer.kv("average_reward", float(train_summary.average_reward))

    # Print compact per-episode trend so audiences can see learning dynamics.
    episode_rewards = np.asarray(train_summary.episode_rewards, dtype=np.float64)
    rolling_window = min(5, len(episode_rewards))
    rolling_mean = np.array(
        [float(np.mean(episode_rewards[max(0, idx - rolling_window + 1) : idx + 1])) for idx in range(len(episode_rewards))],
        dtype=np.float64,
    )
    print("Episode rewards (ASCII bars):")
    for idx, reward in enumerate(train_summary.episode_rewards, start=1):
        print(f"  ep {idx:02d} | {reward:8.3f} | {_bar(reward)} | roll={rolling_mean[idx - 1]:8.3f}")

    # ---------------------------------------------------------------------
    # 5) Evaluation (explicit calls kept here on purpose)
    # - For each split and seed:
    #   - Evaluate trained policy (checkpoint loaded)
    #   - Evaluate baseline policy (no checkpoint)
    # - Aggregate all episodes and compare statistically
    # ---------------------------------------------------------------------
    printer.header("Evaluate Trained vs Untrained")
    eval_rows = []
    for split_name in ("train", "val", "test"):
        printer.info(f"Evaluating split: {split_name}")

        # Build a base config for this split, then clone per seed/policy.
        base_eval_cfg = load_config(split_cfg_paths[split_name])
        apply_shared_hyperparams(base_eval_cfg, state.controls, state.hyperparams)
        base_eval_cfg.seed = state.hyperparams.seed
        base_eval_cfg.training.max_steps = state.controls.train_max_steps
        base_eval_cfg.output_dir = train_cfg.output_dir

        # Hold all episode rewards across seeds for robust statistics.
        trained_rewards_all: list[float] = []
        untrained_rewards_all: list[float] = []
        trained_queue_means: list[float] = []
        untrained_queue_means: list[float] = []

        for seed_offset in range(state.controls.eval_seeds):
            run_seed = int(base_eval_cfg.seed + seed_offset)
            printer.kv("eval_seed", run_seed)

            # Trained policy: loads latest checkpoint from output_dir.
            trained_cfg = copy.deepcopy(base_eval_cfg)
            trained_cfg.seed = run_seed
            printer.info("Calling run_evaluation(..., load_checkpoint=True)")
            trained = run_evaluation(
                trained_cfg,
                episodes=state.controls.eval_episodes,
                checkpoint_path=None,
                replay_file=None,
                load_checkpoint=True,
                show_progress=False,
            )

            # Baseline policy: same environment/setup but random-initialized agent.
            untrained_cfg = copy.deepcopy(base_eval_cfg)
            untrained_cfg.seed = run_seed
            printer.info("Calling run_evaluation(..., load_checkpoint=False)")
            untrained = run_evaluation(
                untrained_cfg,
                episodes=state.controls.eval_episodes,
                checkpoint_path=None,
                replay_file=None,
                load_checkpoint=False,
                show_progress=False,
            )

            # Append per-episode rewards (distribution-level comparison),
            # and queue metrics (operational performance proxy).
            trained_rewards_all.extend(float(v) for v in trained.episode_rewards)
            untrained_rewards_all.extend(float(v) for v in untrained.episode_rewards)
            trained_queue_means.append(float(trained.average_queue))
            untrained_queue_means.append(float(untrained.average_queue))

        # Bootstrap + permutation-based significance test.
        stats = compare_reward_distributions(
            np.asarray(trained_rewards_all, dtype=np.float64),
            np.asarray(untrained_rewards_all, dtype=np.float64),
            seed=int(base_eval_cfg.seed),
            bootstrap_samples=state.controls.significance_bootstrap_samples,
            permutation_samples=state.controls.significance_permutation_samples,
        )

        # One summary row per split for final report.
        row = {
            "split": split_name,
            "num_seeds": int(state.controls.eval_seeds),
            "episodes_per_seed": int(state.controls.eval_episodes),
            "num_samples_per_policy": int(len(trained_rewards_all)),
            "trained_avg_reward": float(stats.trained_mean),
            "untrained_avg_reward": float(stats.untrained_mean),
            "delta": float(stats.mean_diff),
            "trained_avg_queue": float(np.mean(trained_queue_means) if trained_queue_means else 0.0),
            "untrained_avg_queue": float(np.mean(untrained_queue_means) if untrained_queue_means else 0.0),
            "p_value": float(stats.p_value),
            "ci95_low": float(stats.ci_low),
            "ci95_high": float(stats.ci_high),
            "cohen_d": float(stats.cohen_d),
            "is_significant_0_05": bool(stats.p_value < 0.05),
        }
        eval_rows.append(row)
        printer.kv("summary", row)

    # ---------------------------------------------------------------------
    # 6) Persist all results to a presentation-friendly JSON artifact
    # ---------------------------------------------------------------------
    report_path = save_report(printer, state, split_mode, prep_summary, train_summary, eval_rows)

    # ---------------------------------------------------------------------
    # 7) Final completion signal
    # ---------------------------------------------------------------------
    printer.header("Pipeline Finished")
    printer.info("End-to-end flow completed successfully")
    printer.kv("report_path", report_path)
    return report_path


def main() -> None:
    run_presentation_flow()


if __name__ == "__main__":
    main()
