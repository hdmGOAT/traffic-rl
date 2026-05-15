#!/usr/bin/env python3
from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Ensure project src is importable
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
import sys
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from traffic_rl.config import load_config
from traffic_rl.evaluation import run_evaluation
from traffic_rl.analysis import compare_reward_distributions


def main():
    out_dir = REPO_ROOT / "outputs" / "notebook_run"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load notebook-run split configs if present
    split_cfg_paths_file = out_dir / "cityflow_train.yaml"
    # Fallback by scanning notebook_run for cityflow_*.yaml
    split_cfg_paths = {}
    for p in out_dir.glob("cityflow_*.yaml"):
        name = p.stem.replace("cityflow_", "")
        split_cfg_paths[name] = str(p)

    if not split_cfg_paths:
        print("No cityflow_*.yaml configs found in outputs/notebook_run — falling back to mock configs")
        # use mock configs if present in repo configs
        split_cfg_paths = {"train": str(REPO_ROOT / "configs" / "default.yaml"),
                           "val": str(REPO_ROOT / "configs" / "default.yaml"),
                           "test": str(REPO_ROOT / "configs" / "default.yaml")}

    # Load a train_cfg to find output dir and shared hyperparams if needed
    # We'll assume that train checkpoint exists at outputs/notebook_run/agent_checkpoint_cityflow.npz
    checkpoint = out_dir / "agent_checkpoint_cityflow.npz"
    if not checkpoint.exists():
        print("Warning: checkpoint not found:", checkpoint)

    # Settings (match notebook defaults — override via env if needed)
    EVAL_SEEDS = int(os.environ.get("EVAL_SEEDS", "7"))
    EVAL_EPISODES = int(os.environ.get("EVAL_EPISODES", "12"))
    SIGNIFICANCE_BOOTSTRAP_SAMPLES = int(os.environ.get("SIGNIFICANCE_BOOTSTRAP_SAMPLES", "5000"))
    SIGNIFICANCE_PERMUTATION_SAMPLES = int(os.environ.get("SIGNIFICANCE_PERMUTATION_SAMPLES", "20000"))

    per_split = {}
    for split_name, cfg_path in split_cfg_paths.items():
        print(f"Running evaluation for split={split_name} using config={cfg_path}")
        base_eval_cfg = load_config(cfg_path)
        base_eval_cfg.output_dir = str(out_dir)
        trained_all = []
        untrained_all = []
        for seed_offset in range(EVAL_SEEDS):
            run_seed = int(base_eval_cfg.seed + seed_offset if getattr(base_eval_cfg, 'seed', None) is not None else seed_offset)

            tcfg = copy.deepcopy(base_eval_cfg)
            tcfg.seed = run_seed
            trained = run_evaluation(tcfg, episodes=EVAL_EPISODES, checkpoint_path=None, load_checkpoint=True, show_progress=False)

            ucfg = copy.deepcopy(base_eval_cfg)
            ucfg.seed = run_seed
            untrained = run_evaluation(ucfg, episodes=EVAL_EPISODES, checkpoint_path=None, load_checkpoint=False, show_progress=False)

            trained_all.extend(float(v) for v in trained.episode_rewards)
            untrained_all.extend(float(v) for v in untrained.episode_rewards)

        t_arr = np.asarray(trained_all, dtype=np.float64)
        u_arr = np.asarray(untrained_all, dtype=np.float64)
        per_split[split_name] = {"trained": t_arr, "untrained": u_arr}

        stats = compare_reward_distributions(t_arr, u_arr, seed=int(getattr(base_eval_cfg, 'seed', 7)), bootstrap_samples=SIGNIFICANCE_BOOTSTRAP_SAMPLES, permutation_samples=SIGNIFICANCE_PERMUTATION_SAMPLES)
        print(f"Split={split_name}: n_trained={t_arr.size} n_untrained={u_arr.size}")
        print(f"  trained_mean={stats.trained_mean:.3f} untrained_mean={stats.untrained_mean:.3f} delta={stats.mean_diff:.3f}")
        print(f"  p_value={stats.p_value:.5f} ci95=[{stats.ci_low:.3f},{stats.ci_high:.3f}] cohen_d={stats.cohen_d:.3f}")

        # Plot distributions
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        sns.histplot(t_arr, color="C0", label="trained", kde=True, stat="density")
        sns.histplot(u_arr, color="C1", label="untrained", kde=True, stat="density", alpha=0.6)
        plt.legend()
        plt.title(f"{split_name} reward distributions")

        plt.subplot(1, 2, 2)
        sns.boxplot(data=[t_arr, u_arr])
        plt.xticks([0, 1], ["trained", "untrained"])
        plt.title(f"{split_name} boxplot")
        plt.tight_layout()
        fig_path = out_dir / f"{split_name}_reward_dist.png"
        plt.savefig(fig_path)
        plt.close()
        print(f"Saved plot to {fig_path}\n")

    # Save raw arrays for offline inspection
    save_path = out_dir / "per_split_raw_rewards.npz"
    np.savez_compressed(save_path, **{f"{s}_trained": per_split[s]["trained"] for s in per_split}, **{f"{s}_untrained": per_split[s]["untrained"] for s in per_split})
    print("Saved raw reward arrays to", save_path)

    # Also write a short JSON summary
    summary = {s: {"n_trained": int(per_split[s]["trained"].size), "n_untrained": int(per_split[s]["untrained"].size)} for s in per_split}
    (out_dir / "per_split_summary.json").write_text(json.dumps(summary, indent=2))
    print("Wrote per_split_summary.json")


if __name__ == "__main__":
    main()
