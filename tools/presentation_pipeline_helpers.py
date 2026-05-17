from __future__ import annotations

"""Helper utilities for the presentation pipeline runner.

This module intentionally contains implementation details so the main runner can
stay linear and easy to present. Responsibilities are split as follows:

- `presentation_end_to_end.py`:
    Orchestrates the story and surfaces key ML calls (`run_training`,
    `run_evaluation`).
- `presentation_pipeline_helpers.py` (this file):
    Provides reusable setup, data prep, config rewriting, and reporting utilities.
"""

import copy
import importlib.util
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def _find_repo_root(start: Path) -> Path:
    """Find repository root by looking for canonical project markers.

    We search upward from `start` until we find a directory that contains all
    expected top-level paths (`pyproject.toml`, `src`, `configs`).
    """
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").exists() and (candidate / "src").exists() and (candidate / "configs").exists():
            return candidate
    raise FileNotFoundError(
        f"Could not locate repo root from {start}. Expected pyproject.toml/src/configs in a parent directory."
    )


REPO_ROOT = _find_repo_root(Path.cwd()).resolve()
SRC_DIR = REPO_ROOT / "src"
# Make project imports work whether script is launched from repo root or subdir.
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from traffic_rl.analysis import compare_reward_distributions
from traffic_rl.config import load_config
from traffic_rl.evaluation import run_evaluation
from traffic_rl.pems.pipeline import PemsDemandConfig, build_cityflow_demands, load_pems_demand_config
from traffic_rl.training import run_training


@dataclass(slots=True)
class RunControls:
    """Top-level runtime knobs used for the presentation run."""

    quick_mode: bool = True
    reward_type: str = "mixed"
    train_episodes: int = 6
    train_max_steps: int = 80
    eval_episodes: int = 6
    eval_seeds: int = 4
    significance_bootstrap_samples: int = 1000
    significance_permutation_samples: int = 3000


@dataclass(slots=True)
class SharedHyperparams:
    """Single source of truth for train/eval agent hyperparameters."""

    gamma: float = 0.99
    learning_rate: float = 0.0002
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.999
    hidden_dim: int = 64
    batch_size: int = 32
    replay_capacity: int = 10000
    learning_starts: int = 200
    target_update_interval: int = 150
    train_frequency: int = 4
    seed: int = 42


@dataclass(slots=True)
class PathBundle:
    """All input/output paths required by the pipeline."""

    pems_config_path: Path
    cityflow_base_config_path: Path
    mock_base_config_path: Path
    output_root: Path


@dataclass(slots=True)
class PipelineState:
    """Container object for immutable run context."""

    controls: RunControls
    hyperparams: SharedHyperparams
    paths: PathBundle
    has_cityflow: bool


class StepPrinter:
    """Small helper for consistent console output formatting.

    Keeps output structured for presentations by numbering sections and using a
    predictable key/value style.
    """

    def __init__(self) -> None:
        self._step = 0

    def header(self, title: str) -> None:
        self._step += 1
        print("\n" + "=" * 88)
        print(f"STEP {self._step}: {title}")
        print("=" * 88)

    @staticmethod
    def info(message: str) -> None:
        print(f"[INFO] {message}")

    @staticmethod
    def kv(key: str, value: Any) -> None:
        print(f"  - {key}: {value}")


def make_pipeline_state() -> PipelineState:
    """Build default controls, hyperparameters, paths, and environment flags."""

    controls = RunControls()
    hyperparams = SharedHyperparams()
    output_suffix = "presentation_quick" if controls.quick_mode else "presentation_full"
    output_root = REPO_ROOT / "outputs" / output_suffix
    paths = PathBundle(
        pems_config_path=REPO_ROOT / "configs" / "pems04_to_cityflow.example.yaml",
        cityflow_base_config_path=REPO_ROOT / "configs" / "cityflow.quick.yaml",
        mock_base_config_path=REPO_ROOT / "configs" / "default.yaml",
        output_root=output_root,
    )
    return PipelineState(
        controls=controls,
        hyperparams=hyperparams,
        paths=paths,
        has_cityflow=importlib.util.find_spec("cityflow") is not None,
    )


def apply_shared_hyperparams(cfg: Any, controls: RunControls, hp: SharedHyperparams) -> Any:
    """Apply shared RL hyperparameters to a loaded config object.

    Why this exists:
    It guarantees train and evaluation runs use the same agent settings so
    differences are due to learning state, not mismatched hyperparameters.
    """

    cfg.reward.type = controls.reward_type
    cfg.training.agent_type = "dqn"
    cfg.training.gamma = hp.gamma
    cfg.training.learning_rate = hp.learning_rate
    cfg.training.epsilon_start = hp.epsilon_start
    cfg.training.epsilon_end = hp.epsilon_end
    cfg.training.epsilon_decay = hp.epsilon_decay
    cfg.training.hidden_dim = hp.hidden_dim
    cfg.training.batch_size = hp.batch_size
    cfg.training.replay_capacity = hp.replay_capacity
    cfg.training.learning_starts = hp.learning_starts
    cfg.training.target_update_interval = hp.target_update_interval
    cfg.training.train_frequency = hp.train_frequency
    cfg.training.freeze_on_epsilon_end = False
    return cfg


def inspect_pems_tensor(printer: StepPrinter, pems_cfg: PemsDemandConfig) -> np.ndarray:
    """Load PEMS tensor and print sanity-check statistics.

    Returns:
    Raw `[T, N, F]` tensor used by downstream demand generation logic.
    """

    printer.header("Inspect Raw PEMS Tensor")
    npz = np.load(pems_cfg.pems_npz_path)
    tensor = npz["data"]

    printer.info("Loaded PEMS tensor")
    printer.kv("shape", tuple(int(v) for v in tensor.shape))
    printer.kv("dtype", str(tensor.dtype))
    printer.kv("min", float(tensor.min()))
    printer.kv("mean", float(tensor.mean()))
    printer.kv("max", float(tensor.max()))

    flow_feature_idx = pems_cfg.flow_feature_index
    for sensor_idx in pems_cfg.sensor_indices[:4]:
        values = tensor[:, sensor_idx, flow_feature_idx]
        printer.kv(
            f"sensor_{sensor_idx}_stats",
            {
                "min": float(values.min()),
                "mean": float(values.mean()),
                "max": float(values.max()),
            },
        )

    return tensor


def build_demands(printer: StepPrinter, state: PipelineState, pems_cfg: PemsDemandConfig) -> tuple[Any, dict[str, Any]]:
    """Generate train/val/test CityFlow demand files from PEMS data.

    In quick mode we intentionally reduce split size and sensors so demos run
    faster while preserving the full pipeline structure.
    """

    printer.header("Build CityFlow Demand Splits")
    prep_cfg = copy.deepcopy(pems_cfg)

    if state.controls.quick_mode:
        printer.info("Applying quick-mode demand settings")
        if len(prep_cfg.sensor_indices) > 2:
            prep_cfg.sensor_indices = prep_cfg.sensor_indices[:2]
        prep_cfg.split.train_days = 1
        prep_cfg.split.val_days = 1
        prep_cfg.split.test_days = 1
        prep_cfg.arrival_process = "uniform"

    prep_cfg.output_dir = str(state.paths.output_root / "demand")
    # Core conversion from PEMS windows to CityFlow per-vehicle records.
    outputs = build_cityflow_demands(prep_cfg)
    summary = json.loads(Path(outputs.summary_file).read_text(encoding="utf-8"))

    printer.info("Demand files generated")
    printer.kv("train_flow", outputs.train_flow_file)
    printer.kv("val_flow", outputs.val_flow_file)
    printer.kv("test_flow", outputs.test_flow_file)
    printer.kv("summary_file", outputs.summary_file)
    printer.kv("split_stats", summary.get("splits", {}))

    return outputs, summary


def summarize_flow_file(path: Path, preview_n: int = 2) -> dict[str, Any]:
    """Return a compact summary and preview rows for one flow file."""

    entries = json.loads(path.read_text(encoding="utf-8"))
    count = len(entries)
    if count == 0:
        return {
            "path": str(path),
            "vehicle_count": 0,
            "start_time_min": None,
            "start_time_max": None,
            "sample_entries": [],
        }

    start_times = [int(item.get("startTime", 0)) for item in entries]
    sample_entries = []
    for entry in entries[:preview_n]:
        sample_entries.append(
            {
                "startTime": int(entry.get("startTime", 0)),
                "endTime": int(entry.get("endTime", 0)),
                "route": entry.get("route", []),
                "interval": float(entry.get("interval", 1.0)),
            }
        )

    return {
        "path": str(path),
        "vehicle_count": int(count),
        "start_time_min": int(min(start_times)),
        "start_time_max": int(max(start_times)),
        "sample_entries": sample_entries,
    }


def print_postprocessed_preview(printer: StepPrinter, demand_outputs: Any) -> None:
    """Print split-by-split flow summaries for quick visual inspection."""

    printer.header("Postprocessed CityFlow Demand Preview")
    split_flow_files = {
        "train": Path(demand_outputs.train_flow_file),
        "val": Path(demand_outputs.val_flow_file),
        "test": Path(demand_outputs.test_flow_file),
    }

    for split_name, flow_path in split_flow_files.items():
        preview = summarize_flow_file(flow_path)
        printer.info(f"{split_name.upper()} split")
        printer.kv("path", preview["path"])
        printer.kv("vehicle_count", preview["vehicle_count"])
        printer.kv("start_time_range", (preview["start_time_min"], preview["start_time_max"]))
        printer.kv("sample_entries", preview["sample_entries"])


def print_input_to_output_demo(
    printer: StepPrinter,
    tensor: np.ndarray,
    pems_cfg: PemsDemandConfig,
    demand_outputs: Any,
) -> None:
    """Show one concrete example of input signal -> generated vehicles.

    This function is designed for explanation during presentations:
    - input side: one sensor/timestep flow value
    - output side: generated vehicle records in the corresponding time window
    """

    printer.header("Input to Output Transformation Demo")
    train_flow_entries = json.loads(Path(demand_outputs.train_flow_file).read_text(encoding="utf-8"))

    sensor_idx = int(pems_cfg.sensor_indices[0])
    timestep_idx = 0
    raw_flow_value = float(tensor[timestep_idx, sensor_idx, pems_cfg.flow_feature_index])

    if pems_cfg.arrival_process == "uniform":
        rule_text = "round(flow_value)"
        expected_count = int(round(max(0.0, raw_flow_value)))
    else:
        rule_text = "poisson(flow_value)"
        expected_count = float(max(0.0, raw_flow_value))

    window_size = int(pems_cfg.sampling_interval_sec)
    window0_entries = [item for item in train_flow_entries if 0 <= int(item.get("startTime", 0)) < window_size]

    demo_rows = [
        {
            "departure_second": int(item.get("startTime", 0)),
            "route": " -> ".join(item.get("route", [])),
        }
        for item in window0_entries[:5]
    ]

    printer.info("Single-window transformation snapshot")
    printer.kv(
        "input",
        {
            "timestep_index": timestep_idx,
            "sensor_index": sensor_idx,
            "flow_value": raw_flow_value,
            "arrival_process": pems_cfg.arrival_process,
            "sampling_rule": rule_text,
            "expected_generated_vehicles": expected_count,
        },
    )
    printer.kv(
        "output",
        {
            "window_seconds": [0, window_size],
            "actual_records": len(window0_entries),
            "sample_records": demo_rows,
        },
    )


def _resolve_from_yaml(config_path: Path, value: str) -> str:
    """Resolve absolute/relative path values found in yaml config files."""

    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str((config_path.parent / path).resolve())


def _resolve_engine_dir(base_engine_path: Path, engine_dir_value: str, repo_root: Path) -> Path:
    """Resolve CityFlow engine directory with robust fallback behavior.

    Some shared configs can contain machine-specific absolute paths. If that
    path is missing locally, we fall back to this repository's `cityflow_data`.
    """

    engine_dir_path = Path(engine_dir_value)
    if not engine_dir_path.is_absolute():
        engine_dir_path = (base_engine_path.parent / engine_dir_path).resolve()

    if not engine_dir_path.exists():
        fallback = (repo_root / "cityflow_data").resolve()
        if fallback.exists():
            return fallback
    return engine_dir_path.resolve()


def _to_engine_relative(engine_dir: Path, flow_path: Path) -> str:
    """Convert flow path to engine-dir-relative path expected by CityFlow."""

    return os.path.relpath(flow_path.resolve(), engine_dir.resolve())


def _to_cityflow_dir_string(engine_dir: Path) -> str:
    """Return CityFlow-compatible directory string with trailing separator."""

    value = str(engine_dir.resolve())
    if not value.endswith(os.sep):
        value += os.sep
    return value


def create_split_configs(printer: StepPrinter, state: PipelineState, demand_outputs: Any) -> tuple[str, dict[str, str]]:
    """Create per-split runtime configs used for training/evaluation.

    Returns:
    - `split_mode`: `cityflow` if CityFlow engine is available, else `mock`
    - `split_cfg_paths`: mapping from split name to yaml config path
    """

    printer.header("Create Split-Specific Runtime Configs")
    split_cfg_paths: dict[str, str]
    split_mode = "mock"

    if state.has_cityflow and state.paths.cityflow_base_config_path.exists():
        # CityFlow path: rewrite flowFile for each split while preserving
        # all other engine settings from the base config.
        split_mode = "cityflow"
        base_raw = yaml.safe_load(state.paths.cityflow_base_config_path.read_text(encoding="utf-8")) or {}
        base_engine_path = Path(
            _resolve_from_yaml(state.paths.cityflow_base_config_path, str(base_raw["env"]["cityflow_config_path"]))
        )
        engine_raw = json.loads(base_engine_path.read_text(encoding="utf-8"))

        notebook_cfg_dir = state.paths.output_root / "runtime_configs"
        notebook_cfg_dir.mkdir(parents=True, exist_ok=True)

        split_to_flow = {
            "train": Path(demand_outputs.train_flow_file),
            "val": Path(demand_outputs.val_flow_file),
            "test": Path(demand_outputs.test_flow_file),
        }

        split_cfg_paths = {}
        for split_name, flow_path in split_to_flow.items():
            split_engine = copy.deepcopy(engine_raw)
            resolved_engine_dir = _resolve_engine_dir(base_engine_path, str(split_engine.get("dir", ".")), REPO_ROOT)
            split_engine["dir"] = _to_cityflow_dir_string(resolved_engine_dir)
            split_engine["flowFile"] = _to_engine_relative(resolved_engine_dir, flow_path)

            engine_out = notebook_cfg_dir / f"cityflow_engine_{split_name}.json"
            engine_out.write_text(json.dumps(split_engine, indent=2), encoding="utf-8")

            split_cfg_raw = copy.deepcopy(base_raw)
            split_cfg_raw["env"]["cityflow_config_path"] = str(engine_out)
            split_cfg_path = notebook_cfg_dir / f"cityflow_{split_name}.yaml"
            split_cfg_path.write_text(yaml.safe_dump(split_cfg_raw, sort_keys=False), encoding="utf-8")
            split_cfg_paths[split_name] = str(split_cfg_path)
    else:
        # Mock fallback keeps the pipeline runnable without CityFlow installed.
        split_cfg_paths = {
            "train": str(state.paths.mock_base_config_path),
            "val": str(state.paths.mock_base_config_path),
            "test": str(state.paths.mock_base_config_path),
        }

    printer.info("Split-specific configs ready")
    printer.kv("split_mode", split_mode)
    printer.kv("split_configs", split_cfg_paths)
    return split_mode, split_cfg_paths


def _bar(value: float, scale: float = 3.0, width: int = 30) -> str:
    """Render ASCII bar used in helper-level training display."""

    clipped = max(-width, min(width, int(round(value / scale))))
    if clipped >= 0:
        return "+" * clipped
    return "-" * abs(clipped)


def train_agent(printer: StepPrinter, state: PipelineState, split_cfg_paths: dict[str, str]) -> tuple[Any, Any]:
    """Train on train split and print reward trend diagnostics.

    Note:
    The main presentation file now performs training explicitly for readability,
    but this helper is kept for optional reuse.
    """

    printer.header("Train Agent on Train Split")
    train_cfg = load_config(split_cfg_paths["train"])
    apply_shared_hyperparams(train_cfg, state.controls, state.hyperparams)
    train_cfg.seed = state.hyperparams.seed
    train_cfg.training.episodes = state.controls.train_episodes
    train_cfg.training.max_steps = state.controls.train_max_steps
    train_cfg.output_dir = str(state.paths.output_root)

    printer.info("Starting training run")
    printer.kv("episodes", train_cfg.training.episodes)
    printer.kv("max_steps", train_cfg.training.max_steps)
    printer.kv("output_dir", train_cfg.output_dir)

    train_summary = run_training(train_cfg)

    episode_rewards = np.asarray(train_summary.episode_rewards, dtype=np.float64)
    rolling_window = min(5, len(episode_rewards))
    rolling_mean = np.array(
        [float(np.mean(episode_rewards[max(0, idx - rolling_window + 1) : idx + 1])) for idx in range(len(episode_rewards))],
        dtype=np.float64,
    )

    printer.info("Training complete")
    printer.kv("average_reward", float(train_summary.average_reward))
    print("Episode rewards (ASCII bars):")
    for idx, reward in enumerate(train_summary.episode_rewards, start=1):
        print(f"  ep {idx:02d} | {reward:8.3f} | {_bar(reward)} | roll={rolling_mean[idx - 1]:8.3f}")

    return train_cfg, train_summary


def evaluate_across_splits(
    printer: StepPrinter,
    state: PipelineState,
    split_cfg_paths: dict[str, str],
    train_cfg: Any,
) -> list[dict[str, Any]]:
    """Evaluate trained vs untrained policies across all splits.

    For each split we aggregate rewards across multiple seeds, then compute
    significance statistics on the pooled trained/untrained reward samples.
    """

    printer.header("Evaluate Trained vs Untrained")
    eval_rows: list[dict[str, Any]] = []

    for split_name in ("train", "val", "test"):
        printer.info(f"Running evaluation for split: {split_name}")
        base_eval_cfg = load_config(split_cfg_paths[split_name])
        apply_shared_hyperparams(base_eval_cfg, state.controls, state.hyperparams)
        base_eval_cfg.seed = state.hyperparams.seed
        base_eval_cfg.training.max_steps = state.controls.train_max_steps
        base_eval_cfg.output_dir = train_cfg.output_dir

        trained_rewards_all: list[float] = []
        untrained_rewards_all: list[float] = []
        trained_queue_means: list[float] = []
        untrained_queue_means: list[float] = []

        for seed_offset in range(state.controls.eval_seeds):
            run_seed = int(base_eval_cfg.seed + seed_offset)
            printer.kv("eval_seed", run_seed)

            trained_cfg = copy.deepcopy(base_eval_cfg)
            trained_cfg.seed = run_seed
            trained = run_evaluation(
                trained_cfg,
                episodes=state.controls.eval_episodes,
                checkpoint_path=None,
                replay_file=None,
                load_checkpoint=True,
                show_progress=False,
            )

            untrained_cfg = copy.deepcopy(base_eval_cfg)
            untrained_cfg.seed = run_seed
            untrained = run_evaluation(
                untrained_cfg,
                episodes=state.controls.eval_episodes,
                checkpoint_path=None,
                replay_file=None,
                load_checkpoint=False,
                show_progress=False,
            )

            trained_rewards_all.extend(float(v) for v in trained.episode_rewards)
            untrained_rewards_all.extend(float(v) for v in untrained.episode_rewards)
            trained_queue_means.append(float(trained.average_queue))
            untrained_queue_means.append(float(untrained.average_queue))

        stats = compare_reward_distributions(
            np.asarray(trained_rewards_all, dtype=np.float64),
            np.asarray(untrained_rewards_all, dtype=np.float64),
            seed=int(base_eval_cfg.seed),
            bootstrap_samples=state.controls.significance_bootstrap_samples,
            permutation_samples=state.controls.significance_permutation_samples,
        )

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

    return eval_rows


def save_report(
    printer: StepPrinter,
    state: PipelineState,
    split_mode: str,
    prep_summary: dict[str, Any],
    train_summary: Any,
    eval_rows: list[dict[str, Any]],
) -> Path:
    """Persist pipeline outputs to a single JSON artifact for downstream use."""

    printer.header("Save Final Report")
    report = {
        "meta": {
            "quick_mode": state.controls.quick_mode,
            "split_mode": split_mode,
            "reward_type": state.controls.reward_type,
            "train_episodes": state.controls.train_episodes,
            "eval_episodes": state.controls.eval_episodes,
            "eval_seeds": state.controls.eval_seeds,
            "train_max_steps": state.controls.train_max_steps,
            "significance_bootstrap_samples": state.controls.significance_bootstrap_samples,
            "significance_permutation_samples": state.controls.significance_permutation_samples,
        },
        "prep_summary": prep_summary,
        "training": asdict(train_summary),
        "evaluation": eval_rows,
    }

    state.paths.output_root.mkdir(parents=True, exist_ok=True)
    report_path = state.paths.output_root / "presentation_flow_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    printer.info("Presentation report saved")
    printer.kv("path", report_path)
    printer.kv("meta", report["meta"])
    return report_path


__all__ = [
    "REPO_ROOT",
    "PathBundle",
    "PipelineState",
    "RunControls",
    "SharedHyperparams",
    "StepPrinter",
    "apply_shared_hyperparams",
    "build_demands",
    "compare_reward_distributions",
    "create_split_configs",
    "evaluate_across_splits",
    "inspect_pems_tensor",
    "load_config",
    "load_pems_demand_config",
    "make_pipeline_state",
    "print_input_to_output_demo",
    "print_postprocessed_preview",
    "run_evaluation",
    "run_training",
    "save_report",
    "train_agent",
]
