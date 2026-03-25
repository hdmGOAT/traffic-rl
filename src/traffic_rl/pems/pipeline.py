from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import yaml
from tqdm.auto import tqdm


@dataclass(slots=True)
class SplitConfig:
    train_days: int
    val_days: int
    test_days: int
    start_day: int = 0


@dataclass(slots=True)
class PemsDemandConfig:
    pems_npz_path: str
    output_dir: str
    sampling_interval_sec: int
    flow_feature_index: int
    sensor_indices: list[int]
    route_catalog: dict[str, list[str]]
    sensor_to_route_probabilities: dict[int, dict[str, float]]
    split: SplitConfig
    arrival_process: str = "poisson"
    random_seed: int = 7
    vehicle_template: dict | None = None


@dataclass(slots=True)
class PemsDemandOutputs:
    train_flow_file: Path
    val_flow_file: Path
    test_flow_file: Path
    summary_file: Path


DEFAULT_VEHICLE_TEMPLATE = {
    "length": 5.0,
    "width": 2.0,
    "maxPosAcc": 2.0,
    "maxNegAcc": 4.5,
    "usualPosAcc": 2.0,
    "usualNegAcc": 4.5,
    "minGap": 2.5,
    "maxSpeed": 16.67,
    "headwayTime": 1.5,
}


def load_pems_demand_config(config_path: str | Path) -> PemsDemandConfig:
    path = Path(config_path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    split_raw = raw.get("split", {})
    sensor_prob_raw = raw.get("sensor_to_route_probabilities", {})

    return PemsDemandConfig(
        pems_npz_path=_resolve(path, raw["pems_npz_path"]),
        output_dir=_resolve(path, raw.get("output_dir", "outputs/pems")),
        sampling_interval_sec=int(raw.get("sampling_interval_sec", 300)),
        flow_feature_index=int(raw.get("flow_feature_index", 0)),
        sensor_indices=[int(index) for index in raw.get("sensor_indices", [])],
        route_catalog={str(k): [str(road) for road in v] for k, v in raw.get("route_catalog", {}).items()},
        sensor_to_route_probabilities={
            int(sensor): {str(route_id): float(prob) for route_id, prob in probs.items()}
            for sensor, probs in sensor_prob_raw.items()
        },
        split=SplitConfig(
            train_days=int(split_raw.get("train_days", 40)),
            val_days=int(split_raw.get("val_days", 5)),
            test_days=int(split_raw.get("test_days", 5)),
            start_day=int(split_raw.get("start_day", 0)),
        ),
        arrival_process=str(raw.get("arrival_process", "poisson")).lower(),
        random_seed=int(raw.get("random_seed", 7)),
        vehicle_template=raw.get("vehicle_template", None),
    )


def build_cityflow_demands(cfg: PemsDemandConfig) -> PemsDemandOutputs:
    npz = np.load(cfg.pems_npz_path)
    data = npz["data"]
    if data.ndim != 3:
        raise ValueError(f"Expected PEMS data shape [T, N, F], got {data.shape}")

    timesteps, num_sensors, num_features = data.shape
    _validate_config(cfg, timesteps, num_sensors, num_features)

    steps_per_day = int(24 * 3600 // cfg.sampling_interval_sec)
    split_steps = _split_ranges(cfg.split, steps_per_day, timesteps)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg.random_seed)

    train_entries = _build_split_entries(cfg, data, split_steps["train"], rng, split_name="train")
    val_entries = _build_split_entries(cfg, data, split_steps["val"], rng, split_name="val")
    test_entries = _build_split_entries(cfg, data, split_steps["test"], rng, split_name="test")

    train_file = output_dir / "flow_train.json"
    val_file = output_dir / "flow_val.json"
    test_file = output_dir / "flow_test.json"

    train_file.write_text(json.dumps(train_entries, indent=2), encoding="utf-8")
    val_file.write_text(json.dumps(val_entries, indent=2), encoding="utf-8")
    test_file.write_text(json.dumps(test_entries, indent=2), encoding="utf-8")

    summary = {
        "input_shape": [int(timesteps), int(num_sensors), int(num_features)],
        "sampling_interval_sec": cfg.sampling_interval_sec,
        "flow_feature_index": cfg.flow_feature_index,
        "arrival_process": cfg.arrival_process,
        "splits": {
            "train": {"timesteps": len(split_steps["train"]), "vehicles": len(train_entries)},
            "val": {"timesteps": len(split_steps["val"]), "vehicles": len(val_entries)},
            "test": {"timesteps": len(split_steps["test"]), "vehicles": len(test_entries)},
        },
    }
    summary_file = output_dir / "summary.json"
    summary_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return PemsDemandOutputs(
        train_flow_file=train_file,
        val_flow_file=val_file,
        test_flow_file=test_file,
        summary_file=summary_file,
    )


def _build_split_entries(
    cfg: PemsDemandConfig,
    data: np.ndarray,
    step_indices: np.ndarray,
    rng: np.random.Generator,
    split_name: str,
) -> list[dict]:
    entries: list[dict] = []
    vehicle_template = cfg.vehicle_template or DEFAULT_VEHICLE_TEMPLATE

    for split_step_index, step in enumerate(
        tqdm(step_indices, desc=f"Building {split_name} split", unit="window")
    ):
        window_start = int(split_step_index * cfg.sampling_interval_sec)
        window_end = int(window_start + cfg.sampling_interval_sec)

        for sensor_idx in cfg.sensor_indices:
            flow_value = float(data[int(step), sensor_idx, cfg.flow_feature_index])
            flow_value = max(0.0, flow_value)
            count = _sample_vehicle_count(cfg.arrival_process, flow_value, rng)
            if count <= 0:
                continue

            route_probs = cfg.sensor_to_route_probabilities[sensor_idx]
            route_ids = list(route_probs.keys())
            probs = np.array([route_probs[rid] for rid in route_ids], dtype=np.float64)
            probs = probs / probs.sum()
            sampled_route_ids = rng.choice(route_ids, size=count, replace=True, p=probs)

            arrival_times = _sample_arrival_times(cfg.arrival_process, window_start, window_end, count, rng)

            for route_id, start_time in zip(sampled_route_ids, arrival_times):
                entry = {
                    "vehicle": vehicle_template,
                    "route": cfg.route_catalog[route_id],
                    "interval": 1.0,
                    "startTime": int(start_time),
                    "endTime": int(start_time),
                }
                entries.append(entry)

    entries.sort(key=lambda item: item["startTime"])
    return entries


def _sample_vehicle_count(arrival_process: str, flow_value: float, rng: np.random.Generator) -> int:
    if arrival_process == "poisson":
        return int(rng.poisson(flow_value))
    if arrival_process == "uniform":
        return int(round(flow_value))
    raise ValueError(f"Unsupported arrival_process '{arrival_process}'.")


def _sample_arrival_times(
    arrival_process: str,
    window_start: int,
    window_end: int,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if count <= 0:
        return np.array([], dtype=np.int64)

    if arrival_process == "uniform":
        if count == 1:
            return np.array([window_start], dtype=np.int64)
        return np.linspace(window_start, max(window_start, window_end - 1), num=count, dtype=np.int64)

    return np.sort(rng.integers(window_start, max(window_start + 1, window_end), size=count))


def _split_ranges(split_cfg: SplitConfig, steps_per_day: int, total_steps: int) -> dict[str, np.ndarray]:
    start = split_cfg.start_day * steps_per_day
    train_end = start + split_cfg.train_days * steps_per_day
    val_end = train_end + split_cfg.val_days * steps_per_day
    test_end = val_end + split_cfg.test_days * steps_per_day

    if test_end > total_steps:
        raise ValueError(
            "Split exceeds available timesteps. "
            f"Need {test_end}, available {total_steps}. Adjust split days or start_day."
        )

    return {
        "train": np.arange(start, train_end),
        "val": np.arange(train_end, val_end),
        "test": np.arange(val_end, test_end),
    }


def _validate_config(cfg: PemsDemandConfig, timesteps: int, num_sensors: int, num_features: int) -> None:
    if not cfg.sensor_indices:
        raise ValueError("sensor_indices cannot be empty.")
    if cfg.flow_feature_index < 0 or cfg.flow_feature_index >= num_features:
        raise ValueError(f"flow_feature_index must be in [0, {num_features - 1}].")
    for sensor in cfg.sensor_indices:
        if sensor < 0 or sensor >= num_sensors:
            raise ValueError(f"Sensor index {sensor} is out of bounds for {num_sensors} sensors.")
        if sensor not in cfg.sensor_to_route_probabilities:
            raise ValueError(f"Missing route probabilities for sensor {sensor}.")
        probs = cfg.sensor_to_route_probabilities[sensor]
        if not probs:
            raise ValueError(f"Empty route probabilities for sensor {sensor}.")
        for route_id in probs:
            if route_id not in cfg.route_catalog:
                raise ValueError(f"Route '{route_id}' not found in route_catalog.")
    if timesteps <= 0:
        raise ValueError("PEMS data has zero timesteps.")


def _resolve(config_path: Path, value: str) -> str:
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str((config_path.parent / path).resolve())
