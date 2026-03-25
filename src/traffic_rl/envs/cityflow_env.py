from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from traffic_rl.config import EnvironmentConfig
from traffic_rl.envs.base import TrafficEnv
from traffic_rl.reward import queue_length_reward
from traffic_rl.types import Observation


@dataclass(slots=True)
class _CityFlowHandles:
    engine: object


class CityFlowTrafficEnv(TrafficEnv):
    def __init__(self, cfg: EnvironmentConfig, seed: int = 7) -> None:
        if not cfg.cityflow_config_path:
            raise ValueError("cityflow_config_path is required when backend is 'cityflow'.")

        self._engine_cfg = _load_engine_config(cfg.cityflow_config_path)
        if not bool(self._engine_cfg.get("rlTrafficLight", False)):
            raise ValueError("CityFlow engine config must set 'rlTrafficLight: true' for RL control.")
        self._incoming_roads = _load_incoming_roads(self._engine_cfg, cfg.intersection_id)

        try:
            import cityflow  # type: ignore
        except ImportError as error:
            raise ImportError(
                "CityFlow is not installed in this environment. Install it from source, e.g. clone CityFlow and run: /path/to/venv/bin/python -m pip install ."
            ) from error

        self.cfg = cfg
        self.seed_value = seed
        self.cityflow = cityflow
        self.handles = _CityFlowHandles(
            engine=cityflow.Engine(cfg.cityflow_config_path, thread_num=cfg.cityflow_thread_num)
        )
        self._episode_index = 0
        self._phase = 0
        self._elapsed_green = 0
        self._step = 0

    @property
    def action_size(self) -> int:
        return self.cfg.num_phases

    def reset(self) -> Observation:
        episode_seed = int(self.seed_value + self._episode_index)
        self.handles.engine.reset(seed=episode_seed)
        self._episode_index += 1
        self._phase = 0
        self._elapsed_green = 0
        self._step = 0
        return self._build_observation()

    def set_replay_file(self, replay_file: str) -> None:
        self.handles.engine.set_replay_file(replay_file)

    def _build_observation(self) -> Observation:
        lane_waiting_count = self.handles.engine.get_lane_waiting_vehicle_count()
        lane_ids = _select_incoming_lane_ids(lane_waiting_count, self._incoming_roads)
        queue = np.array([float(lane_waiting_count[lane]) for lane in lane_ids], dtype=np.float32)

        if queue.size == 0:
            queue = np.zeros(self.cfg.num_lanes, dtype=np.float32)
        if queue.size < self.cfg.num_lanes:
            pad = np.zeros(self.cfg.num_lanes - queue.size, dtype=np.float32)
            queue = np.concatenate([queue, pad])
        elif queue.size > self.cfg.num_lanes:
            queue = queue[: self.cfg.num_lanes]

        return Observation(
            queue_lengths=queue,
            waiting_vehicles=queue.copy(),
            current_phase=self._phase,
            elapsed_green=self._elapsed_green,
        )

    def step(self, action: int) -> tuple[Observation, float, bool, dict]:
        self._step += 1

        if action != self._phase and self._elapsed_green >= self.cfg.min_green_time:
            self.handles.engine.set_tl_phase(self.cfg.intersection_id, action)
            self._phase = action
            self._elapsed_green = 0
        else:
            self._elapsed_green += self.cfg.decision_interval

        for _ in range(self.cfg.decision_interval):
            self.handles.engine.next_step()

        obs = self._build_observation()
        reward = queue_length_reward(obs)
        done = self._step >= self.cfg.episode_horizon_seconds // self.cfg.decision_interval

        info = {
            "step": self._step,
            "avg_queue": float(obs.queue_lengths.mean()),
            "throughput": float(len(self.handles.engine.get_vehicles(include_waiting=False))),
            "replay_enabled": bool(self._engine_cfg.get("saveReplay", False)),
        }
        return obs, reward, done, info


def _load_engine_config(config_path: str) -> dict:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"CityFlow engine config was not found: {config_path}")
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError("CityFlow engine config must be a JSON object.")
    return config


def _load_incoming_roads(engine_cfg: dict, intersection_id: str) -> set[str]:
    root_dir = Path(engine_cfg.get("dir", ".")).resolve()
    roadnet_file = engine_cfg.get("roadnetFile")
    if not roadnet_file:
        return set()

    roadnet_path = (root_dir / roadnet_file).resolve()
    if not roadnet_path.exists():
        return set()

    roadnet = json.loads(roadnet_path.read_text(encoding="utf-8"))
    roads = roadnet.get("roads", [])
    incoming = {
        str(road.get("id"))
        for road in roads
        if str(road.get("endIntersection")) == intersection_id and road.get("id") is not None
    }
    return incoming


def _select_incoming_lane_ids(lane_waiting_count: dict, incoming_roads: set[str]) -> list[str]:
    if not incoming_roads:
        return sorted(lane_waiting_count.keys())

    selected = [
        lane_id
        for lane_id in sorted(lane_waiting_count.keys())
        if any(lane_id.startswith(f"{road_id}_") for road_id in incoming_roads)
    ]
    return selected if selected else sorted(lane_waiting_count.keys())
