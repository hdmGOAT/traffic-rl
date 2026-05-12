from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from traffic_rl.config import EnvironmentConfig
from traffic_rl.envs.base import TrafficEnv
from traffic_rl.reward import reward_from_type
from traffic_rl.types import Observation


# Thin wrapper that holds the CityFlow engine object.
# Using a dataclass keeps the engine reference named and avoids a bare attribute on the env.
@dataclass(slots=True)
class _CityFlowHandles:
    engine: object  # cityflow.Engine — typed as object to avoid a hard import at module level.


class CityFlowTrafficEnv(TrafficEnv):
    """Traffic environment backed by the CityFlow microsimulator.

    CityFlow models individual vehicles on a real road network, making it
    far more realistic than the mock environment. The agent controls a single
    intersection by selecting which traffic-light phase to activate each step.
    """

    def __init__(self, cfg: EnvironmentConfig, seed: int = 7, reward_type: str = "queue_length") -> None:
        if not cfg.cityflow_config_path:
            raise ValueError("cityflow_config_path is required when backend is 'cityflow'.")

        # Load the CityFlow JSON engine config (thread count, road network file, etc.).
        self._engine_cfg = _load_engine_config(cfg.cityflow_config_path)

        # RL control must be enabled in the engine config — without it CityFlow
        # runs its own built-in signal logic and ignores set_tl_phase() calls.
        if not bool(self._engine_cfg.get("rlTrafficLight", False)):
            raise ValueError("CityFlow engine config must set 'rlTrafficLight: true' for RL control.")

        # Pre-compute which roads feed into our target intersection so we only
        # observe queues on incoming lanes, not the whole network.
        self._incoming_roads = _load_incoming_roads(self._engine_cfg, cfg.intersection_id)

        # CityFlow is an optional C extension — give a helpful error if not installed.
        try:
            import cityflow  # type: ignore
        except ImportError as error:
            raise ImportError(
                "CityFlow is not installed in this environment. Install it from source, e.g. clone CityFlow and run: /path/to/venv/bin/python -m pip install ."
            ) from error

        self.cfg = cfg
        self.reward_type = reward_type
        self.seed_value = seed
        self.cityflow = cityflow
        # Spin up the CityFlow engine with the provided JSON config.
        self.handles = _CityFlowHandles(
            engine=cityflow.Engine(cfg.cityflow_config_path, thread_num=cfg.cityflow_thread_num)
        )
        self._episode_index = 0  # Incremented each reset so each episode gets a unique seed.
        self._phase = 0          # Currently active signal phase.
        self._elapsed_green = 0  # Steps the current phase has been active.
        self._step = 0           # Decision steps taken in the current episode.
        # Track when each vehicle was first seen in the network (for wait time calculation).
        self._vehicle_enter_times = {}  # Maps vehicle_id -> first_observation_time

    @property
    def action_size(self) -> int:
        """Number of distinct signal phases (from config, e.g. 8 for a 4-way intersection)."""
        return self.cfg.num_phases

    def reset(self) -> Observation:
        """Reset CityFlow to the start of a new episode with a varied seed.

        Using episode_index + base_seed gives each episode different traffic
        arrival patterns while still being reproducible across runs.
        """
        episode_seed = int(self.seed_value + self._episode_index)
        self.handles.engine.reset(seed=episode_seed)
        self._episode_index += 1
        self._phase = 0
        self._elapsed_green = 0
        self._step = 0
        self._vehicle_enter_times = {}  # Clear vehicle tracking for new episode
        obs = self._build_observation()
        self._last_obs = obs
        return obs

    def set_replay_file(self, replay_file: str) -> None:
        """Tell CityFlow where to write its vehicle animation replay log."""
        self.handles.engine.set_replay_file(replay_file)

    def _build_observation(self) -> Observation:
        """Read the current intersection state from CityFlow and package it as an Observation.

        CityFlow exposes per-lane waiting counts and vehicle lists for the entire network.
        We filter down to only the lanes on incoming roads so the observation
        vector size matches cfg.num_lanes regardless of network size.
        We also track vehicle enter times to compute per-lane average wait times.
        """
        current_time = self.handles.engine.get_current_time()
        
        # Dict mapping lane_id → number of stopped vehicles in that lane right now.
        lane_waiting_count = self.handles.engine.get_lane_waiting_vehicle_count()
        lane_vehicles = self.handles.engine.get_lane_vehicles()  # Dict: lane_id -> [vehicle_id, ...]

        # Keep only the lanes that feed into our controlled intersection.
        lane_ids = _select_incoming_lane_ids(lane_waiting_count, self._incoming_roads)
        queue = np.array([float(lane_waiting_count[lane]) for lane in lane_ids], dtype=np.float32)
        
        # Compute average wait time per lane.
        wait_times_per_lane = []
        for lane_id in lane_ids:
            vehicle_ids = lane_vehicles.get(lane_id, [])
            if vehicle_ids:
                waits = []
                for vid in vehicle_ids:
                    # Track when we first see this vehicle; if new, record current time.
                    if vid not in self._vehicle_enter_times:
                        self._vehicle_enter_times[vid] = current_time
                    wait = current_time - self._vehicle_enter_times[vid]
                    waits.append(wait)
                avg_wait = float(np.mean(waits))
            else:
                avg_wait = 0.0
            wait_times_per_lane.append(avg_wait)
        
        wait_times = np.array(wait_times_per_lane, dtype=np.float32)

        # Pad or trim to exactly cfg.num_lanes so the arrays are always a fixed size.
        if queue.size == 0:
            queue = np.zeros(self.cfg.num_lanes, dtype=np.float32)
            wait_times = np.zeros(self.cfg.num_lanes, dtype=np.float32)
        if queue.size < self.cfg.num_lanes:
            pad = np.zeros(self.cfg.num_lanes - queue.size, dtype=np.float32)
            queue = np.concatenate([queue, pad])
            wait_times = np.concatenate([wait_times, pad])
        elif queue.size > self.cfg.num_lanes:
            queue = queue[: self.cfg.num_lanes]
            wait_times = wait_times[: self.cfg.num_lanes]

        return Observation(
            queue_lengths=queue,
            waiting_vehicles=queue.copy(),
            wait_times=wait_times,
            current_phase=self._phase,
            elapsed_green=self._elapsed_green,
        )

    def step(self, action: int) -> tuple[Observation, float, bool, dict]:
        """Apply the agent's chosen phase and advance the CityFlow simulation.

        Phase switches are blocked until min_green_time has elapsed to prevent
        rapid flickering (which would be physically unrealistic and destabilise training).
        After setting the phase, CityFlow is ticked forward by decision_interval
        simulation seconds before the next observation is taken.
        """
        self._step += 1
        prev_obs = getattr(self, "_last_obs", None)

        # Only apply a phase change if the request is different from the current phase
        # AND the current phase has been green for the minimum required time.
        if action != self._phase and self._elapsed_green >= self.cfg.min_green_time:
            self.handles.engine.set_tl_phase(self.cfg.intersection_id, action)
            self._phase = action
            self._elapsed_green = 0
        else:
            self._elapsed_green += self.cfg.decision_interval

        # Advance the simulation by decision_interval seconds one tick at a time.
        for _ in range(self.cfg.decision_interval):
            self.handles.engine.next_step()

        obs = self._build_observation()
        self._last_obs = obs
        reward = reward_from_type(self.reward_type, obs, prev_observation=prev_obs)

        # Episode ends when we've simulated the full configured horizon (e.g. 3600 s).
        done = self._step >= self.cfg.episode_horizon_seconds // self.cfg.decision_interval

        info = {
            "step": self._step,
            # Average number of waiting vehicles across incoming lanes this step.
            "avg_queue": float(obs.queue_lengths.mean()),
            # Snapshot count of vehicles currently moving (not stopped) in the network.
            # Note: this is not cumulative — it reflects the moment after the last tick.
            "throughput": float(len(self.handles.engine.get_vehicles(include_waiting=False))),
            # Average time (seconds) vehicles have spent in the network this episode.
            # CityFlow accumulates this as cars complete their trips — the key real-world metric.
            "avg_travel_time": float(self.handles.engine.get_average_travel_time()),
            "replay_enabled": bool(self._engine_cfg.get("saveReplay", False)),
        }
        return obs, reward, done, info


def _load_engine_config(config_path: str) -> dict:
    """Read and validate the CityFlow JSON engine config from disk."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"CityFlow engine config was not found: {config_path}")
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError("CityFlow engine config must be a JSON object.")
    return config


def _load_incoming_roads(engine_cfg: dict, intersection_id: str) -> set[str]:
    """Parse the road network file and return the IDs of roads that feed into our intersection.

    Knowing the incoming roads lets _build_observation filter the full
    network-wide lane counts down to just the lanes relevant to our agent.
    Returns an empty set if the road network file is missing (falls back to all lanes).
    """
    root_dir = Path(engine_cfg.get("dir", ".")).resolve()
    roadnet_file = engine_cfg.get("roadnetFile")
    if not roadnet_file:
        return set()

    roadnet_path = (root_dir / roadnet_file).resolve()
    if not roadnet_path.exists():
        return set()

    roadnet = json.loads(roadnet_path.read_text(encoding="utf-8"))
    roads = roadnet.get("roads", [])
    # A road is "incoming" if its end intersection matches the one we're controlling.
    incoming = {
        str(road.get("id"))
        for road in roads
        if str(road.get("endIntersection")) == intersection_id and road.get("id") is not None
    }
    return incoming


def _select_incoming_lane_ids(lane_waiting_count: dict, incoming_roads: set[str]) -> list[str]:
    """Return a sorted list of lane IDs that belong to incoming roads.

    CityFlow lane IDs follow the format '<road_id>_<lane_index>', so we match
    by prefix. If no incoming roads were found, fall back to all network lanes.
    """
    if not incoming_roads:
        # Fallback: use every lane in the network (works for single-intersection setups).
        return sorted(lane_waiting_count.keys())

    selected = [
        lane_id
        for lane_id in sorted(lane_waiting_count.keys())
        if any(lane_id.startswith(f"{road_id}_") for road_id in incoming_roads)
    ]
    # Second fallback: if the prefix filter matched nothing, use all lanes.
    return selected if selected else sorted(lane_waiting_count.keys())
