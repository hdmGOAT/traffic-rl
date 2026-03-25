from __future__ import annotations

import numpy as np

from traffic_rl.config import EnvironmentConfig
from traffic_rl.envs.base import TrafficEnv
from traffic_rl.reward import queue_length_reward
from traffic_rl.types import Observation


class MockTrafficEnv(TrafficEnv):
    def __init__(self, cfg: EnvironmentConfig, seed: int = 7) -> None:
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self._step = 0
        self._phase = 0
        self._elapsed_green = 0
        self._queues = np.zeros(self.cfg.num_lanes, dtype=np.float32)

    def seed(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)

    @property
    def action_size(self) -> int:
        return self.cfg.num_phases

    def reset(self) -> Observation:
        self._step = 0
        self._phase = 0
        self._elapsed_green = 0
        self._queues = self.rng.integers(0, 3, size=self.cfg.num_lanes).astype(np.float32)
        return self._build_observation()

    def _build_observation(self) -> Observation:
        waiting = self._queues.copy()
        return Observation(
            queue_lengths=self._queues.copy(),
            waiting_vehicles=waiting,
            current_phase=self._phase,
            elapsed_green=self._elapsed_green,
        )

    def step(self, action: int) -> tuple[Observation, float, bool, dict]:
        self._step += 1

        if action != self._phase and self._elapsed_green >= self.cfg.min_green_time:
            self._phase = action
            self._elapsed_green = 0
        else:
            self._elapsed_green += self.cfg.decision_interval

        arrivals = self.rng.poisson(0.8, size=self.cfg.num_lanes).astype(np.float32)
        departures = np.zeros(self.cfg.num_lanes, dtype=np.float32)

        half = max(1, self.cfg.num_lanes // 2)
        if self._phase % 2 == 0:
            departures[:half] = self.rng.integers(0, 2, size=half)
        else:
            departures[half:] = self.rng.integers(0, 2, size=self.cfg.num_lanes - half)

        self._queues = np.maximum(0.0, self._queues + arrivals - departures)

        obs = self._build_observation()
        reward = queue_length_reward(obs)
        done = self._step >= self.cfg.episode_horizon_seconds // self.cfg.decision_interval

        info = {
            "step": self._step,
            "throughput": float(departures.sum()),
            "avg_queue": float(self._queues.mean()),
        }
        return obs, reward, done, info
