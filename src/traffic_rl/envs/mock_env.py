from __future__ import annotations

import numpy as np

from traffic_rl.config import EnvironmentConfig
from traffic_rl.envs.base import TrafficEnv
from traffic_rl.reward import reward_from_type
from traffic_rl.types import Observation


class MockTrafficEnv(TrafficEnv):
    """Synthetic traffic environment that requires no external dependencies.

    Vehicles arrive according to a Poisson process and depart based on which
    phase is active. There is no real road network — it's purely mathematical.
    Used for unit tests and rapid iteration without needing CityFlow installed.
    """

    def __init__(self, cfg: EnvironmentConfig, seed: int = 7, reward_type: str = "queue_length") -> None:
        self.cfg = cfg
        self.reward_type = reward_type
        # Seeded random generator — using a fixed seed makes runs reproducible.
        self.rng = np.random.default_rng(seed)
        self._step = 0          # Counts decision steps within the current episode.
        self._phase = 0         # Currently active traffic-light phase.
        self._elapsed_green = 0 # How many steps this phase has been held.
        # Queue lengths for each lane, initialised to zero at construction.
        self._queues = np.zeros(self.cfg.num_lanes, dtype=np.float32)

    def seed(self, seed: int) -> None:
        """Replace the random generator with a new seeded one (used in tests)."""
        self.rng = np.random.default_rng(seed)

    @property
    def action_size(self) -> int:
        """Number of phases the agent can choose from (set in config)."""
        return self.cfg.num_phases

    def reset(self) -> Observation:
        """Start a fresh episode with random initial queue lengths (0–2 vehicles per lane)."""
        self._step = 0
        self._phase = 0
        self._elapsed_green = 0
        # Randomised starting queues make the agent learn to handle varied conditions.
        self._queues = self.rng.integers(0, 3, size=self.cfg.num_lanes).astype(np.float32)
        obs = self._build_observation()
        self._last_obs = obs
        return obs

    def _build_observation(self) -> Observation:
        """Package the current internal state into an Observation the agent can read."""
        waiting = self._queues.copy()
        # Mock environment: use queue length as dummy wait time (no real vehicle tracking).
        wait_times = self._queues.copy() * 0.5  # Scale by 0.5 seconds per vehicle as placeholder
        return Observation(
            queue_lengths=self._queues.copy(),
            waiting_vehicles=waiting,
            wait_times=wait_times,
            current_phase=self._phase,
            elapsed_green=self._elapsed_green,
        )

    def step(self, action: int) -> tuple[Observation, float, bool, dict]:
        """Advance the simulation by one decision step.

        Phase switching is gated behind min_green_time to prevent the agent from
        flickering phases every single step (which would be unrealistic).
        Vehicles arrive randomly and depart only from lanes served by the current phase.
        """
        self._step += 1
        prev_obs = getattr(self, "_last_obs", None)

        # Only switch phase if the agent requests a different one AND the current
        # phase has been green long enough (minimum green time constraint).
        if action != self._phase and self._elapsed_green >= self.cfg.min_green_time:
            self._phase = action
            self._elapsed_green = 0
        else:
            self._elapsed_green += self.cfg.decision_interval

        # Poisson arrivals: on average 0.8 vehicles arrive per lane per step.
        arrivals = self.rng.poisson(0.8, size=self.cfg.num_lanes).astype(np.float32)

        # Departures: only the lanes served by the active phase can discharge vehicles.
        departures = np.zeros(self.cfg.num_lanes, dtype=np.float32)
        half = max(1, self.cfg.num_lanes // 2)
        if self._phase % 2 == 0:
            # Even phases serve the first half of lanes.
            departures[:half] = self.rng.integers(0, 2, size=half)
        else:
            # Odd phases serve the second half of lanes.
            departures[half:] = self.rng.integers(0, 2, size=self.cfg.num_lanes - half)

        # Update queues: clamp at zero so queues never go negative.
        self._queues = np.maximum(0.0, self._queues + arrivals - departures)

        obs = self._build_observation()
        self._last_obs = obs
        reward = reward_from_type(self.reward_type, obs, prev_observation=prev_obs)

        # Episode ends when we reach the configured time horizon.
        done = self._step >= self.cfg.episode_horizon_seconds // self.cfg.decision_interval

        info = {
            "step": self._step,
            # Total vehicles that departed this step — a proxy for throughput.
            "throughput": float(departures.sum()),
            # Mean queue length across all lanes — used in evaluation summaries.
            "avg_queue": float(self._queues.mean()),
            # Mock environment has no engine to query travel time; 0.0 is a sentinel.
            "avg_travel_time": 0.0,
        }
        return obs, reward, done, info
