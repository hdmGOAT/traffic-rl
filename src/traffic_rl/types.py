from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# A single snapshot of what the intersection looks like at one decision step.
# This is what the agent receives as input before choosing a phase.
@dataclass(slots=True)
class Observation:
    # Number of vehicles stopped (waiting) in each incoming lane.
    # This is the primary signal the agent uses to decide which phase to serve.
    queue_lengths: np.ndarray

    # Redundant copy of queue_lengths kept for potential future use
    # (e.g. separate waiting-time weighting). Currently mirrors queue_lengths.
    waiting_vehicles: np.ndarray

    # Average wait time (in seconds) for vehicles in each incoming lane.
    # Tracks how long vehicles have been waiting since first observed in the lane.
    # Complementary to queue_lengths: high queue with low wait means recent arrivals;
    # low queue with high wait means long-stuck vehicles.
    wait_times: np.ndarray

    # Which traffic-light phase is currently active (0-indexed integer).
    # The agent needs this so it knows its own current state, not just the traffic state.
    current_phase: int

    # How many decision steps the current phase has been held so far.
    # The agent uses this to respect the minimum green-time constraint
    # (it cannot switch phases before min_green_time is reached).
    elapsed_green: int

    def as_vector(self) -> np.ndarray:
        """Flatten the observation into a 1-D float array suitable for neural network input.

        Concatenates: [queue_lengths | waiting_vehicles | wait_times | current_phase | elapsed_green]
        The network never sees raw Python objects — only this flat numeric vector.
        """
        return np.concatenate(
            [
                self.queue_lengths.astype(np.float32),
                self.waiting_vehicles.astype(np.float32),
                self.wait_times.astype(np.float32),
                # Scalar values must be wrapped in an array before concatenation.
                np.array([float(self.current_phase), float(self.elapsed_green)], dtype=np.float32),
            ]
        )
