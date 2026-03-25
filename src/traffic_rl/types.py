from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class Observation:
    queue_lengths: np.ndarray
    waiting_vehicles: np.ndarray
    current_phase: int
    elapsed_green: int

    def as_vector(self) -> np.ndarray:
        return np.concatenate(
            [
                self.queue_lengths.astype(np.float32),
                self.waiting_vehicles.astype(np.float32),
                np.array([float(self.current_phase), float(self.elapsed_green)], dtype=np.float32),
            ]
        )
