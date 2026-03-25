from __future__ import annotations

from .types import Observation


def queue_length_reward(observation: Observation) -> float:
    return -float(observation.queue_lengths.sum())
