from __future__ import annotations

from .types import Observation


def queue_length_reward(observation: Observation) -> float:
    """Convert the current queue state into a training reward signal.

    Returns the negative sum of all waiting vehicles across every incoming lane.
    The agent receives a more negative reward the more vehicles are stuck waiting,
    so minimising this value is the same as minimising congestion.

    Example: 3 vehicles in lane A, 2 in lane B → reward = -(3+2) = -5

    Important caveat: this is a proxy metric used only during training.
    It correlates with delay but does not capture travel time directly
    (see EvaluationSummary.average_travel_time for the real-world metric).
    """
    return -float(observation.queue_lengths.sum())
