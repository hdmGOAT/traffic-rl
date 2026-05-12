from __future__ import annotations

from .types import Observation
import math


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


def mixed_reward(
    observation: Observation,
    prev_observation: Observation | None = None,
    queue_weight: float = 0.3,
    queue_balancing_weight: float = 1.5,
    duration_penalty_weight: float = 0.2,
) -> float:
    """Multi-factor reward focusing on balanced queue reduction.

    Key insight: penalize IMBALANCE between lanes (max-min) more than total queue.
    This forces the agent to serve all directions fairly, preventing starvation.

    Args:
        observation: Current traffic observation.
        prev_observation: Previous traffic observation for delta calculation.
        queue_weight: Weight for total queue reduction (secondary).
        queue_balancing_weight: Weight for preventing queue imbalance between lanes.
        duration_penalty_weight: Penalty weight for keeping the same phase too long.

    Returns a scalar reward where lower (more negative) is worse traffic.
    """
    # 1. Delta Queue (Improvement in total congestion)
    def _total_queue(obs: Observation) -> float:
        return float(obs.queue_lengths.sum())

    if prev_observation is not None:
        delta_queue = _total_queue(prev_observation) - _total_queue(observation)
    else:
        delta_queue = 0.0

    # 2. QUEUE IMBALANCE (max - min): The core anti-starvation metric
    # Penalizes when one lane is neglected while another is served
    # Example: if UP-DOWN has 15 vehicles and LEFT-RIGHT has 1, penalty = 14
    # This forces balanced serving across all directions
    queue_imbalance = float(observation.queue_lengths.max() - observation.queue_lengths.min())

    # 3. Phase-Duration Penalty (Safety Rail to prevent stuck phases)
    duration_penalty = duration_penalty_weight * math.log1p(observation.elapsed_green)

    return (
        queue_weight * delta_queue
        - queue_balancing_weight * queue_imbalance
        - duration_penalty
    )
