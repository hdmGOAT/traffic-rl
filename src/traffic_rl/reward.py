from __future__ import annotations

import math

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


def mixed_reward(
    observation: Observation,
    prev_observation: Observation | None = None,
    queue_weight: float = 0.5,
    queue_balancing_weight: float = 0.8,
    duration_penalty_weight: float = 0.1,
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
    # Penalizes when one lane is neglected while another is served.
    # Example: if UP-DOWN has 15 vehicles and LEFT-RIGHT has 1, penalty = 14.
    # The stronger weight makes unfair phase allocation much less attractive.
    queue_imbalance = float(observation.queue_lengths.max() - observation.queue_lengths.min())

    # 3. Phase-Duration Penalty (Safety Rail to prevent stuck phases)
    duration_penalty = duration_penalty_weight * math.log1p(observation.elapsed_green)

    return (
        queue_weight * delta_queue
        - queue_balancing_weight * queue_imbalance
        - duration_penalty
    )


def reward_from_type(
    reward_type: str,
    observation: Observation,
    prev_observation: Observation | None = None,
) -> float:
    """Resolve the active training reward from a configured reward type."""
    reward_type_normalized = reward_type.strip().lower()
    if reward_type_normalized == "queue_length":
        return queue_length_reward(observation)
    if reward_type_normalized == "mixed":
        return mixed_reward(observation, prev_observation=prev_observation)
    raise ValueError(f"Unsupported reward type '{reward_type}'.")
