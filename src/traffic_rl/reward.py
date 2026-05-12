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


def mixed_reward(
    observation: Observation,
    queue_weight: float = 0.3,
    wait_weight: float = 0.4,
    starvation_weight: float = 0.3,
) -> float:
    """Multi-factor reward combining queue length, average wait time, and starvation.

    Args:
        observation: Current traffic observation.
        queue_weight: Weight for the queue-length term.
        wait_weight: Weight for the average-wait-time term.
        starvation_weight: Weight for the max-wait term that penalizes a lane
            being starved while another phase stays green.

    Returns a single scalar reward where lower (more negative) is worse traffic.

    Rationale: queue_length_reward penalizes accumulation, but wait_times captures
    how long vehicles have been stuck. The max-wait term adds an explicit starvation
    signal so one lane cannot be neglected just because the total queue looks fine.

    Example:
        - 5 vehicles just arrived (low wait): mostly queue penalty, little wait penalty
        - 5 vehicles stuck for 20s (high wait): queue penalty + average wait penalty +
          strong starvation penalty
    """
    queue_penalty = -float(observation.queue_lengths.sum())
    # Average wait time across all lanes; penalize proportionally to average wait.
    avg_wait = float(observation.wait_times.mean())
    wait_penalty = -avg_wait  # Negative: penalize long waits
    max_wait = float(observation.wait_times.max()) if observation.wait_times.size else 0.0
    starvation_penalty = -max_wait  # Negative: penalize a single lane being starved

    # Normalize weights.
    total_weight = queue_weight + wait_weight + starvation_weight
    return (
        queue_weight * queue_penalty
        + wait_weight * wait_penalty
        + starvation_weight * starvation_penalty
    ) / total_weight
