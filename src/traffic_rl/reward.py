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


def mixed_reward(observation: Observation, queue_weight: float = 0.7, wait_weight: float = 0.3) -> float:
    """Multi-factor reward combining queue length and average wait time.

    Args:
        observation: Current traffic observation.
        queue_weight: Weight for the queue-length term (default 0.7).
        wait_weight: Weight for the average-wait-time term (default 0.3).

    Returns a single scalar reward where lower (more negative) is worse traffic.

    Rationale: queue_length_reward penalizes accumulation, but wait_times captures
    how long vehicles have been stuck. Together they discourage both creating new
    backlogs (queue penalty) and letting vehicles age in place (wait penalty).

    Example:
        - 5 vehicles just arrived (low wait): mostly queue penalty, little wait penalty
        - 5 vehicles stuck for 20s (high wait): moderate queue penalty + large wait penalty
    """
    queue_penalty = -float(observation.queue_lengths.sum())
    # Average wait time across all lanes; penalize proportionally to average wait.
    avg_wait = float(observation.wait_times.mean())
    wait_penalty = -avg_wait  # Negative: penalize long waits
    
    # Normalize weights
    total_weight = queue_weight + wait_weight
    return (queue_weight * queue_penalty + wait_weight * wait_penalty) / total_weight
