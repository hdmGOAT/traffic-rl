from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class ComparisonStats:
    trained_mean: float
    untrained_mean: float
    mean_diff: float
    ci_low: float
    ci_high: float
    p_value: float
    cohen_d: float


def compare_reward_distributions(
    trained_rewards: np.ndarray,
    untrained_rewards: np.ndarray,
    *,
    seed: int = 7,
    bootstrap_samples: int = 2000,
    permutation_samples: int = 5000,
) -> ComparisonStats:
    trained = np.asarray(trained_rewards, dtype=np.float64)
    untrained = np.asarray(untrained_rewards, dtype=np.float64)
    if trained.size == 0 or untrained.size == 0:
        raise ValueError("trained_rewards and untrained_rewards must be non-empty.")

    rng = np.random.default_rng(seed)

    trained_mean = float(trained.mean())
    untrained_mean = float(untrained.mean())
    mean_diff = trained_mean - untrained_mean

    ci_low, ci_high = _bootstrap_ci_mean_diff(trained, untrained, rng, bootstrap_samples)
    p_value = _permutation_p_value(trained, untrained, rng, permutation_samples)
    cohen_d = _cohen_d(trained, untrained)

    return ComparisonStats(
        trained_mean=trained_mean,
        untrained_mean=untrained_mean,
        mean_diff=float(mean_diff),
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        p_value=float(p_value),
        cohen_d=float(cohen_d),
    )


def _bootstrap_ci_mean_diff(
    trained: np.ndarray,
    untrained: np.ndarray,
    rng: np.random.Generator,
    num_samples: int,
) -> tuple[float, float]:
    diffs = np.empty(num_samples, dtype=np.float64)
    for i in range(num_samples):
        t_sample = rng.choice(trained, size=trained.size, replace=True)
        u_sample = rng.choice(untrained, size=untrained.size, replace=True)
        diffs[i] = t_sample.mean() - u_sample.mean()
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def _permutation_p_value(
    trained: np.ndarray,
    untrained: np.ndarray,
    rng: np.random.Generator,
    num_samples: int,
) -> float:
    observed = abs(trained.mean() - untrained.mean())
    combined = np.concatenate([trained, untrained])
    n_train = trained.size

    extreme = 0
    for _ in range(num_samples):
        permuted = rng.permutation(combined)
        diff = abs(permuted[:n_train].mean() - permuted[n_train:].mean())
        if diff >= observed:
            extreme += 1
    return (extreme + 1) / (num_samples + 1)


def _cohen_d(trained: np.ndarray, untrained: np.ndarray) -> float:
    n1 = trained.size
    n2 = untrained.size
    if n1 < 2 or n2 < 2:
        return 0.0

    v1 = trained.var(ddof=1)
    v2 = untrained.var(ddof=1)
    pooled_var = ((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2)
    if pooled_var <= 0:
        return 0.0
    return float((trained.mean() - untrained.mean()) / np.sqrt(pooled_var))
