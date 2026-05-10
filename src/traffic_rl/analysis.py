from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class ComparisonStats:
    """Statistical summary comparing two sets of episode rewards (e.g. trained vs baseline).

    All fields are computed from bootstrap resampling and a permutation test
    rather than assuming a particular distribution, which is important because
    RL reward distributions are often skewed and non-normal.
    """

    trained_mean: float    # Mean reward of the trained agent.
    untrained_mean: float  # Mean reward of the baseline agent.
    mean_diff: float       # trained_mean - untrained_mean (positive = trained wins).
    ci_low: float          # Lower bound of the 95% bootstrap confidence interval on mean_diff.
    ci_high: float         # Upper bound of the 95% bootstrap confidence interval on mean_diff.
    p_value: float         # Permutation test p-value: probability of observing this difference
                           # by chance if both agents were actually the same (lower = more significant).
    cohen_d: float         # Effect size: how many standard deviations apart the two means are.
                           # Rule of thumb: 0.2 small, 0.5 medium, 0.8 large.


def compare_reward_distributions(
    trained_rewards: np.ndarray,
    untrained_rewards: np.ndarray,
    *,
    seed: int = 7,
    bootstrap_samples: int = 2000,
    permutation_samples: int = 5000,
) -> ComparisonStats:
    """Run a full statistical comparison between two sets of episode rewards.

    Uses bootstrap resampling for confidence intervals and a permutation test
    for the p-value — both distribution-free methods that work well with small,
    skewed RL reward samples.

    Args:
        trained_rewards:    Episode rewards from the trained agent.
        untrained_rewards:  Episode rewards from the baseline agent.
        seed:               Random seed for reproducibility.
        bootstrap_samples:  Number of bootstrap iterations for the CI.
        permutation_samples: Number of permutation iterations for the p-value.
    """
    trained   = np.asarray(trained_rewards,   dtype=np.float64)
    untrained = np.asarray(untrained_rewards, dtype=np.float64)
    if trained.size == 0 or untrained.size == 0:
        raise ValueError("trained_rewards and untrained_rewards must be non-empty.")

    rng = np.random.default_rng(seed)

    trained_mean   = float(trained.mean())
    untrained_mean = float(untrained.mean())
    mean_diff      = trained_mean - untrained_mean

    ci_low, ci_high = _bootstrap_ci_mean_diff(trained, untrained, rng, bootstrap_samples)
    p_value         = _permutation_p_value(trained, untrained, rng, permutation_samples)
    cohen_d_val     = _cohen_d(trained, untrained)

    return ComparisonStats(
        trained_mean=trained_mean,
        untrained_mean=untrained_mean,
        mean_diff=float(mean_diff),
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        p_value=float(p_value),
        cohen_d=float(cohen_d_val),
    )


def _bootstrap_ci_mean_diff(
    trained: np.ndarray,
    untrained: np.ndarray,
    rng: np.random.Generator,
    num_samples: int,
) -> tuple[float, float]:
    """Estimate the 95% confidence interval of (trained_mean - untrained_mean) via bootstrap.

    For each iteration: resample both arrays with replacement, compute the
    difference in means. The 2.5th and 97.5th percentiles of those differences
    form the 95% CI. A CI that doesn't include zero means the difference is
    statistically significant at the 0.05 level.
    """
    diffs = np.empty(num_samples, dtype=np.float64)
    for i in range(num_samples):
        t_sample = rng.choice(trained,   size=trained.size,   replace=True)
        u_sample = rng.choice(untrained, size=untrained.size, replace=True)
        diffs[i] = t_sample.mean() - u_sample.mean()
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def _permutation_p_value(
    trained: np.ndarray,
    untrained: np.ndarray,
    rng: np.random.Generator,
    num_samples: int,
) -> float:
    """Compute a permutation test p-value for the difference in means.

    Null hypothesis: both groups are drawn from the same distribution.
    We measure how often a random permutation of all rewards produces a
    difference at least as large as what we actually observed. A small p-value
    means the observed gap is unlikely to have occurred by chance.

    Uses a +1 smoothing (Laplace correction) to avoid p=0 with finite samples.
    """
    observed = abs(trained.mean() - untrained.mean())
    combined = np.concatenate([trained, untrained])
    n_train  = trained.size

    # Count how many permutations yield a difference ≥ observed.
    extreme = 0
    for _ in range(num_samples):
        permuted = rng.permutation(combined)
        diff = abs(permuted[:n_train].mean() - permuted[n_train:].mean())
        if diff >= observed:
            extreme += 1

    # +1 in numerator and denominator is the Laplace smoothing correction.
    return (extreme + 1) / (num_samples + 1)


def _cohen_d(trained: np.ndarray, untrained: np.ndarray) -> float:
    """Compute Cohen's d: the standardised effect size of the difference in means.

    d = (mean_trained - mean_untrained) / pooled_std

    Pooled standard deviation weights each group's variance by its sample size.
    Returns 0.0 if either group has fewer than 2 samples (variance undefined).
    """
    n1, n2 = trained.size, untrained.size
    if n1 < 2 or n2 < 2:
        return 0.0

    v1 = trained.var(ddof=1)
    v2 = untrained.var(ddof=1)
    # Pooled variance is the weighted average of both sample variances.
    pooled_var = ((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2)
    if pooled_var <= 0:
        return 0.0
    return float((trained.mean() - untrained.mean()) / np.sqrt(pooled_var))
