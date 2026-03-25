import numpy as np

from traffic_rl.analysis import compare_reward_distributions


def test_compare_reward_distributions_detects_difference() -> None:
    rng = np.random.default_rng(7)
    trained = rng.normal(loc=10.0, scale=1.0, size=80)
    untrained = rng.normal(loc=7.5, scale=1.0, size=80)

    stats = compare_reward_distributions(trained, untrained, seed=7, bootstrap_samples=800, permutation_samples=1200)

    assert stats.mean_diff > 0.0
    assert stats.p_value < 0.05
