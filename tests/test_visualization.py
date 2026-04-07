from __future__ import annotations

from pathlib import Path

from traffic_rl.visualization import write_rl_working_report_html


def test_write_rl_working_report_html_contains_key_metrics(tmp_path: Path) -> None:
    report = {
        "title": "Is RL Working? Trained vs Untrained",
        "trained_mean_reward": -100.0,
        "untrained_mean_reward": -150.0,
        "mean_diff": 50.0,
        "ci95": [30.0, 70.0],
        "p_value": 0.01,
        "cohen_d": 1.2,
        "is_significant_0_05": True,
        "trained_mean_queue": 1.2,
        "untrained_mean_queue": 2.8,
        "trained_mean_throughput": 170.0,
        "untrained_mean_throughput": 120.0,
        "trained_rewards": [-90.0, -110.0, -100.0],
        "untrained_rewards": [-150.0, -155.0, -145.0],
    }
    output = tmp_path / "rl_report.html"

    write_rl_working_report_html(report, output)

    html = output.read_text(encoding="utf-8")
    assert "RL is working: trained policy outperforms untrained baseline." in html
    assert "Episode reward traces" in html
    assert "polyline" in html
    assert "&quot;trained_mean_reward&quot;" in html
