from __future__ import annotations

from html import escape
import json
from pathlib import Path


def write_rl_working_report_html(report: dict, output_file: str | Path) -> Path:
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    html = _build_report_html(report)
    output_path.write_text(html, encoding="utf-8")
    return output_path


def _build_report_html(report: dict) -> str:
    title = escape(str(report.get("title", "RL Comparison Report")))
    trained_rewards = [float(value) for value in report.get("trained_rewards", [])]
    untrained_rewards = [float(value) for value in report.get("untrained_rewards", [])]
    trained_mean = float(report.get("trained_mean_reward", 0.0))
    untrained_mean = float(report.get("untrained_mean_reward", 0.0))
    mean_diff = float(report.get("mean_diff", 0.0))
    ci = report.get("ci95", [0.0, 0.0])
    ci_low = float(ci[0]) if len(ci) > 0 else 0.0
    ci_high = float(ci[1]) if len(ci) > 1 else 0.0
    p_value = float(report.get("p_value", 1.0))
    cohen_d = float(report.get("cohen_d", 0.0))
    trained_queue = float(report.get("trained_mean_queue", 0.0))
    untrained_queue = float(report.get("untrained_mean_queue", 0.0))
    trained_throughput = float(report.get("trained_mean_throughput", 0.0))
    untrained_throughput = float(report.get("untrained_mean_throughput", 0.0))
    significant = bool(report.get("is_significant_0_05", False))

    verdict = (
        "RL is working: trained policy outperforms untrained baseline."
        if significant and mean_diff > 0
        else "No strong evidence yet: trained policy is not clearly better."
    )
    verdict_color = "#0f8a32" if significant and mean_diff > 0 else "#a4370a"

    reward_svg = _build_reward_svg(trained_rewards, untrained_rewards)
    report_json = json.dumps(report, indent=2)

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 24px; }}
    h1 {{ margin-bottom: 8px; }}
    .meta {{ color: #444; margin-bottom: 16px; }}
    .cards {{ display: grid; grid-template-columns: repeat(3, minmax(180px, 1fr)); gap: 12px; }}
    .card {{ border: 1px solid #ddd; border-radius: 8px; padding: 12px; }}
    .k {{ color: #555; font-size: 13px; }}
    .v {{ font-size: 20px; font-weight: 600; }}
    .verdict {{ margin-top: 14px; padding: 10px; border-radius: 8px; color: white;
      background: {verdict_color}; }}
    .legend {{ margin: 8px 0 0; font-size: 13px; color: #333; }}
    .dot {{ display: inline-block; width: 10px; height: 10px; border-radius: 50%;
      margin-right: 6px; }}
    pre {{ background: #f8f8f8; border: 1px solid #eee; border-radius: 8px;
      padding: 10px; overflow: auto; }}
  </style>
</head>
<body>
  <h1>{title}</h1>
  <div class="meta">Automatic RL evidence report from trained vs untrained evaluation.</div>
  <div class="cards">
    <div class="card"><div class="k">Trained mean reward</div><div class="v">{trained_mean:.3f}</div></div>
    <div class="card"><div class="k">Untrained mean reward</div><div class="v">{untrained_mean:.3f}</div></div>
    <div class="card"><div class="k">Mean diff (trained-untrained)</div><div class="v">{mean_diff:.3f}</div></div>
    <div class="card"><div class="k">95% CI</div><div class="v">[{ci_low:.3f}, {ci_high:.3f}]</div></div>
    <div class="card"><div class="k">p-value / Cohen's d</div><div class="v">{p_value:.5f} / {cohen_d:.3f}</div></div>
    <div class="card"><div class="k">Significant @ 0.05</div><div class="v">{str(significant)}</div></div>
    <div class="card"><div class="k">Trained mean queue</div><div class="v">{trained_queue:.3f}</div></div>
    <div class="card"><div class="k">Untrained mean queue</div><div class="v">{untrained_queue:.3f}</div></div>
    <div class="card"><div class="k">Throughput (trained/untrained)</div><div class="v">{trained_throughput:.1f} / {untrained_throughput:.1f}</div></div>
  </div>
  <div class="verdict">{escape(verdict)}</div>

  <h2>Episode reward traces</h2>
  <svg viewBox="0 0 960 320" width="100%" height="320" role="img" aria-label="reward comparison">
    {reward_svg}
  </svg>
  <div class="legend">
    <span class="dot" style="background:#0a66c2;"></span>trained
    &nbsp;&nbsp;
    <span class="dot" style="background:#c2380a;"></span>untrained
  </div>

  <h2>Raw report JSON</h2>
  <pre>{escape(report_json)}</pre>
</body>
</html>
"""


def _build_reward_svg(trained: list[float], untrained: list[float]) -> str:
    width = 960.0
    height = 320.0
    pad_left = 50.0
    pad_right = 20.0
    pad_top = 20.0
    pad_bottom = 30.0

    values = trained + untrained
    if not values:
        return ""

    min_val = min(values)
    max_val = max(values)
    if max_val == min_val:
        max_val = min_val + 1.0

    plot_w = width - pad_left - pad_right
    plot_h = height - pad_top - pad_bottom

    def points(series: list[float]) -> str:
        if not series:
            return ""
        if len(series) == 1:
            x = pad_left + (plot_w * 0.5)
            y = pad_top + (max_val - series[0]) * plot_h / (max_val - min_val)
            return f"{x:.1f},{y:.1f}"
        parts: list[str] = []
        for idx, value in enumerate(series):
            x = pad_left + idx * plot_w / (len(series) - 1)
            y = pad_top + (max_val - value) * plot_h / (max_val - min_val)
            parts.append(f"{x:.1f},{y:.1f}")
        return " ".join(parts)

    trained_points = points(trained)
    untrained_points = points(untrained)

    axis = (
        f'<line x1="{pad_left}" y1="{pad_top}" x2="{pad_left}" '
        f'y2="{height - pad_bottom}" stroke="#888" stroke-width="1" />'
        f'<line x1="{pad_left}" y1="{height - pad_bottom}" x2="{width - pad_right}" '
        f'y2="{height - pad_bottom}" stroke="#888" stroke-width="1" />'
    )
    labels = (
        f'<text x="{pad_left}" y="{pad_top - 4}" fill="#444" font-size="11">{max_val:.1f}</text>'
        f'<text x="{pad_left}" y="{height - pad_bottom + 14}" fill="#444" font-size="11">{min_val:.1f}</text>'
    )
    trained_line = (
        f'<polyline fill="none" stroke="#0a66c2" stroke-width="2" points="{trained_points}" />'
        if trained_points
        else ""
    )
    untrained_line = (
        f'<polyline fill="none" stroke="#c2380a" stroke-width="2" points="{untrained_points}" />'
        if untrained_points
        else ""
    )
    return axis + labels + trained_line + untrained_line
