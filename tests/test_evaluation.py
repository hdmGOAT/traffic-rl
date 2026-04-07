from __future__ import annotations

import json
from pathlib import Path

import pytest

from traffic_rl.config import load_config
from traffic_rl.evaluation import (
    generate_chart_from_replay,
    resolve_cityflow_file_path,
    resolve_replay_file_path,
)


def test_generate_chart_from_replay_counts_vehicles(tmp_path: Path) -> None:
    replay = tmp_path / "replay.txt"
    replay.write_text(
        "\n".join(
            [
                "1 2 0 veh_1 0 5 2,3 4 0 veh_2 0 5 2;road_1_0_1 g g g",
                "1 2 0 veh_1 0 5 2;road_1_0_1 r r r",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    chart = tmp_path / "chart.txt"

    generate_chart_from_replay(replay, chart, title="Vehicle count")

    assert chart.read_text(encoding="utf-8") == "Vehicle count\n2\n1\n"


def test_generate_chart_from_replay_rejects_invalid_format(tmp_path: Path) -> None:
    replay = tmp_path / "replay.txt"
    replay.write_text("invalid line without separator\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing ';'"):
        generate_chart_from_replay(replay, tmp_path / "chart.txt")


def test_resolve_cityflow_paths_from_engine_dir(tmp_path: Path) -> None:
    cfg = load_config(Path("configs/cityflow.quick.yaml"))
    cityflow_dir = tmp_path / "cityflow_data"
    engine_path = tmp_path / "engine.json"
    engine_path.write_text(
        json.dumps(
            {
                "dir": str(cityflow_dir),
                "replayLogFile": "replay/replay_log.txt",
            }
        ),
        encoding="utf-8",
    )
    cfg.env.cityflow_config_path = str(engine_path)

    resolved_chart = resolve_cityflow_file_path(cfg, "replay/chart.txt")
    resolved_replay = resolve_replay_file_path(cfg, None)

    assert resolved_chart == (cityflow_dir / "replay/chart.txt").resolve()
    assert resolved_replay == (cityflow_dir / "replay/replay_log.txt").resolve()
