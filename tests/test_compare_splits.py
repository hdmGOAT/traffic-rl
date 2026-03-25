from pathlib import Path

from traffic_rl.cli.compare_splits import _resolve_from_yaml, _to_engine_relative


def test_to_engine_relative_builds_relpath(tmp_path: Path) -> None:
    engine_dir = tmp_path / "cityflow_data"
    flow_path = tmp_path / "outputs" / "pems" / "flow_train.json"
    engine_dir.mkdir(parents=True)
    flow_path.parent.mkdir(parents=True)
    flow_path.write_text("[]", encoding="utf-8")

    rel = _to_engine_relative(engine_dir, flow_path)
    assert rel == "../outputs/pems/flow_train.json"


def test_resolve_from_yaml_handles_relative_path(tmp_path: Path) -> None:
    cfg = tmp_path / "configs" / "cityflow.yaml"
    cfg.parent.mkdir(parents=True)
    cfg.write_text("env: {}", encoding="utf-8")

    resolved = _resolve_from_yaml(cfg, "./cityflow_engine.json")
    assert resolved == str((cfg.parent / "cityflow_engine.json").resolve())
