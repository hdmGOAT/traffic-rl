from pathlib import Path

from traffic_rl.config import load_config


def test_load_default_config() -> None:
    cfg = load_config(Path("configs/default.yaml"))
    assert cfg.env.backend == "mock"
    assert cfg.env.num_lanes == 4
    assert cfg.training.episodes > 0
    assert cfg.training.hidden_dim > 0
    assert cfg.training.batch_size > 0


def test_cityflow_config_path_resolves_relative_to_yaml() -> None:
    cfg = load_config(Path("configs/cityflow.quick.yaml"))
    assert cfg.env.cityflow_config_path is not None
    assert Path(cfg.env.cityflow_config_path).is_absolute()
