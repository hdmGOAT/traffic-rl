from __future__ import annotations

from traffic_rl.config import AppConfig
from traffic_rl.envs.base import TrafficEnv
from traffic_rl.envs.cityflow_env import CityFlowTrafficEnv
from traffic_rl.envs.mock_env import MockTrafficEnv


def build_env(cfg: AppConfig) -> TrafficEnv:
    backend = cfg.env.backend.lower()
    if backend == "mock":
        return MockTrafficEnv(cfg.env, seed=cfg.seed)
    if backend == "cityflow":
        return CityFlowTrafficEnv(cfg.env, seed=cfg.seed)
    raise ValueError(f"Unsupported backend '{cfg.env.backend}'.")
