from __future__ import annotations

from traffic_rl.config import AppConfig
from traffic_rl.envs.base import TrafficEnv
from traffic_rl.envs.cityflow_env import CityFlowTrafficEnv
from traffic_rl.envs.mock_env import MockTrafficEnv


def build_env(cfg: AppConfig) -> TrafficEnv:
    """Instantiate the correct traffic environment based on the config's backend setting.

    'mock'     — fast, dependency-free synthetic environment used for unit tests
                 and quick smoke-runs. No CityFlow installation required.
    'cityflow' — full microsimulation using the CityFlow engine and real road
                 network / flow files. Required for any meaningful RL training.

    Raises ValueError for any unrecognised backend name so misconfigured YAML
    files fail loudly rather than silently using the wrong environment.
    """
    backend = cfg.env.backend.lower()

    if backend == "mock":
        return MockTrafficEnv(cfg.env, seed=cfg.seed, reward_type=cfg.reward.type)

    if backend == "cityflow":
        return CityFlowTrafficEnv(cfg.env, seed=cfg.seed, reward_type=cfg.reward.type)

    raise ValueError(f"Unsupported backend '{cfg.env.backend}'.")
