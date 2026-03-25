from pathlib import Path

import json
import numpy as np

from traffic_rl.pems.pipeline import PemsDemandConfig, SplitConfig, build_cityflow_demands


def test_build_cityflow_demands(tmp_path: Path) -> None:
    data = np.zeros((24, 4, 3), dtype=np.float32)
    data[:, :, 0] = 1.0
    npz_path = tmp_path / "pems.npz"
    np.savez(npz_path, data=data)

    cfg = PemsDemandConfig(
        pems_npz_path=str(npz_path),
        output_dir=str(tmp_path / "out"),
        sampling_interval_sec=3600,
        flow_feature_index=0,
        sensor_indices=[0, 1],
        route_catalog={"r1": ["a", "b"], "r2": ["c", "d"]},
        sensor_to_route_probabilities={
            0: {"r1": 1.0},
            1: {"r2": 1.0},
        },
        split=SplitConfig(train_days=1, val_days=0, test_days=0),
        arrival_process="uniform",
        random_seed=7,
    )

    outputs = build_cityflow_demands(cfg)
    train_entries = json.loads(outputs.train_flow_file.read_text(encoding="utf-8"))

    assert outputs.train_flow_file.exists()
    assert outputs.summary_file.exists()
    assert len(train_entries) > 0
    assert "route" in train_entries[0]
    assert "startTime" in train_entries[0]
