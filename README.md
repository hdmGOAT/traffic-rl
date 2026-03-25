# traffic-rl

Reinforcement-learning traffic signal control in CityFlow, with PEMS-04 demand conversion and split-based significance evaluation.

## What Works End-to-End

- DQN training/evaluation (`mock` and `cityflow` backends)
- Checkpointed policy comparison (trained vs untrained)
- PEMS-04 (`PEMS04.npz`) → CityFlow `flow_train/val/test.json`
- Statistical significance reports across temporal splits

## Recommended Environment

CityFlow builds reliably here with Python `3.10`.

```bash
cd /home/hd/projects/traffic-rl
uv venv --python 3.10 .venv310
uv pip install --python .venv310/bin/python -e '.[dev]'
```

## Install CityFlow (from source)

```bash
cd /home/hd/projects
git clone https://github.com/cityflow-project/CityFlow.git
cd CityFlow

# Required for modern CMake/Python compatibility in this setup
sed -i 's/cmake_minimum_required(VERSION 3.0)/cmake_minimum_required(VERSION 3.5)/' CMakeLists.txt
sed -i 's/cmake_minimum_required(VERSION 2.8.12)/cmake_minimum_required(VERSION 3.5)/g' extern/pybind11/CMakeLists.txt extern/pybind11/tools/pybind11Tools.cmake

uv pip install --python /home/hd/projects/traffic-rl/.venv310/bin/python .
```

## Quick Smoke Run

```bash
cd /home/hd/projects/traffic-rl
/home/hd/projects/traffic-rl/.venv310/bin/python -m traffic_rl.cli.train --config configs/cityflow.quick.yaml
/home/hd/projects/traffic-rl/.venv310/bin/python -m traffic_rl.cli.evaluate --config configs/cityflow.quick.yaml --episodes 3
```

## Train / Evaluate (Main Protocol)

```bash
cd /home/hd/projects/traffic-rl

# Train
/home/hd/projects/traffic-rl/.venv310/bin/python -m traffic_rl.cli.train --config configs/cityflow.more_cycles.yaml

# Evaluate trained checkpoint
/home/hd/projects/traffic-rl/.venv310/bin/python -m traffic_rl.cli.evaluate \
  --config configs/cityflow.more_cycles.yaml \
  --episodes 10 \
  --checkpoint outputs/agent_checkpoint_cityflow.npz \
  --replay-file replay/eval_trained.txt

# Evaluate untrained baseline
/home/hd/projects/traffic-rl/.venv310/bin/python -m traffic_rl.cli.evaluate \
  --config configs/cityflow.more_cycles.yaml \
  --episodes 10 \
  --no-checkpoint \
  --replay-file replay/eval_untrained.txt
```

## PEMS-04 Demand Conversion

Convert `Pems_Dataset/PEMS04/PEMS04.npz` into split demand files:

```bash
cd /home/hd/projects/traffic-rl
/home/hd/projects/traffic-rl/.venv310/bin/python -m traffic_rl.cli.pems_build \
  --config configs/pems04_to_cityflow.example.yaml
```

Generated outputs:

- `outputs/pems04/flow_train.json`
- `outputs/pems04/flow_val.json`
- `outputs/pems04/flow_test.json`
- `outputs/pems04/summary.json`

## Statistical Comparison

### Single Demand Setup

```bash
cd /home/hd/projects/traffic-rl
/home/hd/projects/traffic-rl/.venv310/bin/python -m traffic_rl.cli.compare \
  --config configs/cityflow.more_cycles.yaml \
  --episodes 15 \
  --seeds 5 \
  --checkpoint outputs/agent_checkpoint_cityflow.npz \
  --report-file outputs/compare_report.json
```

### PEMS Train/Val/Test Split Generalization

```bash
cd /home/hd/projects/traffic-rl
/home/hd/projects/traffic-rl/.venv310/bin/python -m traffic_rl.cli.compare_splits \
  --config configs/cityflow.more_cycles.yaml \
  --flow-train outputs/pems04/flow_train.json \
  --flow-val outputs/pems04/flow_val.json \
  --flow-test outputs/pems04/flow_test.json \
  --episodes 10 \
  --seeds 5 \
  --checkpoint outputs/agent_checkpoint_cityflow.npz \
  --report-dir outputs/compare_splits_big
```

This writes:

- `outputs/compare_splits_big/report_train.json`
- `outputs/compare_splits_big/report_val.json`
- `outputs/compare_splits_big/report_test.json`
- `outputs/compare_splits_big/aggregate_report.json`

## Replay Viewer

```bash
cd /home/hd/projects/CityFlow/frontend
python -m http.server 8080
```

Open `http://localhost:8080/index.html` and load:

- roadnet log: `cityflow_data/replay/roadnet_log.json`
- replay log: one of `cityflow_data/replay/*.txt`

## Findings

Latest recorded findings are in:

- `findings/pems_cityflow_split_significance.md`

## Troubleshooting

- If progress looks stalled at `0/3` in split comparison, watch the per-episode bars (`train seed=... trained/untrained`) — each split can take minutes.
- If CityFlow cannot find roads from generated flows, regenerate demands with `configs/pems04_to_cityflow.example.yaml` and ensure route IDs match `cityflow_data/roadnet.json`.
- If `pip` is unavailable in `.venv310`, use `uv pip --python /home/hd/projects/traffic-rl/.venv310/bin/python ...`.