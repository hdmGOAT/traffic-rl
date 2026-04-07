# traffic-rl

Reinforcement-learning traffic signal control in CityFlow, with PEMS-04 demand conversion and split-based significance evaluation.

## What Works End-to-End

- DQN training/evaluation (`mock` and `cityflow` backends)
- Checkpointed policy comparison (trained vs untrained)
- Evaluation replay chart generation (`--chart-file`, `--chart-title`)
- HTML/JSON evidence report proving RL effectiveness (`traffic_rl.cli.visualize`)
- PEMS-04 (`PEMS04.npz`) → CityFlow `flow_train/val/test.json`
- Statistical significance reports across temporal splits
- Notebook end-to-end run with split-wise significance table (p-value, 95% CI, Cohen's d)

## Recommended Environment

CityFlow builds reliably here with Python `3.10`.

```bash
cd /path/to/traffic-rl
uv venv --python 3.10 .venv310
uv pip install --python .venv310/bin/python -e '.[dev]'
```

## Install CityFlow (from source)

```bash
cd /path/to/workspace
git clone https://github.com/cityflow-project/CityFlow.git
cd CityFlow

# Required for modern CMake/Python compatibility in this setup
sed -i 's/cmake_minimum_required(VERSION 3.0)/cmake_minimum_required(VERSION 3.5)/' CMakeLists.txt
sed -i 's/cmake_minimum_required(VERSION 2.8.12)/cmake_minimum_required(VERSION 3.5)/g' extern/pybind11/CMakeLists.txt extern/pybind11/tools/pybind11Tools.cmake

uv pip install --python /path/to/traffic-rl/.venv310/bin/python .
```

## Quick Smoke Run

```bash
cd /path/to/traffic-rl
.venv310/bin/python -m traffic_rl.cli.train --config configs/cityflow.quick.yaml
.venv310/bin/python -m traffic_rl.cli.evaluate \
  --config configs/cityflow.quick.yaml \
  --episodes 3 \
  --replay-file replay/quick_eval.txt \
  --chart-file replay/quick_eval_chart.txt \
  --chart-title "Vehicle count"
```

## Notebook End-to-End Run

Run the full pipeline in Jupyter (data prep -> training -> trained vs untrained significance):

1. Open `notebooks/main_data_prep_train_test_eval.ipynb`
2. Run all cells from top to bottom
3. Check Step 6 output table for split-wise metrics:
   - trained/untrained mean reward
   - trained/untrained mean queue
   - p-value, 95% CI, Cohen's d, significance flag
4. Saved report is written to:
   - `outputs/notebook_run/notebook_flow_report.json`

Default notebook quick settings currently use `EVAL_EPISODES=10` and `EVAL_SEEDS=5`
for stronger significance estimates.

## Path Portability (Important)

CityFlow engine JSON files are sensitive to `dir` path format and path validity.

- Avoid machine-specific absolute paths in committed configs when possible.
- If a copied config references a missing machine path, notebook runtime config generation
  now falls back to this repository's `cityflow_data/` directory.
- Generated notebook engine files are stored in `outputs/notebook_run/` and set `dir`
  to a valid local path with an explicit trailing separator.

## Train / Evaluate (Main Protocol)

```bash
cd /path/to/traffic-rl

# Train
.venv310/bin/python -m traffic_rl.cli.train --config configs/cityflow.more_cycles.yaml

# Evaluate trained checkpoint
.venv310/bin/python -m traffic_rl.cli.evaluate \
  --config configs/cityflow.more_cycles.yaml \
  --episodes 10 \
  --checkpoint outputs/agent_checkpoint_cityflow.npz \
  --replay-file replay/eval_trained.txt \
  --chart-file replay/eval_trained_chart.txt \
  --chart-title "Vehicle count"

# Evaluate untrained baseline
.venv310/bin/python -m traffic_rl.cli.evaluate \
  --config configs/cityflow.more_cycles.yaml \
  --episodes 10 \
  --no-checkpoint \
  --replay-file replay/eval_untrained.txt \
  --chart-file replay/eval_untrained_chart.txt \
  --chart-title "Vehicle count"
```

Training now prints a per-episode terminal line with:

- episode reward
- running average reward
- an ASCII trend bar (higher/less-negative reward fills more `#`)

## RL Working Visualization Report

Generate a standalone HTML + JSON report that compares trained vs untrained policy performance, including:

- mean reward difference
- confidence interval / p-value / effect size
- mean queue and mean throughput comparison
- episode reward traces (inline chart)

```bash
cd /path/to/traffic-rl
.venv310/bin/python -m traffic_rl.cli.visualize \
  --config configs/cityflow.more_cycles.yaml \
  --episodes 10 \
  --seeds 5 \
  --checkpoint outputs/agent_checkpoint_cityflow.npz \
  --html-file outputs/rl_working_cityflow_strong.html \
  --json-file outputs/rl_working_cityflow_strong.json
```

Outputs:

- `outputs/rl_working_cityflow_strong.html`
- `outputs/rl_working_cityflow_strong.json`

## PEMS-04 Demand Conversion

Convert `Pems_Dataset/PEMS04/PEMS04.npz` into split demand files:

```bash
cd /path/to/traffic-rl
.venv310/bin/python -m traffic_rl.cli.pems_build \
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
cd /path/to/traffic-rl
.venv310/bin/python -m traffic_rl.cli.compare \
  --config configs/cityflow.more_cycles.yaml \
  --episodes 15 \
  --seeds 5 \
  --checkpoint outputs/agent_checkpoint_cityflow.npz \
  --report-file outputs/compare_report.json
```

### PEMS Train/Val/Test Split Generalization

```bash
cd /path/to/traffic-rl
.venv310/bin/python -m traffic_rl.cli.compare_splits \
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
cd /path/to/CityFlow/frontend
python -m http.server 8080
```

Open `http://localhost:8080/index.html` and load:

- roadnet log: `cityflow_data/replay/roadnet_log.json`
- replay log: one of `cityflow_data/replay/*.txt`
- chart file (optional): one of `cityflow_data/replay/*_chart.txt`

## Findings

Latest recorded findings are in:

- `findings/pems_cityflow_split_significance.md`

## Troubleshooting

- If progress looks stalled at `0/3` in split comparison, watch the per-episode bars (`train seed=... trained/untrained`) — each split can take minutes.
- If CityFlow cannot find roads from generated flows, regenerate demands with `configs/pems04_to_cityflow.example.yaml` and ensure route IDs match `cityflow_data/roadnet.json`.
- If the notebook kernel crashes when training starts, inspect `outputs/notebook_run/cityflow_engine_train.json` and verify:
  - `dir` exists locally and points to this repo's `cityflow_data/`
  - `roadnetFile` and `flowFile` resolve to existing files from that `dir`
- If `pip` is unavailable in `.venv310`, use `uv pip --python .venv310/bin/python ...`.
- If replay logs fail with `write roadnet log file error`, create `cityflow_data/replay/` first.
- If newly added CLI flags are missing (for example `--chart-file` or `traffic_rl.cli.visualize`), reinstall editable package:
  `.venv310/bin/python -m pip install -e .`