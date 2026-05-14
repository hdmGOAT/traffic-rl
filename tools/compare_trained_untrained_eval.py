from pathlib import Path
import sys

REPO_ROOT = Path.cwd()
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from traffic_rl.config import load_config
from traffic_rl.evaluation import run_evaluation, _resolve_checkpoint
import numpy as np

cfg_path = REPO_ROOT / 'traffic-rl' / 'outputs' / 'notebook_run' / 'cityflow_train.yaml'
print('using cfg:', cfg_path)
cfg = load_config(cfg_path)
cfg.output_dir = str(REPO_ROOT / 'traffic-rl' / 'outputs' / 'notebook_run')

print('running trained evaluation...')
trained = run_evaluation(cfg, episodes=5, load_checkpoint=True, show_progress=False)
print('trained average reward, avg_queue:', trained.average_reward, trained.average_queue)

print('running untrained evaluation...')
untrained = run_evaluation(cfg, episodes=5, load_checkpoint=False, show_progress=False)
print('untrained average reward, avg_queue:', untrained.average_reward, untrained.average_queue)

print('\ntrained episode rewards:', trained.episode_rewards)
print('untrained episode rewards:', untrained.episode_rewards)
