from pathlib import Path
import sys

REPO_ROOT = Path.cwd()
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from traffic_rl.config import load_config
from traffic_rl.evaluation import _resolve_checkpoint
from traffic_rl.envs.factory import build_env
from traffic_rl.agents.factory import build_agent
import numpy as np

cfg_path = REPO_ROOT / "traffic-rl" / "outputs" / "notebook_run" / "cityflow_train.yaml"
print('cfg_path exists:', cfg_path.exists())
if not cfg_path.exists():
    # fallback: try to find any cityflow_*.yaml under outputs/notebook_run
    candidates = list((REPO_ROOT / 'traffic-rl' / 'outputs' / 'notebook_run').glob('cityflow_*.yaml'))
    cfg_path = candidates[0] if candidates else None
    print('fallback cfg_path:', cfg_path)

if cfg_path is None:
    print('No per-split engine config found; aborting')
    raise SystemExit(1)

cfg = load_config(cfg_path)
# ensure output_dir matches where the checkpoint was saved in notebook
cfg.output_dir = str(REPO_ROOT / 'traffic-rl' / 'outputs' / 'notebook_run')
print('cfg.env.backend:', cfg.env.backend)
print('cfg.output_dir:', cfg.output_dir)

resolved = _resolve_checkpoint(cfg, None)
print('_resolve_checkpoint ->', resolved)

if resolved is None:
    print('No checkpoint found for this config')
    raise SystemExit(1)

data = np.load(resolved, allow_pickle=True)
print('checkpoint keys:', data.files)
if 'epsilon' in data.files:
    print('epsilon:', float(data['epsilon'][0]))

# Try loading into an agent and get greedy action/Qs
env = build_env(cfg)
agent = build_agent(cfg, env.action_size)
try:
    agent.load(resolved)
    obs = env.reset()
    s = obs.as_vector().astype('float32')
    print('greedy action:', agent.act(s, train=False))
    if hasattr(agent, 'online_net') and agent.online_net is not None:
        print('Q-values:', agent.online_net.predict(s.reshape(1, -1)))
except Exception as e:
    print('error during agent load/action:', e)
