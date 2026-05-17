#!/usr/bin/env python3
"""Standalone script to generate notebook figures from outputs/notebook_run/notebook_flow_report.json
Creates PNGs under outputs/notebook_run/figures
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO_ROOT = Path.cwd()
report_path = REPO_ROOT / 'outputs' / 'notebook_run' / 'notebook_flow_report.json'
figures_dir = REPO_ROOT / 'outputs' / 'notebook_run' / 'figures'
figures_dir.mkdir(parents=True, exist_ok=True)

if not report_path.exists():
    raise FileNotFoundError(f'Expected report at {report_path}; run the evaluation cell in the notebook first.')

report = json.loads(report_path.read_text(encoding='utf-8'))

# 1) Training episode rewards plot
train_summary = report.get('training', {})
episode_rewards = np.asarray(train_summary.get('episode_rewards', []), dtype=np.float64)
plt.figure(figsize=(10,4))
if episode_rewards.size > 0:
    rolling = np.convolve(episode_rewards, np.ones(5)/5, mode='same')
    plt.plot(episode_rewards, marker='o', alpha=0.6, label='Episode reward')
    plt.plot(rolling, color='C1', lw=2, label='Rolling mean (5)')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Training Episode Rewards')
    plt.grid(alpha=0.3)
    plt.legend()
    out1 = figures_dir / 'training_episode_rewards.png'
    plt.tight_layout()
    plt.savefig(out1, dpi=150)
    plt.close()
else:
    print('No episode rewards found in report; skipping training reward plot')

# 2) Evaluation comparison: trained vs untrained per split with 95% CI
eval_rows = report.get('evaluation', [])
if eval_rows:
    df = pd.DataFrame(eval_rows)
    plt.figure(figsize=(8,4))
    x = np.arange(len(df))
    width = 0.35
    plt.bar(x - width/2, df['trained_avg_reward'], width, yerr=(df['ci95_high'] - df['ci95_low'])/2, capsize=4, label='Trained')
    plt.bar(x + width/2, df['untrained_avg_reward'], width, label='Untrained')
    plt.xticks(x, df['split'])
    plt.ylabel('Average Reward')
    plt.title('Trained vs Untrained: Average Reward by Split (95% CI shown)')
    plt.legend()
    out2 = figures_dir / 'eval_rewards_comparison.png'
    plt.tight_layout()
    plt.savefig(out2, dpi=150)
    plt.close()
else:
    print('No evaluation rows found in report; skipping eval comparison plot')

# 3) Scatter: delta vs trained average queue (useful to see reward-queue tradeoff)
if eval_rows:
    plt.figure(figsize=(6,4))
    plt.scatter(df['delta'], df['trained_avg_queue'], s=80, c='C2')
    for i, r in df.iterrows():
        plt.text(r['delta'], r['trained_avg_queue'], r['split'], fontsize=9, va='bottom', ha='right')
    plt.axvline(0, color='k', linewidth=0.6, linestyle='--')
    plt.xlabel('Reward Delta (Trained - Untrained)')
    plt.ylabel('Trained Average Queue')
    plt.title('Reward Delta vs Trained Avg Queue by Split')
    out3 = figures_dir / 'reward_delta_vs_queue.png'
    plt.tight_layout()
    plt.savefig(out3, dpi=150)
    plt.close()

print(f'Figures written to: {figures_dir}')
print('Files:')
for p in sorted(figures_dir.glob('*.png')):
    print(' -', p)
