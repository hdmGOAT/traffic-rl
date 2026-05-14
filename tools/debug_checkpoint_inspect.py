from pathlib import Path
import numpy as np

REPO_ROOT = Path.cwd()
print('cwd:', REPO_ROOT)

# Find checkpoint files in the repo
ck_files = list(REPO_ROOT.rglob('agent_checkpoint*.npz'))
print('found checkpoint files:')
for p in ck_files:
    print('-', p)

for p in ck_files:
    print('\n----', p)
    try:
        data = np.load(p, allow_pickle=True)
        print('keys:', data.files)
        if 'epsilon' in data.files:
            try:
                print('epsilon:', float(data['epsilon'][0]))
            except Exception:
                print('epsilon: (could not parse)')
        if 'online_w1' in data.files:
            print('online_w1 shape:', data['online_w1'].shape)
        if 'keys' in data.files and 'values' in data.files:
            print('tabular checkpoint detected (keys/values)')
    except Exception as e:
        print('error reading', p, e)

print('\nDone')
