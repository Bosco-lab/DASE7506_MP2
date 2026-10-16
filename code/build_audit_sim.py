"""Build a synthetic 'audit' bank to check the frozen method generalises.

The instructor's audit bank has no references. This simulation copies a subset of
validation candidates into a new data directory, with shuffled question order, new
question IDs and a question count that shares no factor with the supplied splits,
so any dependence on IDs, order or sample count shows up as a failure.

Usage (from code/): python build_audit_sim.py --n 137 --output runs/audit_sim
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from common import PROTOCOL, load_inputs

parser = argparse.ArgumentParser()
parser.add_argument('--n', type=int, default=137)
parser.add_argument('--output', type=Path, default=Path('runs/audit_sim'))
args = parser.parse_args()

bank, arrays = load_inputs('validation')
rng = np.random.RandomState(7)
idx = rng.choice(len(bank), size=min(args.n, len(bank)), replace=False)

out = args.output
(out/'candidates').mkdir(parents=True, exist_ok=True)
rows, feats = [], []
for k, i in enumerate(idx):
    row = {'id': f'audit:{k:04d}', 'question': bank[i]['question'],
           'candidates': [{'text': c['text'], 'code': None, 'index': j}
                          for j, c in enumerate(bank[i]['candidates'])]}
    rows.append(row)
    feats.append(arrays['embeddings'][i])
feats = np.stack(feats).reshape(-1, 1024).astype(np.float32)

jsonl = out/'candidates/gsm8k_audit.jsonl'
jsonl.write_text(''.join(json.dumps(r)+'\n' for r in rows), encoding='utf-8')
npz = out/'candidates/gsm8k_audit.npz'
np.savez(npz, embeddings=feats)

manifest = {'protocol': PROTOCOL, 'splits': {'audit': len(rows)},
            'sha256': {'candidates/gsm8k_audit.jsonl': hashlib.sha256(jsonl.read_bytes()).hexdigest(),
                       'candidates/gsm8k_audit.npz': hashlib.sha256(npz.read_bytes()).hexdigest()}}
(out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
print(f'audit simulation: {len(rows)} questions -> {out}  (shuffled order, new IDs, no references)')
