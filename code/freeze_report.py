"""Freeze report — asset sizes, hashes, and the inference import chain."""
import hashlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BUNDLE = HERE/'artifacts'/'model_bundle'

print('== inference assets ==')
total = 0
for f in sorted(BUNDLE.iterdir()):
    b = f.read_bytes()
    total += len(b)
    print(f'  {f.name:<14}{len(b):>8} B   sha256 {hashlib.sha256(b).hexdigest()[:32]}…')
print(f'  total {total} B = {total/2**20:.4f} MiB   (limit 16 MiB)')

print('\n== inference import chain (no sklearn / scipy / torch allowed) ==')
sys.path.insert(0, str(HERE))
before = set(sys.modules)
import submission  # noqa: E402
bank_like = []
submission.predict(bank_like, {'embeddings': __import__('numpy').zeros((0, 4, 1024), 'float32')}, BUNDLE)
new = sorted(m for m in set(sys.modules)-before if '.' not in m)
heavy = [m for m in new if m in ('sklearn', 'scipy', 'torch', 'pandas', 'joblib')]
print('  newly imported top-level modules:', ', '.join(new))
print('  heavy libraries pulled in:', heavy or 'none')

print('\n== source files the frozen method needs ==')
for name in ('submission.py', 'scorer_method.py', 'features.py', 'group_features.py', 'verify_features.py'):
    print(f'  {name:<22}{(HERE/name).stat().st_size:>8} B')
print('  common.py is only used by predict.py / the scorer, not by submission.py')
