"""Measure the frozen method end to end, the way the resource limit defines it.

Times one fresh prediction process plus the independent scoring process on the
complete split, with the four-thread environment set by the caller. Validation has
more questions than test, so the number below is a conservative upper bound.
"""
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ENV = {**os.environ, 'OMP_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '4',
       'MKL_NUM_THREADS': '4', 'OMP_DYNAMIC': 'FALSE', 'NUMEXPR_NUM_THREADS': '4'}


def run(split, reps=3):
    times = []
    for r in range(reps):
        t = time.perf_counter()
        subprocess.run([sys.executable, 'predict.py', '--method', 'submission.py',
                        '--checkpoint', 'artifacts/model_bundle', '--split', split,
                        '--output', f'runs/timing-{split}-pred.json'], check=True,
                       capture_output=True, cwd=HERE, env=ENV)
        subprocess.run([sys.executable, 'score_predictions.py', '--split', split,
                        '--predictions', f'runs/timing-{split}-pred.json',
                        '--output', f'runs/timing-{split}-score.json'], check=True,
                       capture_output=True, cwd=HERE, env=ENV)
        times.append(time.perf_counter()-t)
    return times


for split in ('validation',):
    ts = run(split)
    print(f'{split}: predict+score incl. two fresh processes -> '
          + ', '.join(f'{t:.2f}s' for t in ts)
          + f'   min {min(ts):.2f}s  median {statistics.median(ts):.2f}s')
