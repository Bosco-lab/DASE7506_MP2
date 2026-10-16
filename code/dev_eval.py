"""Development evaluator for train/validation only (not part of the official protocol).

Reports accuracy on the whole split and on the "hard" subset: groups whose four
candidates contain at least two different valid parsed answers, the condition
every test group satisfies. Refuses the test split so it cannot leak into
method selection. Official numbers still come from predict.py + score_predictions.py.

Usage (from code/):
    python dev_eval.py --method submission.py --checkpoint artifacts/model_bundle \
        --split validation --label my-method --output runs/dev/my-method.json
"""
import os
for _var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_var, '4')

import argparse
import datetime
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np

from common import DATA, load_inputs, parse_answer, read_jsonl, score_choices, validate_choices

DEV_SPLITS = ('train', 'validation')
LOG = Path(__file__).resolve().parent/'runs'/'dev_log.jsonl'


def load_labels(split, data_dir=DATA):
    """Return bank, arrays, correct [N,4] bool and hard [N] bool for a dev split."""
    if split not in DEV_SPLITS:
        raise ValueError(f'dev_eval supports {DEV_SPLITS} only; test is reserved for the final run')
    bank, arrays = load_inputs(split, data_dir)
    gold = read_jsonl(Path(data_dir)/f'gsm8k_{split}.jsonl')
    if [g['id'] for g in gold] != [row['id'] for row in bank]:
        raise ValueError('Reference/bank alignment mismatch')
    correct = np.zeros((len(bank), 4), dtype=bool)
    hard = np.zeros(len(bank), dtype=bool)
    for i, (row, ref) in enumerate(zip(bank, gold)):
        answers = [parse_answer(c['text']) for c in row['candidates']]
        target = parse_answer(ref['answer'])
        correct[i] = [a is not None and a == target for a in answers]
        hard[i] = len({a for a in answers if a is not None}) >= 2
    return bank, arrays, correct, hard


def summarize(choices, correct, hard):
    """Accuracy (%) with binomial standard error on all / hard / easy questions."""
    choices = np.asarray(choices, dtype=int)
    hit = correct[np.arange(len(choices)), choices]

    def block(mask):
        n = int(mask.sum())
        k = int(hit[mask].sum())
        p = k/n if n else float('nan')
        se = np.sqrt(p*(1-p)/n) if n else float('nan')
        return {'n': n, 'correct': k, 'accuracy': 100*p, 'se': 100*se}

    return {'all': block(np.ones_like(hard)), 'hard': block(hard), 'easy': block(~hard)}, hit


def load_method(path):
    """Import a method file the same way predict.py does."""
    path = Path(path).resolve()
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location('student_method', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def reference_rows(correct, hard):
    """Label-free reference points (always pick 0) and the oracle ceiling."""
    rows = {'always-0': summarize(np.zeros(len(hard), dtype=int), correct, hard)[0]}
    oracle = {}
    for name, mask in (('all', np.ones_like(hard)), ('hard', hard), ('easy', ~hard)):
        n = int(mask.sum())
        k = int(correct[mask].any(1).sum())
        oracle[name] = {'n': n, 'correct': k, 'accuracy': 100*k/n if n else float('nan'), 'se': 0.0}
    rows['oracle'] = oracle
    return rows


def print_table(rows):
    print(f"{'method':<28}{'all %':>14}{'hard %':>14}{'easy %':>14}")
    for name, r in rows.items():
        cells = ''.join(f"{r[k]['accuracy']:>8.2f}±{r[k]['se']:<5.2f}" for k in ('all', 'hard', 'easy'))
        print(f'{name:<28}{cells}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--split', choices=DEV_SPLITS, default='validation')
    parser.add_argument('--data-dir', type=Path, default=DATA)
    parser.add_argument('--label', help='Name recorded in the log (default: method file stem)')
    parser.add_argument('--output', type=Path, help='Write per-question results JSON here')
    parser.add_argument('--no-log', action='store_true', help='Do not append to runs/dev_log.jsonl')
    args = parser.parse_args()
    label = args.label or args.method.stem

    bank, arrays, correct, hard = load_labels(args.split, args.data_dir)
    method = load_method(args.method)
    tick = time.perf_counter()
    choices = method.predict(bank, arrays, args.checkpoint)
    seconds = time.perf_counter()-tick
    ids = [row['id'] for row in bank]
    validate_choices(bank, ids, choices)

    summary, hit = summarize(choices, correct, hard)
    official = score_choices(args.split, ids, choices, args.data_dir)
    if official['correct'] != summary['all']['correct']:
        raise RuntimeError('dev_eval disagrees with the official scorer')

    rows = reference_rows(correct, hard)
    rows = {'always-0': rows['always-0'], label: summary, 'oracle': rows['oracle']}
    print(f'split={args.split}  n={len(bank)}  hard={int(hard.sum())}  predict_seconds={seconds:.2f}')
    print_table(rows)
    print(f"official scorer accuracy: {official['accuracy']:.4f} (matches)")

    record = {'time': datetime.datetime.now().isoformat(timespec='seconds'), 'label': label,
              'method': str(args.method), 'checkpoint': str(args.checkpoint) if args.checkpoint else None,
              'split': args.split, 'predict_seconds': seconds, **{k: summary[k] for k in ('all', 'hard', 'easy')}}
    if not args.no_log:
        LOG.parent.mkdir(parents=True, exist_ok=True)
        with LOG.open('a', encoding='utf-8') as f:
            f.write(json.dumps(record)+'\n')
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        detail = {**record, 'ids': ids, 'selected': [int(c) for c in choices],
                  'hit': hit.tolist(), 'hard_mask': hard.tolist()}
        args.output.write_text(json.dumps(detail)+'\n')


if __name__ == '__main__':
    main()
