"""Answer-level aggregation with a temperature T, chosen on train out-of-fold scores.

For a trained ranker checkpoint, recompute 5-fold out-of-fold scores on train with
the checkpoint's own C (same folds/seed as train_ranker.py), sweep T, pick the T with
the best train-CV hard accuracy (ties -> larger T, i.e. closer to plain voting), and
write a new checkpoint = same weights + config['aggregate_T']. Validation is not used.

Usage (from code/):
    python fit_aggregate.py --checkpoint runs/rank_etgv_list --output runs/agg_etgv_list
"""
import os
for _var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_var, '4')

import argparse
import json
from pathlib import Path
import shutil

import numpy as np
from sklearn.model_selection import KFold

from dev_eval import load_labels
from features import parse_answer
from scorer_method import answer_aggregate, build_inputs, feature_blocks
from train_ranker import SEED, fit_listwise, fit_pointwise, standardize

T_GRID = (0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, float('inf'))


def oof_scores(bank, arrays, correct, hard, config):
    X = build_inputs(bank, arrays, feature_blocks(config))
    n = len(bank)
    fit = fit_listwise if config.get('loss') == 'listwise' else fit_pointwise
    oof = np.zeros((n, 4))
    for tr, te in KFold(n_splits=5, shuffle=True, random_state=SEED).split(np.arange(n)):
        if config.get('hard_only'):
            tr = tr[hard[tr]]
        mean, std = standardize(X[tr])
        w, b = fit((X[tr]-mean)/std, correct[tr], config['C'])
        oof[te] = ((X[te]-mean)/std) @ w + b
    return oof


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads((args.checkpoint/'config.json').read_text())

    bank, arrays, correct, hard = load_labels('train')
    oof = oof_scores(bank, arrays, correct, hard, config)
    answers = [[parse_answer(c['text']) for c in row['candidates']] for row in bank]
    rows = np.arange(len(bank))

    sweep = {}
    for T in T_GRID:
        pick = np.array(answer_aggregate(oof, answers, T))
        hit = correct[rows, pick]
        sweep[T] = {'hard': 100*hit[hard].mean(), 'all': 100*hit.mean()}
    base = correct[rows, oof.argmax(1)]
    print(f"argmax (no aggregation): cv hard={100*base[hard].mean():.2f} all={100*base.mean():.2f}")
    for T, r in sweep.items():
        print(f"T={T:<6} cv hard={r['hard']:.2f} all={r['all']:.2f}")
    best_T = max(T_GRID, key=lambda t: (round(sweep[t]['hard'], 6), t))

    args.output.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.checkpoint/'model.npz', args.output/'model.npz')
    config['aggregate_T'] = 'inf' if not np.isfinite(best_T) else best_T
    config['aggregate_source'] = str(args.checkpoint)
    config['aggregate_cv'] = {'argmax': {'hard': 100*base[hard].mean(), 'all': 100*base.mean()},
                              **{('inf' if not np.isfinite(t) else str(t)): r for t, r in sweep.items()}}
    (args.output/'config.json').write_text(json.dumps(config, indent=2)+'\n')
    print(f'best T={best_T}  saved to {args.output}')


if __name__ == '__main__':
    main()
