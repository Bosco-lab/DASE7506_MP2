"""Select C on the decision metric itself (train CV with answer-level aggregation).

train_ranker picks C by the raw-argmax CV, but the frozen method aggregates answers
before choosing, and the two metrics do not agree on C. For a decision that matches
what is actually deployed, sweep both C and T on train out-of-fold scores and report
the best aggregated CV hard accuracy per C.

Usage (from code/): python select_by_agg_cv.py --features emb,text,verify,process
"""
import os
for _var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_var, '4')

import argparse

import numpy as np
from sklearn.model_selection import KFold

from ablate import acc
from dev_eval import load_labels
from features import parse_answer
from fit_aggregate import T_GRID
from scorer_method import answer_aggregate, build_inputs
from train_ranker import SEED, fit_listwise, standardize

C_GRID = (0.003, 0.01, 0.03, 0.1, 0.3, 1.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--features', required=True)
    args = parser.parse_args()
    blocks = [b.strip() for b in args.features.split(',')]

    bank, arrays, correct, hard = load_labels('train')
    n = len(bank)
    X = build_inputs(bank, arrays, blocks)
    ans = [[parse_answer(c['text']) for c in r['candidates']] for r in bank]
    folds = list(KFold(5, shuffle=True, random_state=SEED).split(np.arange(n)))
    rows = np.arange(n)

    print(f'features {blocks}   dim {X.shape[-1]}')
    best = None
    for C in C_GRID:
        oof = np.zeros((n, 4))
        for tr, te in folds:
            mean, std = standardize(X[tr])
            w, _ = fit_listwise((X[tr]-mean)/std, correct[tr], C)
            oof[te] = ((X[te]-mean)/std) @ w
        sweep = {T: acc(answer_aggregate(oof, ans, T), correct, hard) for T in T_GRID}
        T_best = max(T_GRID, key=lambda t: (round(sweep[t], 6), t))
        raw = acc(oof.argmax(1), correct, hard)
        print(f'  C={C:<6} argmax {raw:6.2f}   best T={str(T_best):<5} aggregated {sweep[T_best]:6.2f}')
        if best is None or sweep[T_best] > best[2]:
            best = (C, T_best, sweep[T_best])
    print(f'BEST C={best[0]}  T={best[1]}  aggregated CV hard {best[2]:.2f}')


if __name__ == '__main__':
    main()
