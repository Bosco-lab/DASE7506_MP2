"""Candidate scorer with cross-candidate features and/or a listwise loss.

  --features  any of emb,text,group (comma separated)
  --loss      pointwise : logistic regression on each candidate (as in train_scorer.py)
              listwise  : softmax over the 4 candidates of a question, maximising
                          the probability mass on correct candidates (L2 = 1/(2C))
Regularisation C is chosen by 5-fold grouped CV on train (hard-subset selection
accuracy); validation is not touched. Inference: scorer_method.py.

Usage (from code/):
    python train_ranker.py --features emb,text,group --loss listwise --output runs/rank_etg_list
"""
import os
for _var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_var, '4')

import argparse
import json
from pathlib import Path
import time

import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

from dev_eval import load_labels
from features import TEXT_FEATURE_NAMES
from group_features import GROUP_FEATURE_NAMES
from process_features import PROCESS_FEATURE_NAMES
from verify_features import VERIFY_FEATURE_NAMES
from scorer_method import build_inputs

SEED = 0
C_GRID = (0.003, 0.01, 0.03, 0.1, 0.3, 1.0)


def standardize(X):
    flat = X.reshape(-1, X.shape[-1])
    return flat.mean(0), flat.std(0)+1e-6


def fit_pointwise(Z, correct, C):
    n, k, d = Z.shape
    clf = LogisticRegression(C=C, max_iter=3000, random_state=SEED)
    clf.fit(Z.reshape(-1, d), correct.reshape(-1).astype(int))
    return clf.coef_[0], float(clf.intercept_[0])


def fit_listwise(Z, correct, C, w0=None):
    """Minimise -sum log P(correct set) + ||w||^2/(2C) over questions."""
    n, k, d = Z.shape
    mask = correct.astype(np.float64)
    usable = mask.sum(1) > 0
    Z, mask = Z[usable].astype(np.float64), mask[usable]
    lam = 1.0/(2*C)

    def objective(w):
        s = Z @ w
        s -= s.max(1, keepdims=True)
        e = np.exp(s)
        p = e/e.sum(1, keepdims=True)
        pc = (p*mask).sum(1)
        loss = -np.log(pc+1e-12).sum() + lam*(w @ w)
        # d/ds of -log sum_{c} p_c = p - p*mask/pc
        g_s = p - (p*mask)/pc[:, None]
        grad = np.einsum('nk,nkd->d', g_s, Z) + 2*lam*w
        return loss, grad

    w0 = np.zeros(d) if w0 is None else w0
    res = minimize(objective, w0, jac=True, method='L-BFGS-B', options={'maxiter': 2000})
    return res.x, 0.0


def selection_accuracy(scores, correct, mask):
    pick = scores.argmax(1)
    return 100*correct[np.arange(len(pick)), pick][mask].mean()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--features', required=True, help='comma list of emb,text,group,verify,process')
    parser.add_argument('--loss', choices=('pointwise', 'listwise'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--hard-only', action='store_true',
                        help='fit only on groups with >=2 distinct parsed answers (label-free condition)')
    args = parser.parse_args()
    blocks = [b.strip() for b in args.features.split(',')]
    assert set(blocks) <= {'emb', 'text', 'group', 'verify', 'process'} and blocks
    fit = fit_pointwise if args.loss == 'pointwise' else fit_listwise
    tick = time.perf_counter()

    bank, arrays, correct, hard = load_labels('train')
    X = build_inputs(bank, arrays, blocks)
    n, k, d = X.shape
    folds = list(KFold(n_splits=5, shuffle=True, random_state=SEED).split(np.arange(n)))

    cv = {}
    for C in C_GRID:
        oof = np.zeros((n, k))
        for tr, te in folds:
            if args.hard_only:
                tr = tr[hard[tr]]
            mean, std = standardize(X[tr])
            w, b = fit((X[tr]-mean)/std, correct[tr], C)
            oof[te] = ((X[te]-mean)/std) @ w + b
        cv[C] = {'hard': selection_accuracy(oof, correct, hard),
                 'all': selection_accuracy(oof, correct, np.ones(n, bool))}
        print(f"C={C:<6} cv hard={cv[C]['hard']:.2f} all={cv[C]['all']:.2f}", flush=True)
    best_C = max(C_GRID, key=lambda c: cv[c]['hard'])

    fit_rows = np.flatnonzero(hard) if args.hard_only else np.arange(n)
    mean, std = standardize(X[fit_rows])
    w, b = fit((X[fit_rows]-mean)/std, correct[fit_rows], best_C)
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez(args.output/'model.npz', mean=mean.astype(np.float32), std=std.astype(np.float32),
             coef=w.astype(np.float32), intercept=np.float32(b))
    names = {'emb': [f'emb_{i}' for i in range(1024)], 'text': TEXT_FEATURE_NAMES, 'group': GROUP_FEATURE_NAMES,
             'verify': VERIFY_FEATURE_NAMES,
             'process': PROCESS_FEATURE_NAMES}
    config = {'features': blocks, 'loss': args.loss, 'hard_only': args.hard_only, 'C': best_C, 'seed': SEED, 'dim': d,
              'feature_names': [f for blk in blocks for f in names[blk]],
              'cv': {str(c): v for c, v in cv.items()}, 'train_seconds': time.perf_counter()-tick}
    (args.output/'config.json').write_text(json.dumps(config, indent=2)+'\n')
    print(f"best C={best_C}  saved to {args.output}  ({config['train_seconds']:.1f}s)")


if __name__ == '__main__':
    main()
