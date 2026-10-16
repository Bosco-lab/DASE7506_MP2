"""Train a per-candidate correctness scorer (logistic regression).

Each candidate is scored on its own (no cross-candidate features); the method
picks the highest-scoring candidate. Three input groups are compared:
  emb  - fixed 1024-d embedding only
  text - hand-made text features (features.py) only
  both - concatenation
C is chosen by 5-fold grouped CV on train (groups = questions), using
hard-subset selection accuracy, so validation stays untouched until evaluation.

Usage (from code/):
    python train_scorer.py --features emb  --output runs/scorer_emb
    python train_scorer.py --features text --output runs/scorer_text
    python train_scorer.py --features both --output runs/scorer_both
"""
import os
for _var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_var, '4')

import argparse
import json
from pathlib import Path
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from common import parse_answer as official_parse
from dev_eval import load_labels
from features import TEXT_FEATURE_NAMES, parse_answer, text_feature_matrix

SEED = 0
C_GRID = (0.003, 0.01, 0.03, 0.1, 0.3, 1.0)


def build_inputs(bank, arrays, kind):
    parts = []
    if kind in ('emb', 'both'):
        parts.append(arrays['embeddings'].astype(np.float32))
    if kind in ('text', 'both'):
        parts.append(text_feature_matrix(bank))
    return np.concatenate(parts, axis=-1)


def selection_accuracy(scores, correct, mask):
    pick = scores.argmax(1)
    return 100*correct[np.arange(len(pick)), pick][mask].mean()


def fit(X, y, C):
    mean = X.mean(0)
    std = X.std(0) + 1e-6
    clf = LogisticRegression(C=C, max_iter=3000, random_state=SEED)
    clf.fit((X-mean)/std, y)
    return mean, std, clf.coef_[0].astype(np.float32), float(clf.intercept_[0])


def score(X, mean, std, coef, intercept):
    return ((X-mean)/std) @ coef + intercept


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--features', choices=('emb', 'text', 'both'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    tick = time.perf_counter()

    bank, arrays, correct, hard = load_labels('train')
    # guard: the copied parser in features.py must agree with the official one
    for row in bank[:500]:
        for c in row['candidates']:
            assert parse_answer(c['text']) == official_parse(c['text'])
    X = build_inputs(bank, arrays, args.features)
    n, k, d = X.shape
    Xf, yf = X.reshape(-1, d), correct.reshape(-1).astype(int)
    groups = np.repeat(np.arange(n), k)

    cv = {}
    for C in C_GRID:
        oof = np.zeros(n*k)
        for tr, te in GroupKFold(n_splits=5).split(Xf, yf, groups):
            oof[te] = score(Xf[te], *fit(Xf[tr], yf[tr], C))
        oof = oof.reshape(n, k)
        cv[C] = {'hard': selection_accuracy(oof, correct, hard),
                 'all': selection_accuracy(oof, correct, np.ones(n, bool)),
                 'auc': roc_auc_score(yf, oof.reshape(-1))}
        print(f"C={C:<6} cv hard={cv[C]['hard']:.2f} all={cv[C]['all']:.2f} auc={cv[C]['auc']:.4f}")
    best_C = max(C_GRID, key=lambda c: cv[c]['hard'])

    mean, std, coef, intercept = fit(Xf, yf, best_C)
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez(args.output/'model.npz', mean=mean.astype(np.float32), std=std.astype(np.float32),
             coef=coef, intercept=np.float32(intercept))
    config = {'features': args.features, 'C': best_C, 'seed': SEED, 'dim': d,
              'text_features': TEXT_FEATURE_NAMES if args.features != 'emb' else [],
              'cv': {str(c): v for c, v in cv.items()}, 'train_seconds': time.perf_counter()-tick}
    (args.output/'config.json').write_text(json.dumps(config, indent=2)+'\n')
    print(f"best C={best_C}  saved to {args.output}  ({config['train_seconds']:.1f}s)")


if __name__ == '__main__':
    main()
