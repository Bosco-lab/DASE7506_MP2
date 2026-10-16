"""Headroom probe: do untried signals carry information? Judged by train CV only.

Probes, all with fold discipline (a fold's model and neighbour index never see the
held-out fold's labels):
  1. neighbour prior   - mean correctness of the k nearest train candidates in
                         embedding space (same-question neighbours excluded)
  2. whitened embedding - PCA-whitened embeddings replacing the raw 1024 dims
  3. interactions      - the best of the above combined with the current blocks
Reference: the frozen model's train-CV hard accuracy is 82.26.
"""
import os
for _var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_var, '4')

import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold

from ablate import acc
from dev_eval import load_labels
from features import parse_answer
from scorer_method import answer_aggregate, build_inputs
from train_ranker import SEED, fit_listwise, standardize

C, T, K = 0.03, 1.0, 25


def neighbour_prior(emb, correct, rows, folds):
    """Fold-wise mean correctness of the k nearest neighbours (other questions only)."""
    unit = emb/(np.linalg.norm(emb, axis=-1, keepdims=True)+1e-8)
    prior = np.zeros(correct.shape)
    for tr, te in folds:
        idx_rows, idx_vec, idx_lab, idx_key = [], [], [], []
        for r in tr:
            for j in range(4):
                idx_rows.append(r); idx_vec.append(unit[r, j]); idx_lab.append(correct[r, j])
        idx_vec = np.stack(idx_vec); idx_lab = np.array(idx_lab); idx_rows = np.array(idx_rows)
        for r in te:
            sims = np.stack([unit[r, j] @ idx_vec.T for j in range(4)])
            mask = idx_rows[None, :] == r
            sims = np.where(mask, -np.inf, sims)
            for j in range(4):
                top = np.argpartition(-sims[j], K)[:K]
                prior[r, j] = idx_lab[top].mean()
    return prior


def cv_scores(X, correct, folds):
    n = len(X)
    oof = np.zeros((n, 4))
    for tr, te in folds:
        mean, std = standardize(X[tr])
        w, _ = fit_listwise((X[tr]-mean)/std, correct[tr], C)
        oof[te] = ((X[te]-mean)/std) @ w
    return oof


bank, arrays, correct, hard = load_labels('train')
n = len(bank)
rows = np.arange(n)
folds = list(KFold(5, shuffle=True, random_state=SEED).split(rows))
ans = [[parse_answer(c['text']) for c in r['candidates']] for r in bank]
emb = arrays['embeddings']
txt = build_inputs(bank, arrays, ['text', 'verify'])

print('== 1) neighbour prior alone ==')
prior = neighbour_prior(emb, correct, rows, folds)
print(f'   selection by prior alone: CV hard {acc(np.array(answer_aggregate(prior, ans, T)), correct, hard):.2f}')
print(f'   (compare: vote 69.9, frozen model 82.26)')

print('\n== 2) frozen blocks + neighbour prior as an extra column ==')
X = np.concatenate([emb, txt, prior[..., None]], -1)
print(f'   CV hard {acc(np.array(answer_aggregate(cv_scores(X, correct, folds), ans, T)), correct, hard):.2f}')

print('\n== 3) PCA-whitened embeddings (fit on train folds only) ==')
for k in (128, 256):
    Xw = np.zeros((n, 4, k), dtype=np.float32)
    for tr, te in folds:
        pca = PCA(n_components=k, whiten=True, random_state=SEED)
        pca.fit(emb[tr].reshape(-1, emb.shape[-1]))
        for part in (tr, te):
            Xw[part] = pca.transform(emb[part].reshape(-1, emb.shape[-1])).reshape(-1, 4, k)
    X = np.concatenate([Xw, txt], -1)
    print(f'   {k} comps: CV hard {acc(np.array(answer_aggregate(cv_scores(X, correct, folds), ans, T)), correct, hard):.2f}')

np.save('runs/prior_train_oof.npy', prior)
