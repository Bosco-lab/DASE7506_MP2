"""Does a small MLP beat the linear ranker? (train CV decides)

One hidden ReLU layer over [embedding, text, verify], trained with the same
listwise softmax objective as the final linear model but with analytic gradients
and Adam. Weights are stored as plain arrays so inference stays NumPy-only.

Usage (from code/): python train_mlp.py --hidden 64 --iters 200 --output runs/mlp_v
"""
import os
for _var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_var, '4')

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.model_selection import KFold

from ablate import acc
from dev_eval import load_labels
from features import TEXT_FEATURE_NAMES, parse_answer
from scorer_method import answer_aggregate, build_inputs
from train_ranker import SEED
from verify_features import VERIFY_FEATURE_NAMES

BLOCKS = ['emb', 'text', 'verify']
C = 0.03          # kept for comparability with the linear model: L2 = 1/(2*C)
T = 1.0


def relu(x):
    return np.maximum(x, 0.0)


def forward(params, X):
    W1, b1, w2, b2 = params
    z = X @ W1 + b1
    h = relu(z)
    return h, h @ w2 + b2, z


def loss_and_grad(params, X, mask, lam):
    """mask: [n,4] float, 1 for correct candidates (a question may have several)."""
    _, s, _ = forward(params, X)
    s = s - s.max(1, keepdims=True)
    e = np.exp(s)
    p = e/e.sum(1, keepdims=True)
    pc = (p*mask).sum(1)
    loss = -np.log(pc+1e-12).sum()
    g = p - (p*mask)/pc[:, None]                     # dL/ds
    W1, b1, w2, b2 = params
    z = X @ W1 + b1
    h = relu(z)
    gw2 = np.einsum('nk,nke->e', g, h)
    gb2 = g.sum()
    gh = g[..., None]*w2
    gz = gh*(z > 0)
    gW1 = np.einsum('nkd,nke->de', X, gz)
    gb1 = gz.sum((0, 1))
    loss += lam*((W1**2).sum() + (w2**2).sum())
    gW1 += 2*lam*W1
    gw2 += 2*lam*w2
    return loss, [gW1, gb1, gw2, gb2]


def fit_mlp(X, correct, hidden, iters, lr=3e-3, seed=SEED, lam=None):
    rng = np.random.RandomState(seed)
    d = X.shape[-1]
    lam = 1.0/(2*C) if lam is None else lam
    W1 = rng.randn(d, hidden).astype(np.float64)*np.sqrt(2.0/d)
    b1 = np.zeros(hidden)
    w2 = rng.randn(hidden).astype(np.float64)*np.sqrt(2.0/hidden)
    b2 = 0.0
    params = [W1, b1, w2, b2]
    mask = correct.astype(np.float64)
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    for t in range(1, iters+1):
        loss, grads = loss_and_grad(params, X, mask, lam)
        for i, g in enumerate(grads):
            m[i] = 0.9*m[i] + 0.1*g
            v[i] = 0.999*v[i] + 0.001*g*g
            mh = m[i]/(1-0.9**t)
            vh = v[i]/(1-0.999**t)
            params[i] = params[i] - lr*mh/(np.sqrt(vh)+1e-8)
    return params, loss


def score(params, X):
    return forward(params, X)[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--iters', type=int, default=200)
    parser.add_argument('--lr', type=float, default=3e-3)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--folds', type=int, default=5)
    args = parser.parse_args()
    tick = time.perf_counter()

    bank, arrays, correct, hard = load_labels('train')
    X = build_inputs(bank, arrays, BLOCKS)
    n = len(bank)
    ans = [[parse_answer(c['text']) for c in r['candidates']] for r in bank]

    oof = np.zeros((n, 4))
    done = np.zeros(n, dtype=bool)
    for k, (tr, te) in enumerate(KFold(args.folds, shuffle=True, random_state=SEED).split(np.arange(n))):
        mean, std = X[tr].reshape(-1, X.shape[-1]).mean(0), X[tr].reshape(-1, X.shape[-1]).std(0)+1e-6
        params, loss = fit_mlp((X[tr]-mean)/std, correct[tr], args.hidden, args.iters, args.lr)
        oof[te] = score(params, (X[te]-mean)/std)
        done[te] = True
        part = acc(answer_aggregate(oof[done], [a for a, d in zip(ans, done) if d], T),
                   correct[done], hard[done])
        print(f'fold {k} loss {loss:.1f}  partial CV hard {part:.2f}', flush=True)
    cv_hard = acc(answer_aggregate(oof, ans, T), correct, hard)
    cv_all = acc(answer_aggregate(oof, ans, T), correct, np.ones(n, bool))
    print(f'MLP h={args.hidden} iters={args.iters}: CV hard {cv_hard:.2f}  CV all {cv_all:.2f}')

    mean, std = X.reshape(-1, X.shape[-1]).mean(0), X.reshape(-1, X.shape[-1]).std(0)+1e-6
    params, _ = fit_mlp((X-mean)/std, correct, args.hidden, args.iters, args.lr)
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez(args.output/'model.npz', mean=mean.astype(np.float32), std=std.astype(np.float32),
             W1=params[0].astype(np.float32), b1=params[1].astype(np.float32),
             w2=params[2].astype(np.float32), b2=np.float32(params[3]))
    config = {'kind': 'mlp', 'blocks': BLOCKS, 'hidden': args.hidden, 'iters': args.iters, 'lr': args.lr,
              'C': C, 'aggregate_T': T, 'seed': SEED,
              'feature_names': ([f'emb.{i}' for i in range(1024)] + TEXT_FEATURE_NAMES + VERIFY_FEATURE_NAMES),
              'cv': {'hard': cv_hard, 'all': cv_all}, 'train_seconds': time.perf_counter()-tick}
    (args.output/'config.json').write_text(json.dumps(config, indent=2)+'\n')
    print(f'saved to {args.output} ({config["train_seconds"]:.0f}s)')


if __name__ == '__main__':
    main()
