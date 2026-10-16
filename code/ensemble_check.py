"""Does an ensemble of rankers beat the single final model? (train CV only)

Each member is the same listwise linear ranker over a different feature block set,
trained with 5-fold CV on train (seed 0, C=0.03). Member scores are standardised
within each question and averaged, then answer-level aggregation is applied.
"""
import os
for _var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_var, '4')

import itertools
import json

import numpy as np
from sklearn.model_selection import KFold

from dev_eval import load_labels
from features import parse_answer
from scorer_method import answer_aggregate, build_inputs
from train_ranker import SEED, fit_listwise, standardize
from ablate import acc

C, T = 0.03, 1.0
MEMBERS = {
    'e+t+v': ['emb', 'text', 'verify'],
    'e+t': ['emb', 'text'],
    'emb': ['emb'],
    't+v': ['text', 'verify'],
}


def oof(bank, arrays, correct, spec):
    X = build_inputs(bank, arrays, spec)
    n = len(X)
    out = np.zeros((n, 4))
    for tr, te in KFold(5, shuffle=True, random_state=SEED).split(np.arange(n)):
        mean, std = standardize(X[tr])
        w, _ = fit_listwise((X[tr]-mean)/std, correct[tr], C)
        out[te] = ((X[te]-mean)/std) @ w
    return out


def zs(s):
    return (s - s.mean(1, keepdims=True))/(s.std(1, keepdims=True)+1e-9)


def main():
    bank, arrays, correct, hard = load_labels('train')
    ans = [[parse_answer(c['text']) for c in r['candidates']] for r in bank]
    scores = {k: oof(bank, arrays, correct, v) for k, v in MEMBERS.items()}
    for k, s in scores.items():
        print(f'{k:<8} CV hard {acc(answer_aggregate(s, ans, T), correct, hard):.2f}')
    print()
    results = {}
    for n_members in (2, 3, 4):
        for combo in itertools.combinations(MEMBERS, n_members):
            avg = np.mean([zs(scores[m]) for m in combo], axis=0)
            a = acc(answer_aggregate(avg, ans, T), correct, hard)
            results['+'.join(combo)] = a
            print(f"{'+'.join(combo):<28} CV hard {a:.2f}")
    best = max(results, key=results.get)
    print(f'\nbest ensemble: {best}  {results[best]:.2f}   (single e+t+v: {acc(answer_aggregate(scores["e+t+v"], ans, T), correct, hard):.2f})')
    json.dump({'members': {k: acc(answer_aggregate(v, ans, T), correct, hard) for k, v in scores.items()},
               'ensembles': results}, open('runs/ensemble_cv.json', 'w'), indent=2)


if __name__ == '__main__':
    main()
