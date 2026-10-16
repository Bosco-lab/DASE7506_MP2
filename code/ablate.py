"""Ablations: how much does each evidence block contribute?

Every variant is trained with the same procedure (listwise loss, C from the final
model, 5-fold CV on train with seed 0), scored two ways: train-CV out-of-fold and
validation, both with answer-level aggregation at T=1. Feature blocks can be
dropped as a whole or reduced to named columns (e.g. only the length features),
which gives the "surface feature" controls.

Usage (from code/):  python ablate.py            # writes runs/ablate.json
"""
import os
for _var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_var, '4')

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import KFold

from dev_eval import load_labels
from features import TEXT_FEATURE_NAMES, parse_answer
from group_features import GROUP_FEATURE_NAMES
from process_features import PROCESS_FEATURE_NAMES
from scorer_method import answer_aggregate, build_inputs
from train_ranker import SEED, fit_listwise, standardize
from verify_features import VERIFY_FEATURE_NAMES

C_FIXED = 0.03
T_FIXED = 1.0
NAMES = {'text': TEXT_FEATURE_NAMES, 'group': GROUP_FEATURE_NAMES,
         'verify': VERIFY_FEATURE_NAMES, 'process': PROCESS_FEATURE_NAMES}
LENGTH_COLS = ['log_chars', 'log_lines']
FORMAT_COLS = ['log_chars', 'log_lines', 'n_steps', 'n_equals', 'log_numbers', 'n_bold',
               'has_hash', 'has_boxed', 'has_final_phrase', 'n_hedge']

VARIANTS = [
    ('full (e+t+v)', ['emb', 'text', 'verify']),
    ('no-emb (t+v)', ['text', 'verify']),
    ('no-text (e+v)', ['emb', 'verify']),
    ('no-verify (e+t)', ['emb', 'text']),
    ('emb only', ['emb']),
    ('text only', ['text']),
    ('verify only', ['verify']),
    ('e+t+g+v', ['emb', 'text', 'group', 'verify']),
    ('process only', ['process']),
    ('full + process (e+t+v+p)', ['emb', 'text', 'verify', 'process']),
    ('e+t+length only', ['emb', 'text:log_chars,log_lines']),
    ('e+t+format only', ['emb', f'text:{",".join(FORMAT_COLS)}']),
    ('length only (no emb)', ['text:log_chars,log_lines']),
    ('full minus integer prior', ['emb', 'text:log_chars,log_lines,n_steps,n_equals,log_numbers,n_bold,'
                                         'has_hash,has_boxed,has_final_phrase,n_hedge,ans_negative,'
                                         'ans_log_abs,ans_in_question,question_num_coverage,log_question_chars',
                                  'verify']),
]


def build(bank, arrays, spec):
    parts, cols = [], []
    for block in spec:
        if ':' in block:
            kind, wanted = block.split(':')
            keep = [NAMES[kind].index(f) for f in wanted.split(',')]
            parts.append(build_inputs(bank, arrays, [kind])[:, :, keep])
            cols += [f'{kind}.{NAMES[kind][i]}' for i in keep]
        else:
            parts.append(build_inputs(bank, arrays, [block]))
            cols += ([f'emb.{i}' for i in range(1024)] if block == 'emb' else list(NAMES[block]))
    return np.concatenate(parts, -1), cols


def cv_scores(X, correct, hard, C=C_FIXED):
    n = len(X)
    oof = np.zeros((n, 4))
    for tr, te in KFold(5, shuffle=True, random_state=SEED).split(np.arange(n)):
        mean, std = standardize(X[tr])
        w, _ = fit_listwise((X[tr]-mean)/std, correct[tr], C)
        oof[te] = ((X[te]-mean)/std) @ w
    return oof


def acc(pick, correct, mask):
    return 100*correct[np.arange(len(pick)), pick][mask].mean()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('runs/ablate.json'))
    parser.add_argument('--save-hits', type=Path, default=Path('runs/dev'))
    args = parser.parse_args()

    bank, arrays, correct, hard = load_labels('train')
    vbank, varrays, vcorrect, vhard = load_labels('validation')
    vans = [[parse_answer(c['text']) for c in row['candidates']] for row in vbank]
    print(f'train n={len(bank)} hard={hard.sum()}   val n={len(vbank)} hard={vhard.sum()}\n')
    print(f"{'variant':<24}{'dim':>6}{'cv hard':>9}{'cv all':>8}{'val hard':>10}{'val all':>8}")

    results = {}
    for label, spec in VARIANTS:
        X, cols = build(bank, arrays, spec)
        Xv, _ = build(vbank, varrays, spec)
        oof = cv_scores(X, correct, hard)
        mean, std = standardize(X)
        w, _ = fit_listwise((X-mean)/std, correct, C_FIXED)
        vs = ((Xv-mean)/std) @ w
        cv_pick = np.array(answer_aggregate(oof, [[parse_answer(c['text']) for c in r['candidates']]
                                                  for r in bank], T_FIXED))
        val_pick = np.array(answer_aggregate(vs, vans, T_FIXED))
        row = {'dim': X.shape[-1], 'cv_hard': acc(cv_pick, correct, hard), 'cv_all': acc(cv_pick, correct, np.ones(len(bank), bool)),
               'val_hard': acc(val_pick, vcorrect, vhard), 'val_all': acc(val_pick, vcorrect, np.ones(len(vbank), bool))}
        results[label] = row
        np.save(args.save_hits/f'ablate-{label.split(" ")[0].replace("/", "_")}.npy', val_pick)
        print(f"{label:<24}{row['dim']:>6}{row['cv_hard']:>9.2f}{row['cv_all']:>8.2f}{row['val_hard']:>10.2f}{row['val_all']:>8.2f}", flush=True)

    args.output.write_text(json.dumps({'C': C_FIXED, 'T': T_FIXED, 'results': results}, indent=2)+'\n')
    print(f'\nwritten to {args.output}')


if __name__ == '__main__':
    main()
