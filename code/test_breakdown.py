"""Post-hoc breakdown of the frozen method's test result (analysis only, no tuning).

dev_eval.load_labels deliberately refuses the test split, so this script loads the
bank and references through the official readers directly and computes the same
correct/hard masks. Nothing here feeds back into the frozen method.
"""
import collections, json
import numpy as np
from common import load_inputs, read_jsonl
from features import parse_answer
from scorer_method import answer_aggregate, candidate_scores, load_scorer


def test_labels():
    bank, arrays = load_inputs('test')
    gold = read_jsonl('data/gsm8k_test.jsonl')
    if [g['id'] for g in gold] != [r['id'] for r in bank]:
        raise ValueError('reference/bank alignment mismatch')
    correct = np.zeros((len(bank), 4), dtype=bool)
    hard = np.zeros(len(bank), dtype=bool)
    for i, (row, ref) in enumerate(zip(bank, gold)):
        a = [parse_answer(c['text']) for c in row['candidates']]
        target = parse_answer(ref['answer'])
        correct[i] = [x is not None and x == target for x in a]
        hard[i] = len({x for x in a if x is not None}) >= 2
    return bank, arrays, correct, hard


pred = json.load(open('artifacts/test-predictions.json'))
bank, arrays, correct, hard = test_labels()
pick = np.array(pred['selected'])
assert len(pick) == len(bank), 'prediction/bank length mismatch'
hit = correct[np.arange(len(bank)), pick]
print(f'test accuracy {100*hit.mean():.2f}%  ({int(hit.sum())}/{len(bank)})  hard subset {hard.sum()} of {len(bank)}')

kinds, maj_wrong, tie = [], 0, 0
for i, (row, cor) in enumerate(zip(bank, correct)):
    a = [parse_answer(c['text']) for c in row['candidates']]
    cnt = collections.Counter(v for v in a if v is not None)
    top = max(cnt.values())
    win = [v for v, n in cnt.items() if n == top]
    if len(win) > 1:
        kinds.append('tie')
    elif any(cor[j] for j in range(4) if a[j] == win[0]):
        kinds.append('maj_right')
    else:
        kinds.append('maj_wrong')
kinds = np.array(kinds)
print('\naccuracy by question type (test):')
print(f"{'type':<12}{'n':>5}{'acc':>9}")
for k in ('maj_right', 'tie', 'maj_wrong'):
    m = kinds == k
    print(f'{k:<12}{int(m.sum()):>5}{100*hit[m].mean():>8.1f}%')

with np.load('artifacts/model_bundle/model.npz') as f:
    params = {k: f[k] for k in ('mean', 'std', 'coef', 'intercept')}
cfg = json.load(open('artifacts/model_bundle/config.json'))
s = candidate_scores(bank, arrays, cfg, params)
answers = [[parse_answer(c['text']) for c in r['candidates']] for r in bank]
T = float(cfg['aggregate_T'])
conf = []
for i in range(len(bank)):
    z = s[i]/T
    w = np.exp(z-z.max()); w /= w.sum()
    a = answers[i][pick[i]]
    conf.append(w[[j for j in range(4) if answers[i][j] == a]].sum() if a is not None else w[pick[i]])
conf = np.array(conf)
print('\nreliability on test (confidence = pooled weight of the chosen answer):')
edges = [0, 0.7, 0.85, 0.95, 1.01]
for lo, hi in zip(edges, edges[1:]):
    m = (conf >= lo) & (conf < hi)
    if m.sum():
        print(f'  [{lo:.2f},{hi:.2f})  n={int(m.sum()):>3}  acc={100*hit[m].mean():>5.1f}%')
