"""Dump failures of the final method on validation, grouped, for manual classification."""
import collections, json, sys
import numpy as np
from dev_eval import load_labels
from features import parse_answer
from scorer_method import answer_aggregate, candidate_scores, load_scorer

bank, arrays, correct, hard = load_labels('validation')
cfg, p = load_scorer('artifacts/model_bundle')      # frozen checkpoint (agg_etv_list, T=1.0)
s = candidate_scores(bank, arrays, cfg, p)
answers = [[parse_answer(c['text']) for c in row['candidates']] for row in bank]
pick = np.array(answer_aggregate(s, answers, cfg['aggregate_T']))
hit = correct[np.arange(len(bank)), pick]

kinds, wrong = [], []
for i, (a, cor) in enumerate(zip(answers, correct)):
    cnt = collections.Counter(v for v in a if v is not None)
    if len(cnt) < 2:
        k = 'easy'
    else:
        top = max(cnt.values()); win = [v for v, n in cnt.items() if n == top]
        if len(win) > 1:
            k = 'tie'
        else:
            k = 'maj_right' if any(cor[j] for j in range(4) if a[j] == win[0]) else 'maj_wrong'
    kinds.append(k)
    if not hit[i]:
        wrong.append(i)
kinds = np.array(kinds)
print('errors by kind:', collections.Counter(kinds[np.array(wrong)]))
print('total wrong', len(wrong), 'of', len(bank), ' val acc', f'{100*hit.mean():.2f}%')


def show(i, budget=380):
    row = bank[i]
    print('=' * 96)
    print(f'[{kinds[i]}] {i}  Q: {row["question"][:260]}')
    print(f'  answers {answers[i]}   picked {pick[i]} {"OK" if hit[i] else "WRONG"}')
    print(f'  -- picked ({row["candidates"][pick[i]]["text"][:budget].replace(chr(10), " | ")})')
    good = [j for j in range(4) if correct[i, j]]
    if good:
        print(f'  -- correct cand {good[0]} ({row["candidates"][good[0]]["text"][:budget].replace(chr(10), " | ")})')


limit = int(sys.argv[1]) if len(sys.argv) > 1 else 24
order = sorted(wrong, key=lambda i: {'maj_wrong': 0, 'tie': 1, 'maj_right': 2, 'easy': 3}[kinds[i]])
for i in order[:limit]:
    show(i)
