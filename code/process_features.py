"""Process-consistency features: did a candidate stop early, or carry on past another answer?

A frequent error in the hard groups is a solution whose arithmetic is fine but which
stops one step short (it reports the original price instead of the price paid, or
the cookies on one tray instead of all four). Such a wrong answer then usually shows
up as an intermediate value inside a competing solution that carries on. This block
compares the derived numbers of the four candidates, i.e. numbers that occur in a
solution but not in the question (step labels and list numbering are removed first):

  am_intermediate       share of the other answers whose solutions contain my answer
                        as a derived value (I may have stopped early)
  has_intermediate      share of the other answers that occur among my derived values
                        (I carried on past them)
  net_continue          has_intermediate - am_intermediate
  am_/has_intermediate_big
                        the same, counting only answers that are not small integers
                        (|v| >= 10 or non-integer), where coincidental matches are rare
  shared_derived        share of my derived values that another candidate also derives
  shared_derived_other_answer
                        the same, counting only candidates with a different answer
  log_derived           number of derived values (a proxy for the number of steps)
  derived_gap_to_max    my count minus the largest count in the question (<= 0)
  is_most_derived       1 if no candidate derives more values
"""
import re
from fractions import Fraction

import numpy as np

from features import NUM_RE, parse_answer

PROCESS_FEATURE_NAMES = [
    'am_intermediate', 'has_intermediate', 'net_continue',
    'am_intermediate_big', 'has_intermediate_big',
    'shared_derived', 'shared_derived_other_answer',
    'log_derived', 'derived_gap_to_max', 'is_most_derived',
]

_LABELS = re.compile(r'(?im)(?:\bstep\s*\d+\b|^\s*#*\s*\d+[.)]\s)')


def _values(text):
    out = set()
    for m in NUM_RE.findall(_LABELS.sub(' ', text)):
        try:
            out.add(Fraction(m.replace(',', '')))
        except (ValueError, ZeroDivisionError):
            pass
    return out


def _big(v):
    return v is not None and (abs(v) >= 10 or v.denominator != 1)


def process_feature_matrix(bank):
    """Return float32 array [N, 4, P] of process-consistency features."""
    out = np.zeros((len(bank), 4, len(PROCESS_FEATURE_NAMES)), dtype=np.float32)
    for i, row in enumerate(bank):
        question = _values(row['question'])
        texts = [c['text'] for c in row['candidates']]
        answers = [parse_answer(t) for t in texts]
        values = [Fraction(a) if a is not None else None for a in answers]
        derived = [_values(t) - question for t in texts]
        counts = np.array([len(d) for d in derived], dtype=np.float32)
        for j in range(4):
            a = values[j]
            others = [k for k in range(4) if k != j and values[k] is not None and values[k] != a]
            other_answers = {values[k] for k in others}
            am = has = am_big = has_big = 0.0
            if a is not None and other_answers:
                hit_am = [b for b in other_answers if any(a in derived[k] for k in others if values[k] == b)]
                hit_has = [b for b in other_answers if b in derived[j]]
                am = len(hit_am)/len(other_answers)
                has = len(hit_has)/len(other_answers)
                am_big = am if _big(a) else 0.0
                has_big = sum(_big(b) for b in hit_has)/len(other_answers)
            mine = derived[j]
            pool_all = set().union(*(derived[k] for k in range(4) if k != j))
            pool_diff = set().union(*(derived[k] for k in others))
            out[i, j] = [
                am, has, has-am, am_big, has_big,
                len(mine & pool_all)/len(mine) if mine else 0.0,
                len(mine & pool_diff)/len(mine) if mine else 0.0,
                np.log1p(len(mine)),
                counts[j]-counts.max(),
                float(counts[j] == counts.max()),
            ]
    return out
