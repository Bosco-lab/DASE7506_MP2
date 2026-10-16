"""Baseline (kept for comparison): majority vote with longest-solution tiebreak.

Tiebreak strategy (val hard-subset accuracy, seed 42):
  first    69.01%
  integer  70.42%
  boxed    69.30%
  longest  70.70%  <-- selected
  shortest 65.07%
  random   67.04%

Differences are within noise (±2.45%); 'longest' is the best point estimate
on validation. Length is itself a surface feature, and the rule was selected
on validation. It serves as the baseline for comparisons with later methods.
"""
from __future__ import annotations

import collections
import re
from fractions import Fraction
from pathlib import Path


# ---------------------------------------------------------------------------
# helpers (local copies so submission.py is self-contained at inference time)
# ---------------------------------------------------------------------------

def _parse_answer(text: str):
    """Replicate common.parse_answer without importing common at inference."""
    text = str(text)
    tail = text.rsplit('####', 1)[1] if '####' in text else None
    if tail is None:
        boxed = re.findall(r'\\boxed\{([^{}]+)\}', text)
        tail = boxed[-1] if boxed else ''
    m = re.search(r'[-+]?\d[\d,]*(?:\.\d+)?(?:/\d+)?', tail)
    if not m:
        return None
    try:
        return str(Fraction(m.group().replace(',', '')))
    except (ValueError, ZeroDivisionError):
        return None


# ---------------------------------------------------------------------------
# public interface
# ---------------------------------------------------------------------------

def predict(bank, arrays, checkpoint):
    """Return one Python integer in 0..3 per question, in bank order.

    Strategy: majority vote over parsed final answers; ties broken by
    choosing the candidate with the longest solution text.

    bank: list of question dicts with four candidate texts each.
    arrays['embeddings']: float32 NumPy array of shape [N, 4, 1024].
    checkpoint: pathlib.Path or None (unused by this baseline).
    """
    choices = []
    for row in bank:
        cands = row['candidates']
        answers = [_parse_answer(c['text']) for c in cands]
        cnt = collections.Counter(a for a in answers if a is not None)

        if not cnt:
            # no candidate produced a parseable answer — fall back to index 0
            choices.append(0)
            continue

        top_count = cnt.most_common(1)[0][1]
        top_answer_set = {a for a, c in cnt.items() if c == top_count}
        top_idxs = [i for i, a in enumerate(answers) if a in top_answer_set]

        if len(top_idxs) == 1:
            choices.append(top_idxs[0])
        else:
            # tiebreak: pick the candidate with the longest solution text
            choices.append(max(top_idxs, key=lambda i: len(cands[i]['text'])))

    return choices
