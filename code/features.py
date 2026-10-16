"""Per-candidate text features shared by training and inference.

Only surface/format statistics of a single candidate and its question; no
cross-candidate information and no arithmetic re-checking, so an
ablation of a single candidate's own features stays clean.
"""
import re
from fractions import Fraction

import numpy as np

NUM_RE = re.compile(r'[-+]?\d[\d,]*(?:\.\d+)?(?:/\d+)?')
HEDGE_RE = re.compile(r'\b(assume|assuming|approximately|approx|wait|however|actually|maybe|perhaps|unclear|ambiguous|interpret)\b', re.I)

TEXT_FEATURE_NAMES = [
    'log_chars', 'log_lines', 'n_steps', 'n_equals', 'log_numbers', 'n_bold',
    'has_hash', 'has_boxed', 'has_final_phrase', 'n_hedge',
    'parsed', 'ans_integer', 'ans_negative', 'ans_log_abs', 'ans_in_question',
    'question_num_coverage', 'log_question_chars',
]


def parse_answer(text):
    """Same rule as common.parse_answer (copied so inference needs no extra import)."""
    text = str(text)
    tail = text.rsplit('####', 1)[1] if '####' in text else None
    if tail is None:
        boxed = re.findall(r'\\boxed\{([^{}]+)\}', text)
        tail = boxed[-1] if boxed else ''
    match = NUM_RE.search(tail)
    if not match:
        return None
    try:
        return str(Fraction(match.group().replace(',', '')))
    except (ValueError, ZeroDivisionError):
        return None


def _numbers(text):
    out = set()
    for m in NUM_RE.findall(text):
        try:
            out.add(Fraction(m.replace(',', '')))
        except (ValueError, ZeroDivisionError):
            pass
    return out


def candidate_text_features(question, text, question_numbers=None):
    if question_numbers is None:
        question_numbers = _numbers(question)
    answer = parse_answer(text)
    value = Fraction(answer) if answer is not None else None
    solution_numbers = _numbers(text)
    coverage = (len(question_numbers & solution_numbers)/len(question_numbers)) if question_numbers else 1.0
    return [
        np.log1p(len(text)),
        np.log1p(text.count('\n')),
        len(re.findall(r'\bstep\b', text, re.I)),
        text.count('='),
        np.log1p(len(NUM_RE.findall(text))),
        text.count('**')/2,
        float('####' in text),
        float('\\boxed' in text),
        float(bool(re.search(r'final answer', text, re.I))),
        len(HEDGE_RE.findall(text)),
        float(value is not None),
        float(value is not None and value.denominator == 1),
        float(value is not None and value < 0),
        float(np.log1p(abs(float(value)))) if value is not None else 0.0,
        float(value is not None and value in question_numbers),
        coverage,
        np.log1p(len(question)),
    ]


def text_feature_matrix(bank):
    """Return float32 array [N, 4, F] of per-candidate text features."""
    out = np.zeros((len(bank), 4, len(TEXT_FEATURE_NAMES)), dtype=np.float32)
    for i, row in enumerate(bank):
        qnums = _numbers(row['question'])
        for j, cand in enumerate(row['candidates']):
            out[i, j] = candidate_text_features(row['question'], cand['text'], qnums)
    return out
