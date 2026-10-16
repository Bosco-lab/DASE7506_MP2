"""Numeric verification features for one candidate.

The checker is deliberately conservative: it only judges equations whose
left-hand side reduces to pure arithmetic over numbers (after normalising LaTeX,
units and markdown), evaluated exactly with Fractions via a tiny AST walker
(no eval). Anything it cannot reduce is skipped, never counted as an error.
"""
import ast
from fractions import Fraction
import operator
import re

import numpy as np

from features import parse_answer

VERIFY_FEATURE_NAMES = [
    'n_checked', 'n_wrong', 'frac_wrong', 'any_wrong',
    'last_eq_matches_answer', 'answer_in_results', 'answer_in_text_body',
    'unused_question_numbers', 'log_ratio_answer_to_max_q',
]

NUM = r'\d[\d,]*(?:\.\d+)?'
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}


def _normalise(s):
    # a word/unit stays a word ("U"), so "96 minutes = 96/60" is never read as "96 = 96/60"
    s = re.sub(r'\\(?:text|mathrm|textbf|mathbf|operatorname)\{[^{}]*\}', ' U ', s)
    s = re.sub(r'\\(?:Rightarrow|implies|to|rightarrow|approx|quad|qquad)', ' U ', s)
    s = re.sub(r'\\frac\{([^{}]+)\}\{([^{}]+)\}', r'((\1)/(\2))', s)
    s = re.sub(r'\\boxed\{([^{}]+)\}', r'\1', s)
    for a, b in (('\\times', '*'), ('\\cdot', '*'), ('×', '*'), ('·', '*'), ('\\div', '/'), ('÷', '/'),
                 ('−', '-'), ('\\left', ''), ('\\right', ''), ('\\,', ' '), ('\\$', ' '), ('$', ' '),
                 ('**', ' '), ('{', '('), ('}', ')'), ('\\%', '%')):
        s = s.replace(a, b)
    s = re.sub(r'(?<=\d),(?=\d{3})', '', s)
    s = re.sub(rf'({NUM})\s*%', r'(\1/100)', s)
    return s


def _eval(node):
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return Fraction(str(node.value))
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_eval(node.operand)
    raise ValueError('unsupported')


def _arith(expr):
    """Exact value of a pure-arithmetic string, or None."""
    expr = expr.strip()
    if not expr or not re.search(r'\d', expr) or re.search(r'[A-Za-z_\\]', expr) or len(expr) > 80:
        return None
    try:
        return _eval(ast.parse(expr, mode='eval'))
    except (ValueError, SyntaxError, ZeroDivisionError, RecursionError, TypeError):
        return None


ARITH = r'[\d\.\s\(\)\+\-\*/]+'


def _tail_run(seg):
    """Trailing pure-arithmetic run of a segment, or None if it continues a symbolic expression."""
    m = re.search(rf'{ARITH}$', seg)
    if not m:
        return None
    run, before = m.group(0).strip(), seg[:m.start()].rstrip()
    if not run or run[0] in '+*/)' or (before and re.search(r'[A-Za-z0-9\)\]]$', before)):
        return None
    if run[0] == '-' and before:
        return None  # "... foo - 3" : minus belongs to something we cannot see
    if re.search(r'[A-Za-z]', seg[:m.start()][-3:]):
        return None
    return run


def _head_run(seg):
    """Leading pure-arithmetic run of a segment, or None if it is followed by a variable."""
    m = re.match(ARITH, seg)
    if not m or not re.search(r'\d', m.group(0)):
        return None
    run, after = m.group(0).strip(), seg[m.end():]
    if re.match(r'[A-Za-z\(]', after):
        return None  # "4x", "3(", ...
    return run


def _segment_value(seg, first):
    """Arithmetic string that a whole '='-segment stands for, or None.

    Middle segments must be pure arithmetic up to trailing words/units ("= 96/60 = 1.6 hours"
    is fine); a segment where words sit *between* numbers ("0.5 hours = 0.5*3600") is skipped
    on the side that carries the unit, so a quantity is never compared with its conversion.
    """
    if first:
        return _tail_run(seg)
    head = _head_run(seg)
    if head is None:
        return None
    rest = seg[seg.find(head)+len(head):] if head in seg else ''
    # unit right after a value that is then continued by more maths => mixed segment, skip
    if re.search(r'[\+\-\*/]', re.sub(r'^\s*U\b', '', rest)) and re.match(r'\s*U\b', rest):
        return None
    return head


def check_equations(text):
    """Return list of (claimed, computed) for equations the checker can judge.

    A line "e0 = e1 = e2" is split on "="; every pair of adjacent segments whose
    arithmetic values are both known and at least one side contains an operator
    is checked, with the right side treated as the claimed value.
    """
    out = []
    for line in text.split('\n'):
        line = re.sub(r'^\s*(?:[-*•>]|\d+\.)\s+', '', line)  # markdown bullets / numbering
        parts = _normalise(line).split('=')
        if len(parts) < 2:
            continue
        # value of each segment when it is the right side (claim) and when it is the left side
        as_right = [None] + [_segment_value(p, False) for p in parts[1:]]
        as_left = [_tail_run(parts[0]) if not re.search(r'U\s*$', parts[0]) else None]
        for p in parts[1:]:
            if re.search(r'[,;]|\bU\b', p.strip()):
                # "= 4/12, U 1/6 = ..." : the next equation starts after the last separator
                tail = re.split(r'[,;]|\bU\b', p)[-1]
                as_left.append(_tail_run(tail) if tail.strip() else None)
            else:
                as_left.append(_segment_value(p, False))
        for k in range(len(parts)-1):
            left, right = as_left[k], as_right[k+1]
            if left is None or right is None:
                continue
            # a claim is "<arithmetic> = <single number>"; rewrites like "a/b = a/b*4" are skipped
            if re.search(r'[\+\*/]|(?<=\S)\s*-', right) or not re.search(r'[\+\-\*/]', left.lstrip('-')):
                continue
            if left.lstrip()[0] in '+-*/':
                continue  # "Jayden - 15": operand hidden behind a word
            if re.match(rf'\s*{re.escape(right)}\s*,', parts[k+1]):
                continue  # "= 1,680," : number cut by a trailing comma, value unknown
            if not re.search(r'[\+\-\*/]', left.lstrip('-')+right.lstrip('-')):
                continue  # "5 = 5", nothing to check
            lv, rv = _arith(left), _arith(right)
            if lv is not None and rv is not None:
                out.append((rv, lv))
    return out


def _near(a, b):
    return abs(a-b) <= max(Fraction(1, 100), abs(b)*Fraction(1, 1000))


def _close(claimed, computed):
    # "(x/y) * 100 = 24%" : the % on the claim was normalised to /100
    return _near(claimed, computed) or _near(claimed*100, computed)


def _numbers(text):
    vals = set()
    for m in re.findall(NUM, text):
        try:
            vals.add(Fraction(m.replace(',', '')))
        except (ValueError, ZeroDivisionError):
            pass
    return vals


def candidate_verify_features(question, text, question_numbers=None):
    if question_numbers is None:
        question_numbers = _numbers(question)
    checks = check_equations(text)
    wrong = sum(not _close(c, v) for c, v in checks)
    answer = parse_answer(text)
    value = Fraction(answer) if answer is not None else None
    results = [v for _, v in checks]
    body = text.rsplit('####', 1)[0] if '####' in text else text
    body_numbers = _numbers(body.replace(f'\\boxed{{{answer}}}', ' ')) if answer else set()
    unused = len(question_numbers - _numbers(text))
    max_q = max((abs(q) for q in question_numbers), default=Fraction(0))
    return [
        len(checks),
        wrong,
        wrong/len(checks) if checks else 0.0,
        float(wrong > 0),
        float(value is not None and bool(results) and _close(results[-1], value)),
        float(value is not None and any(_close(r, value) for r in results)),
        float(value is not None and value in body_numbers),
        unused,
        float(np.log((abs(float(value))+1)/(float(max_q)+1))) if value is not None else 0.0,
    ]


def verify_feature_matrix(bank):
    out = np.zeros((len(bank), 4, len(VERIFY_FEATURE_NAMES)), dtype=np.float32)
    for i, row in enumerate(bank):
        qnums = _numbers(row['question'])
        for j, cand in enumerate(row['candidates']):
            out[i, j] = candidate_verify_features(row['question'], cand['text'], qnums)
    return out
