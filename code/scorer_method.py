"""Linear candidate scorer: pick the candidate with the highest score.

Inference uses NumPy only (no sklearn import). The checkpoint directory comes
from train_scorer.py or train_ranker.py and holds
model.npz + config.json.
"""
import json
from pathlib import Path

import numpy as np

from features import parse_answer, text_feature_matrix
from group_features import group_feature_matrix
from process_features import process_feature_matrix
from verify_features import verify_feature_matrix

LEGACY_BLOCKS = {'emb': ['emb'], 'text': ['text'], 'both': ['emb', 'text']}


def feature_blocks(config):
    blocks = config['features']
    return LEGACY_BLOCKS[blocks] if isinstance(blocks, str) else list(blocks)


def build_inputs(bank, arrays, blocks):
    makers = {'emb': lambda: arrays['embeddings'].astype(np.float32),
              'text': lambda: text_feature_matrix(bank),
              'group': lambda: group_feature_matrix(bank, arrays),
              'verify': lambda: verify_feature_matrix(bank),
              'process': lambda: process_feature_matrix(bank)}
    return np.concatenate([makers[b]() for b in blocks], axis=-1)


def load_scorer(checkpoint):
    checkpoint = Path(checkpoint)
    config = json.loads((checkpoint/'config.json').read_text())
    with np.load(checkpoint/'model.npz') as f:
        params = {k: f[k] for k in ('mean', 'std', 'coef', 'intercept')}
    return config, params


def candidate_scores(bank, arrays, config, params):
    X = build_inputs(bank, arrays, feature_blocks(config))
    return ((X-params['mean'])/params['std']) @ params['coef'] + params['intercept']


def answer_aggregate(scores, answers, T):
    """Choose an answer, not a candidate.

    Each candidate gets weight softmax(score/T) within its question; candidates
    sharing a parsed answer pool their weight, and the best-scoring candidate of
    the heaviest answer is returned. T -> 0 recovers the plain argmax of scores;
    T = inf gives uniform weights, i.e. majority vote with the score breaking ties.
    Unparseable candidates form singleton groups.
    """
    choices = []
    for s, ans in zip(scores, answers):
        z = s/T if np.isfinite(T) else np.zeros_like(s)
        w = np.exp(z - z.max())
        w /= w.sum()
        groups = {}
        for j, a in enumerate(ans):
            groups.setdefault(a if a is not None else ('none', j), []).append(j)
        best = max(groups.values(), key=lambda idx: (round(float(w[idx].sum()), 9), float(s[idx].max())))
        choices.append(int(max(best, key=lambda j: s[j])))
    return choices


def predict(bank, arrays, checkpoint):
    if checkpoint is None:
        raise ValueError('scorer_method needs --checkpoint (run train_scorer.py or train_ranker.py first)')
    config, params = load_scorer(checkpoint)
    scores = candidate_scores(bank, arrays, config, params)
    if 'aggregate_T' in config:
        answers = [[parse_answer(c['text']) for c in row['candidates']] for row in bank]
        T = float(config['aggregate_T']) if config['aggregate_T'] != 'inf' else float('inf')
        return answer_aggregate(scores, answers, T)
    return [int(i) for i in scores.argmax(1)]
