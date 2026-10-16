"""Within-question (cross-candidate) features.

For a linear scorer, subtracting the group mean from a feature never changes
the argmax inside a group, so cross-candidate information has to enter through
non-linear group statistics: answer agreement, embedding similarity to the
other candidates, and within-group ranks.
"""
import numpy as np

from features import parse_answer

GROUP_FEATURE_NAMES = [
    'vote_share', 'is_plurality', 'is_strict_top', 'vote_margin',
    'sim_mean_others', 'sim_max_others', 'sim_same_answer', 'sim_diff_answer',
    'len_rank', 'len_z',
]


def group_feature_matrix(bank, arrays):
    """Return float32 array [N, 4, G] of within-question features."""
    emb = arrays['embeddings'].astype(np.float32)
    unit = emb/(np.linalg.norm(emb, axis=-1, keepdims=True)+1e-8)
    sims = np.einsum('nid,njd->nij', unit, unit)
    out = np.zeros((len(bank), 4, len(GROUP_FEATURE_NAMES)), dtype=np.float32)
    for i, row in enumerate(bank):
        cands = row['candidates']
        answers = [parse_answer(c['text']) for c in cands]
        counts = {}
        for a in answers:
            if a is not None:
                counts[a] = counts.get(a, 0)+1
        top = max(counts.values()) if counts else 0
        n_top = sum(v == top for v in counts.values())
        lengths = np.array([len(c['text']) for c in cands], dtype=np.float32)
        ranks = lengths.argsort().argsort()/3.0
        len_std = lengths.std()
        for j, a in enumerate(answers):
            own = counts.get(a, 0) if a is not None else 0
            best_other = max([v for b, v in counts.items() if b != a], default=0)
            others = [k for k in range(4) if k != j]
            same = [k for k in others if a is not None and answers[k] == a]
            diff = [k for k in others if k not in same]
            out[i, j] = [
                own/4,
                float(a is not None and own == top),
                float(a is not None and own == top and n_top == 1),
                (own-best_other)/4,
                sims[i, j, others].mean(),
                sims[i, j, others].max(),
                sims[i, j, same].mean() if same else 0.0,
                sims[i, j, diff].mean() if diff else 0.0,
                ranks[j],
                (lengths[j]-lengths.mean())/len_std if len_std > 0 else 0.0,
            ]
    return out
