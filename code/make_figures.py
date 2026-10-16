"""Generate the report figures from saved artifacts (no test-time selection, plotting only).

Sources: runs/dev/*.json (per-question hits), artifacts/test-*.json, artifacts/model_bundle.
Palette: #047857 / #b45309 / #4f46e5, validated for CVD separation and contrast.
"""
import collections
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binomtest

import os
os.environ.setdefault('OMP_NUM_THREADS', '4')

from common import load_inputs, read_jsonl
from features import parse_answer
from scorer_method import candidate_scores, load_scorer

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT/'report'/'figures'
FIG.mkdir(parents=True, exist_ok=True)

TEAL, AMBER, INDIGO = '#047857', '#b45309', '#4f46e5'
INK, MUTED, GRID = '#1f2937', '#6b7280', '#e5e7eb'
plt.rcParams.update({
    'figure.dpi': 160, 'savefig.dpi': 160, 'font.size': 9.5,
    'axes.edgecolor': GRID, 'axes.labelcolor': INK, 'axes.titlecolor': INK,
    'text.color': INK, 'xtick.color': MUTED, 'ytick.color': MUTED,
    'axes.spines.top': False, 'axes.spines.right': False,
    'font.family': 'DejaVu Sans', 'figure.facecolor': 'white', 'axes.facecolor': 'white',
})


def load_dev(name):
    d = json.load(open(ROOT/'code'/'runs'/'dev'/f'{name}.json'))
    return np.array(d['hit']), np.array(d['hard_mask']), np.array(d['selected'])


def test_labels():
    bank, arrays = load_inputs('test')
    gold = read_jsonl(ROOT/'code'/'data'/'gsm8k_test.jsonl')
    correct = np.zeros((len(bank), 4), dtype=bool)
    for i, (row, ref) in enumerate(zip(bank, gold)):
        a = [parse_answer(c['text']) for c in row['candidates']]
        t = parse_answer(ref['answer'])
        correct[i] = [x is not None and x == t for x in a]
    return bank, arrays, correct


def kinds(bank, correct):
    out = []
    for row, cor in zip(bank, correct):
        a = [parse_answer(c['text']) for c in row['candidates']]
        cnt = collections.Counter(v for v in a if v is not None)
        top = max(cnt.values())
        win = [v for v, n in cnt.items() if n == top]
        if len(win) > 1:
            out.append('tie')
        elif any(cor[j] for j in range(4) if a[j] == win[0]):
            out.append('majority right')
        else:
            out.append('majority wrong')
    return np.array(out)


# ---------------------------------------------------------------- fig 1: T sweep
T_SWEEP = {'0.05': 81.22, '0.1': 81.22, '0.2': 81.27, '0.3': 81.32, '0.5': 81.69,
           '0.75': 81.84, '1.0': 82.26, '1.5': 81.93, '2.0': 81.46, '3.0': 80.09,
           '5.0': 77.91, 'inf': 75.78}
xs = [0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0]
ys = list(T_SWEEP.values())
fig, ax = plt.subplots(figsize=(5.4, 2.9))
ax.plot(xs, ys, color=TEAL, lw=2, zorder=3)
ax.scatter([1.0], [82.26], s=52, color=TEAL, zorder=4, edgecolor='white', lw=1.5)
ax.annotate('T = 1.0\ntrain CV 82.26', (1.0, 82.26), textcoords='offset points', xytext=(12, -20),
            fontsize=8.5, color=INK, arrowprops=dict(arrowstyle='-', color=MUTED, lw=0.8))
ax.scatter([8.0], [75.78], s=40, color=AMBER, zorder=4, edgecolor='white', lw=1.5)
ax.annotate('T = ∞ : vote alone\n75.78', (8.0, 75.78), textcoords='offset points', xytext=(-92, 10),
            fontsize=8.5, color=INK, arrowprops=dict(arrowstyle='-', color=MUTED, lw=0.8))
ax.scatter([0.05], [81.22], s=40, color=INDIGO, zorder=4, edgecolor='white', lw=1.5)
ax.annotate('T → 0 : scores alone\n81.22', (0.05, 81.22), textcoords='offset points', xytext=(6, 10),
            fontsize=8.5, color=INK, arrowprops=dict(arrowstyle='-', color=MUTED, lw=0.8))
ax.set_xscale('log')
ax.set_xlim(0.04, 9.5)
ax.set_ylim(74.5, 83.5)
ax.set_xticks([0.05, 0.1, 0.5, 1, 2, 5, 8])
ax.set_xticklabels(['0.05', '0.1', '0.5', '1', '2', '5', '∞'])
ax.set_xlabel('answer-aggregation temperature T  (log scale)')
ax.set_ylabel('train CV accuracy, %')
ax.set_title('Pooling candidate scores by answer: the optimum sits between the two extremes',
             fontsize=9.5, loc='left', pad=8)
ax.grid(axis='y', color=GRID, lw=0.7)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(FIG/'fig1_temperature.png', bbox_inches='tight')
plt.close(fig)

# ------------------------------------------------------- fig 2: per-type accuracy
vbank, _, vcorrect, vhard = None, None, None, None
vhit_final, vhard_mask, _ = load_dev('model_bundle')
vhit_vote, _, _ = load_dev('vote-longest')
vhit_scorer, _, _ = load_dev('scorer-both')

from dev_eval import load_labels
vb, va, vcorrect, vhard = load_labels('validation')
vk = kinds(vb, vcorrect)

tbank, tarrays, tcorrect = test_labels()
tk = kinds(tbank, tcorrect)
tsel = np.array(json.load(open(ROOT/'code'/'artifacts'/'test-predictions.json'))['selected'])
thit = tcorrect[np.arange(len(tbank)), tsel]

order = ['majority right', 'tie', 'majority wrong']
groups = {
    'majority vote': [100*vhit_vote[vhard & (vk == k)].mean() for k in order],
    'single-candidate scorer': [100*vhit_scorer[vhard & (vk == k)].mean() for k in order],
    'final method': [100*vhit_final[vhard & (vk == k)].mean() for k in order],
}
counts_val = {k: int((vhard & (vk == k)).sum()) for k in order}
counts_test = {k: int((tk == k).sum()) for k in order}

fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.0), gridspec_kw={'width_ratios': [1.35, 1]})
ax = axes[0]
x = np.arange(len(order))
w = 0.26
for idx, (name, vals) in enumerate(groups.items()):
    color = [TEAL, INDIGO, AMBER][idx]
    off = (idx-1)*w
    ax.bar(x+off, vals, w-0.03, color=color, label=name, zorder=3)
    for xi, v in zip(x+off, vals):
        ax.text(xi, v+1.2, f'{v:.0f}', ha='center', fontsize=7.6, color=INK)
ax.set_xticks(x)
ax.set_xticklabels([f'{k}\n(n={counts_val[k]})' for k in order], fontsize=8.5)
ax.set_ylim(0, 112)
ax.set_ylabel('validation accuracy, %')
ax.set_title('Where the gain comes from — hard questions only', fontsize=9.5, loc='left', pad=8)
ax.legend(frameon=False, fontsize=8, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.02))
ax.grid(axis='y', color=GRID, lw=0.7)
ax.set_axisbelow(True)

ax = axes[1]
tv = [100*thit[tk == k].mean() for k in order]
ax.bar(x, tv, 0.5, color=[TEAL, INDIGO, AMBER], zorder=3)
for xi, v, k in zip(x, tv, order):
    ax.text(xi, v+1.5, f'{v:.1f}', ha='center', fontsize=8.2, color=INK)
    ax.text(xi, 3, f'{counts_test[k]}/467', ha='center', fontsize=7.4, color='white')
ax.set_xticks(x)
ax.set_xticklabels(['majority\nright', 'tie', 'majority\nwrong'], fontsize=8.5)
ax.set_ylim(0, 112)
ax.set_title('Frozen method on test', fontsize=9.5, loc='left', pad=8)
ax.grid(axis='y', color=GRID, lw=0.7)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(FIG/'fig2_by_type.png', bbox_inches='tight')
plt.close(fig)

# ------------------------------------------------------------ fig 3: calibration
def reliability(bank, arrays, pick):
    """Confidence = pooled weight of the chosen answer, plus whether it was correct."""
    cfg, params = load_scorer(ROOT/'code'/'artifacts'/'model_bundle')
    s = candidate_scores(bank, arrays, cfg, params)
    T = float(cfg['aggregate_T'])
    ans = [[parse_answer(c['text']) for c in r['candidates']] for r in bank]
    conf = np.zeros(len(bank))
    for i in range(len(bank)):
        z = s[i]/T
        w = np.exp(z - z.max())
        w /= w.sum()
        a = ans[i][pick[i]]
        conf[i] = w[[k for k in range(4) if ans[i][k] == a]].sum() if a is not None else w[pick[i]]
    return conf


vpick = np.array(json.load(open(ROOT/'code'/'runs'/'dev'/'model_bundle.json'))['selected'])
vconf = reliability(vb, va, vpick)
vhit_arr = vcorrect[np.arange(len(vb)), vpick]
tconf = reliability(tbank, tarrays, tsel)
thit_arr = thit

edges = [0.0, 0.5, 0.7, 0.85, 0.95, 1.001]
labels = ['<0.5', '0.5–0.7', '0.7–0.85', '0.85–0.95', '≥0.95']
vx, tx, vn, tn = [], [], [], []
for lo, hi in zip(edges, edges[1:]):
    m = (vconf >= lo) & (vconf < hi) & vhard
    vx.append(100*vhit_arr[m].mean() if m.sum() else np.nan); vn.append(int(m.sum()))
    m2 = (tconf >= lo) & (tconf < hi)
    tx.append(100*thit_arr[m2].mean() if m2.sum() else np.nan); tn.append(int(m2.sum()))

fig, ax = plt.subplots(figsize=(5.4, 3.0))
xs2 = np.arange(len(labels))
ax.plot(xs2, vx, color=INDIGO, lw=2, marker='o', ms=6, mec='white', mew=1.4, label='validation (n=355)')
ax.plot(xs2, tx, color=TEAL, lw=2, marker='s', ms=6, mec='white', mew=1.4, label='test (n=467)')
for xi, v, n in zip(xs2, tx, tn):
    ax.annotate(f'{v:.0f}', (xi, v), textcoords='offset points', xytext=(0, 9), ha='center', fontsize=8.2, color=INK)
for xi, v, n in zip(xs2, vx, vn):
    ax.annotate(f'{v:.0f}', (xi, v), textcoords='offset points', xytext=(0, -15), ha='center', fontsize=8.2, color=INK)
ax.set_xticks(xs2)
ax.set_xticklabels(labels)
ax.set_ylim(20, 108)
ax.set_xlabel('confidence = pooled weight of the chosen answer')
ax.set_ylabel('accuracy, %')
ax.set_title('High confidence means high accuracy — monotone on unseen test data',
             fontsize=9.5, loc='left', pad=8)
ax.legend(frameon=False, fontsize=8.5, loc='lower right')
ax.grid(axis='y', color=GRID, lw=0.7)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(FIG/'fig3_reliability.png', bbox_inches='tight')
plt.close(fig)

# --------------------------------------------------------------- fig 4: ablation
ABL = [('embedding', -6.86), ('text features', -1.51), ('verification', -0.61),
       ('process-consistency', +0.66), ('within-question', -0.24)]
fig, ax = plt.subplots(figsize=(5.4, 2.4))
names = [a for a, _ in ABL]
vals = [b for _, b in ABL]
y = np.arange(len(names))[::-1]
ax.barh(y, vals, 0.5, color=[TEAL if v < 0 else AMBER for v in vals], zorder=3)
for yi, v in zip(y, vals):
    ax.text(v + (-0.25 if v < 0 else 0.25), yi, f'{v:+.2f}', va='center',
            ha='right' if v < 0 else 'left', fontsize=8.2, color=INK)
ax.axvline(0, color=MUTED, lw=1)
ax.set_yticks(y)
ax.set_yticklabels(names)
ax.set_xlim(-8.2, 2.4)
ax.set_xlabel('change in train-CV hard accuracy when the block is removed (points)')
ax.set_title('Contribution of each evidence block', fontsize=9.5, loc='left', pad=8)
ax.grid(axis='x', color=GRID, lw=0.7)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(FIG/'fig4_ablation.png', bbox_inches='tight')
plt.close(fig)

# ------------------------------------------------------------------- numbers.json
out = {'temperature_sweep': T_SWEEP, 'by_type_val': groups, 'counts_val': counts_val,
       'by_type_test': dict(zip(order, tv)), 'counts_test': counts_test,
       'reliability': {'bins': labels, 'val': {'acc': vx, 'n': vn}, 'test': {'acc': tx, 'n': tn}},
       'ablation': dict(ABL)}
(FIG.parent/'figures_data.json').write_text(json.dumps(out, indent=2)+'\n')
print('figures written to', FIG)
for f in sorted(FIG.glob('*.png')):
    print(' ', f.name, f.stat().st_size, 'B')
print()
print('val by type  :', groups)
print('test by type :', dict(zip(order, [round(v, 1) for v in tv])))
print('reliability test:', list(zip(labels, [round(v, 1) for v in tx], tn)))
