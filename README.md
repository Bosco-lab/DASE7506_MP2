# DASE7506 MP2: Listwise Linear Verifier with Answer-Level Pooling

This repository selects one of four candidate solutions per GSM8K question. The method is a linear scorer over the supplied embeddings, text statistics and numeric-verification features. It is trained with a listwise softmax loss, and candidate scores are pooled by parsed answer before the final choice.

| | |
|---|---|
| Official test accuracy | **83.30%** (389 / 467), from [`code/artifacts/test-score.json`](code/artifacts/test-score.json) |
| Protocol | `dase7506-mp2-open-hard-verifier-v5` |
| Report | [`report/report.pdf`](report/report.pdf) (10 pages) |
| Code version | `<repo-url>` at commit `<commit-hash>` |
| Checkpoint | [`code/artifacts/model_bundle/`](code/artifacts/model_bundle/) (14,081 B in total) |

The task description and the official running instructions are in [`guide/GUIDE.md`](guide/GUIDE.md) and [`code/README.md`](code/README.md). Those files are unchanged from the original release.

## 1. Setup

Run all commands from `code/` with Python 3.12. The complete data is included in `code/data/`.

```bash
cd code
python -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python check_data.py
```

`check_data.py` must report `status: passed`. Training uses NumPy, SciPy and scikit-learn (pinned in `requirements.txt`). Inference uses NumPy and the Python standard library only.

## 2. Reproduce the test score without retraining

The submitted checkpoint is in `code/artifacts/model_bundle/`. These commands read it and write their outputs to `code/runs/`:

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
python predict.py --method submission.py --checkpoint artifacts/model_bundle \
  --split test --output runs/test-predictions.json
python score_predictions.py --split test \
  --predictions runs/test-predictions.json --output runs/test-score.json
```

The `accuracy` field in `runs/test-score.json` must equal the value in `artifacts/test-score.json` (`83.29764453961457`). On Windows, set the three thread variables in the shell first, then run the same two Python commands.

The commands in `code/README.md` use `--checkpoint runs/model_bundle`. For this repository, use `--checkpoint artifacts/model_bundle` instead.

## 3. Retrain the checkpoint

Training runs on CPU and takes about two minutes. It reads the train split only and writes to `code/runs/`, so the submitted checkpoint in `artifacts/` is not changed.

```bash
python train_ranker.py --features emb,text,verify --loss listwise --output runs/rank_etv_list
python fit_aggregate.py --checkpoint runs/rank_etv_list --output runs/model_bundle
```

- `train_ranker.py` selects `C` by 5-fold cross-validation on train (seed 0) and fits the linear scorer.
- `fit_aggregate.py` selects the pooling temperature `T` on the train out-of-fold scores and writes the final checkpoint.

The retrained `runs/model_bundle/model.npz` is byte-identical to the submitted one. Check this with the hash in section 5. The retrained `config.json` has the same `features`, `loss`, `C`, `seed` and `aggregate_T`, but it also stores the full cross-validation tables, so its hash differs from the submitted `config.json`. The submitted file keeps only the fields needed for inference, plus a provenance record. To evaluate the retrained checkpoint, run the section 2 commands with `--checkpoint runs/model_bundle`. Its test predictions are identical to the submitted ones.

## 4. Repository layout

```
.
├── README.md                 this file
├── report/
│   ├── report.pdf            the report
│   ├── figures/              report figures
│   └── figures_data.json     numbers behind the figures
├── guide/GUIDE.md            original assignment guide (unchanged)
└── code/
    ├── README.md             original running instructions (unchanged)
    ├── artifacts/            submitted checkpoint and official results (read-only)
    ├── data/                 original data (unchanged)
    └── *.py                  original tools and the method (see tables below)
```

`code/runs/` holds training and evaluation outputs. It is excluded by `.gitignore` and is created when you run the commands above.

**Submitted artifacts (`code/artifacts/`)**

| File | Content |
|---|---|
| `model_bundle/model.npz` | Learned weights: feature mean and std, coefficients, intercept (13,594 B) |
| `model_bundle/config.json` | Feature blocks, `C`, seed, `aggregate_T` and provenance (487 B) |
| `test-predictions.json` | Official test predictions of the frozen method |
| `test-score.json` | Official scorer output for the test split |
| `audit-predictions.json` | Predictions on a simulated audit bank (see section 6) |
| `ensemble_cv.json` | Train cross-validation results of the ensemble experiment (a negative result) |

**Original files (unchanged unless noted)**

| File | Role |
|---|---|
| `common.py`, `predict.py`, `score_predictions.py`, `check_data.py` | Official data loading, prediction runner, scorer and data checker |
| `submission.py` | Entry point `predict(bank, arrays, checkpoint)`. **Implemented by me**; the original raised `NotImplementedError` |
| `requirements.txt` | **Modified**: SciPy and scikit-learn added for training |
| `README.md`, `PACKAGE_MANIFEST.json`, `data/` | Original instructions, release hashes and data |

**Inference** (`submission.py` imports these; all must be present)

| File | Role |
|---|---|
| `scorer_method.py` | Loads the checkpoint, builds features, scores candidates and pools scores by answer |
| `features.py` | 17 per-candidate text features |
| `verify_features.py` | 9 numeric-verification features: exact re-evaluation of the solution's arithmetic |
| `group_features.py` | 10 within-question features. Not used by the frozen configuration, but imported by `scorer_method.py` |
| `process_features.py` | 10 process-consistency features. Not used by the frozen configuration, but imported by `scorer_method.py` |

**Training**

| File | Role |
|---|---|
| `train_ranker.py` | Trains the listwise or pointwise linear scorer on any set of feature blocks. Produces the final model |
| `fit_aggregate.py` | Selects the pooling temperature `T` on train out-of-fold scores |
| `train_scorer.py` | Per-candidate logistic-regression scorer (an earlier comparison method) |
| `vote_method.py` | Baseline: majority vote with a longest-solution tie-break. Needs no checkpoint |

**Experiments and analysis** (used for the report; not needed for inference)

| File | Role |
|---|---|
| `dev_eval.py` | Development evaluator for train and validation: full, hard and easy subsets. Refuses the test split |
| `ablate.py` | Feature-block ablations |
| `ensemble_check.py` | Ensemble of rankers (a negative result) |
| `train_mlp.py` | One-hidden-layer MLP (a negative result) |
| `probe_headroom.py` | Nearest-neighbour prior and whitened-embedding probes (negative results) |
| `select_by_agg_cv.py` | Selects `C` and `T` on the aggregated metric (report section on the selection metric) |
| `make_v2.py` | Builds the alternative checkpoint from that selection. Not adopted |
| `failures.py` | Lists validation errors of the frozen method by question type |
| `test_breakdown.py` | Post-hoc test breakdown by question type and confidence. Run after freezing only |
| `make_figures.py` | Draws the report figures. Needs the validation outputs from `dev_eval.py` in `runs/dev/` |
| `build_audit_sim.py` | Builds a simulated audit bank |
| `time_frozen.py` | Measures prediction plus scoring end to end |
| `freeze_report.py` | Prints asset sizes and hashes, and checks the inference import chain |

## 5. Hashes and seeds

| File | sha256 |
|---|---|
| `artifacts/model_bundle/model.npz` | `b5ddef691e285872f21983d2c52c047adb1f2f4543b55be12d5ec54283319818` |
| `artifacts/model_bundle/config.json` | `0dd16cb0bb39c2f3cd0ae0089e1033b5b2bd0224f55e1100b06cb065c31e3d5a` |

`config.json` also records the `model.npz` hash. Compute the hashes with `sha256sum` (Linux, macOS) or `Get-FileHash` (Windows PowerShell).

All cross-validation uses 5 folds with seed 0. `build_audit_sim.py` samples questions with seed 7. No other step is random.

## 6. Resources and checks

| Item | Limit | Measured |
|---|---|---|
| Prediction plus scoring, 4 threads | ≤ 30 s | 2.02 s (median of 3 runs on validation, 798 questions; both processes started fresh) |
| Inference assets | ≤ 16 MiB | 0.0134 MiB (14,081 B) |
| Inference dependencies | — | NumPy and the standard library only |
| Training | CPU | CPU only, about 65 s per model |
| Local hardware | — | AMD Ryzen 9 7945HX (16 cores, 32 threads), 15.7 GB RAM, Windows 11 |

Validation has more questions than test, so the timing on validation is an upper bound for test. To measure the time, and to check the asset sizes and the import chain:

```bash
python time_frozen.py
python freeze_report.py
```

The method does not depend on question IDs, order or count. To check this on a simulated audit bank (137 validation questions in a new order, with new IDs and no references):

```bash
python build_audit_sim.py --n 137 --output runs/audit_sim
python predict.py --method submission.py --checkpoint artifacts/model_bundle \
  --split audit --data-dir runs/audit_sim --output runs/audit-predictions.json
```

## 7. Data use

The training scripts read the train split only. The ablations also read validation. The test references were opened only after the method was frozen, for the final score and the post-hoc breakdowns in the report.

## 8. AI disclosure

This project was carried out with substantial help from Claude (Anthropic), used as a coding and analysis assistant throughout. Claude:

- wrote the code: feature extraction, the numeric verifier, training, evaluation, ablation and figure scripts;
- proposed most method components and experiments, including the listwise loss, answer-level pooling, the verification and process-consistency features, and the probes that produced the negative results;
- ran the experiments and summarised the results after each step;
- carried out the failure analysis, quantifying the error types and reading and grouping the majority-wrong cases;
- drafted the report and its figures, and cross-checked the reported numbers against the experiment records.

I directed the work step by step, made the decisions at each choice point (for example the tie-breaking rule, the final configuration, and stopping further tuning instead of re-running the test bank), and reviewed the results and the report. I take responsibility for all of its content.

## 9. Attribution

Questions and references derive from [GSM8K](https://github.com/openai/grade-school-math) (MIT license). Candidate solutions and embeddings were prepared by the course with [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B). See `code/data/LICENSE.txt` and `code/data/NOTICE.txt`.
