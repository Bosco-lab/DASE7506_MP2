"""Submission: listwise linear ranker + answer-level aggregation (agg_etv_list).

This file is the entry point only. The real work lives in four modules that sit
next to it, and all five must be shipped together:

  scorer_method.py    scores every candidate and turns scores into a choice
                      - load_scorer()        reads model.npz + config.json
                      - build_inputs()       concatenates the feature blocks
                      - candidate_scores()   standardise, then linear score
                      - answer_aggregate()   pool softmax(score / T) over
                                             candidates that share an answer,
                                             then pick the best candidate of
                                             the heaviest answer
  features.py         per-candidate text features (17), e.g. whether the answer
                      is an integer, is parseable, covers the question numbers
  group_features.py   within-question features (10), e.g. vote share, embedding
                      similarity to the other candidates
  verify_features.py  numeric verification (9): re-evaluates the arithmetic in
                      the solution exactly (Fraction + AST, no eval), and checks
                      whether the stated answer matches the last computed step

Checkpoint: artifacts/model_bundle/ holds model.npz (13,594 B, the learned weights)
and config.json (487 B: feature blocks, C, seed, aggregate_T, provenance).

How this method was produced (not needed at inference, but for reproduction):
  train_ranker.py    fits the linear model (features emb,text,verify; listwise
                     softmax loss; C chosen by 5-fold CV on train)
  fit_aggregate.py   picks the temperature T on train out-of-fold scores
  ablate.py          the feature ablations
  train_scorer.py    the per-candidate scorer
  vote_method.py     the baseline (majority vote, longest tiebreak)

Inference uses NumPy and the standard library only - no sklearn, scipy or torch.
"""
from scorer_method import predict as _predict


def predict(bank, arrays, checkpoint):
    """Return one Python integer in 0..3 per question, in bank order.

    bank: list of question dictionaries with four candidate texts each.
    arrays['embeddings']: float32 NumPy array of shape [N, 4, 1024].
    checkpoint: pathlib.Path to artifacts/model_bundle (required).
    """
    return _predict(bank, arrays, checkpoint)
