"""Synthetic known-signal gate for the FAITHFUL flat PC topic model (plan Task
A2/A4). This runs the faithful ``analysis.pc.model.PCTopicModel`` (label-free
NEF-MAP pi, autograd-through-inference, binary logistic head, single label C=1),
NOT the free-pi variant.

We plant a corpus in the classic Hughes motivating regime: a handful of
*dominant, label-irrelevant* topics carry most of the tokens and dominate the
word co-occurrence structure, plus ONE *subtle predictive* topic that carries a
minority of the tokens and whose per-document weight alone drives a binary
label. We fit with FEWER topics than there are dominant structural topics
(``K_fit=3`` < ``K_dom=4``), so an unsupervised fit is tempted to spend all its
topics compressing the dominant structure and never isolates the low-mass
predictive topic. Prediction-Constrained training, guided by the labels,
reshapes the *global* topics so one scarce topic captures the predictive
direction.

We compare three models on a held-out split by ROC AUC:

  1. **PC** (``weight_y > 0``): the faithful constrained objective.
  2. **Two-stage**: the SAME class with ``weight_y=0`` (unsupervised LDA-MAP) for
     the representation, then a plain ``LogisticRegression`` on its label-free
     topic proportions — the Hughes baseline PC is meant to beat. Because the
     faithful model's ``transform`` is the *identical* label-free routine used at
     train time, there is no train/test representation mismatch here.
  3. **LR-on-codes**: logistic regression on the raw counts — a sanity reference
     confirming the signal is genuinely present in the data.

The gate asserts PC clears chance by a wide margin AND beats the two-stage
baseline by a real margin.

Regime (fixed): D=600 docs, V=60 vocab, K_dom=4 dominant topics, K_fit=3 fitted
topics, signal block = 10 unique words, dominant token share ~0.55, logistic
label slope=28, 180 tokens/doc, 70/30 split, data seed 0. Reference AUCs at this
seed (weight_y=50, pi_iters=100, max_iter=150): PC~0.956, two-stage~0.767,
LR-on-codes~0.949 (see module test output).
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from analysis.pc.model import PCTopicModel

# --- fixed synthetic regime ------------------------------------------------
SEED = 0
D, V = 600, 60
K_DOM = 4          # dominant (label-irrelevant) structural topics
K_FIT = 3          # fitted topics: deliberately fewer than K_DOM
SIG_BLOCK = 10     # unique words owned by the subtle predictive topic
DOM_MASS = 0.55    # average token share of the dominant topics per doc
LABEL_SLOPE = 28.0 # logistic slope mapping the signal-topic weight to y
N_TOK = 180        # tokens per document
TEST_FRAC = 0.30

# fixed fit controls (see module docstring for the resulting reference AUCs)
WEIGHT_Y = 50.0
ALPHA = 1.1        # authors' default; alpha == 1 is degenerate for the NEF MAP
PI_ITERS = 100     # faithful unroll count = authors' pi_max_iters default
MAX_ITER = 150


def _make_corpus(seed):
    """Bag-of-words corpus with K_DOM dominant topics on disjoint word blocks
    plus one subtle predictive topic on its own SIG_BLOCK words. The predictive
    topic's per-doc weight (a minority of the mass) drives a binary label via a
    logistic link centered so the classes are balanced.
    """
    rng = np.random.default_rng(seed)
    K_true = K_DOM + 1
    sig = K_DOM  # index of the predictive topic

    # Topic-word distributions (rows on the V-simplex).
    topics = np.full((K_true, V), 0.01)
    dom_region = V - SIG_BLOCK
    dom_bl = dom_region // K_DOM
    for k in range(K_DOM):
        topics[k, k * dom_bl:(k + 1) * dom_bl] += 1.0   # dominant disjoint blocks
    topics[sig, V - SIG_BLOCK:] += 1.0                  # predictive topic's own block
    topics /= topics.sum(axis=1, keepdims=True)

    # Doc-topic weights: dominant topics carry DOM_MASS on average; the
    # predictive topic carries the rest but VARIES doc-to-doc (that variation
    # is the signal).
    theta = np.zeros((D, K_true))
    theta[:, :K_DOM] = rng.dirichlet(np.full(K_DOM, 0.4), size=D) * DOM_MASS
    theta[:, sig] = rng.uniform(0.0, 2 * (1 - DOM_MASS), size=D)
    theta /= theta.sum(axis=1, keepdims=True)

    X = np.zeros((D, V))
    for d in range(D):
        X[d] = rng.multinomial(N_TOK, theta[d] @ topics)

    center = np.median(theta[:, sig])
    p = 1.0 / (1.0 + np.exp(-LABEL_SLOPE * (theta[:, sig] - center)))
    y = rng.binomial(1, p)
    return X, y


def test_pc_recovers_planted_signal_and_beats_two_stage(capsys):
    X, y = _make_corpus(SEED)
    n_te = int(TEST_FRAC * D)
    n_tr = D - n_te
    Xtr, Xte = X[:n_tr], X[n_tr:]
    ytr, yte = y[:n_tr], y[n_tr:]

    # 1) PC (supervised, constrained; single binary label => C=1).
    pc = PCTopicModel(
        K=K_FIT, C=1, weight_y=WEIGHT_Y, alpha=ALPHA,
        pi_iters=PI_ITERS, max_iter=MAX_ITER, seed=0,
    ).fit(Xtr, ytr)
    pc_auc = roc_auc_score(yte, pc.predict_proba(Xte)[:, 0])

    # 2) Two-stage: unsupervised (weight_y=0) representation -> logistic regression
    # on the label-free topic proportions (train and test via the same routine).
    unsup = PCTopicModel(
        K=K_FIT, C=1, weight_y=0.0, alpha=ALPHA,
        pi_iters=PI_ITERS, max_iter=MAX_ITER, seed=0,
    ).fit(Xtr, ytr)  # label ignored at weight_y=0
    lr_topics = LogisticRegression(max_iter=1000).fit(unsup.Pi_, ytr)
    two_stage_auc = roc_auc_score(
        yte, lr_topics.predict_proba(unsup.transform(Xte))[:, 1]
    )

    # 3) LR on raw codes (sanity: the signal really is in the data).
    lr_codes = LogisticRegression(max_iter=2000).fit(Xtr, ytr)
    lr_codes_auc = roc_auc_score(yte, lr_codes.predict_proba(Xte)[:, 1])

    with capsys.disabled():
        print(
            f"\n[synthetic signal / faithful PC] heldout ROC AUC — "
            f"PC={pc_auc:.4f}  two-stage={two_stage_auc:.4f}  "
            f"LR-on-codes={lr_codes_auc:.4f}  (pos rate te={yte.mean():.2f})"
        )

    # PC clears chance by a wide margin.
    assert pc_auc > 0.80, f"PC heldout AUC {pc_auc:.3f} not clearly above chance"
    # PC beats the two-stage baseline by a real margin (the Hughes result).
    assert pc_auc > two_stage_auc + 0.05, (
        f"PC {pc_auc:.3f} did not beat two-stage {two_stage_auc:.3f} by margin"
    )
    # Sanity: the planted signal is present in the raw features.
    assert lr_codes_auc > 0.72, (
        f"LR-on-codes {lr_codes_auc:.3f}: signal not present in raw data"
    )
