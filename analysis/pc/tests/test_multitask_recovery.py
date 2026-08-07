"""Multi-task recovery gate for the joint / index-drug PC fit (per-cell missing
labels). Companion to the masked-core gates in ``test_multitask_masking.py``.

We plant a multi-outcome corpus in the Hughes motivating regime, then hand each
document a label for exactly ONE of the ``C`` outcomes (the index-drug pattern —
an almost-all-missing ``D x C`` label matrix with ~one observed cell per row) and
ask whether a SINGLE shared faithful PC, jointly fit across all heads under that
sparse per-cell supervision, still recovers each outcome's heldout signal and
beats the unsupervised two-stage baseline.

Regime: ``C = 2`` outcomes; ``K_DOM = 3`` dominant, label-irrelevant topics carry
most tokens (``dom_mass = 0.55``) and dominate the word co-occurrence structure;
``C`` subtle predictive topics each carry a minority of the mass and their
per-doc weight alone drives one outcome via a logistic link. We fit ``K = 4``
topics — enough for PC, guided by the sparse labels, to repurpose topics onto the
two predictive directions, but few enough that an unsupervised fit spends its
budget on the dominant structure and underperforms. Each outcome's label is
observed on ~half the docs (one of the two per row), so the two-stage / LR
baselines are fit on each column's own observed rows — the exact same supervision
the shared PC sees, isolating the benefit of jointly shaping the representation.

Deterministic given the seeds; sklearn is used only in the eval harness.
"""
from __future__ import annotations

import numpy as np

from analysis.pc.evaluate import evaluate_pc_multitask, format_results_table

# --- fixed multi-task synthetic regime (see module docstring) --------------
C = 2               # outcomes / heads
K_DOM = 3           # dominant, label-irrelevant structural topics
K_FIT = 4           # fitted topics
SIG_BLOCK = 8       # unique words per topic block
DOM_MASS = 0.55     # average token share of the dominant topics per doc
SLOPE = 26.0        # logistic slope mapping a predictive-topic weight to y
N_TOK = 220
D_TR, D_TE = 260, 140

WEIGHT_Y = 50.0
PI_ITERS = 50
MAX_ITER = 100


def _planted_topics():
    """The ``K_DOM + C`` planted topic-word rows on disjoint word blocks
    (``K_DOM`` dominant + ``C`` subtle predictive). Returns ``(topics, V)``."""
    K_true = K_DOM + C
    V = K_true * SIG_BLOCK
    topics = np.full((K_true, V), 0.01)
    for k in range(K_true):
        topics[k, k * SIG_BLOCK:(k + 1) * SIG_BLOCK] += 1.0   # disjoint blocks
    topics /= topics.sum(axis=1, keepdims=True)
    return topics, V


def _sample(seed, D, topics, V):
    """Draw ``D`` docs + their ``C`` outcome labels from the planted topics."""
    rng = np.random.default_rng(seed)
    K_true = K_DOM + C
    theta = np.zeros((D, K_true))
    theta[:, :K_DOM] = rng.dirichlet(np.full(K_DOM, 0.4), size=D) * DOM_MASS
    for j in range(C):
        theta[:, K_DOM + j] = rng.uniform(0.0, 2.0 * (1.0 - DOM_MASS) / C, size=D)
    theta /= theta.sum(axis=1, keepdims=True)

    X = np.zeros((D, V))
    for d in range(D):
        X[d] = rng.multinomial(N_TOK, theta[d] @ topics)

    y = np.zeros((D, C))
    for j in range(C):
        w = theta[:, K_DOM + j]
        center = np.median(w)
        p = 1.0 / (1.0 + np.exp(-SLOPE * (w - center)))
        y[:, j] = rng.binomial(1, p)
    return X, y


def _index_drug_mask(D, seed):
    """One observed outcome per document (the index-drug pattern): a ``(D, C)``
    mask with exactly one True per row, the observed outcome chosen at random."""
    rng = np.random.default_rng(seed)
    obs = rng.integers(0, C, size=D)
    m = np.zeros((D, C))
    m[np.arange(D), obs] = 1.0
    return m


def test_joint_pc_recovers_per_outcome_and_beats_two_stage(capsys):
    """One shared PC, jointly fit under one-observed-cell-per-row supervision,
    recovers each outcome's heldout signal and beats the unsupervised two-stage."""
    topics, V = _planted_topics()
    Xtr, Ytr = _sample(0, D_TR, topics, V)
    Xte, Yte = _sample(1, D_TE, topics, V)

    mask_tr = _index_drug_mask(D_TR, seed=10)          # index-drug: 1 obs/row
    mask_te = np.ones((D_TE, C))                        # score every test cell

    # exactly one observed cell per train row (the index-drug invariant)
    assert np.array_equal(mask_tr.sum(axis=1), np.ones(D_TR))

    res = evaluate_pc_multitask(
        Xtr, Ytr, mask_tr, Xte, Yte, mask_te,
        K=K_FIT, weight_y=WEIGHT_Y, pi_iters=PI_ITERS, max_iter=MAX_ITER, seed=0,
    )

    pc = res["PC"]["macro"]["auc"]
    ts = res["two_stage"]["macro"]["auc"]

    with capsys.disabled():
        print()
        print(format_results_table(res))

    # every outcome was scored (nothing degenerate on this balanced heldout set)
    assert res["PC"]["macro"]["n_labels_scored"] == C
    # each per-outcome heldout column recovers signal well above chance
    for c in range(C):
        assert res["PC"]["per_label"][c]["auc"] > 0.65, (
            f"outcome {c} PC AUC {res['PC']['per_label'][c]['auc']:.3f} too low"
        )
    # macro: PC recovers good AUC AND beats the unsupervised two-stage by a margin
    assert pc > 0.72, f"joint PC macro-AUC {pc:.3f} not clearly above chance"
    assert pc > ts + 0.03, (
        f"joint PC {pc:.3f} did not beat two-stage {ts:.3f} by margin"
    )


def test_multitask_recovery_is_deterministic():
    """Two identical multi-task runs are bit-for-bit identical (seeded init)."""
    topics, V = _planted_topics()
    Xtr, Ytr = _sample(0, D_TR, topics, V)
    Xte, Yte = _sample(1, D_TE, topics, V)
    mask_tr = _index_drug_mask(D_TR, seed=10)
    mask_te = np.ones((D_TE, C))
    kw = dict(K=K_FIT, weight_y=WEIGHT_Y, pi_iters=PI_ITERS, max_iter=MAX_ITER, seed=0)

    a = evaluate_pc_multitask(Xtr, Ytr, mask_tr, Xte, Yte, mask_te, **kw)
    b = evaluate_pc_multitask(Xtr, Ytr, mask_tr, Xte, Yte, mask_te, **kw)
    for key in ("PC", "two_stage", "lr_codes"):
        assert a[key]["macro"]["auc"] == b[key]["macro"]["auc"]
        assert a[key]["macro"]["ap"] == b[key]["macro"]["ap"]
