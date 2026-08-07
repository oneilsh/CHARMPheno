"""Unit tests for the ``PCTopicModel`` wrapper: fit decreases the objective,
``lam=0`` gives a chance-level / zero-influence head (unsupervised LDA-MAP),
``transform`` is label-free and shaped right, and refits are deterministic."""
from __future__ import annotations

import numpy as np
import pytest

from analysis.pc.model import PCTopicModel


def _toy_corpus(seed=0, D=40, V=12, K_true=3):
    """A small bag-of-words corpus with genuine topic co-occurrence structure
    and a binary label loosely tied to one topic's weight."""
    rng = np.random.default_rng(seed)
    # K_true topics over V words, each concentrated on a disjoint word block.
    topics = np.full((K_true, V), 0.02)
    block = V // K_true
    for k in range(K_true):
        topics[k, k * block:(k + 1) * block] += 1.0
    topics /= topics.sum(axis=1, keepdims=True)

    theta = rng.dirichlet(np.full(K_true, 0.3), size=D)
    n_tok = 60
    X = np.zeros((D, V))
    for d in range(D):
        wt = theta[d] @ topics
        X[d] = rng.multinomial(n_tok, wt)
    y = (theta[:, 0] > np.median(theta[:, 0])).astype(int)
    return X, y


def test_fit_runs_and_decreases_objective():
    X, y = _toy_corpus()
    m = PCTopicModel(K=3, C=2, lam=1.0, max_iter=200, seed=0).fit(X, y)
    assert m.final_obj_ < m.init_obj_
    assert np.isfinite(m.final_obj_)


def test_shapes():
    X, y = _toy_corpus()
    D, V = X.shape
    m = PCTopicModel(K=4, C=2, lam=1.0, max_iter=100, seed=1).fit(X, y)
    assert m.beta_.shape == (4, V)
    assert m.eta_.shape == (2, 4)
    assert m.b_.shape == (2,)
    assert m.Pi_.shape == (D, 4)
    # simplex rows
    assert np.allclose(m.beta_.sum(axis=1), 1.0)
    assert np.allclose(m.Pi_.sum(axis=1), 1.0)

    P = m.predict_proba(X[:5])
    assert P.shape == (5, 2)
    assert np.allclose(P.sum(axis=1), 1.0)


def test_lam0_head_is_zero_and_predict_proba_is_chance():
    """lam=0 leaves the zero-initialized head untouched, so predict_proba is
    exactly uniform (chance) — the unsupervised LDA-MAP regime."""
    X, y = _toy_corpus()
    m = PCTopicModel(K=3, C=2, lam=0.0, max_iter=200, seed=0).fit(X, y)
    assert np.allclose(m.eta_, 0.0)
    assert np.allclose(m.b_, 0.0)
    P = m.predict_proba(X[:10])
    assert np.allclose(P, 0.5)


def test_lam0_ignores_labels():
    """With lam=0 the representation must not depend on y at all."""
    X, y = _toy_corpus()
    m0 = PCTopicModel(K=3, C=2, lam=0.0, max_iter=150, seed=2).fit(X, y)
    m1 = PCTopicModel(K=3, C=2, lam=0.0, max_iter=150, seed=2).fit(X, 1 - y)
    assert np.allclose(m0.beta_, m1.beta_)
    assert np.allclose(m0.Pi_, m1.Pi_)


def test_transform_is_label_free_and_deterministic():
    """transform never sees y and is reproducible; its output depends only on
    X_new and the fitted beta_."""
    X, y = _toy_corpus()
    m = PCTopicModel(K=3, C=2, lam=1.0, max_iter=150, seed=3).fit(X, y)
    Xte = _toy_corpus(seed=99, D=15)[0]

    Pi_a = m.transform(Xte)
    Pi_b = m.transform(Xte)
    assert np.allclose(Pi_a, Pi_b)  # deterministic
    assert Pi_a.shape == (15, 3)
    assert np.allclose(Pi_a.sum(axis=1), 1.0)

    # Two models with identical beta_ but that were fit with different labels
    # produce identical transforms (transform ignores the head entirely).
    m2 = PCTopicModel(K=3, C=2, lam=1.0, max_iter=150, seed=3)
    m2.beta_ = m.beta_.copy()
    m2.V_ = m.V_
    Pi_c = m2.transform(Xte)
    assert np.allclose(Pi_a, Pi_c)


def test_refit_same_seed_is_deterministic():
    X, y = _toy_corpus()
    m1 = PCTopicModel(K=3, C=2, lam=1.0, max_iter=150, seed=7).fit(X, y)
    m2 = PCTopicModel(K=3, C=2, lam=1.0, max_iter=150, seed=7).fit(X, y)
    assert np.allclose(m1.beta_, m2.beta_)
    assert np.allclose(m1.eta_, m2.eta_)
    assert np.allclose(m1.b_, m2.b_)
    assert np.allclose(m1.Pi_, m2.Pi_)
    assert np.allclose(m1.predict_proba(X[:8]), m2.predict_proba(X[:8]))


def test_labeled_mask_semisupervised():
    """A partially-labeled fit runs and still produces a usable head."""
    X, y = _toy_corpus()
    D = X.shape[0]
    mask = np.zeros(D, dtype=bool)
    mask[: D // 2] = True
    m = PCTopicModel(K=3, C=2, lam=1.0, max_iter=150, seed=0).fit(X, y, labeled_mask=mask)
    assert m.final_obj_ < m.init_obj_
    P = m.predict_proba(X[:5])
    assert P.shape == (5, 2)
