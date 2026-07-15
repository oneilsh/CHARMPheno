"""Unit tests for the K-1 reference-topic parameterization (opt-in)."""
from __future__ import annotations

import numpy as np
import pytest

from spark_vi.models.topic.stm import (
    OnlineSTM,
    _stm_doc_inference,
    _softmax,
)
from spark_vi.models.topic.types import STMDocument
from spark_vi.models.topic.partition import TopicBlockPartition


# A tiny deterministic 3-topic / 6-word beta. Three "pure" topics over disjoint
# 2-word blocks. Positivity is all the inference math needs, so we can pass this
# directly as expElogbeta in the primitive-level tests.
_BETA3 = np.array([
    [.45, .45, .02, .02, .03, .03],
    [.02, .02, .45, .45, .03, .03],
    [.03, .03, .02, .02, .45, .45],
])


def test_reference_inference_pins_reference_to_zero():
    """With reference=0, topic 0's eta is exactly 0, it has no variance, and
    theta is still a valid distribution that gives the reference positive mass.

    allowed=None means all 3 topics; marginal precision over full set = the full
    diagonal precision matrix."""
    K = 3
    full_prec = np.diag(np.full(K, 1.0 / 5.0))
    eta_hat, nu_d, _ = _stm_doc_inference(
        indices=np.array([2, 3], dtype=np.int32),
        counts=np.array([5.0, 5.0]),
        expElogbeta=_BETA3,
        Gamma=np.zeros((1, 3)),
        Sigma_inv_allowed=full_prec,
        x=np.array([1.0]),
        reference=0,
    )
    assert eta_hat[0] == 0.0
    assert nu_d[0, 0] == 0.0
    assert np.all(nu_d[0, :] == 0.0) and np.all(nu_d[:, 0] == 0.0)
    theta = _softmax(eta_hat)
    assert abs(float(theta.sum()) - 1.0) < 1e-12
    assert theta[0] > 0.0


def test_reference_inference_recovers_dominant_topic():
    """A doc made of pure topic-1 words puts most theta mass on topic 1 even
    though topic 0 is pinned as the reference — the free optimization works.

    allowed=None means all 3 topics; marginal precision over full set = full_prec."""
    K = 3
    full_prec = np.diag(np.full(K, 1.0 / 5.0))
    eta_hat, _, _ = _stm_doc_inference(
        indices=np.array([2, 3], dtype=np.int32),   # topic-1's words
        counts=np.array([8.0, 8.0]),
        expElogbeta=_BETA3,
        Gamma=np.zeros((1, 3)),
        Sigma_inv_allowed=full_prec,
        x=np.array([1.0]),
        reference=0,
    )
    theta = _softmax(eta_hat)
    assert int(np.argmax(theta)) == 1


def test_reference_inference_respects_allowed():
    """reference must sit inside allowed; disallowed topics stay at theta=0 and
    the reference is still pinned to 0.

    allowed=[0,1] (topic 2 disallowed); marginal precision over {0,1} is the
    2x2 sub-block of the full diagonal precision matrix."""
    K = 3
    full_prec = np.diag(np.full(K, 1.0 / 5.0))
    allowed = np.array([0, 1], dtype=np.int64)
    prec_allowed = full_prec[np.ix_(allowed, allowed)]
    eta_hat, nu_d, _ = _stm_doc_inference(
        indices=np.array([0, 1], dtype=np.int32),
        counts=np.array([4.0, 4.0]),
        expElogbeta=_BETA3,
        Gamma=np.zeros((1, 3)),
        Sigma_inv_allowed=prec_allowed,
        x=np.array([1.0]),
        allowed=allowed,   # topic 2 disallowed
        reference=0,
    )
    assert eta_hat[0] == 0.0
    assert eta_hat[2] == -np.inf
    theta = _softmax(eta_hat)
    assert theta[2] == 0.0
    assert abs(float(theta.sum()) - 1.0) < 1e-12


def test_reference_topic_requires_k_at_least_two():
    """reference_topic needs at least one free topic besides the reference."""
    with pytest.raises(ValueError, match="reference_topic requires K >= 2"):
        OnlineSTM(K=1, vocab_size=4, P=1, reference_topic=True)


def test_reference_index_toggles():
    assert OnlineSTM(K=3, vocab_size=4, P=1, reference_topic=False)._reference_index() is None
    assert OnlineSTM(K=3, vocab_size=4, P=1)._reference_index() == 0  # reference is now the default
    assert OnlineSTM(K=3, vocab_size=4, P=1, reference_topic=True)._reference_index() == 0


def test_reference_not_in_allowed_raises():
    """When reference is not in allowed, _stm_doc_inference must raise ValueError
    rather than silently misplacing the pin via np.searchsorted.

    allowed=[0,1,2] = all 3 topics; marginal precision over full set = full_prec."""
    K = 3
    full_prec = np.diag(np.full(K, 1.0 / 5.0))
    with pytest.raises(ValueError, match="reference"):
        _stm_doc_inference(
            indices=np.array([0, 1], dtype=np.int32),
            counts=np.array([4.0, 4.0]),
            expElogbeta=_BETA3,
            Gamma=np.zeros((1, 3)),
            Sigma_inv_allowed=full_prec,
            x=np.array([1.0]),
            allowed=np.array([0, 1, 2], dtype=np.int64),
            reference=5,
        )


def _toy_docs(rng, *, V, D, doc_len, K_blocks):
    """D docs over V words; each doc concentrates on one of K_blocks word
    blocks plus light noise. Integer token ids only (domain-agnostic)."""
    block = V // K_blocks
    docs = []
    for _ in range(D):
        b = int(rng.integers(K_blocks))
        toks = np.concatenate([
            rng.integers(b * block, (b + 1) * block, size=doc_len - 3),
            rng.integers(0, V, size=3),
        ])
        u, c = np.unique(toks, return_counts=True)
        docs.append(STMDocument(indices=u.astype(np.int32),
                                counts=c.astype(np.float64),
                                length=int(c.sum()), x=np.array([1.0])))
    return docs


def test_reference_gamma_column_zero_and_sigma_diagonal_pinned():
    """After full-batch updates with reference_topic, the reference topic's
    Gamma column stays 0 and its Sigma diagonal is pinned to 1 (block-wise
    unit-diagonal M-step pins every diagonal to 1 each step, independent of
    sigma_init)."""
    rng = np.random.default_rng(0)
    V, K = 30, 4
    docs = _toy_docs(rng, V=V, D=60, doc_len=20, K_blocks=K)
    m = OnlineSTM(K=K, vocab_size=V, P=1, sigma_init=3.0,
                  random_seed=1, reference_topic=True)
    gp = m.initialize_global(None)
    for _ in range(8):
        gp = m.update_global(gp, m.local_update(docs, gp), learning_rate=1.0)
    assert np.allclose(gp["Gamma"][:, 0], 0.0)
    # Unit-diagonal M-step pins Σ_ii = 1 for every topic (incl. the reference),
    # so the reference diagonal is 1 regardless of sigma_init=3.0.
    assert gp["Sigma"][0, 0] == 1.0


def test_reference_topic_still_learns_content():
    """The reference is a real topic: its lambda row carries vocabulary mass."""
    rng = np.random.default_rng(0)
    V, K = 30, 4
    docs = _toy_docs(rng, V=V, D=60, doc_len=20, K_blocks=K)
    m = OnlineSTM(K=K, vocab_size=V, P=1, random_seed=1, reference_topic=True)
    gp = m.initialize_global(None)
    for _ in range(8):
        gp = m.update_global(gp, m.local_update(docs, gp), learning_rate=1.0)
    assert gp["lambda"][0].sum() > V  # row mass above the eta=1/K prior floor


def test_reference_elbo_finite():
    """KL over the free subspace stays finite (the reference row/col of nu_d is
    excluded, so the sub-covariance is non-singular)."""
    rng = np.random.default_rng(0)
    V, K = 30, 4
    docs = _toy_docs(rng, V=V, D=60, doc_len=20, K_blocks=K)
    m = OnlineSTM(K=K, vocab_size=V, P=1, random_seed=1, reference_topic=True)
    gp = m.initialize_global(None)
    for _ in range(5):
        stats = m.local_update(docs, gp)
        gp = m.update_global(gp, stats, learning_rate=1.0)
    elbo = m.compute_elbo(gp, m.local_update(docs, gp))
    assert np.isfinite(elbo)


def test_reference_on_is_the_default():
    """reference_topic now defaults to True (validated default, insight 0030):
    omitting the kwarg is identical to reference_topic=True, and differs from the
    explicit-off (legacy full-K) fit."""
    rng = np.random.default_rng(0)
    V, K = 30, 4
    docs = _toy_docs(rng, V=V, D=40, doc_len=20, K_blocks=K)
    default = OnlineSTM(K=K, vocab_size=V, P=1, random_seed=7)
    on = OnlineSTM(K=K, vocab_size=V, P=1, random_seed=7, reference_topic=True)
    off = OnlineSTM(K=K, vocab_size=V, P=1, random_seed=7, reference_topic=False)
    gpd, gpon, gpoff = default.initialize_global(None), on.initialize_global(None), off.initialize_global(None)
    for _ in range(5):
        gpd = default.update_global(gpd, default.local_update(docs, gpd), learning_rate=1.0)
        gpon = on.update_global(gpon, on.local_update(docs, gpon), learning_rate=1.0)
        gpoff = off.update_global(gpoff, off.local_update(docs, gpoff), learning_rate=1.0)
    # Default == reference-on, in every global param.
    assert np.array_equal(gpd["Gamma"], gpon["Gamma"])
    assert np.array_equal(gpd["Sigma"], gpon["Sigma"])
    assert np.array_equal(gpd["lambda"], gpon["lambda"])
    # And differs from the explicit-off full-K fit.
    assert not np.array_equal(gpd["Gamma"], gpoff["Gamma"])


def test_reference_gated_infer_local_pins_reference():
    """infer_local must use the same parameterization as training, so exported
    theta has the reference pinned to eta=0."""
    rng = np.random.default_rng(0)
    V = 24
    part = TopicBlockPartition(group_var="g", background_k=2,
                               foreground=(("rare", 2),))
    K = part.K
    docs = []
    for i in range(80):
        is_rare = (i % 4 == 0)
        toks = rng.integers(0, V, size=12)
        docs.append(STMDocument(indices=np.unique(toks).astype(np.int32),
                                counts=np.ones(len(np.unique(toks))),
                                length=len(np.unique(toks)),
                                x=np.array([1.0]),
                                groups=frozenset({"rare"}) if is_rare else frozenset()))
    m = OnlineSTM(K=K, vocab_size=V, P=1, random_seed=3,
                  topic_blocks=part, reference_topic=True)
    gp = m.initialize_global(None)
    for _ in range(10):
        gp = m.update_global(gp, m.local_update(docs, gp), learning_rate=0.5)
    out = m.infer_local(docs[0], gp)
    assert out["eta"][0] == 0.0
    assert abs(float(out["theta"].sum()) - 1.0) < 1e-12
