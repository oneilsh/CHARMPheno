"""Unit tests for the K-1 reference-topic parameterization (opt-in)."""
from __future__ import annotations

import numpy as np
import pytest

from spark_vi.models.topic.stm import (
    OnlineSTM,
    _stm_doc_inference,
    _softmax,
)


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
    theta is still a valid distribution that gives the reference positive mass."""
    eta_hat, nu_d, _ = _stm_doc_inference(
        indices=np.array([2, 3], dtype=np.int32),
        counts=np.array([5.0, 5.0]),
        expElogbeta=_BETA3,
        Gamma=np.zeros((1, 3)),
        Sigma_diag=np.full(3, 5.0),
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
    though topic 0 is pinned as the reference — the free optimization works."""
    eta_hat, _, _ = _stm_doc_inference(
        indices=np.array([2, 3], dtype=np.int32),   # topic-1's words
        counts=np.array([8.0, 8.0]),
        expElogbeta=_BETA3,
        Gamma=np.zeros((1, 3)),
        Sigma_diag=np.full(3, 5.0),
        x=np.array([1.0]),
        reference=0,
    )
    theta = _softmax(eta_hat)
    assert int(np.argmax(theta)) == 1


def test_reference_inference_respects_allowed():
    """reference must sit inside allowed; disallowed topics stay at theta=0 and
    the reference is still pinned to 0."""
    eta_hat, nu_d, _ = _stm_doc_inference(
        indices=np.array([0, 1], dtype=np.int32),
        counts=np.array([4.0, 4.0]),
        expElogbeta=_BETA3,
        Gamma=np.zeros((1, 3)),
        Sigma_diag=np.full(3, 5.0),
        x=np.array([1.0]),
        allowed=np.array([0, 1], dtype=np.int64),   # topic 2 disallowed
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
    assert OnlineSTM(K=3, vocab_size=4, P=1).  _reference_index() is None
    assert OnlineSTM(K=3, vocab_size=4, P=1, reference_topic=True)._reference_index() == 0
