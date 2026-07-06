"""Tests for the COLD (reference) single-document predictive-gain oracle.

``doc_predictive_gain`` computes the leave-one-topic-out held-out predictive
gain Delta_k = LL(allowed) - LL(allowed \\ {k}) for a single STM document,
by brute-force re-inference under each topic's ablation (see
spark_vi/mllib/topic/predictive_gain.py for the design). This file plants a
tiny hand-checkable, non-gated, disjoint-vocabulary two-topic model so the
expected sign and rough magnitude of every Delta_k can be reasoned about
directly: a document built entirely from topic 0's signature words should
show a large Delta for topic 0 (removing the generating topic hurts held-out
prediction) and a near-zero Delta for topic 1 (removing an irrelevant topic
costs ~nothing -- the "auto-floor" property) -- and symmetrically for a
topic-1-only document.
"""
from __future__ import annotations

import numpy as np
import pytest

from spark_vi.models.topic.partition import TopicBlockPartition
from spark_vi.models.topic.types import STMDocument


K, V = 2, 8


def _sharp_global_params():
    """2-topic, disjoint-vocab (V=8) global_params: topic 0 owns words
    {0,1,2,3}, topic 1 owns words {4,5,6,7}. lambda carries a large
    pseudo-count on-signature and a tiny one off-signature so both
    expElogbeta (inference) and beta_prob = E[beta] (scoring) are sharply
    peaked on the owning topic's words. Gamma=0 (neutral prior mean),
    Sigma=I (unit generative scale, non-gated: R == Sigma already)."""
    lam = np.full((K, V), 0.01)
    lam[0, 0:4] = 1000.0
    lam[1, 4:8] = 1000.0
    Gamma = np.zeros((1, K))
    Sigma = np.eye(K)
    return {"lambda": lam, "Gamma": Gamma, "Sigma": Sigma}


# A large scale c (weak Gaussian prior precision 1/c) lets eta_hat follow the
# strong multinomial evidence to its (near-)extreme MAP value instead of
# being pulled back toward the neutral Gamma^T x = 0 prior mean; this is what
# makes the auto-floor property ("removing an irrelevant topic costs ~0")
# hold to a tight numeric tolerance in a tiny hand-built fixture. At c=1 the
# posterior mode is a genuine ~91/9 mix (the Gaussian prior meaningfully
# regularizes with only 4 signature words of evidence) and delta[1] is a
# real, non-negligible -0.5 nats, not a numerical near-zero -- so the
# auto-floor assertion needs the weaker prior to isolate the property being
# tested from ordinary shrinkage.
_C_WEAK_PRIOR = 1000.0


def _non_gated_partition():
    # background_k == K, foreground empty: allowed_indices(frozenset()) == [0, 1]
    # for every doc regardless of doc.groups (non-gated model).
    return TopicBlockPartition(group_var="g", background_k=K, foreground=())


def _doc(indices, groups=frozenset()):
    """A document whose tokens are all drawn from `indices`, count 5 each
    (20 tokens total -- enough for heldout_split(holdout_frac=0.3) to yield
    a non-empty visible AND held half)."""
    indices = np.asarray(indices, dtype=np.int32)
    counts = np.full(indices.shape, 5.0, dtype=np.float64)
    return STMDocument(
        indices=indices, counts=counts, length=int(counts.sum()),
        x=np.array([1.0]), groups=groups,
    )


class TestColdAutoFloor:
    def test_cold_topic0_doc_delta_signs(self):
        from spark_vi.mllib.topic.predictive_gain import doc_predictive_gain

        gp = _sharp_global_params()
        part = _non_gated_partition()
        doc = _doc([0, 1, 2, 3])  # all topic-0 signature words

        dg = doc_predictive_gain(doc, gp, part, c=_C_WEAK_PRIOR, seed=0)

        assert dg is not None
        assert np.array_equal(dg.allowed, np.array([0, 1]))
        assert len(dg.delta) == len(dg.allowed) == 2
        assert np.all(np.isfinite(dg.delta))

        # Removing topic 0 (the generating topic) should hurt held-out
        # prediction a lot; removing topic 1 (irrelevant) should cost ~0.
        assert dg.delta[0] > 1.0
        assert dg.delta[1] == pytest.approx(0.0, abs=1e-2)

    def test_cold_topic1_doc_delta_signs_mirror(self):
        """Mirror case: a document built entirely from topic 1's signature
        words should show the opposite pattern -- delta[1] large, delta[0]
        ~0."""
        from spark_vi.mllib.topic.predictive_gain import doc_predictive_gain

        gp = _sharp_global_params()
        part = _non_gated_partition()
        doc = _doc([4, 5, 6, 7])  # all topic-1 signature words

        dg = doc_predictive_gain(doc, gp, part, c=_C_WEAK_PRIOR, seed=0)

        assert dg is not None
        assert dg.delta[1] > 1.0
        assert dg.delta[0] == pytest.approx(0.0, abs=1e-2)


class TestColdFiniteness:
    def test_cold_finite_and_theta_full_is_a_gated_simplex(self):
        from spark_vi.mllib.topic.predictive_gain import doc_predictive_gain

        gp = _sharp_global_params()
        part = _non_gated_partition()
        doc = _doc([0, 1, 2, 3])

        dg = doc_predictive_gain(doc, gp, part, c=_C_WEAK_PRIOR, seed=0)

        assert dg is not None
        assert np.isfinite(dg.ll_full)
        assert np.all(np.isfinite(dg.delta))
        assert np.all(np.isfinite(dg.dedup_delta))
        assert dg.n_held > 0

        # theta_full: a full-K vector, mass only on `allowed`, summing to 1.
        assert dg.theta_full.shape == (K,)
        assert dg.theta_full.sum() == pytest.approx(1.0, abs=1e-8)
        off_allowed = np.setdiff1d(np.arange(K), dg.allowed)
        assert np.all(dg.theta_full[off_allowed] == 0.0)


class TestColdDegenerateSplit:
    def test_cold_too_short_doc_returns_none(self):
        """A 1-token document can't be split into a non-empty visible AND
        held half (heldout_split's guard); doc_predictive_gain must mirror
        that skip by returning None rather than raising."""
        from spark_vi.mllib.topic.predictive_gain import doc_predictive_gain

        gp = _sharp_global_params()
        part = _non_gated_partition()
        doc = STMDocument(
            indices=np.array([0], dtype=np.int32),
            counts=np.array([1.0]),
            length=1, x=np.array([1.0]), groups=frozenset(),
        )

        dg = doc_predictive_gain(doc, gp, part, c=1.0, seed=0)
        assert dg is None
