"""Tests for the local (pure numpy) concentration-recovery diagnostic.

Plant synthetic documents at a KNOWN per-document topic concentration over a
shared-term topic matrix, then check that STM (at Sigma scale c) and LDA (at
Dirichlet alpha) recover it. No Spark, no mllib -- see
spark_vi/eval/topic/concentration_recovery.py.
"""
from __future__ import annotations

import numpy as np
import pytest

from spark_vi.eval.topic.concentration import doc_concentration
from spark_vi.eval.topic.concentration_recovery import (
    corpus_concentration_summary,
    lda_optimize_alpha,
    lda_recover_theta,
    make_shared_beta,
    plant_corpus,
    stm_recover_theta,
)


def _median_top_mass(theta_matrix: np.ndarray) -> float:
    tops = [doc_concentration(row)[0] for row in theta_matrix]
    return float(np.median(tops))


def _median_eff_topics(theta_matrix: np.ndarray) -> float:
    effs = [doc_concentration(row)[1] for row in theta_matrix]
    return float(np.median(effs))


class TestMakeSharedBeta:
    def test_shared_beta_valid_and_overlapping(self):
        K, V = 6, 300
        beta = make_shared_beta(K, V, pool_frac=0.5, shared_mass=0.5, seed=0)
        assert beta.shape == (K, V)
        np.testing.assert_allclose(beta.sum(axis=1), np.ones(K), atol=1e-9)
        assert (beta >= 0).all()

        C = round(0.5 * V)
        # Every topic has nonzero mass somewhere in the shared pool.
        assert (beta[:, :C] > 0).all(axis=0).any()
        # Signature blocks are topic-specific: topic k's own block should carry
        # more of its mass than any other topic's block of the same size.
        sig = (V - C) // K
        for k in range(K):
            lo = C + k * sig
            hi = lo + sig
            own_mass = beta[k, lo:hi].sum()
            other_mass = beta[(k + 1) % K, lo:hi].sum()
            assert own_mass > other_mass


class TestPlantCorpus:
    def test_plant_logistic_normal_monotone(self):
        beta = make_shared_beta(K=6, V=300, seed=0)
        medians = []
        for level in (1, 4, 9):
            _, theta_true = plant_corpus(
                beta, D=200, doc_len=80, mechanism="logistic_normal",
                level=level, seed=1,
            )
            medians.append(_median_top_mass(theta_true))
        assert medians[0] < medians[1] < medians[2]

        effs = []
        for level in (1, 4, 9):
            _, theta_true = plant_corpus(
                beta, D=200, doc_len=80, mechanism="logistic_normal",
                level=level, seed=1,
            )
            effs.append(_median_eff_topics(theta_true))
        assert effs[0] > effs[1] > effs[2]

    def test_plant_dirichlet_monotone(self):
        beta = make_shared_beta(K=6, V=300, seed=0)
        medians = []
        for level in (1.0, 0.3, 0.05):
            _, theta_true = plant_corpus(
                beta, D=200, doc_len=80, mechanism="dirichlet",
                level=level, seed=2,
            )
            medians.append(_median_top_mass(theta_true))
        # smaller level -> peakier -> higher top_mass
        assert medians[0] < medians[1] < medians[2]

    def test_plant_returns_valid_docs(self):
        K, V, D, doc_len = 6, 300, 50, 80
        beta = make_shared_beta(K, V, seed=0)
        docs, theta_true = plant_corpus(
            beta, D=D, doc_len=doc_len, mechanism="logistic_normal",
            level=4, seed=3,
        )
        assert len(docs) == D
        assert theta_true.shape == (D, K)
        np.testing.assert_allclose(theta_true.sum(axis=1), np.ones(D), atol=1e-9)
        for doc in docs:
            assert doc.indices.dtype == np.int32
            assert np.all(np.diff(doc.indices) > 0)  # sorted, unique
            assert np.all(doc.indices >= 0) and np.all(doc.indices < V)
            assert np.all(doc.counts > 0)
            assert doc.counts.sum() == doc_len
            assert doc.length == doc_len
            assert doc.x.shape == (1,)
            assert doc.x[0] == 1.0
            assert doc.groups == frozenset()


class TestSTMRecovery:
    def test_stm_recovers_planted_concentration(self):
        beta = make_shared_beta(K=6, V=300, seed=0)
        docs, theta_true = plant_corpus(
            beta, D=150, doc_len=120, mechanism="logistic_normal",
            level=9, seed=4,
        )
        planted_median = _median_top_mass(theta_true)

        theta_hat_c9 = stm_recover_theta(docs, beta, c=9)
        theta_hat_c1 = stm_recover_theta(docs, beta, c=1)

        recovered_median = _median_top_mass(theta_hat_c9)
        smeared_median = _median_top_mass(theta_hat_c1)

        assert abs(recovered_median - planted_median) < 0.12
        assert recovered_median > smeared_median


class TestLDARecovery:
    def test_lda_recovers_and_small_alpha_sharpens(self):
        beta = make_shared_beta(K=6, V=300, seed=0)
        docs, _ = plant_corpus(
            beta, D=150, doc_len=120, mechanism="logistic_normal",
            level=9, seed=4,
        )
        theta_small_alpha = lda_recover_theta(docs, beta, alpha=0.05)
        theta_large_alpha = lda_recover_theta(docs, beta, alpha=1.0)

        assert _median_top_mass(theta_small_alpha) > _median_top_mass(theta_large_alpha)

    def test_lda_optimize_alpha_runs(self):
        K = 6
        beta = make_shared_beta(K=K, V=300, seed=0)
        docs, _ = plant_corpus(
            beta, D=150, doc_len=120, mechanism="logistic_normal",
            level=9, seed=4,
        )
        alpha = lda_optimize_alpha(docs, beta, K)
        assert alpha.shape == (K,)
        assert np.all(np.isfinite(alpha))
        assert np.all(alpha > 0)
        assert np.all(alpha < 1.0)


class TestCorpusConcentrationSummary:
    def test_matches_lda_concentration_readout(self):
        from spark_vi.eval.topic.concentration import lda_concentration_readout

        rng = np.random.default_rng(0)
        theta = rng.dirichlet(np.full(5, 0.3), size=20)
        expected = lda_concentration_readout(theta)
        got = corpus_concentration_summary(theta)
        assert got == expected
