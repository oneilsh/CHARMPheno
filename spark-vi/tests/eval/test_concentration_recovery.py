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
    _predictive_loglik,
    corpus_concentration_summary,
    heldout_split,
    lda_heldout_ll,
    lda_optimize_alpha,
    lda_recover_theta,
    make_shared_beta,
    plant_corpus,
    stm_recover_theta,
    sweep_heldout,
)
from spark_vi.models.topic.types import STMDocument


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


class TestHeldoutSplit:
    def test_heldout_split_conserves_tokens(self):
        doc = STMDocument(
            indices=np.array([2, 5, 9, 20], dtype=np.int32),
            counts=np.array([3.0, 5.0, 4.0, 8.0], dtype=np.float64),
            length=20,
            x=np.array([1.0]),
            groups=frozenset(),
        )
        total = int(doc.counts.sum())

        split = heldout_split(doc, holdout_frac=0.4, seed=123)
        assert split is not None
        visible_doc, held_indices, held_counts = split

        visible_tokens = int(visible_doc.counts.sum())
        held_tokens = int(held_counts.sum())
        assert visible_tokens + held_tokens == total
        assert held_tokens == round(0.4 * total)

        assert set(visible_doc.indices.tolist()) <= set(doc.indices.tolist())
        assert set(held_indices.tolist()) <= set(doc.indices.tolist())

        # visible_doc carries over doc metadata unchanged.
        assert visible_doc.x is doc.x or np.array_equal(visible_doc.x, doc.x)
        assert visible_doc.groups == doc.groups
        assert visible_doc.length == visible_tokens

    def test_short_doc_returns_none(self):
        doc = STMDocument(
            indices=np.array([1], dtype=np.int32),
            counts=np.array([1.0], dtype=np.float64),
            length=1,
            x=np.array([1.0]),
            groups=frozenset(),
        )
        assert heldout_split(doc, holdout_frac=0.4, seed=0) is None

        doc2 = STMDocument(
            indices=np.array([1, 2], dtype=np.int32),
            counts=np.array([1.0, 1.0], dtype=np.float64),
            length=2,
            x=np.array([1.0]),
            groups=frozenset(),
        )
        # 2-token doc is fine as long as the visible half is nonempty.
        assert heldout_split(doc2, holdout_frac=0.4, seed=0) is not None


class TestPredictiveLoglik:
    def test_predictive_loglik_peaks_for_correct_theta(self):
        K, V = 3, 30
        beta = make_shared_beta(K=K, V=V, pool_frac=0.2, shared_mass=0.1, seed=0)
        # Topic 0's signature block carries most of its mass -- find its
        # highest-probability terms and hold out tokens concentrated there.
        top_terms_topic0 = np.argsort(-beta[0])[:5]
        held_indices = top_terms_topic0.astype(np.int32)
        held_counts = np.full(5, 4.0)

        theta_peaked = np.zeros(K)
        theta_peaked[0] = 1.0
        theta_uniform = np.full(K, 1.0 / K)

        ll_peaked = _predictive_loglik(theta_peaked, beta, held_indices, held_counts)
        ll_uniform = _predictive_loglik(theta_uniform, beta, held_indices, held_counts)
        assert ll_peaked > ll_uniform


class TestSweepHeldout:
    def test_sweep_uses_same_split_across_knobs(self, monkeypatch):
        import spark_vi.eval.topic.concentration_recovery as cr

        beta = make_shared_beta(K=4, V=100, seed=0)
        docs, _ = plant_corpus(
            beta, D=10, doc_len=40, mechanism="logistic_normal", level=4, seed=1,
        )

        calls = []
        original_split = cr.heldout_split

        def spy(doc, *, holdout_frac=0.3, seed):
            result = original_split(doc, holdout_frac=holdout_frac, seed=seed)
            held_sum = None if result is None else float(result[2].sum())
            calls.append((seed, held_sum))
            return result

        monkeypatch.setattr(cr, "heldout_split", spy)

        cr.sweep_heldout(docs, beta, method="stm", knobs=[1.0, 4.0], seed=0)

        n = len(docs)
        assert len(calls) == 2 * n
        first_pass, second_pass = calls[:n], calls[n:]
        assert first_pass == second_pass


class TestHeldoutGoldStandard:
    def test_heldout_gold_standard_recovers_planted_concentration(self):
        beta = make_shared_beta(K=6, V=300, seed=0)
        docs, theta_true = plant_corpus(
            beta, D=200, doc_len=80, mechanism="logistic_normal", level=7, seed=5,
        )
        planted_median = _median_top_mass(theta_true)

        knobs = [1, 2, 4, 7, 12]
        result = sweep_heldout(docs, beta, method="stm", knobs=knobs, seed=0)
        argmax_c = result["argmax_knob"]

        if argmax_c == knobs[-1]:
            # Boundary argmax is not a validated peak -- widen the grid so
            # the maximum lands interior.
            knobs = knobs + [20, 30]
            result = sweep_heldout(docs, beta, method="stm", knobs=knobs, seed=0)
            argmax_c = result["argmax_knob"]

        assert argmax_c != knobs[0]  # diffuse c=1 must not win on a peaky corpus
        assert argmax_c != knobs[-1], "argmax at grid boundary -- not a validated peak"

        theta_hat_argmax = stm_recover_theta(docs, beta, c=argmax_c)
        theta_hat_c1 = stm_recover_theta(docs, beta, c=1)

        recovered_median = _median_top_mass(theta_hat_argmax)
        smeared_median = _median_top_mass(theta_hat_c1)

        err_argmax = abs(recovered_median - planted_median)
        err_c1 = abs(smeared_median - planted_median)

        assert err_argmax < err_c1
        # The gold standard should PIN the concentration, not merely beat the
        # c=1 baseline: recovery at the argmax c must land close to the
        # planted median top_mass in absolute terms (observed ~0.0066 on this
        # corpus; 0.06 leaves ample margin).
        assert err_argmax < 0.06, (
            f"argmax c={argmax_c} recovered median top_mass {recovered_median:.4f} "
            f"vs planted {planted_median:.4f} (err {err_argmax:.4f})"
        )


class TestLDAHeldoutGoldStandard:
    def test_lda_heldout_gold_standard_recovers_planted_concentration(self):
        beta = make_shared_beta(K=6, V=300, seed=0)
        docs, theta_true = plant_corpus(
            beta, D=200, doc_len=80, mechanism="logistic_normal", level=7, seed=5,
        )
        planted_median = _median_top_mass(theta_true)

        # LDA alpha runs the OPPOSITE direction from STM's c: SMALL alpha is
        # peaky, LARGE alpha is diffuse.
        knobs = [0.02, 0.1, 0.5, 1.0, 3.0]
        result = sweep_heldout(docs, beta, method="lda", knobs=knobs, seed=0)
        argmax_alpha = result["argmax_knob"]

        if argmax_alpha == knobs[0]:
            # Boundary argmax at the peakiest end is not a validated peak --
            # widen the grid downward so the maximum lands interior.
            knobs = [0.005, 0.01] + knobs
            result = sweep_heldout(docs, beta, method="lda", knobs=knobs, seed=0)
            argmax_alpha = result["argmax_knob"]

        assert argmax_alpha != knobs[-1], "diffuse alpha=3.0 must not win on a peaky corpus"
        assert argmax_alpha != knobs[0], "argmax at grid boundary -- not a validated peak"

        theta_hat_argmax = lda_recover_theta(docs, beta, alpha=argmax_alpha)
        theta_hat_diffuse = lda_recover_theta(docs, beta, alpha=3.0)

        recovered_median = _median_top_mass(theta_hat_argmax)
        diffuse_median = _median_top_mass(theta_hat_diffuse)

        err_argmax = abs(recovered_median - planted_median)
        err_diffuse = abs(diffuse_median - planted_median)

        assert err_argmax < err_diffuse
        assert err_argmax < 0.06, (
            f"argmax alpha={argmax_alpha} recovered median top_mass "
            f"{recovered_median:.4f} vs planted {planted_median:.4f} "
            f"(err {err_argmax:.4f})"
        )


class TestLDAHeldoutLLSmoke:
    def test_lda_heldout_ll_smoke(self):
        beta = make_shared_beta(K=4, V=100, seed=0)
        docs, _ = plant_corpus(
            beta, D=20, doc_len=40, mechanism="logistic_normal", level=4, seed=2,
        )

        ll = lda_heldout_ll(docs, beta, alpha=0.1, seed=0)
        assert isinstance(ll, float)
        assert np.isfinite(ll)
        assert ll < 0

        result = sweep_heldout(docs, beta, method="lda", knobs=[0.1, 1.0], seed=0)
        assert set(result["lls"].keys()) == {0.1, 1.0}
        assert len(result["lls"]) == 2
        assert result["argmax_knob"] in (0.1, 1.0)
