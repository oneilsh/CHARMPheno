"""Tests for the gated held-out predictive-LL generative-scale sweep.

corpus_heldout_scale_sweep_gated{,_rdd} sweep the STM generative covariance
scale c (Sigma_gen = c*R, R the fit's diagonal-normalized correlation) and
score held-out per-token predictive log-likelihood, recovering the faithful
generative scale via argmax_c. This is the GATED analog of the LOCAL,
non-gated diagnostic validated by insight 0038
(spark_vi/eval/topic/concentration_recovery.py): inference here runs the
gated E-step (Gamma^T x prior mean, per-doc allowed topic sets restricted by
a TopicBlockPartition) while the held-out split/scoring machinery
(heldout_split, _predictive_loglik) is reused directly from that module.

Corpus-planting helpers (_planted_beta, _make_global_params) mirror the
pattern already validated in tests/test_corpus_eta_scale.py.
"""
from __future__ import annotations

import numpy as np
import pytest

from spark_vi.models.topic.partition import TopicBlockPartition
from spark_vi.models.topic.types import STMDocument

from tests._stm_synth import synthetic_gated_corpus, fit_stm


def _planted_beta(K, V):
    """Peaked topic-word beta: topic k owns a disjoint signature block, so a
    document sampled from softmax(eta)@beta carries real token evidence
    (mirrors tests/test_corpus_eta_scale.py::_planted_beta)."""
    beta = np.full((K, V), 1e-3)
    blk = V // K
    for k in range(K):
        beta[k, k * blk:(k + 1) * blk] += 2.0
    beta /= beta.sum(axis=1, keepdims=True)
    return beta


def _make_global_params(K, V, Gamma, Sigma, *, beta=None):
    """global_params with an INFORMATIVE lambda: lambda = beta * concentration
    so both expElogbeta (inference) and the lambda-normalized beta_prob
    (scoring) are near-identical to the true planted beta -- isolating the
    scale-recovery question from beta-estimation error (mirrors
    tests/test_corpus_eta_scale.py::_make_global_params)."""
    if beta is None:
        beta = _planted_beta(K, V)
    lam = beta * (500.0 * V) + 0.01
    return {"lambda": lam, "Gamma": Gamma, "Sigma": Sigma}


def _global_params_from_fit(gp):
    return {"lambda": gp["lambda"], "Gamma": gp["Gamma"], "Sigma": gp["Sigma"]}


def _build_fitted_corpus(*, seed=0):
    docs, planted, part = synthetic_gated_corpus(
        groups=("A", "B"), fg_per_group=1, bg_k=2, V=40, D=40, doc_len=25,
        bg_frac=0.5, seed=seed,
    )
    K = part.K
    gp = fit_stm(docs, K=K, V=40, sigma_init=1.0, n_iter=20,
                 partition=part, seed=seed)
    return docs, part, gp, K


class TestNumpyRddParity:
    def test_numpy_rdd_parity(self, spark):
        from spark_vi.mllib.topic.stm import (
            corpus_heldout_scale_sweep_gated, corpus_heldout_scale_sweep_gated_rdd,
        )

        docs, part, gp, K = _build_fitted_corpus(seed=3)
        global_params = _global_params_from_fit(gp)
        c_grid = [1, 2, 4, 8]

        expected = corpus_heldout_scale_sweep_gated(
            docs, global_params, part, c_grid=c_grid, seed=0,
        )

        rdd = spark.sparkContext.parallelize(docs, numSlices=3)
        result = corpus_heldout_scale_sweep_gated_rdd(
            rdd, global_params, part, c_grid=c_grid, seed=0,
        )

        assert result["n_docs"] == expected["n_docs"]
        assert set(result["lls"].keys()) == set(expected["lls"].keys())
        for c in c_grid:
            np.testing.assert_allclose(
                result["lls"][c], expected["lls"][c], rtol=1e-8, atol=1e-10
            )
        assert result["argmax_c"] == expected["argmax_c"]


class TestArgmaxRecoversPlantedScale:
    def test_argmax_recovers_planted_scale(self):
        """KEY test: plant a GATED corpus at a known scalar generative scale s
        and confirm the sweep's argmax_c recovers it (not the diffuse end)."""
        from spark_vi.mllib.topic.stm import corpus_heldout_scale_sweep_gated

        part = TopicBlockPartition(
            group_var="g", background_k=2, foreground=(("A", 1), ("B", 1)))
        K = part.K  # 4
        V = 60
        s = 5.0
        rng = np.random.default_rng(42)
        beta = _planted_beta(K, V)
        Gamma = np.zeros((1, K))
        Sigma = np.eye(K)  # R == I: the sweep's c IS the generative eta-variance
        gp = _make_global_params(K, V, Gamma, Sigma, beta=beta)

        groups_cycle = [frozenset(), frozenset({"A"}), frozenset({"B"})]
        docs = []
        D_total, doc_len = 240, 60
        for i in range(D_total):
            g = groups_cycle[i % 3]
            allowed = np.sort(part.allowed_indices(g))
            draw = rng.normal(scale=np.sqrt(s), size=allowed.shape[0])
            z = draw - draw.max()
            w = np.exp(z)
            theta = np.zeros(K)
            theta[allowed] = w / w.sum()
            toks = rng.choice(V, size=doc_len, p=theta @ beta)
            u, c = np.unique(toks, return_counts=True)
            docs.append(STMDocument(
                indices=u.astype(np.int32), counts=c.astype(np.float64),
                length=int(c.sum()), x=np.array([1.0]), groups=g))

        c_grid = [1, 2, 5, 10, 20]
        result = corpus_heldout_scale_sweep_gated(
            docs, gp, part, c_grid=c_grid, holdout_frac=0.3, seed=0,
        )
        argmax_c = result["argmax_c"]

        if argmax_c in (c_grid[0], c_grid[-1]):
            # Boundary argmax is not a validated peak -- widen so the max is interior.
            c_grid = [1, 2, 5, 10, 20, 30, 40]
            result = corpus_heldout_scale_sweep_gated(
                docs, gp, part, c_grid=c_grid, holdout_frac=0.3, seed=0,
            )
            argmax_c = result["argmax_c"]

        assert argmax_c != c_grid[0], (
            f"argmax landed at the diffuse end c={argmax_c} -- not recovering "
            f"the planted scale s={s}"
        )
        assert argmax_c != c_grid[-1], "argmax at grid boundary -- not a validated peak"

        nearest = min(c_grid, key=lambda c: abs(c - s))
        assert argmax_c == nearest, (
            f"argmax_c={argmax_c} is not the grid value nearest planted s={s} "
            f"(nearest={nearest}); lls={result['lls']}"
        )


class TestEmptyRddRaises:
    def test_empty_rdd_raises(self, spark):
        from spark_vi.mllib.topic.stm import corpus_heldout_scale_sweep_gated_rdd

        part = TopicBlockPartition(
            group_var="g", background_k=2, foreground=(("A", 1),))
        K = part.K
        gp = _make_global_params(K, 20, np.zeros((1, K)), np.eye(K))
        empty = spark.sparkContext.parallelize([], numSlices=1)
        with pytest.raises(ValueError, match="empty"):
            corpus_heldout_scale_sweep_gated_rdd(empty, gp, part, c_grid=[1, 2])


class TestReferenceTopicHandled:
    def test_reference_topic_handled(self):
        from spark_vi.mllib.topic.stm import corpus_heldout_scale_sweep_gated

        docs, part, gp, K = _build_fitted_corpus(seed=9)
        global_params = _global_params_from_fit(gp)

        result = corpus_heldout_scale_sweep_gated(
            docs, global_params, part, c_grid=[1, 4], reference=0, seed=0,
        )
        assert set(result["lls"].keys()) == {1, 4}
        for ll in result["lls"].values():
            assert np.isfinite(ll)
        assert result["argmax_c"] in (1, 4)
