"""Tests for corpus_theta_gated_rdd — the per-document gated MAP theta
collector used to build the dashboard's per-doc theta ("topic mass
distribution") histogram for a GATED STM.

corpus_theta_gated_rdd mirrors corpus_concentration_stm_rdd's per-doc gated
E-step (_stm_doc_inference -> _gated_mode_theta at the model's OWN fitted Sigma,
FULL document — no held-out split, no c-rescaling), but instead of accumulating
into a ConcentrationAcc it down-samples the corpus and COLLECTS the raw
(N_sampled, K) theta array to the driver so compute_theta_aggregates can bin it.

Corpus-planting helpers mirror the pattern already validated in
tests/test_heldout_scale_sweep.py / tests/_stm_synth.py.
"""
from __future__ import annotations

import numpy as np
import pytest

from spark_vi.models.topic.partition import TopicBlockPartition
from spark_vi.models.topic.types import STMDocument

from tests._stm_synth import gated_ln_corpus, fit_stm


def _global_params_from_fit(gp):
    return {"lambda": gp["lambda"], "Gamma": gp["Gamma"], "Sigma": gp["Sigma"]}


def _build_fitted_gated_corpus(*, D=60, seed=0):
    """Small fitted gated logistic-normal corpus (two groups A/B, one
    foreground topic each, two background topics)."""
    docs, part, Sigma_true, beta = gated_ln_corpus(
        group_weights={"A": 1.0, "B": 1.0}, fg_per_group=1, bg_k=2,
        V=40, D=D, doc_len=25, seed=seed,
    )
    K = part.K
    gp = fit_stm(docs, K=K, V=40, sigma_init=1.0, n_iter=20,
                 partition=part, seed=seed)
    return docs, part, _global_params_from_fit(gp), K


class TestShapeAndNormalization:
    def test_shape_and_rows_sum_to_one(self, spark):
        from spark_vi.mllib.topic.stm import corpus_theta_gated_rdd

        docs, part, gp, K = _build_fitted_gated_corpus(D=60, seed=1)
        rdd = spark.sparkContext.parallelize(docs, numSlices=3)

        theta = corpus_theta_gated_rdd(rdd, gp, part, sample_cap=10_000, seed=0)

        assert theta.ndim == 2
        assert theta.shape[1] == K
        assert theta.shape[0] == len(docs)  # cap >= N -> all docs
        np.testing.assert_allclose(theta.sum(axis=1), 1.0, atol=1e-6)
        assert (theta >= 0.0).all()


class TestStructuralMask:
    def test_out_of_group_foreground_is_exactly_zero(self, spark):
        """A gated doc in group g must be EXACTLY 0 on the foreground topics of
        OTHER groups (hard structural mask) and may be positive only on
        background ∪ g's foreground."""
        from spark_vi.mllib.topic.stm import corpus_theta_gated_rdd

        # One doc per group so we can check the mask per row deterministically.
        part = TopicBlockPartition(
            group_var="g", background_k=2, foreground=(("A", 1), ("B", 1)))
        K = part.K  # 4
        V = 40
        beta = np.full((K, V), 1e-3)
        blk = V // K
        for kk in range(K):
            beta[kk, kk * blk:(kk + 1) * blk] += 2.0
        beta /= beta.sum(axis=1, keepdims=True)
        lam = beta * (500.0 * V) + 0.01
        gp = {"lambda": lam, "Gamma": np.zeros((1, K)), "Sigma": np.eye(K)}

        rng = np.random.default_rng(0)
        docs = []
        groups_in_order = ["A", "B"]
        for g in groups_in_order:
            fg = part.block_indices(g)[0]
            toks = rng.choice(V, size=40, p=beta[fg])
            u, c = np.unique(toks, return_counts=True)
            docs.append(STMDocument(
                indices=u.astype(np.int32), counts=c.astype(np.float64),
                length=int(c.sum()), x=np.array([1.0]),
                groups=frozenset({g})))

        # numSlices=1 keeps input order so row i == docs[i]
        rdd = spark.sparkContext.parallelize(docs, numSlices=1)
        theta = corpus_theta_gated_rdd(rdd, gp, part, sample_cap=10_000, seed=0)

        assert theta.shape[0] == 2
        for i, g in enumerate(groups_in_order):
            allowed = set(int(a) for a in part.allowed_indices(frozenset({g})))
            for kk in range(K):
                if kk not in allowed:
                    assert theta[i, kk] == 0.0, (
                        f"doc {i} (group {g}) has nonzero mass {theta[i, kk]} on "
                        f"disallowed topic {kk}")
            # positive mass somewhere in the allowed set
            assert theta[i, sorted(allowed)].sum() > 0.99


class TestSamplingCap:
    def test_cap_below_n_downsamples(self, spark):
        from spark_vi.mllib.topic.stm import corpus_theta_gated_rdd

        docs, part, gp, K = _build_fitted_gated_corpus(D=120, seed=2)
        rdd = spark.sparkContext.parallelize(docs, numSlices=4)

        theta = corpus_theta_gated_rdd(rdd, gp, part, sample_cap=40, seed=0)
        # Bernoulli sample -> N' is random around the cap but strictly < N and > 0
        assert 0 < theta.shape[0] < len(docs)
        assert theta.shape[1] == K

    def test_cap_at_or_above_n_returns_all(self, spark):
        from spark_vi.mllib.topic.stm import corpus_theta_gated_rdd

        docs, part, gp, K = _build_fitted_gated_corpus(D=60, seed=3)
        rdd = spark.sparkContext.parallelize(docs, numSlices=3)

        theta = corpus_theta_gated_rdd(rdd, gp, part, sample_cap=10_000, seed=0)
        assert theta.shape[0] == len(docs)  # deterministic: no sampling


class TestEmptyRddRaises:
    def test_empty_rdd_raises(self, spark):
        from spark_vi.mllib.topic.stm import corpus_theta_gated_rdd

        part = TopicBlockPartition(
            group_var="g", background_k=2, foreground=(("A", 1),))
        K = part.K
        gp = {"lambda": np.full((K, 20), 0.5), "Gamma": np.zeros((1, K)),
              "Sigma": np.eye(K)}
        empty = spark.sparkContext.parallelize([], numSlices=1)
        with pytest.raises(ValueError, match="empty"):
            corpus_theta_gated_rdd(empty, gp, part)
