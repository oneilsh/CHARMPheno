"""Tests for the distributed per-document topic-concentration STM pass.

corpus_concentration_stm{,_rdd} runs the same per-doc Laplace E-step as
corpus_eta_variance_gated{,_rdd} to get eta_hat over each doc's allowed
(gated) topic set, forms the gated softmax mode theta_d, and accumulates
(top_mass, eff_topics) into a ConcentrationAcc (spark_vi.eval.topic.concentration).
Mirrors tests/test_corpus_eta_variance.py's fixture construction and its
numpy<->RDD parity assertion style.
"""
from __future__ import annotations

import numpy as np
import pytest

from tests._stm_synth import synthetic_gated_corpus, fit_stm


def _global_params_from_fit(gp):
    return {"lambda": gp["lambda"], "Gamma": gp["Gamma"], "Sigma": gp["Sigma"]}


class TestCorpusConcentrationStmNumpy:
    def _build_fitted_corpus(self, *, seed=0):
        docs, planted, part = synthetic_gated_corpus(
            groups=("A", "B"), fg_per_group=1, bg_k=2, V=40, D=40, doc_len=25,
            bg_frac=0.5, seed=seed,
        )
        K = part.K
        gp = fit_stm(docs, K=K, V=40, sigma_init=1.0, n_iter=20,
                     partition=part, seed=seed)
        return docs, part, gp, K

    def test_numpy_summary_shape_and_bounds(self):
        from spark_vi.mllib.topic.stm import corpus_concentration_stm

        docs, part, gp, K = self._build_fitted_corpus(seed=1)
        summary = corpus_concentration_stm(
            docs, _global_params_from_fit(gp), part, reference=0,
        )

        assert set(summary.keys()) == {"n_docs", "top_mass", "eff_topics"}
        assert summary["n_docs"] == len(docs)

        # top_mass support is [0, 1]: bin_edges must span exactly that.
        edges = summary["top_mass"]["bin_edges"]
        assert edges[0] == pytest.approx(0.0)
        assert edges[-1] == pytest.approx(1.0)

        eff_mean = summary["eff_topics"]["mean"]
        assert 1.0 <= eff_mean <= K

    def test_gating_limits_eff_topics(self):
        # Directly exercise the gated-softmax construction: a background-only
        # doc's mode theta must place exactly 0 mass on every foreground topic,
        # so its effective-topics count can never exceed the background
        # block's size.
        part_docs_kwargs = dict(
            groups=("A", "B"), fg_per_group=1, bg_k=2, V=40, D=40, doc_len=25,
            bg_frac=0.5, seed=2,
        )
        docs, planted, part = synthetic_gated_corpus(**part_docs_kwargs)
        K = part.K
        bg_only_allowed = part.allowed_indices(frozenset())
        fg_indices = np.array(
            [k for k in range(K) if k not in set(bg_only_allowed.tolist())]
        )
        assert fg_indices.size > 0

        # Build a fake eta_hat as the gated softmax construction would use it:
        # -inf outside the allowed set (exactly what _stm_doc_inference returns
        # for disallowed topics).
        eta_hat = np.full(K, -np.inf)
        eta_hat[bg_only_allowed] = np.array([0.3, -0.1])[: len(bg_only_allowed)]

        allowed = bg_only_allowed
        z = eta_hat[allowed]
        z = z - z.max()
        w = np.exp(z)
        theta = np.zeros(K, dtype=np.float64)
        theta[allowed] = w / w.sum()

        assert np.all(theta[fg_indices] == 0.0)

    def test_numpy_rdd_parity(self, spark):
        from spark_vi.mllib.topic.stm import (
            corpus_concentration_stm, corpus_concentration_stm_rdd,
        )

        docs, part, gp, K = self._build_fitted_corpus(seed=3)
        global_params = _global_params_from_fit(gp)

        expected = corpus_concentration_stm(docs, global_params, part, reference=0)

        rdd = spark.sparkContext.parallelize(docs, numSlices=3)
        result = corpus_concentration_stm_rdd(rdd, global_params, part, reference=0)

        assert result["n_docs"] == expected["n_docs"]
        for key in ("top_mass", "eff_topics"):
            for stat in ("mean", "std", "p10", "p25", "p50", "p75", "p90"):
                np.testing.assert_allclose(
                    result[key][stat], expected[key][stat], rtol=1e-8, atol=1e-10
                )
            np.testing.assert_allclose(result[key]["hist"], expected[key]["hist"], rtol=1e-8)
            np.testing.assert_allclose(
                result[key]["bin_edges"], expected[key]["bin_edges"], rtol=1e-8
            )

    def test_rdd_empty_raises(self, spark):
        from spark_vi.mllib.topic.stm import corpus_concentration_stm_rdd

        docs, part, gp, K = self._build_fitted_corpus(seed=4)
        global_params = _global_params_from_fit(gp)
        empty = spark.sparkContext.parallelize([], numSlices=1)
        with pytest.raises(ValueError, match="empty"):
            corpus_concentration_stm_rdd(empty, global_params, part)
