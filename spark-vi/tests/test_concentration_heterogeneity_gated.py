"""Tests for the gated distributed concentration-heterogeneity pass.

Three tests:
  - TestSummarizeExtractedMatchesCore: guards the behavior-preserving refactor
    of concentration_raw_vs_dedup into summarize_concentration_heterogeneity
    (Deliverable 1) -- the extracted aggregation must produce byte-identical
    output to the pre-refactor monolith, given the same per-doc arrays.
  - TestGatedAdapterThetaValid: gated_infer_theta (Deliverable 2) returns a
    valid theta on the simplex, disallowed topics exactly 0, reference alive.
  - TestNumpyRddParity: corpus_concentration_heterogeneity_gated (numpy) vs
    corpus_concentration_heterogeneity_rdd (local spark) on the same docs
    (Deliverable 3) -- deterministic MAP inference, no RNG, so exact parity.
"""
from __future__ import annotations

import numpy as np
import pytest

from spark_vi.eval.topic.concentration_heterogeneity import (
    concentration_raw_vs_dedup,
    summarize_concentration_heterogeneity,
)
from spark_vi.models.topic.partition import TopicBlockPartition
from spark_vi.models.topic.types import STMDocument

from tests._stm_synth import synthetic_gated_corpus, fit_stm


def _doc(indices, counts, x=None, groups=frozenset()):
    return STMDocument(
        indices=np.asarray(indices, dtype=np.int32),
        counts=np.asarray(counts, dtype=np.float64),
        length=int(np.sum(counts)),
        x=np.array([1.0]) if x is None else np.asarray(x, dtype=np.float64),
        groups=groups,
    )


class TestSummarizeExtractedMatchesCore:
    """Same per-doc arrays fed two ways: (a) through the pre-existing
    concentration_raw_vs_dedup with a mock infer_theta that yields the
    prescribed theta, (b) directly into summarize_concentration_heterogeneity.
    Every summary field must match -- this is the refactor's safety net."""

    def _docs(self):
        return [
            _doc([0, 1, 2], [3, 1, 1]),
            _doc([0, 1, 3], [2, 1, 1]),
            _doc([0, 2, 3], [1, 1, 1]),
            _doc([1, 2, 3], [4, 2, 1]),
        ]

    def _theta_lookup(self):
        return {
            ((0, 1, 2), (3.0, 1.0, 1.0)): np.array([0.7, 0.2, 0.1]),
            ((0, 1, 2), (1.0, 1.0, 1.0)): np.array([0.5, 0.3, 0.2]),
            ((0, 1, 3), (2.0, 1.0, 1.0)): np.array([0.6, 0.3, 0.1]),
            ((0, 1, 3), (1.0, 1.0, 1.0)): np.array([0.4, 0.35, 0.25]),
            ((0, 2, 3), (1.0, 1.0, 1.0)): np.array([0.5, 0.3, 0.2]),
            ((1, 2, 3), (4.0, 2.0, 1.0)): np.array([0.8, 0.15, 0.05]),
            ((1, 2, 3), (1.0, 1.0, 1.0)): np.array([0.6, 0.25, 0.15]),
        }

    def _infer_theta(self, indices, counts):
        lookup = self._theta_lookup()
        key = (tuple(int(i) for i in indices), tuple(float(c) for c in counts))
        return lookup[key]

    def test_summarize_extracted_matches_core(self):
        from spark_vi.eval.topic.concentration import doc_concentration
        from spark_vi.eval.topic.concentration_heterogeneity import (
            dedup_counts, doc_burstiness,
        )

        docs = self._docs()
        core_result = concentration_raw_vs_dedup(docs, self._infer_theta)

        # Recompute the per-doc arrays independently (mirrors what a
        # distributed per-doc pass would hand to the driver) and feed them
        # straight into the extracted aggregation function.
        top_mass_raw, top_mass_dedup = [], []
        eff_topics_raw, eff_topics_dedup = [], []
        repeat_fraction = []
        n_skipped = 0
        for doc in docs:
            burst = doc_burstiness(doc.indices, doc.counts)
            if burst["total"] < 2.0 or burst["unique"] <= 1:
                n_skipped += 1
                continue
            theta_raw = self._infer_theta(doc.indices, doc.counts)
            theta_dedup = self._infer_theta(doc.indices, dedup_counts(doc.counts))
            top_raw, eff_raw = doc_concentration(theta_raw)
            top_dedup, eff_dedup = doc_concentration(theta_dedup)
            top_mass_raw.append(top_raw)
            top_mass_dedup.append(top_dedup)
            eff_topics_raw.append(eff_raw)
            eff_topics_dedup.append(eff_dedup)
            repeat_fraction.append(burst["repeat_fraction"])

        extracted_result = summarize_concentration_heterogeneity(
            top_mass_raw=np.array(top_mass_raw),
            top_mass_dedup=np.array(top_mass_dedup),
            eff_topics_raw=np.array(eff_topics_raw),
            eff_topics_dedup=np.array(eff_topics_dedup),
            repeat_fraction=np.array(repeat_fraction),
            n_skipped=n_skipped,
        )

        assert set(extracted_result.keys()) == set(core_result.keys())
        for key, val in core_result.items():
            if isinstance(val, np.ndarray):
                assert np.allclose(extracted_result[key], val)
            elif isinstance(val, dict):
                _dicts_close(extracted_result[key], val)
            elif isinstance(val, float) and np.isnan(val):
                assert np.isnan(extracted_result[key])
            else:
                assert extracted_result[key] == val


def _dicts_close(a, b):
    assert set(a.keys()) == set(b.keys())
    for k, v in b.items():
        if v is None:
            assert a[k] is None
        else:
            assert a[k] == pytest.approx(v)
    return True


def _build_fitted_corpus(*, seed=0):
    docs, planted, part = synthetic_gated_corpus(
        groups=("A", "B"), fg_per_group=1, bg_k=2, V=40, D=40, doc_len=25,
        bg_frac=0.5, seed=seed,
    )
    K = part.K
    gp = fit_stm(docs, K=K, V=40, sigma_init=1.0, n_iter=20,
                 partition=part, seed=seed)
    return docs, part, gp, K


def _global_params_from_fit(gp):
    return {"lambda": gp["lambda"], "Gamma": gp["Gamma"], "Sigma": gp["Sigma"]}


class TestGatedAdapterThetaValid:
    def test_gated_adapter_theta_valid(self):
        from spark_vi.mllib.topic.stm import gated_infer_theta

        docs, part, gp, K = _build_fitted_corpus(seed=5)
        global_params = _global_params_from_fit(gp)

        infer_theta = gated_infer_theta(global_params, part, c=4.0, reference=0)

        doc = docs[0]
        theta = infer_theta(doc.indices, doc.counts, doc.groups, doc.x)

        assert theta.shape == (K,)
        assert np.all(theta >= 0.0)
        assert theta.sum() == pytest.approx(1.0)

        allowed = part.allowed_indices(doc.groups)
        disallowed = np.setdiff1d(np.arange(K), allowed)
        assert np.allclose(theta[disallowed], 0.0)
        # reference topic (0) is always in the background block, hence
        # always allowed, and must carry nonzero mass (it is "alive": its
        # exp(0)=1 sits in the softmax denominator, not zeroed out).
        assert theta[0] > 0.0


class TestNumpyRddParity:
    def test_numpy_rdd_parity(self, spark):
        from spark_vi.mllib.topic.stm import (
            corpus_concentration_heterogeneity_gated,
            corpus_concentration_heterogeneity_rdd,
        )

        docs, part, gp, K = _build_fitted_corpus(seed=7)
        global_params = _global_params_from_fit(gp)

        expected = corpus_concentration_heterogeneity_gated(
            docs, global_params, part, c=4.0, reference=0,
        )

        rdd = spark.sparkContext.parallelize(docs, numSlices=3)
        result = corpus_concentration_heterogeneity_rdd(
            rdd, global_params, part, c=4.0, reference=0, sample_frac=None,
        )

        assert result["n_docs"] == expected["n_docs"]
        assert result["n_skipped"] == expected["n_skipped"]
        assert result["c"] == expected["c"]

        for key in (
            "top_mass_raw", "top_mass_dedup", "eff_topics_raw",
            "eff_topics_dedup", "repeat_fraction",
        ):
            np.testing.assert_allclose(result[key], expected[key], rtol=1e-9, atol=1e-9)

        for key in (
            "spread_ratio_top_mass", "rank_corr_top_mass", "burstiness_corr_top_mass",
        ):
            np.testing.assert_allclose(result[key], expected[key], rtol=1e-9, atol=1e-9)

        for summary_key in (
            "top_mass_raw_summary", "top_mass_dedup_summary",
            "eff_topics_raw_summary", "eff_topics_dedup_summary",
            "repeat_fraction_summary",
        ):
            for stat_key, stat_val in expected[summary_key].items():
                if stat_val is None:
                    assert result[summary_key][stat_key] is None
                else:
                    np.testing.assert_allclose(
                        result[summary_key][stat_key], stat_val, rtol=1e-9, atol=1e-9
                    )

    def test_numpy_rdd_parity_with_skipped_docs(self, spark):
        """Same as test_numpy_rdd_parity, but with a handful of degenerate
        docs (total < 2 tokens, or a single unique token) spliced in among
        normal ones -- the base parity test above has zero skips, so it
        never exercises whether the skip guard drops the SAME docs, in the
        SAME relative order, on both the numpy oracle and the distributed
        RDD path. A desync here would silently misalign the per-doc arrays
        the two paths compare (or corrupt the rank/burstiness correlations)
        without necessarily changing n_skipped, so this asserts n_skipped>0
        on both sides AND full elementwise/summary parity."""
        from spark_vi.mllib.topic.stm import (
            corpus_concentration_heterogeneity_gated,
            corpus_concentration_heterogeneity_rdd,
        )

        docs, part, gp, K = _build_fitted_corpus(seed=11)
        global_params = _global_params_from_fit(gp)

        degenerate = [
            _doc([0], [1.0], groups=frozenset({"A"})),   # total < 2 -> skip
            _doc([3], [5.0], groups=frozenset({"B"})),   # unique <= 1 -> skip
            _doc([1], [1.0], groups=frozenset({"A"})),   # total < 2 -> skip
        ]
        mixed_docs = docs[:20] + degenerate + docs[20:]

        expected = corpus_concentration_heterogeneity_gated(
            mixed_docs, global_params, part, c=4.0, reference=0,
        )
        assert expected["n_skipped"] > 0

        rdd = spark.sparkContext.parallelize(mixed_docs, numSlices=3)
        result = corpus_concentration_heterogeneity_rdd(
            rdd, global_params, part, c=4.0, reference=0, sample_frac=None,
        )
        assert result["n_skipped"] > 0

        assert result["n_docs"] == expected["n_docs"]
        assert result["n_skipped"] == expected["n_skipped"]
        assert result["c"] == expected["c"]

        for key in (
            "top_mass_raw", "top_mass_dedup", "eff_topics_raw",
            "eff_topics_dedup", "repeat_fraction",
        ):
            np.testing.assert_allclose(result[key], expected[key], rtol=1e-9, atol=1e-9)

        for key in (
            "spread_ratio_top_mass", "rank_corr_top_mass", "burstiness_corr_top_mass",
        ):
            np.testing.assert_allclose(result[key], expected[key], rtol=1e-9, atol=1e-9)

        for summary_key in (
            "top_mass_raw_summary", "top_mass_dedup_summary",
            "eff_topics_raw_summary", "eff_topics_dedup_summary",
            "repeat_fraction_summary",
        ):
            for stat_key, stat_val in expected[summary_key].items():
                if stat_val is None:
                    assert result[summary_key][stat_key] is None
                else:
                    np.testing.assert_allclose(
                        result[summary_key][stat_key], stat_val, rtol=1e-9, atol=1e-9
                    )
