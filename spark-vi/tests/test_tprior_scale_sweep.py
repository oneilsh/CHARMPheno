from __future__ import annotations

import json
import math
import numpy as np

from tests._stm_synth import synthetic_gated_corpus, fit_stm


def _build_fitted_corpus(seed=0):
    docs, planted, part = synthetic_gated_corpus(
        groups=("A", "B"), fg_per_group=1, bg_k=2, V=40, D=40, doc_len=25,
        bg_frac=0.5, seed=seed,
    )
    gp = fit_stm(docs, K=part.K, V=40, sigma_init=1.0, n_iter=20,
                 partition=part, seed=seed)
    return docs, part, {"lambda": gp["lambda"], "Gamma": gp["Gamma"],
                        "Sigma": gp["Sigma"]}


def test_nu_inf_column_matches_gaussian_sweep():
    from spark_vi.mllib.topic.stm import (
        corpus_tprior_scale_sweep_gated, corpus_heldout_scale_sweep_gated,
    )
    docs, part, gp = _build_fitted_corpus(seed=3)
    c_grid = [1, 2, 4, 8]
    gauss = corpus_heldout_scale_sweep_gated(
        docs, gp, part, c_grid=c_grid, holdout_frac=0.3, seed=0)
    t = corpus_tprior_scale_sweep_gated(
        docs, gp, part, c_grid=c_grid, nu_grid=[math.inf],
        holdout_frac=0.3, seed=0)
    by_c = {row["c"]: row["ll"] for row in t["grid"] if row["nu"] == "inf"}
    for c in c_grid:
        assert abs(by_c[c] - gauss["lls"][c]) < 1e-6


def test_grid_argmax_and_structure():
    from spark_vi.mllib.topic.stm import corpus_tprior_scale_sweep_gated
    docs, part, gp = _build_fitted_corpus(seed=5)
    t = corpus_tprior_scale_sweep_gated(
        docs, gp, part, c_grid=[2, 4, 8], nu_grid=[2.5, 5, math.inf],
        holdout_frac=0.3, drift_fracs=(0.2, 0.3), seed=0)
    assert len(t["grid"]) == 9
    assert t["argmax"]["c"] in {2, 4, 8}
    assert t["argmax"]["nu"] in {2.5, 5, "inf"}
    assert all(np.isfinite(r["ll"]) for r in t["grid"])
    assert t["n_docs"] > 0


def test_drift_and_sd_readouts_present():
    from spark_vi.mllib.topic.stm import corpus_tprior_scale_sweep_gated
    docs, part, gp = _build_fitted_corpus(seed=7)
    t = corpus_tprior_scale_sweep_gated(
        docs, gp, part, c_grid=[2, 4, 8], nu_grid=[5, math.inf],
        holdout_frac=0.3, drift_fracs=(0.2, 0.3, 0.5), seed=0)
    d = t["drift"]
    assert [p["frac"] for p in d["gaussian"]] == [0.2, 0.3, 0.5]
    assert [p["frac"] for p in d["tprior"]] == [0.2, 0.3, 0.5]
    assert d["gaussian_spread"] >= 0.0 and d["tprior_spread"] >= 0.0
    s = t["sd_readout"]
    assert s["n_docs"] > 0
    for q in ("p10", "p25", "p50", "p75", "p90"):
        assert q in s["sd_quantiles"] and q in s["sd_c_quantiles"]
    # sd_c = sd * c_star: median should be a positive scale
    assert s["sd_c_quantiles"]["p50"] > 0.0


def test_output_is_json_safe():
    from spark_vi.mllib.topic.stm import corpus_tprior_scale_sweep_gated
    docs, part, gp = _build_fitted_corpus(seed=9)
    t = corpus_tprior_scale_sweep_gated(
        docs, gp, part, c_grid=[2, 4], nu_grid=[5, math.inf],
        holdout_frac=0.3, drift_fracs=(0.2, 0.3), seed=0)
    json.dumps(t)   # must not raise


class TestRddParity:
    def test_rdd_matches_numpy(self, spark):
        import math
        from spark_vi.mllib.topic.stm import (
            corpus_tprior_scale_sweep_gated,
            corpus_tprior_scale_sweep_gated_rdd,
        )
        docs, part, gp = _build_fitted_corpus(seed=3)
        c_grid = [2, 4, 8]
        nu_grid = [5, math.inf]
        expected = corpus_tprior_scale_sweep_gated(
            docs, gp, part, c_grid=c_grid, nu_grid=nu_grid,
            holdout_frac=0.3, drift_fracs=(0.2, 0.3), seed=0)
        rdd = spark.sparkContext.parallelize(docs, numSlices=3)
        got = corpus_tprior_scale_sweep_gated_rdd(
            rdd, gp, part, c_grid=c_grid, nu_grid=nu_grid,
            holdout_frac=0.3, drift_fracs=(0.2, 0.3), seed=0)
        # grid LLs match doc-for-doc (same splits, same cold start per (doc, c))
        exp_by = {(r["c"], r["nu"]): r["ll"] for r in expected["grid"]}
        got_by = {(r["c"], r["nu"]): r["ll"] for r in got["grid"]}
        assert set(exp_by) == set(got_by)
        for k in exp_by:
            assert abs(exp_by[k] - got_by[k]) < 1e-6
        # argmax selection (which grid point wins) must match exactly; the
        # winning ll itself is subject to the same distributed-reduction
        # ULP-level reordering noise as the grid (treeReduce sums partition
        # totals in a different grouping than the numpy sequential loop --
        # not a correctness issue, see the grid check above), so compare it
        # with the same tolerance rather than dict `==`.
        assert got["argmax"]["c"] == expected["argmax"]["c"]
        assert got["argmax"]["nu"] == expected["argmax"]["nu"]
        assert abs(got["argmax"]["ll"] - expected["argmax"]["ll"]) < 1e-6
        assert got["n_docs"] == expected["n_docs"]
        assert abs(got["sd_readout"]["sd_c_quantiles"]["p50"]
                   - expected["sd_readout"]["sd_c_quantiles"]["p50"]) < 1e-6

    def test_rdd_output_json_safe(self, spark):
        import json, math
        from spark_vi.mllib.topic.stm import corpus_tprior_scale_sweep_gated_rdd
        docs, part, gp = _build_fitted_corpus(seed=9)
        rdd = spark.sparkContext.parallelize(docs, numSlices=2)
        got = corpus_tprior_scale_sweep_gated_rdd(
            rdd, gp, part, c_grid=[2, 4], nu_grid=[5, math.inf],
            holdout_frac=0.3, drift_fracs=(0.2, 0.3), seed=0)
        json.dumps(got)
