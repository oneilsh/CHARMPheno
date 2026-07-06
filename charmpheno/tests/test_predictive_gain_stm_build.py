"""End-to-end (build-path) test for the STM predictive-gain aggregates
(Phase-2 Task 5: threading spark_vi.mllib.topic.predictive_gain's per-topic
corpus aggregates into the dashboard build).

Clones test_theta_histogram_stm_build.py's synthetic gated-model fixture and
drives the exact chain the dashboard build phase now runs for a gated STM:

    corpus_predictive_gain_gated_rdd     (spark_vi: per-topic presence/depth/
                                           prominence aggregates, fast=True)
    predictive_gain_downdate_audit       (spark_vi: cold-vs-fast reliability)
      -> builder ndarray -> JSON conversion (NaN -> None)
      -> write_phenotypes_bundle         (charmpheno serialization)

and asserts the produced phenotypes.json carries a well-formed
"predictive_gain" object: per-topic arrays at the right length, the
bundle-level diagnostics (null_band, observed_delta_range, downdate_audit),
and — because a gated foreground topic is a structural no-op for the OTHER
group's documents (it never appears in their `allowed` set, so it never
accrues count_k/depth/presence from them) — that a foreground topic's
count_k is restricted to roughly half the corpus (its own group).

There is no full build_dashboard harness (it needs a real checkpoint + BQ),
so this drives the same helpers on a small in-process synthetic gated model,
mirroring the theta-histogram e2e test.
"""
from __future__ import annotations

import json

import numpy as np

from spark_vi.models.topic.partition import TopicBlockPartition
from spark_vi.models.topic.types import STMDocument


def _synthetic_gated_model(*, D=120, seed=0):
    """Two groups A/B (one foreground topic each) + two background topics,
    disjoint per-topic vocab so beta separates topics cleanly. Returns
    (docs, partition, global_params, K). Sigma is the identity (unit
    correlation R with unit diagonal), matching a fitted ADR-0034 checkpoint;
    Gamma is all-zeros (no covariate effects) so x=[1.0] docs are generic."""
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

    bg_rows = part.background_indices()
    rng = np.random.default_rng(seed)
    docs = []
    for i in range(D):
        g = "A" if i % 2 == 0 else "B"
        fg = part.block_indices(g)[0]
        bg = int(bg_rows[rng.integers(len(bg_rows))])
        toks = np.concatenate([
            rng.choice(V, size=13, p=beta[bg]),
            rng.choice(V, size=12, p=beta[fg]),
        ])
        u, c = np.unique(toks, return_counts=True)
        docs.append(STMDocument(
            indices=u.astype(np.int32), counts=c.astype(np.float64),
            length=int(c.sum()), x=np.array([1.0]), groups=frozenset({g})))
    return docs, part, gp, K


def test_stm_predictive_gain_end_to_end(spark, tmp_path):
    from spark_vi.mllib.topic.predictive_gain import (
        corpus_predictive_gain_gated_rdd,
        predictive_gain_downdate_audit,
    )
    from charmpheno.export.dashboard import write_phenotypes_bundle

    docs, part, gp, K = _synthetic_gated_model(D=120, seed=0)
    n_bins = 50

    rdd = spark.sparkContext.parallelize(docs, numSlices=3)
    pg = corpus_predictive_gain_gated_rdd(
        rdd, gp, part, c=1.0, reference=None, fast=True,
        sample_cap=10_000, seed=0,
    )
    for key in ("mean_gain", "depth", "presence", "prominence_hist",
                "length_corr", "dedup_mean_gain", "count_k"):
        assert key in pg
    assert pg["mean_gain"].shape == (K,)
    assert pg["prominence_hist"].shape == (K, n_bins)
    assert pg["n_docs"] > 0

    # Cold-vs-fast downdate audit on a small in-memory sample.
    audit_docs = rdd.takeSample(False, 30, seed=0)
    audit = predictive_gain_downdate_audit(
        audit_docs, gp, part, c=1.0, reference=None,
    )
    assert audit["n_docs_audited"] > 0
    assert np.isfinite(audit["max_abs_overall"])

    # Mirror the builder's ndarray -> JSON conversion (NaN -> None).
    def _nan_to_none(arr):
        return [None if np.isnan(v) else float(v) for v in arr.tolist()]

    presence = _nan_to_none(pg["presence"])
    mean_gain = _nan_to_none(pg["mean_gain"])
    depth = _nan_to_none(pg["depth"])
    length_corr = _nan_to_none(pg["length_corr"])
    dedup_gain = _nan_to_none(pg["dedup_mean_gain"])
    prominence_hist = [
        [None if np.isnan(v) else float(v) for v in row]
        for row in pg["prominence_hist"].tolist()
    ]
    downdate_audit = {
        "max_abs_overall": float(audit["max_abs_overall"]),
        "n_docs_audited": int(audit["n_docs_audited"]),
    }

    out = tmp_path / "phenotypes.json"
    write_phenotypes_bundle(
        out,
        npmi=[0.0] * K,
        pair_coverage=[0.0] * K,
        corpus_prevalence=[0.25] * K,
        topic_indices=list(range(K)),
        labels=None,
        presence=presence,
        mean_gain=mean_gain,
        depth=depth,
        prominence_hist=prominence_hist,
        length_corr=length_corr,
        dedup_gain=dedup_gain,
        prominence_bin_edges=pg["prominence_bin_edges"].tolist(),
        null_band=pg["null_band"],
        observed_delta_range=list(pg["observed_delta_range"]),
        predictive_gain_downdate_audit=downdate_audit,
        predictive_gain_scale=1.0,
        predictive_gain_n_docs=int(pg["n_docs"]),
    )
    payload = json.loads(out.read_text())
    assert "NaN" not in out.read_text()

    pgj = payload["predictive_gain"]
    assert len(pgj["presence"]) == K
    assert len(pgj["mean_gain"]) == K
    assert len(pgj["depth"]) == K
    assert len(pgj["prominence_hist"]) == K
    assert all(len(row) == n_bins for row in pgj["prominence_hist"])
    assert len(pgj["length_corr"]) == K
    assert len(pgj["dedup_gain"]) == K
    assert len(pgj["prominence_bin_edges"]) == n_bins + 1

    # Bundle-level diagnostics survive the round trip.
    assert set(pgj["null_band"].keys()) == {"mean", "std", "n", "hist", "p95"}
    assert len(pgj["observed_delta_range"]) == 2
    assert pgj["downdate_audit"]["n_docs_audited"] == audit["n_docs_audited"]
    assert pgj["scale"] == 1.0
    assert pgj["n_docs"] == int(pg["n_docs"])

    # A gated foreground topic only ever appears in its OWN group's `allowed`
    # set (never the other group's), so count_k restricts it to ~half the
    # corpus -- the within-group denominator the module docstring promises.
    fg_a = int(part.block_indices("A")[0])
    assert 0 < pg["count_k"][fg_a] <= (len(docs) // 2) + 5
