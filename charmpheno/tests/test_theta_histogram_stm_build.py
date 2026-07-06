"""End-to-end (build-path) test for the STM per-doc theta histogram.

Exercises the exact chain the dashboard build step now runs for a gated STM:

    corpus_theta_gated_rdd            (spark_vi: per-doc gated MAP theta)
      -> compute_theta_aggregates    (charmpheno: bin + percentiles + suppress)
      -> _parse_theta_histogram / _parse_theta_percentiles  (-> ndarrays)
      -> (builder ndarray->JSON conversion, NaN->None)
      -> write_phenotypes_bundle     (charmpheno serialization)

and asserts the produced phenotypes.json now carries the fields the frontend's
"topic mass distribution" panel gates on: top-level theta_histogram_bin_edges,
per-phenotype theta_histogram / theta_percentiles, and — because a gated
foreground topic is a structural 0 for out-of-group docs — that a foreground
topic puts most of its mass in the lowest histogram bin.

There is no full build_dashboard harness (it needs a real checkpoint + BQ), so
this drives the same helpers on a small in-process synthetic gated model.
"""
from __future__ import annotations

import json

import numpy as np

from spark_vi.models.topic.partition import TopicBlockPartition
from spark_vi.models.topic.types import STMDocument


def _synthetic_gated_model(*, D=120, seed=0):
    """Two groups A/B (one foreground topic each) + two background topics,
    disjoint per-topic vocab so beta separates topics cleanly. Returns
    (docs, partition, global_params, K)."""
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


def test_stm_theta_histogram_end_to_end(spark, tmp_path):
    from spark_vi.mllib.topic.stm import corpus_theta_gated_rdd
    from charmpheno.export.theta_aggregates import compute_theta_aggregates
    from charmpheno.export.model_adapter import (
        _parse_theta_histogram, _parse_theta_percentiles,
    )
    from charmpheno.export.dashboard import write_phenotypes_bundle

    docs, part, gp, K = _synthetic_gated_model(D=120, seed=0)
    n_bins = 50
    min_count = 5  # small so the D=120 synthetic corpus isn't fully suppressed

    rdd = spark.sparkContext.parallelize(docs, numSlices=3)
    theta_arr = corpus_theta_gated_rdd(rdd, gp, part, sample_cap=10_000, seed=0)
    assert theta_arr.shape == (len(docs), K)

    agg = compute_theta_aggregates(theta_arr, min_count=min_count)
    hist_np = _parse_theta_histogram(agg["theta_histogram"])       # (K, n_bins), NaN=suppressed
    pct_np = _parse_theta_percentiles(agg["theta_percentiles"])    # (K, 5)
    assert hist_np.shape == (K, n_bins)
    assert pct_np.shape == (K, 5)

    # Mirror the builder's ndarray -> JSON conversion (NaN -> None).
    hist = [[None if np.isnan(v) else float(v) for v in row]
            for row in hist_np.tolist()]
    pct = [{"p5": float(r[0]), "p25": float(r[1]), "p50": float(r[2]),
            "p75": float(r[3]), "p95": float(r[4])} for r in pct_np]

    out = tmp_path / "phenotypes.json"
    write_phenotypes_bundle(
        out,
        npmi=[0.0] * K,
        pair_coverage=[0.0] * K,
        corpus_prevalence=theta_arr.mean(axis=0).tolist(),
        topic_indices=list(range(K)),
        theta_histogram=hist,
        theta_percentiles=pct,
        n_bins=n_bins,
        min_count=min_count,
        labels=None,
    )
    payload = json.loads(out.read_text())

    # Top-level bin edges present, correct length + range.
    bin_edges = payload["theta_histogram_bin_edges"]
    assert len(bin_edges) == n_bins + 1
    assert bin_edges[0] == 0.0 and bin_edges[-1] == 1.0
    assert payload["theta_histogram_min_count"] == min_count

    # Every phenotype carries the two theta fields at the right shapes.
    assert len(payload["phenotypes"]) == K
    for p in payload["phenotypes"]:
        assert len(p["theta_histogram"]) == n_bins
        assert set(p["theta_percentiles"].keys()) == {
            "p5", "p25", "p50", "p75", "p95"}

    # A foreground topic (slot for group A) is a structural 0 for every B doc
    # (~half the corpus), so most of its histogram mass sits in the lowest bin.
    fg_a = int(part.block_indices("A")[0])
    fg_row = hist_np[fg_a]
    lowest = fg_row[0]
    assert not np.isnan(lowest)
    # Lowest bin is the modal bin and holds a large fraction of patients.
    assert lowest >= 0.4
    assert lowest == np.nanmax(fg_row)
