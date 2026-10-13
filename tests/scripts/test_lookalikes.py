"""Framing 3 — undiagnosed look-alikes (lookalikes.py and gated_pc_readout.run_lookalikes)."""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(REPO_ROOT / "analysis" / "cloud"), str(REPO_ROOT / "charmpheno"),
           str(REPO_ROOT / "spark-vi"), str(REPO_ROOT / "tests" / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.environ["PYTHONPATH"] = os.pathsep.join(
    p for p in (str(REPO_ROOT), str(REPO_ROOT / "analysis" / "cloud"),
                str(REPO_ROOT / "charmpheno"), str(REPO_ROOT / "spark-vi"),
                os.environ.get("PYTHONPATH", "")) if p)

import lookalikes as lk  # noqa: E402


def test_auc_handles_ties_and_extremes():
    assert lk.auc([0.1, 0.2, 0.9, 0.8], [0, 0, 1, 1]) == 1.0
    assert lk.auc([0.5, 0.5], [0, 1]) == 0.5
    assert np.isnan(lk.auc([0.1, 0.2], [0, 0]))


def test_top_lookalikes_skip_the_labelled():
    keys, cut = lk.top_lookalikes([0.9, 0.8, 0.7, 0.1], [1, 0, 0, 0], ["a", "b", "c", "d"], 2)
    assert keys == ["b", "c"] and cut == 0.7


def test_feature_profile_keeps_x_like_codes_on_enough_patients():
    top = [{0: {1: 1.0, 2: 1.0}}] * 25 + [{0: {3: 1.0}}] * 5
    pos = [{0: {1: 1.0}}] * 10 + [{0: {9: 1.0}}] * 10
    bg = [{0: {2: 1.0}}] * 50 + [{0: {}}] * 50
    rows = lk.feature_profile(top, pos, bg)
    keys = [r[0] for r in rows]
    assert (0, 1) in keys            # look-alike AND X-like
    assert (0, 2) not in keys        # common in background, not X-like
    assert (0, 3) not in keys        # fewer than 20 look-alikes carry it
    assert lk.feature_profile(top, pos, bg, excl_by_dom={0: {1}}) == []


@pytest.mark.slow
def test_run_lookalikes_end_to_end(spark, tmp_path):
    from pyspark.ml.linalg import Vectors
    from pyspark.sql import Row
    from test_inspect_topics import _make_census_run
    import gated_pc_readout as gpr
    _make_census_run(tmp_path)                        # K = 10; node1's signature = topic 1
    C, K = 4, 10
    parent_int = {0: [], 1: [0], 2: [0], 3: [0]}
    (tmp_path / "readout_dag.json").write_text(json.dumps({
        "version": 1, "C_fit": 4, "C": 4,
        "parent_int": {"1": [0], "2": [0], "3": [0]},
        "int2cid": {str(e): 1000 + e for e in range(4)},
        "name_by_id": {str(1000 + e): f"node{e}" for e in range(4)},
        "pos_sources": [[0], [1], [2], [3]]}))
    (tmp_path / "bundle_meta.json").write_text(json.dumps({
        "int2cid": {str(e): 1000 + e for e in range(4)},
        "vocab_maps": [{str(500 + i): i for i in range(12)}],
        "name_by_id": {str(1000 + e): f"node{e}" for e in range(4)}}))
    (tmp_path / "concept_names.csv").write_text(
        "concept_id,concept_name\n504,node1\n505,marker five\n500,marker zero\n")
    Vh = np.zeros((C, K)); Vh[1, 1] = 12.0; Vh[2, 4] = 12.0
    np.savez(tmp_path / "readout_heads_gated_pc_t.npz", V=Vh,
             b_raw=np.array([5.0, -4.0, -4.0, -4.0]), const=np.zeros(C),
             degenerate=np.zeros(C, bool), C=np.int64(C), K=np.int64(K),
             theta_topm=np.int64(0))
    rng = np.random.default_rng(0)
    rows = []
    for i in range(400):
        kind = "pos1" if i < 60 else "look" if i < 100 else "pos2" if i < 130 else "bg"
        th = rng.dirichlet(np.ones(K))
        codes = set(rng.choice(np.arange(6, 12), 3, replace=False).tolist())
        if kind in ("pos1", "look"):
            th[1] += 2.0
            codes |= {0, 5}
        if kind == "look":
            codes.add(4)                              # the unmapped node1-named concept
        if kind == "pos2":
            th[4] += 2.0
        th /= th.sum()
        lab = [1.0, float(kind == "pos1"), float(kind == "pos2"), 0.0]
        rows.append(Row(person_id=i, topicDistribution=Vectors.dense(th), label=lab,
                        labelMask=[1.0] * C,
                        features_0=Vectors.sparse(12, sorted(codes), [1.0] * len(codes))))
    df = spark.createDataFrame(rows)
    js = gpr.run_lookalikes(df, tmp_path, "^node1$|^node2$", C=C, parent_int=parent_int,
                            adhoc=True, heads_tag="t", top_n=40, bg_n=300, tag="t")
    rep = (tmp_path / "lookalikes_t.md").read_text()
    assert js["node1"]["auc"] > 0.8
    assert js["node1"]["named_top"][0] >= 30            # the look-alikes carry it
    prof = [r[0] for r in js["node1"]["profile"]]
    assert "marker five" in prof or "marker zero" in prof
    assert "unmapped X-named concept" in rep and "## node2" in rep
