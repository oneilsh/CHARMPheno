"""Insight 0100 measurement: AUC with prior-diagnosis patients set aside
(prevalence_check.py, gated_pc_readout.run_prevalence_check)."""
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

import prevalence_check as pv  # noqa: E402


def test_vocab_to_nodes_reaches_every_readout_ancestor_of_the_label():
    # readout node 3 is an ancestor head whose positive sources are labels 1 and 2
    v2n = pv.vocab_to_nodes({1: [10], 2: [11]}, [[0], [1], [2], [1, 2]])
    assert v2n == {10: [1, 3], 11: [2, 3]}


def test_prior_flags_and_the_incident_auc_drop_the_prevalent():
    rng = np.random.default_rng(0)
    D = 400
    keys = list(range(D))
    y = np.zeros((D, 2), dtype=np.uint8)
    prior = np.zeros(D, dtype=bool)
    score = rng.random((D, 2)) * 0.5
    # prevalent patients: prior code, label usually re-coded, score high (code in features)
    prior[:80] = True
    y[:60, 1] = 1
    score[:80, 1] += 0.5
    # incident positives: weakly separable
    y[100:160, 1] = 1
    score[100:160, 1] += 0.1
    codes = {k: ([7] if prior[k] else []) for k in keys}
    M = pv.prior_matrix(keys, codes, {7: [1]}, 2)
    assert M[:, 1].sum() == 80 and not M[:, 0].any()
    per = pv.node_aucs(score, y, M, [1])
    v = per[1]
    assert v["auc_all"] > v["auc_incident"] + 0.05
    assert v["n_pos_all"] == 120 and v["n_pos_incident"] == 60
    assert abs(v["prior_share_pos"] - 0.5) < 1e-9
    sm = pv.summarize(per)
    assert sm["n_nodes"] == 1 and sm["n_gap_over_0p05"] == 1


@pytest.mark.slow
def test_run_prevalence_check_end_to_end(spark, tmp_path):
    from pyspark.ml.linalg import Vectors
    from pyspark.sql import Row
    from test_inspect_topics import _make_census_run
    import gated_pc_readout as gpr
    _make_census_run(tmp_path)
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
    (tmp_path / "code_map.tsv").write_text("std_cid\tnode_cid\n503\t1001\n")
    Vh = np.zeros((C, K)); Vh[1, 1] = 12.0
    np.savez(tmp_path / "readout_heads_gated_pc_t.npz", V=Vh,
             b_raw=np.array([5.0, -4.0, -4.0, -4.0]), const=np.zeros(C),
             degenerate=np.zeros(C, bool), C=np.int64(C), K=np.int64(K),
             theta_topm=np.int64(0))
    rng = np.random.default_rng(0)
    rows = []
    for i in range(500):
        kind = "prev" if i < 80 else "inc" if i < 140 else "neg"
        th = rng.dirichlet(np.ones(K))
        codes = set(rng.choice(np.arange(6, 12), 3, replace=False).tolist())
        if kind == "prev":
            th[1] += 3.0
            codes.add(3)                                  # node1's own code, pre-index
        if kind == "inc":
            th[1] += 0.4
        th /= th.sum()
        lab = [1.0, float(kind == "inc" or (kind == "prev" and i < 60)), 0.0, 0.0]
        rows.append(Row(person_id=i, topicDistribution=Vectors.dense(th), label=lab,
                        labelMask=[1.0] * C,
                        features_0=Vectors.sparse(12, sorted(codes), [1.0] * len(codes))))
    df = spark.createDataFrame(rows)
    res = gpr.run_prevalence_check(df, tmp_path, C=C, parent_int=parent_int,
                                   heads_tag="t", tag="t")
    v = res["stacked"]["per_node"][1]
    # the 20 un-re-coded prevalent patients score high as NEGATIVES here, so setting
    # them aside can move the AUC either way; the mechanics are what is pinned
    assert v["auc_all"] != v["auc_incident"]
    assert v["n_pos_all"] == 120 and v["n_pos_incident"] == 60 and abs(v["prior_share_pos"] - 0.5) < 1e-9
    assert "largest gaps (stacked)" in (tmp_path / "prevalence_check_t.md").read_text()


def test_incident_mask_kernel_masks_prior_nodes_but_never_the_root():
    from pyspark.ml.linalg import Vectors
    import gated_pc_readout as gpr
    k = gpr.incident_mask_kernel(lambda: {3: [0, 1, 3], 7: [2]})
    out = k(Vectors.sparse(12, [3, 5], [1.0, 2.0]), [1.0, 1.0, 1.0, 1.0])
    assert out == [1.0, 0.0, 1.0, 0.0]                 # root kept, 1 and 3 masked
    assert k(Vectors.sparse(12, [5], [1.0]), [1.0, 0.0, 1.0, 1.0]) == [1.0, 0.0, 1.0, 1.0]
    assert k(Vectors.sparse(12, [7], [1.0]), None) is None


def test_mask_prior_cells_on_a_spark_frame(spark):
    from pyspark.ml.linalg import Vectors
    from pyspark.sql import Row
    import gated_pc_readout as gpr
    df = spark.createDataFrame([
        Row(person_id=1, features_0=Vectors.sparse(12, [3], [1.0]), labelMask=[1.0] * 4),
        Row(person_id=2, features_0=Vectors.sparse(12, [9], [1.0]), labelMask=[1.0] * 4)])
    got = {r.person_id: r.labelMask for r in
           gpr.mask_prior_cells(df, {3: [1, 3]}).collect()}
    assert got == {1: [1.0, 0.0, 1.0, 0.0], 2: [1.0] * 4}
