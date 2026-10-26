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




def test_sidecar_nodes_roll_up_the_mondo_is_a_to_readout_nodes():
    parent_adj = {"MONDO:0000005": ["MONDO:0000001"], "MONDO:0000001": ["MONDO:0000009"]}
    int2cid = {0: -1, 1: 1, 2: 2, 3: 9}
    got = pv.sidecar_nodes_to_readout([5, 2, 77], int2cid, parent_adj)
    assert got == {5: [1, 3], 2: [2]}                   # 77 hits no readout node


def test_prior_from_sets_and_containment():
    M = pv.prior_from_sets([64, 128, 192], {1: [2], 3: [1, 2]}, 3, lambda k: k // 64)
    assert M.tolist() == [[False, False, True], [False] * 3, [False, True, True]]
    a = np.array([[True, False], [True, True]])
    assert pv.containment(a, a) == 1.0 and pv.containment(a, ~a) == 0.0


def test_run_prevalence_check_whole_history(spark, tmp_path):
    import datetime as dt
    from pyspark.ml.linalg import Vectors
    from pyspark.sql import Row
    from test_inspect_topics import _make_census_run
    import gated_pc_readout as gpr
    _make_census_run(tmp_path)
    C, K = 4, 10
    (tmp_path / "readout_dag.json").write_text(json.dumps({
        "version": 1, "C_fit": 4, "C": 4,
        "parent_int": {"1": [0], "2": [0], "3": [0]},
        "int2cid": {"0": -1, "1": 1001, "2": 1002, "3": 1003},
        "name_by_id": {str(1000 + e): f"node{e}" for e in range(4)},
        "pos_sources": [[0], [1], [2], [3]]}))
    (tmp_path / "bundle_meta.json").write_text(json.dumps({
        "int2cid": {"0": -1, "1": 1001, "2": 1002, "3": 1003},
        "vocab_maps": [{str(500 + i): i for i in range(12)}],
        "name_by_id": {str(1000 + e): f"node{e}" for e in range(4)}}))
    (tmp_path / "code_map.tsv").write_text("std_cid\tnode_cid\n503\t1001\n")
    Vh = np.zeros((C, K)); Vh[1, 1] = 12.0
    np.savez(tmp_path / "readout_heads_gated_pc_t.npz", V=Vh,
             b_raw=np.array([5.0, -4.0, -4.0, -4.0]), const=np.zeros(C),
             degenerate=np.zeros(C, bool), C=np.int64(C), K=np.int64(K),
             theta_topm=np.int64(0))
    rng = np.random.default_rng(1)
    idx = dt.date(2020, 1, 1)
    rows, first, hz = [], [], []
    for i in range(500):
        kind = "prev" if i < 80 else "old" if i < 110 else "inc" if i < 170 else "neg"
        th = rng.dirichlet(np.ones(K))
        codes = set(rng.choice(np.arange(6, 12), 3, replace=False).tolist())
        if kind == "prev":
            th[1] += 3.0
            codes.add(3)
            first.append(Row(person_id=i, node_cid=1005,
                             first_attested_date=dt.date(2018, 5, 1)))
        if kind == "old":           # diagnosed 8 years back: outside the lookback
            th[1] += 2.0
            first.append(Row(person_id=i, node_cid=1001,
                             first_attested_date=dt.date(2012, 5, 1)))
        if kind == "inc":
            th[1] += 0.4
            first.append(Row(person_id=i, node_cid=1001,       # AFTER the index
                             first_attested_date=dt.date(2020, 6, 1)))
        th /= th.sum()
        lab = [1.0, float(kind in ("inc", "old") or (kind == "prev" and i < 60)), 0.0, 0.0]
        rows.append(Row(person_id=i, topicDistribution=Vectors.dense(th), label=lab,
                        labelMask=[1.0] * C,
                        features_0=Vectors.sparse(12, sorted(codes), [1.0] * len(codes))))
        hz.append(Row(person_id=i, index_date=idx,
                      observation_period_end_date=dt.date(2023, 1, 1)))
    uri = tmp_path / "sidecar"
    spark.createDataFrame(first).write.parquet(str(uri / "k1" / "first_attestation.parquet"))
    spark.createDataFrame(hz).write.parquet(str(uri / "k1" / "index_horizon.parquet"))
    (tmp_path / "conversion_sidecar.json").write_text(json.dumps(
        {"key": "k1", "sidecar_uri": str(uri)}))
    df = spark.createDataFrame(rows)
    padj = {"MONDO:0001005": ["MONDO:0001001"]}
    res = gpr.run_prevalence_check(
        df, tmp_path, C=C, parent_int={0: [], 1: [0], 2: [0], 3: [0]}, heads_tag="t",
        tag="h", history=dict(readout_int2cid={0: -1, 1: 1001, 2: 1002, 3: 1003},
                              parent_adj=padj))
    assert res["containment_lookback_in_history"] == 1.0
    assert res["history_receipt"]["n_with_index"] == 500
    lb, hi = res["stacked"]["per_node"][1], res["history_stacked"]["per_node"][1]
    assert lb["n_pos_incident"] == 90          # 30 old + 60 incident; old look new
    assert hi["n_pos_incident"] == 60          # old ones now set aside, post-index ignored
    assert "WHOLE-HISTORY prior" in (tmp_path / "prevalence_check_h.md").read_text()


def test_ever_member_frame_and_collect_with_ever(spark, tmp_path):
    import datetime as dt
    from pyspark.ml.linalg import Vectors
    from pyspark.sql import Row
    import gated_pc_readout as gpr
    uri = tmp_path / "sc"
    spark.createDataFrame([
        Row(person_id=1, node_cid=1005, first_attested_date=dt.date(2010, 1, 1)),
        Row(person_id=2, node_cid=1002, first_attested_date=dt.date(2030, 1, 1)),
        Row(person_id=3, node_cid=7777, first_attested_date=dt.date(2010, 1, 1)),
    ]).write.parquet(str(uri / "k" / "first_attestation.parquet"))
    (tmp_path / "conversion_sidecar.json").write_text(json.dumps(
        {"key": "k", "sidecar_uri": str(uri)}))
    padj = {"MONDO:0001005": ["MONDO:0001001"]}
    ever = gpr.ever_member_frame(spark, tmp_path, [1, 2], [1001, 1002], padj)
    got = {r["person_id"]: sorted(r["__ever"]) for r in ever.collect()}
    assert got == {1: [0], 2: [1]}                       # 1005 rolls up to member 0
    df_ = spark.createDataFrame([
        Row(person_id=i, label=[1.0, float(i == 4), 0.0],
            features_0=Vectors.sparse(5, [0], [1.0])) for i in (1, 2, 3, 4)])
    out, n = gpr.collect_set_patients(df_, [1, 2], 1, ever_df=ever)
    assert sorted(t for t, _, _ in out) == [[0], [0], [1]] and n == 3


def test_incident_test_frame_drops_prior_diagnosed_persons(spark, tmp_path):
    import datetime as dt
    from pyspark.sql import Row
    import gated_pc_readout as gpr
    uri = tmp_path / "sc"
    spark.createDataFrame([
        Row(person_id=1, node_cid=1005, first_attested_date=dt.date(2015, 1, 1)),  # prior, under member
        Row(person_id=2, node_cid=1001, first_attested_date=dt.date(2021, 1, 1)),  # after index
        Row(person_id=3, node_cid=1009, first_attested_date=dt.date(2015, 1, 1)),  # prior, group node
        Row(person_id=4, node_cid=7777, first_attested_date=dt.date(2015, 1, 1)),  # unrelated
    ]).write.parquet(str(uri / "k" / "first_attestation.parquet"))
    spark.createDataFrame([Row(person_id=i, index_date=dt.date(2020, 1, 1),
                               observation_period_end_date=dt.date(2023, 1, 1))
                           for i in (1, 2, 3, 4)]).write.parquet(
        str(uri / "k" / "index_horizon.parquet"))
    (tmp_path / "conversion_sidecar.json").write_text(json.dumps(
        {"key": "k", "sidecar_uri": str(uri)}))
    test = spark.createDataFrame([Row(person_id=i, x=1) for i in (1, 2, 3, 4)])
    pp = gpr._test_prior_pairs(test, tmp_path)
    nodes = [int(r[0]) for r in pp.select("node_cid").distinct().collect()]
    padj = {"MONDO:0001005": ["MONDO:0001001"], "MONDO:0001001": ["MONDO:0001009"]}
    out = gpr._incident_test_frame(test, pp, nodes, [1001, 1002, 1009], padj)
    assert sorted(r.person_id for r in out.collect()) == [2, 4]


def test_readout_node_keep_takes_ancestors_and_root():
    import gated_pc_readout as gpr
    parent = {1: [0], 2: [1], 3: [0], 4: [3]}
    int2cid = {0: -1, 1: 100, 2: 200, 3: 300, 4: 400}
    assert gpr.readout_node_keep(parent, int2cid, [200], 5) == [True, True, True, False, False]


def test_mask_to_nodes(spark):
    from pyspark.sql import Row
    import gated_pc_readout as gpr
    df = spark.createDataFrame([Row(person_id=1, labelMask=[1.0, 1.0, 0.0, 1.0])])
    got = gpr.mask_to_nodes(df, [True, False, True, True]).collect()[0].labelMask
    assert got == [1.0, 0.0, 0.0, 1.0]
