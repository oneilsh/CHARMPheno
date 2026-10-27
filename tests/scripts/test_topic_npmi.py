"""Exp 0159: gated NPMI (analysis/cloud/topic_npmi.py)."""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CLOUD = REPO_ROOT / "analysis" / "cloud"
for _p in (str(CLOUD), str(REPO_ROOT / "spark-vi")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.environ["PYTHONPATH"] = os.pathsep.join(
    p for p in (str(CLOUD), str(REPO_ROOT / "spark-vi"), os.environ.get("PYTHONPATH", ""))
    if p)

import topic_npmi as tn  # noqa: E402
from spark_vi.dag import DagLayout  # noqa: E402

PARENT = {1: 0, 2: 1, 3: 1}          # 1 -> {2, 3}


def test_doc_topic_hits_respects_the_gate():
    lay = DagLayout(PARENT, n_bg=1, tpn=1)
    tops = np.array([[0, 1], [2, 3], [4, 5], [6, 7]])        # bg, node1, node2, node3
    idx = tn.build_index(tops, range(4))
    gated, every, admitted = tn.doc_topic_hits([0, 2, 4, 6], [2], idx, lay, 1)
    # frontier {2}: closure {2, 1} -> topics 1 (node1), 2 (node2) + bg 0; not node3
    assert admitted == {0, 1, 2}
    assert set(gated) == {0, 1, 2} and 3 in every and 3 not in gated


def test_npmi_from_counts_perfect_pair():
    dfc = np.array([5, 5])
    pairs = np.array([[5, 5], [5, 5]])
    v, n = tn.npmi_from_counts(10, dfc, pairs)
    assert n == 1 and abs(v - 1.0) < 1e-9


@pytest.mark.slow
def test_gated_counts_and_floor(spark):
    from pyspark.ml.linalg import Vectors
    lay = DagLayout(PARENT, n_bg=1, tpn=1)
    tops = np.array([[0, 1], [2, 3], [4, 5], [6, 7]])
    rows = []
    for i in range(30):                      # 30 docs at node 2 with words 4, 5
        rows.append((Vectors.sparse(8, [4, 5], [1.0, 1.0]), [2]))
    for i in range(10):                      # 10 background docs with words 0, 1
        rows.append((Vectors.sparse(8, [0, 1], [1.0, 1.0]), []))
    df = spark.createDataFrame(rows, ["features_0", "frontier"])
    gated, every = tn.gated_npmi_counts(df, lay, tops, feat_col="features_0", n_bg=1)
    assert gated[2][0] == 30 and gated[3][0] == 0 and gated[0][0] == 40
    assert every[2][0] == 40
    assert gated[2][2][0, 1] == 30
    sc = tn.score(gated, every)
    assert abs(sc[2]["npmi_gated"] - 1.0) < 1e-9          # co-occur in every admitted doc
    assert np.isnan(sc[3]["npmi_gated"])                  # under the egress floor


def test_compare_pairs_by_mondo_id(tmp_path, capsys):
    for name, v in (("a", 0.1), ("b", 0.3)):
        d = tmp_path / name
        d.mkdir()
        (d / "npmi.json").write_text(json.dumps({"rows": [
            {"domain": 0, "topic": 1, "kind": "node", "node": 1, "cid": 99, "name": "x",
             "depth": 1, "n_ref": 50, "npmi_gated": v, "npmi_all": v,
             "pairs_gated": 3, "pairs_all": 3}]}))
    tn.compare(tmp_path / "a", tmp_path / "b")
    out = capsys.readouterr().out
    assert "median B−A +0.2000" in out and "B better on 1/1" in out


def test_own_code_exclusions_cover_descendants():
    lay = DagLayout(PARENT, n_bg=1, tpn=1)
    ex = tn.own_code_exclusions(lay, {1: [7], 2: [5]})
    assert ex[lay.block[1][0]] == [5, 7] and ex[lay.block[2][0]] == [5]
    tops = tn.top_words(np.array([[1, 1, 1, 1, 1, 9, 1, 1]] * 4, float), 2,
                        exclude={lay.block[2][0]: [5]})
    assert tops[0][0] == 5 and 5 not in tops[lay.block[2][0]]


@pytest.mark.slow
def test_local_mode_matches_the_spark_pass(spark, tmp_path):
    from pyspark.ml.linalg import Vectors
    lay = DagLayout(PARENT, n_bg=1, tpn=1)
    rows = [(Vectors.sparse(8, [4, 5], [1.0, 1.0]), [2]) for _ in range(30)]
    rows += [(Vectors.sparse(8, [0, 1], [1.0, 1.0]), []) for _ in range(10)]
    df = spark.createDataFrame(rows, ["features_0", "frontier"])
    df.write.parquet(str(tmp_path / "train.parquet"))
    lam = np.full((4, 8), 0.01)
    lam[0, [0, 1]] = 5; lam[1, [2, 3]] = 5; lam[2, [4, 5]] = 5; lam[3, [6, 7]] = 5
    run = tmp_path / "run"
    run.mkdir()
    np.savez(run / "gated_pc_result.npz", **{"lambda_0": lam})
    (run / "manifest.json").write_text(json.dumps({"n_bg": 1, "tpn": 1}))
    (run / "bundle_meta.json").write_text(json.dumps({
        "parent_int": {"1": [0], "2": [1], "3": [1]},
        "int2cid": {"0": 100, "1": 101, "2": 102, "3": 103},
        "name_by_id": {"102": "node two"},
        "vocab_maps": [{str(500 + i): i for i in range(8)}]}))
    (run / "code_map.tsv").write_text("std_cid\tnode_cid\n504\t102\n")
    tn.run_local(run, tmp_path / "train.parquet", top_n=2, tag="t")
    out = json.loads((run / "npmi_t.json").read_text())["rows"]
    two = next(r for r in out if r["node"] == 2)
    assert two["n_ref"] == 30 and abs(two["npmi_gated"] - 1.0) < 1e-9
    tn.run_local(run, tmp_path / "train.parquet", top_n=2, tag="x", exclude_own=True)
    two_x = next(r for r in json.loads((run / "npmi_x.json").read_text())["rows"]
                 if r["node"] == 2)
    assert two_x["pairs_gated"] == 0          # word 4 excluded: no co-occurring pair left
