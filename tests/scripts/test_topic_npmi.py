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
from spark_vi.models.topic.dag_placement import DagLayout  # noqa: E402

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
