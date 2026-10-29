"""Gated NPMI (spark_vi.eval.topic.gated_npmi, exp 0159): the gate, the counts, the
floor, the own-code exclusions."""
import numpy as np
import pytest

from spark_vi.dag import DagLayout
from spark_vi.eval.topic import gated_npmi as gn

PARENT = {1: 0, 2: 1, 3: 1}          # 1 -> {2, 3}


def test_doc_topic_hits_respects_the_gate():
    lay = DagLayout(PARENT, n_bg=1, tpn=1)
    tops = np.array([[0, 1], [2, 3], [4, 5], [6, 7]])        # bg, node1, node2, node3
    idx = gn.build_index(tops, range(4))
    gated, every, admitted = gn.doc_topic_hits([0, 2, 4, 6], [2], idx, lay, 1)
    # frontier {2}: closure {2, 1} -> topics 1 (node1), 2 (node2) + bg 0; not node3
    assert admitted == {0, 1, 2}
    assert set(gated) == {0, 1, 2} and 3 in every and 3 not in gated


def test_npmi_from_counts_perfect_pair():
    dfc = np.array([5, 5])
    pairs = np.array([[5, 5], [5, 5]])
    v, n = gn.npmi_from_counts(10, dfc, pairs)
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
    gated, every = gn.gated_npmi_counts(df, lay, tops, feat_col="features_0", n_bg=1)
    assert gated[2][0] == 30 and gated[3][0] == 0 and gated[0][0] == 40
    assert every[2][0] == 40
    assert gated[2][2][0, 1] == 30
    sc = gn.score(gated, every, floor=20)
    assert abs(sc[2]["npmi_gated"] - 1.0) < 1e-9          # co-occur in every admitted doc
    assert np.isnan(sc[3]["npmi_gated"])                  # under the floor


def test_own_code_exclusions_cover_descendants():
    lay = DagLayout(PARENT, n_bg=1, tpn=1)
    ex = gn.own_code_exclusions(lay, {1: [7], 2: [5]})
    assert ex[lay.block[1][0]] == [5, 7] and ex[lay.block[2][0]] == [5]
    tops = gn.top_words(np.array([[1, 1, 1, 1, 1, 9, 1, 1]] * 4, float), 2,
                        exclude={lay.block[2][0]: [5]})
    assert tops[0][0] == 5 and 5 not in tops[lay.block[2][0]]
