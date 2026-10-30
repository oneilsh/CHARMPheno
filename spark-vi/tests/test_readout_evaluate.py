"""`spark_vi.readout.evaluate`: the readout's metrics from score blocks (ADR 0051 step
4). In memory against per-column metrics on the dense arrays; one slow test checks
the Spark score blocks give the same readout as the local backend's."""
import numpy as np
import pytest

from spark_vi.eval.metrics import core
from spark_vi.readout import evaluate as ev

C, D = 6, 400
RT, FQ = (0.5, 0.9), (0.1, 0.5)


def _arrays(seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random((D, C)) < 0.3).astype(np.uint8)
    y[:, 0] = (rng.random(D) < 0.5).astype(np.uint8)
    p = np.clip(rng.random((D, C)) * 0.6 + 0.4 * y, 0, 1).astype(np.float32)
    p[:, 5] = 0.25                                            # a constant column
    m = (rng.random((D, C)) < 0.8).astype(np.uint8)
    return p, y, m


def _blocks(p, y, m, ids=None, parts=3):
    ids = np.arange(D, dtype=np.int64) if ids is None else ids
    return ev.Blocks([{"ids": ids[r], "proba": p[r], "y": y[r], "mask": m[r]}
                      for r in np.array_split(np.arange(D), parts)])


def test_readout_metrics_equal_per_column_metrics_on_the_whole_arrays():
    p, y, m = _arrays()
    out = ev.readout_metrics(_blocks(p, y, m), C, recall_targets=RT, fdr_targets=FQ)
    for c, r in out["per_node"].items():
        rows = m[:, c].astype(bool)
        want = core.node_metrics(core.tally(p[rows, c], y[rows, c]))
        assert r["auc"] == pytest.approx(want["auc"], abs=1e-12)
        assert r["n_pos"] == want["n_pos"]
    assert out["ranking"]["n_labels_scored"] + out["ranking"]["n_labels_skipped"] == C
    # detection: max over the INFORMATIVE disease columns (5 is constant)
    det = out["detection"]
    assert det["n_constant_nodes"] == 1 and det["grain"] == "document"
    want = core.node_metrics(core.tally(p[:, 1:5].max(axis=1), y[:, 0]))
    assert det["auc"] == pytest.approx(want["auc"], abs=1e-12)


def test_detection_pools_rows_onto_units_by_max():
    p, y, m = _arrays(1)
    ids = np.arange(D, dtype=np.int64)                        # units of 4 rows
    det = ev.detection(_blocks(p, y, m, ids), RT, unit_divisor=4)
    s = p[:, 1:5].max(axis=1).reshape(-1, 4).max(axis=1)
    yy = y[:, 0].reshape(-1, 4).max(axis=1)
    want = core.node_metrics(core.tally(s, yy))
    assert det["grain"] == "unit" and det["n_units"] == D // 4
    assert det["auc"] == pytest.approx(want["auc"], abs=1e-12)


def test_stacked_and_calibration_groups():
    from spark_vi.readout.stacked import stacked_proba

    p, y, m = _arrays(2)
    M = np.eye(C)
    M[1:, 0] = 1.0                                            # every node under the root
    sb = ev.with_stacked(_blocks(p, y, m), M)
    got = ev.column_metrics(sb, score="stacked", mask=None)
    P = stacked_proba(p, M)
    for c in range(C):
        assert got[c]["n_pos"] == int(y[:, c].sum())
        if got[c]["auc"] is not None:
            want = core.node_metrics(core.tally(P[:, c], y[:, c]))["auc"]
            assert got[c]["auc"] == pytest.approx(want, abs=1e-12)
    ece = ev.calibration_by_group(sb, {"a": [1, 2]}, scores=("proba", "stacked"))
    want = core.ece_from_bins(core.calibration_bins(p[:, [1, 2]].ravel(),
                                                    y[:, [1, 2]].ravel()))
    assert ece[("proba", "a")] == pytest.approx(want, abs=1e-12)


@pytest.mark.slow
def test_spark_score_blocks_give_the_local_readout(spark):
    from pyspark.ml.linalg import Vectors
    from spark_vi.readout import LocalReadoutData, SparkReadoutData

    rng = np.random.default_rng(3)
    K = 5
    theta = rng.dirichlet(np.ones(K), size=D)
    y = (rng.random((D, C)) < 0.3).astype(np.uint8)
    m = (rng.random((D, C)) < 0.8).astype(np.uint8)
    rows = [(int(d) * 64, Vectors.dense(theta[d]), [float(v) for v in y[d]],
             [float(v) for v in m[d]]) for d in range(D)]
    from pyspark.ml.linalg import VectorUDT
    from pyspark.sql.types import (ArrayType, DoubleType, LongType, StructField,
                                   StructType)
    schema = StructType([StructField("doc_key", LongType()),
                         StructField("topicDistribution", VectorUDT()),
                         StructField("label", ArrayType(DoubleType())),
                         StructField("labelMask", ArrayType(DoubleType()))])
    df = spark.createDataFrame(rows, schema).repartition(4)
    V, b = rng.normal(size=(C, K)), rng.normal(size=C)
    deg = np.zeros(C, dtype=bool)
    deg[2] = True
    const = np.full(C, 0.7)
    sb = SparkReadoutData(df, C, K).score_blocks(V, b, degenerate=deg, const=const)
    lb = LocalReadoutData.from_spark(df, C, K).score_blocks(V, b, degenerate=deg,
                                                            const=const)
    try:
        kw = dict(recall_targets=RT, fdr_targets=FQ, unit_divisor=64)
        a, z = ev.readout_metrics(sb, C, **kw), ev.readout_metrics(lb, C, **kw)
    finally:
        sb.unpersist()
    assert set(a["per_node"]) == set(z["per_node"])
    for c in a["per_node"]:
        assert a["per_node"][c]["auc"] == pytest.approx(z["per_node"][c]["auc"], abs=1e-9)
    assert a["detection"]["auc"] == pytest.approx(z["detection"]["auc"], abs=1e-9)
    assert a["detection"]["n_units"] == D
    assert a["ranking"]["skipped_by_reason"] == z["ranking"]["skipped_by_reason"]
    assert a["ranking"]["auc"] == pytest.approx(z["ranking"]["auc"], abs=1e-9)
