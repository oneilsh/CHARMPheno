"""Spectral (anchor-word) lambda init wired into the Gated-PC path.

The Gated-PC estimator historically always constructed its gated topic engine
with init="random" (a flat random-Gamma lambda) — the deep-node flat-start
starvation trap of insight 0079. These tests pin the wiring that lets a
`docs/experiments/*.md` set `init: spectral` and have the block-aligned
anchor-word seed (spark_vi.models.topic.gated_init, Arora et al. 2013) actually
reach the fit through `data_summary`, exactly as the node-affinity estimator
already does. The seed machinery itself is tested in test_gated_init.py /
test_gated_lda_shim.py; here we test only that the PC path threads it.
"""
import json

import numpy as np
import pytest

pyspark = pytest.importorskip("pyspark")


# --------------------------------------------------------------------------- #
# Spark-free: param plumbing + init threading into the gated engine            #
# --------------------------------------------------------------------------- #
def test_pc_estimator_accepts_spectral_params_and_threads_init():
    from spark_vi.mllib.topic.pc import (
        OnlinePCLDAEstimator, _build_model_and_config,
    )
    parent = {1: 0, 2: 0, 3: 1, 4: 1}
    est = OnlinePCLDAEstimator(
        numLabels=5, weightY=0.0,
        gateParent=json.dumps({str(c): p for c, p in parent.items()}),
        gateNBg=2, gateTpn=1,
        init="spectral", spectralMethod="scalable", spectralMaxVocab=4096,
        spectralD=64, spectralMinDocFreq=3, anchorScope="frontier",
        spectralTopoOrder="reverse")
    # the seven params round-trip through the estimator
    assert est.getOrDefault("init") == "spectral"
    assert est.getOrDefault("spectralMethod") == "scalable"
    assert est.getOrDefault("spectralMaxVocab") == 4096
    assert est.getOrDefault("spectralD") == 64
    assert est.getOrDefault("spectralMinDocFreq") == 3
    assert est.getOrDefault("anchorScope") == "frontier"
    assert est.getOrDefault("spectralTopoOrder") == "reverse"
    # and init reaches the gated engine (was hardcoded "random" before this wiring)
    model, _ = _build_model_and_config(est, vocab_size=30, domains=None)
    assert model._lda.init == "spectral"


def test_pc_estimator_default_init_is_random():
    from spark_vi.mllib.topic.pc import (
        OnlinePCLDAEstimator, _build_model_and_config,
    )
    est = OnlinePCLDAEstimator(
        numLabels=3, weightY=0.0,
        gateParent=json.dumps({"1": 0, "2": 0}), gateNBg=2, gateTpn=1)
    assert est.getOrDefault("init") == "random"
    model, _ = _build_model_and_config(est, vocab_size=20, domains=None)
    assert model._lda.init == "random"


# --------------------------------------------------------------------------- #
# Spark: the seed actually flows through data_summary into the fit             #
# --------------------------------------------------------------------------- #
def _two_node_df(spark, V=6):
    from pyspark.ml.linalg import SparseVector
    # node 1 attests tokens {0,1}; node 2 attests {2,3}. weightY=0 (unsupervised
    # topics), so the label is inert but present (mirrors the driver's columns).
    rows = []
    for _ in range(24):
        rows.append((SparseVector(V, [0, 1], [3.0, 2.0]), [0.0, 0.0, 0.0], [1]))
        rows.append((SparseVector(V, [2, 3], [3.0, 2.0]), [0.0, 0.0, 0.0], [2]))
    return spark.createDataFrame(rows, ["features", "label", "frontier"])


def _pc_est(**kw):
    from spark_vi.mllib.topic.pc import OnlinePCLDAEstimator
    est = OnlinePCLDAEstimator(
        featuresCol="features", frontierCol="frontier", labelCol="label",
        numLabels=3, weightY=0.0, maxIter=2, seed=0, subsamplingRate=1.0, **kw)
    return est.setGateParent({1: 0, 2: 0})._set(gateNBg=2, gateTpn=1)


def test_pc_gated_spectral_seeds_lambda_dense(spark):
    """init='spectral' (dense route, V below the threshold) builds the seed from
    the collected corpus and fits; the resulting lambda differs from a random-init
    fit on the same corpus/seed — the spectral seed changed the trajectory."""
    df = _two_node_df(spark)
    m_rand = _pc_est(init="random").fit(df)
    m_spec = _pc_est(init="spectral", spectralMethod="dense").fit(df)
    lam_r = m_rand._result.global_params["lambda"]
    lam_s = m_spec._result.global_params["lambda"]
    assert lam_r.shape == lam_s.shape == (4, 6)   # K = nBg(2) + 2 nodes
    assert not np.allclose(lam_r, lam_s)


def test_pc_gated_spectral_auto_routes_scalable(spark):
    """spectralMethod='auto' with a tiny spectralMaxVocab routes the (concatenated)
    V >= threshold corpus to the distributed scalable seed (ADR 0032) and fits with
    no NotImplementedError — the gated PC RDD (GatedPCDocument) is duck-compatible
    with scalable_block_aligned_lambda's .indices/.counts/.frontier reads."""
    df = _two_node_df(spark)
    m = _pc_est(init="spectral", spectralMethod="auto", spectralMaxVocab=4,
                spectralMinDocFreq=1).fit(df)                 # V=6 >= 4 -> scalable
    assert m._result.global_params["lambda"].shape == (4, 6)


def test_pc_spectral_without_gate_raises(spark):
    """Spectral init needs the gate (there are no per-node blocks to seed without
    it); an ungated init='spectral' fit fails fast with a clear message."""
    from spark_vi.mllib.topic.pc import OnlinePCLDAEstimator
    from pyspark.ml.linalg import SparseVector
    df = spark.createDataFrame(
        [(SparseVector(6, [0, 1], [1.0, 1.0]), [0.0]) for _ in range(8)],
        ["features", "label"])
    est = OnlinePCLDAEstimator(featuresCol="features", labelCol="label",
                               numLabels=1, weightY=0.0, k=3, maxIter=2, seed=0,
                               init="spectral")               # no gateParent
    with pytest.raises(ValueError, match="requires gateParent"):
        est.fit(df)
