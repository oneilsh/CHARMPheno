"""DagGatedLDAEstimator / DagGatedLDAModel (ADR 0050): the one MLlib shim over the
DAG-gated LDA engine.

Ported from the two shims it replaces (the node-affinity GatedLDAEstimator and the
OnlinePCLDAEstimator at weightY = 0): fit/transform, a content-seeded (so
deterministic) transform, mini-batch SVI, the per-iteration callback, the
empirical-Bayes alpha, the leaves-first schedule wiring, multi-domain features,
and persistence (save/load, class-tag rejection, resume continues the counter).
"""
from __future__ import annotations

import json

import numpy as np
import pytest

pyspark = pytest.importorskip("pyspark")
from pyspark.ml.linalg import Vectors  # noqa: E402

PARENT = {1: 0, 2: 0, 3: 1, 4: 1, 5: 2, 6: 2}


def _df(spark, n=40, V=30, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        leaf = int(rng.choice([3, 4, 5, 6]))
        idx = sorted(rng.choice(V, size=6, replace=False).tolist())
        rows.append((Vectors.sparse(V, idx, [1.0] * len(idx)), [leaf]))
    return spark.createDataFrame(rows, ["features", "frontier"])


def _est(**kw):
    from spark_vi.mllib.topic.dag_gated_lda import DagGatedLDAEstimator
    kw.setdefault("maxIter", 3)
    kw.setdefault("seed", 0)
    kw.setdefault("subsamplingRate", 1.0)
    est = DagGatedLDAEstimator(gateNBg=2, gateTpn=1, **kw)
    return est.setGateParent(PARENT)


def test_defaults_carry_the_live_params_only():
    from spark_vi.mllib.topic.dag_gated_lda import DagGatedLDAEstimator
    est = DagGatedLDAEstimator()
    assert est.getOrDefault("gateSeedMode") == "balanced"      # ADR 0050 default
    assert est.getOrDefault("gateWarm") == 0
    assert est.getOrDefault("gateOpenSeed") is True
    assert est.getOrDefault("optimizeDocConcentration") is False
    names = {p.name for p in est.params}
    for gone in ("weightY", "numLabels", "labelCol", "init", "spectralMethod",
                 "anchorScope", "etaBoost", "countTransform", "alphaInit", "omega",
                 "etaPerDomain", "nodeAffinityCol", "warmStartFrom", "localizeHead"):
        assert gone not in names, gone


def test_fit_transform_smoke(spark):
    df = _df(spark)
    model = _est().fit(df)
    out = model.transform(df)
    assert "topicDistribution" in out.columns
    theta = np.asarray(out.select("topicDistribution").head()[0].toArray())
    assert theta.shape == (2 + 6,)                 # K = n_bg + nodes * tpn
    np.testing.assert_allclose(theta.sum(), 1.0, atol=1e-6)
    assert model.result.metadata["model_class"] == "DagGatedOnlineLDA"


def test_requires_the_gate_and_the_frontier_column(spark):
    from spark_vi.mllib.topic.dag_gated_lda import DagGatedLDAEstimator
    df = _df(spark, n=6)
    with pytest.raises(ValueError, match="gateParent"):
        DagGatedLDAEstimator(maxIter=1).fit(df)
    with pytest.raises(ValueError, match="frontierCol"):
        _est(frontierCol="nope", maxIter=1).fit(df)


def test_transform_is_deterministic(spark):
    # Content-seeded gamma_init: two transforms of the same model + data agree.
    df = _df(spark)
    model = _est().fit(df)
    a1 = [r[0] for r in model.transform(df).select("topicDistribution").collect()]
    a2 = [r[0] for r in model.transform(df).select("topicDistribution").collect()]
    assert len(a1) == len(a2) == 40
    for v1, v2 in zip(a1, a2):
        assert np.array_equal(v1.toArray(), v2.toArray())


def test_minibatch_fit_smoke(spark):
    model = _est(subsamplingRate=0.5, maxIter=4).fit(_df(spark))
    assert model.result.n_iterations == 4


def test_on_iteration_callback_fires(spark):
    seen = []
    est = _est(maxIter=3)
    est.setOnIteration(lambda it, gp, elbo: seen.append(it))
    est.fit(_df(spark))
    assert len(seen) == 3


def test_doc_concentration_scalar_and_vector(spark):
    df = _df(spark)
    m = _est(docConcentration=[0.5], maxIter=1).fit(df)
    np.testing.assert_allclose(m.trainedAlpha(), 0.5)
    with pytest.raises(ValueError, match="length K"):
        _est(docConcentration=[0.1, 0.2], maxIter=1).fit(df)


def test_optimize_alpha_learns_asymmetric(spark):
    # node 1 common, node 2 rare: the empirical-Bayes alpha learns a(1) > a(2).
    from spark_vi.mllib.topic.dag_gated_lda import DagGatedLDAEstimator
    from spark_vi.dag import DagLayout
    parent = {1: 0, 2: 0}
    V = 24
    rng = np.random.default_rng(0)
    rows = []
    for _ in range(120):
        leaf = 1 if rng.random() < 0.75 else 2
        idx = sorted(rng.choice(V, size=5, replace=False).tolist())
        rows.append((Vectors.sparse(V, idx, [1.0] * len(idx)), [leaf]))
    for _ in range(60):
        idx = sorted(rng.choice(V, size=5, replace=False).tolist())
        rows.append((Vectors.sparse(V, idx, [1.0] * len(idx)), []))
    df = spark.createDataFrame(rows, ["features", "frontier"])
    est = DagGatedLDAEstimator(gateNBg=2, gateTpn=1, maxIter=8, seed=0,
                               subsamplingRate=1.0, learningOffset=1.0,
                               learningDecay=0.7, optimizeDocConcentration=True)
    model = est.setGateParent(parent).fit(df)
    lay = DagLayout(parent, n_bg=2, tpn=1)
    alpha = model.result.global_params["alpha"]
    assert alpha[lay.block[1][0]] > alpha[lay.block[2][0]]


def test_leaves_first_schedule_with_balanced_and_leaf_seeds(spark):
    df = _df(spark, n=60)
    est = _est(gateWarm=1, gateSeedLeaves=True, maxIter=4)
    est.setGateSeedOwnCodes({1: [0], 2: [1]})
    model = est.fit(df)
    gp = model.result.global_params
    assert "gate_iter" in gp and int(gp["gate_iter"]) == 4
    assert (np.asarray(gp["gate_open_at"]) >= 0).all()     # every block opened


def test_profile_seed_round_trips_and_fits(spark):
    est = _est(gateWarm=1, maxIter=3)
    est.setGateSeedProfile({3: {0: 2.0, 1: 1.0}, 1: {5: 1.0}})
    assert est.getGateSeedProfile() == {3: {0: 2.0, 1: 1.0}, 1: {5: 1.0}}
    est.fit(_df(spark))


def test_multidomain_fit_and_transform(spark):
    V0, V1 = 10, 6
    rng = np.random.default_rng(1)
    rows = []
    for _ in range(30):
        leaf = int(rng.choice([3, 4, 5, 6]))
        i0 = sorted(rng.choice(V0, size=3, replace=False).tolist())
        i1 = sorted(rng.choice(V1, size=2, replace=False).tolist())
        rows.append((Vectors.sparse(V0, i0, [1.0] * 3),
                     Vectors.sparse(V1, i1, [2.0] * 2), [leaf]))
    df = spark.createDataFrame(rows, ["features_0", "features_1", "frontier"])
    model = _est(featuresCols=["features_0", "features_1"]).fit(df)
    lam = model.result.global_params["lambda"]
    assert isinstance(lam, dict) and lam[0].shape[1] == V0 and lam[1].shape[1] == V1
    assert model.result.metadata["domains"] == [V0, V1]
    assert model.vocabSize() == V0 + V1
    out = model.transform(df).select("topicDistribution").head()[0]
    np.testing.assert_allclose(out.toArray().sum(), 1.0, atol=1e-6)
    with pytest.raises(ValueError, match="domainBounds"):
        _est(featuresCols=["features_0", "features_1"],
             domainBounds=[0, V0]).fit(df)


def test_transform_matches_the_engine_cavi(spark):
    """The transform is the engine's ungated full-K CAVI under a content seed."""
    import hashlib
    from spark_vi.mllib.topic._common import _vector_to_bow_document
    from spark_vi.models.topic.dag_gated_lda import expElogbeta_from_lambda
    from spark_vi.models.topic.lda import _cavi_doc_inference
    df = _df(spark, n=8)
    model = _est().fit(df)
    row = df.head()
    got = model.transform(df.limit(1)).head()["topicDistribution"].toArray()
    doc = _vector_to_bow_document(row["features"])
    h = hashlib.blake2b(digest_size=8)
    h.update(np.ascontiguousarray(doc.indices, dtype=np.int32).tobytes())
    h.update(np.ascontiguousarray(doc.counts, dtype=np.float64).tobytes())
    rng = np.random.default_rng(int.from_bytes(h.digest(), "little"))
    gp = model.result.global_params
    gamma, _, _, _ = _cavi_doc_inference(
        indices=doc.indices, counts=doc.counts,
        expElogbeta=expElogbeta_from_lambda(gp["lambda"]), alpha=gp["alpha"],
        gamma_init=rng.gamma(100.0, 1.0 / 100.0, size=len(gp["alpha"])),
        max_iter=100, tol=1e-3)
    np.testing.assert_allclose(got, gamma / gamma.sum(), rtol=1e-12)


# --------------------------------------------------------------------------- #
# Persistence: save/load, class-tag rejection, resume continues the counter.   #
# --------------------------------------------------------------------------- #

def test_model_save_load_round_trip_and_transform(tmp_path, spark):
    from spark_vi.mllib.topic.dag_gated_lda import DagGatedLDAModel
    df = _df(spark)
    model = _est().fit(df)
    model.save(str(tmp_path / "m"))
    loaded = DagGatedLDAModel.load(str(tmp_path / "m"))
    for name, arr in model.result.global_params.items():
        np.testing.assert_array_equal(loaded.result.global_params[name], arr)
    assert "topicDistribution" in loaded.transform(df).columns


def test_model_load_rejects_another_model_class(tmp_path):
    from spark_vi.mllib.topic.dag_gated_lda import DagGatedLDAModel
    d = tmp_path / "lda"
    (d / "params").mkdir(parents=True)
    np.save(d / "params" / "lambda.npy", np.zeros((2, 3)))
    (d / "manifest.json").write_text(json.dumps({
        "format_version": 1, "elbo_trace": [], "n_iterations": 0,
        "converged": False, "metadata": {"model_class": "OnlineLDA", "K": 2, "V": 3},
        "param_names": ["lambda"], "diagnostic_traces": {}}))
    with pytest.raises(ValueError, match="DagGatedOnlineLDA"):
        DagGatedLDAModel.load(str(d))


def test_resume_continues_the_iteration_counter(tmp_path, spark):
    df = _df(spark)
    save_dir = tmp_path / "run1"
    a = _est(maxIter=3, seed=2)
    a.setSaveDir(str(save_dir))
    ra = a.fit(df).result
    lam_a = np.array(ra.global_params["lambda"], copy=True)
    b = _est(maxIter=3, seed=2)
    b.setSaveDir(str(save_dir))
    b.setResumeFrom(str(save_dir))
    rb = b.fit(df).result
    assert rb.n_iterations == 6 and len(rb.elbo_trace) == 6
    assert rb.elbo_trace[:3] == ra.elbo_trace
    assert not np.allclose(rb.global_params["lambda"], lam_a)


def test_profile_seed_lands_in_the_primary_domain_slice():
    """gateSeedDomain names the domain the profile / own codes index; its local ids
    are shifted to that domain's span of the concatenated vocabulary."""
    from spark_vi.dag import DagLayout
    lay = DagLayout(PARENT, n_bg=2, tpn=1)
    sizes = [4, 3]
    for d, (lo, hi) in ((0, (0, 4)), (1, (4, 7))):
        est = _est(gateWarm=1, gateOpenSeed=False, gateSeedDomain=d)
        est.setGateSeedProfile({3: {0: 2.0, 2: 1.0}, 1: {1: 1.0}})
        assert est._primary_slice(7, sizes) == (lo, hi)
        seed, init, _ = est._gate_seeds(None, lay, 7, sizes)
        k3, k1 = lay.block[3][0], lay.block[1][0]            # 3 is a leaf, 1 is not
        np.testing.assert_array_equal(init[k3, lo:lo + 3], [2.0, 0.0, 1.0])
        assert init[k3].sum() == 3.0                          # nothing outside the slice
        assert seed[k1, lo + 1] == 1.0 and seed[k1].sum() == 1.0
    with pytest.raises(ValueError, match="gateSeedDomain"):
        _est(gateSeedDomain=2)._primary_slice(7, sizes)
