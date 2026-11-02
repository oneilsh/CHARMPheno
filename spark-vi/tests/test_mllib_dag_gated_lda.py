"""DagGatedLDAEstimator / DagGatedLDAModel (ADR 0050; user API, spec 2026-10-30
R2-R5, R7, R9): the one MLlib shim over the DAG-gated LDA engine.

The user gives a DAG and a labels column in their own ids; the node set (prune,
land, reduce, splice) is learned at fit and the model answers in user ids. Also:
fit/transform, a content-seeded (so deterministic) transform, mini-batch SVI, the
per-iteration callback, the empirical-Bayes alpha, the leaves-first schedule and the
seed extras in user / term ids, multi-domain features with vocabularies, and
persistence (save/load keeps the node set, resume; MLlib persistence in
test_mllib_dag_persistence.py).
"""
from __future__ import annotations

import json

import numpy as np
import pytest

pyspark = pytest.importorskip("pyspark")
from pyspark.ml.linalg import Vectors  # noqa: E402

# A plain int DAG: two top-level nodes, two leaves under each.
PARENT = {1: None, 2: None, 3: 1, 4: 1, 5: 2, 6: 2}

# A mixed str / int DAG with a multi-parent node (2), a chain to splice (1 -> "c":
# 1 has one kept child and no documents of its own), and a rare node to prune.
MIXED = {"A": None, "B": None, 1: "A", "c": 1, 2: ["A", "B"], 3: 2, "rare": 3}
V0_TERMS = [f"t{i}" for i in range(8)]
V1_TERMS = [100, 101, 102, 103, "t0"]          # "t0" is in both vocabularies


def _df(spark, n=40, V=30, seed=0, col="labels"):
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        leaf = int(rng.choice([3, 4, 5, 6]))
        idx = sorted(rng.choice(V, size=6, replace=False).tolist())
        rows.append((Vectors.sparse(V, idx, [1.0] * len(idx)), [leaf]))
    return spark.createDataFrame(rows, ["features", col])


def _est(**kw):
    from spark_vi.mllib.topic.dag_gated_lda import DagGatedLDAEstimator
    kw.setdefault("maxIter", 3)
    kw.setdefault("seed", 0)
    kw.setdefault("subsamplingRate", 1.0)
    kw.setdefault("optimizeDocConcentration", False)   # fixed alpha unless asked
    kw.setdefault("dag", PARENT)
    return DagGatedLDAEstimator(gateNBg=2, gateTpn=1, **kw)


def _mixed_rows(seed=0, n=48, specific_only=False):
    """Rows for the MIXED DAG over two domains. Labels carry ancestors and
    duplicates unless `specific_only` (then only the most-specific labels: the same
    gate)."""
    rng = np.random.default_rng(seed)
    full = {  # label list as written -> its most-specific labels
        "c": (["c", 1, "A", "c"], ["c"]),
        "3": ([3, 2, "B", 3], [3]),
        "2": ([2], [2]),
        "B": (["B", "B"], ["B"]),
        "A+3": (["A", 3, "c"], [3, "c"]),
    }
    keys = list(full)
    rows = []
    for i in range(n):
        if i % 8 == 7:
            lab = [] if i % 16 == 7 else None                 # background docs
        else:
            k = keys[i % len(keys)]
            lab = full[k][1] if specific_only else full[k][0]
        i0 = sorted(rng.choice(8, size=3, replace=False).tolist())
        i1 = sorted(rng.choice(5, size=2, replace=False).tolist())
        rows.append((Vectors.sparse(8, i0, [1.0] * 3), Vectors.sparse(5, i1, [2.0] * 2),
                     lab))
    rare = [3, "rare"] if not specific_only else ["rare"]
    rows.append((Vectors.sparse(8, [0], [1.0]), Vectors.sparse(5, [1], [1.0]), rare))
    return rows


def _str_dag_df(spark, specific_only=False, seed=0):
    """The MIXED DAG with every id written as a str (Spark array columns are
    homogeneous), and its labels as array<string>."""
    from pyspark.sql.types import ArrayType, StringType, StructField, StructType
    from pyspark.ml.linalg import VectorUDT
    schema = StructType([StructField("f0", VectorUDT()), StructField("f1", VectorUDT()),
                         StructField("labels", ArrayType(StringType()))])
    rows = [(a, b, None if lab is None else [str(x) for x in lab])
            for a, b, lab in _mixed_rows(seed=seed, specific_only=specific_only)]
    return spark.createDataFrame(rows, schema)


STR_DAG = {str(k): (None if v is None else
                    [str(x) for x in v] if isinstance(v, list) else str(v))
           for k, v in MIXED.items()}


def _str_est(**kw):
    from spark_vi.mllib.topic.dag_gated_lda import DagGatedLDAEstimator
    kw.setdefault("maxIter", 3)
    kw.setdefault("seed", 0)
    kw.setdefault("subsamplingRate", 1.0)
    kw.setdefault("optimizeDocConcentration", False)   # fixed alpha unless asked
    kw.setdefault("minDocsPerNode", 2)
    kw.setdefault("dag", STR_DAG)
    # String term ids throughout, as a vectorizer's vocabularies are.
    kw.setdefault("vocabularies", [V0_TERMS, [str(t) for t in V1_TERMS]])
    return DagGatedLDAEstimator(gateNBg=2, gateTpn=1, featuresCols=["f0", "f1"], **kw)


# --------------------------------------------------------------------------- #
# Params (no Spark)                                                            #
# --------------------------------------------------------------------------- #

def test_defaults_and_removed_params():
    from spark_vi.mllib.topic.dag_gated_lda import DagGatedLDAEstimator
    est = DagGatedLDAEstimator()
    assert est.getOrDefault("gateSeedMode") == "balanced"      # ADR 0050 default
    assert est.getOrDefault("gateWarm") == 0
    assert est.getOrDefault("gateOpenSeed") is True
    assert est.getOrDefault("optimizeDocConcentration") is False     # not MLlib's True: deep blocks
    assert est.getOrDefault("gateNBg") == 10 and est.getOrDefault("gateTpn") == 1
    assert est.getOrDefault("labelsCol") == "labels"
    assert est.getOrDefault("minDocsPerNode") == 1
    assert est.getOrDefault("unknownLabels") == "error"
    assert not est.isSet("topicConcentration")
    assert est.getDag() == {} and est.getVocabularies() == []
    names = {p.name for p in est.params}
    for gone in ("gateParent", "frontierCol", "domainBounds", "gateSeedDomain",
                 "gateSeedOwnCodes", "weightY", "numLabels", "labelCol", "init",
                 "spectralMethod", "anchorScope", "etaBoost", "nodeAffinityCol"):
        assert gone not in names, gone


def test_dag_param_keeps_id_types_and_validates():
    from spark_vi.mllib.topic.dag_gated_lda import DagGatedLDAEstimator
    est = DagGatedLDAEstimator(dag=MIXED)
    got = est.getDag()
    assert got[2] == ["A", "B"] and got["c"] == [1] and got["A"] == []
    assert 1 in got and "1" not in got
    # The stored form is JSON and re-settable (copy / Pipeline carry the string).
    est2 = DagGatedLDAEstimator().setDag(est.getOrDefault("dagEdges"))
    assert est2.getDag() == got
    assert est.copy().getDag() == got
    with pytest.raises(ValueError, match="cycle"):
        DagGatedLDAEstimator().setDag({1: 2, 2: 1})


def test_dag_from_pandas():
    pd = pytest.importorskip("pandas")
    from spark_vi.mllib.topic.dag_gated_lda import DagGatedLDAEstimator
    df = pd.DataFrame({"child": [3, 4, 1], "parent": [1, 1, None]})
    assert DagGatedLDAEstimator().setDag(df).getDag() == {1: [], 3: [1], 4: [1]}


def test_seed_extras_params_round_trip():
    from spark_vi.mllib.topic.dag_gated_lda import DagGatedLDAEstimator
    est = DagGatedLDAEstimator(
        nodeTerms={"c": ["t1", ("f1", "t0")], 3: [101]},
        gateSeedProfile={3: {"t2": 2.0, 100: 1.0, "t3": 0.0}})
    assert est.getNodeTerms() == {3: [101], "c": ["t1", ("f1", "t0")]}
    assert est.getGateSeedProfile() == {3: {100: 1.0, "t2": 2.0}}   # w <= 0 dropped
    copy = DagGatedLDAEstimator()
    copy.setNodeTerms(est.getOrDefault("nodeTerms"))
    copy.setGateSeedProfile(est.getOrDefault("gateSeedProfile"))
    assert copy.getNodeTerms() == est.getNodeTerms()
    assert copy.getGateSeedProfile() == est.getGateSeedProfile()
    with pytest.raises(ValueError, match="repeats"):
        DagGatedLDAEstimator(vocabularies=[["a", "a"]])


def test_term_index_ambiguity_rule():
    from spark_vi.mllib.topic.dag_gated_lda import _TermIndex
    ti = _TermIndex(["f0", "f1"], [V0_TERMS, V1_TERMS], [8, 5])
    assert ti.resolve("t3") == 3
    assert ti.resolve(102) == 8 + 2
    assert ti.resolve(("f0", "t0")) == 0 and ti.resolve(("f1", "t0")) == 8 + 4
    assert ti.resolve("nope") is None and ti.resolve(("f1", "t3")) is None
    with pytest.raises(ValueError, match="matched exactly"):   # 100 != "100"
        ti.resolve("100")
    ti_s = _TermIndex(["f0"], [["10", "x"]], [2])          # a vectorizer vocab: strs
    assert ti_s.resolve("10") == 0
    with pytest.raises(ValueError, match="pass .10."):
        ti_s.resolve(10)
    with pytest.raises(ValueError, match="matched exactly"):
        ti_s.resolve(("f0", 10))
    with pytest.raises(ValueError, match="featureColumn"):
        ti.resolve("t0")
    with pytest.raises(ValueError, match="not a feature column"):
        ti.resolve(("f9", "t0"))


def test_seed_extras_translate_to_engine_ids_and_concat_indices():
    """nodeTerms / profile in user node + term ids -> engine ids and concatenated
    vocabulary indices; entries of unkept nodes and unknown terms are dropped."""
    from spark_vi.dag.nodeset import fit_node_set
    from spark_vi.mllib.topic.dag_gated_lda import DagGatedLDAEstimator
    ns = fit_node_set(MIXED, [lab for _, _, lab in _mixed_rows()], min_docs_per_node=2)
    assert ns.pruned == {"rare": (3,)} and ns.spliced == (1,)
    est = DagGatedLDAEstimator(
        nodeTerms={"c": ["t1", ("f1", "t0")], 3: [101, "zzz"], "rare": ["t5"],
                   1: ["t6"]},
        gateSeedProfile={2: {"t2": 2.0, 100: 1.0}, "nope": {"t1": 1.0}})
    own, prof, summary = est._translate_seed_extras(
        ns, ["f0", "f1"], [8, 5], [V0_TERMS, V1_TERMS])
    e = ns.engine_id
    assert own == {e["c"]: [1, 12], e[3]: [9]}
    assert prof == {e[2]: {2: 2.0, 8: 1.0}}
    assert "2 not kept dropped" in summary and "1 term(s) in no vocabulary" in summary
    bad = DagGatedLDAEstimator(nodeTerms={"c": ["t0"]})
    with pytest.raises(ValueError, match="featureColumn"):
        bad._translate_seed_extras(ns, ["f0", "f1"], [8, 5], [V0_TERMS, V1_TERMS])
    with pytest.raises(ValueError, match="vocabularies"):
        bad._translate_seed_extras(ns, ["f0", "f1"], [8, 5], None)


def test_profile_seed_replaces_only_the_domains_it_touches():
    """A profile row replaces the seed in the domain(s) its terms belong to and
    leaves every other domain's slice alone."""
    from spark_vi.dag.layout import DagLayout
    lay = DagLayout({1: [0], 2: [0], 3: [1], 4: [1]}, n_bg=2, tpn=1)
    bounds = [0, 4, 7]
    est = _est(gateWarm=1, gateOpenSeed=False)
    # node 3 (leaf): terms in domain 1 only; node 1 (late): terms in both domains.
    seed, init, _ = est._gate_seeds(None, lay, 7, bounds, own={},
                                    prof={3: {5: 2.0}, 1: {0: 1.0, 6: 3.0}})
    k3, k1 = lay.block[3][0], lay.block[1][0]
    np.testing.assert_array_equal(init[k3], [0, 0, 0, 0, 0, 2.0, 0])
    np.testing.assert_array_equal(seed[k1], [1.0, 0, 0, 0, 0, 0, 3.0])
    assert seed[k3].sum() == 0 and init[k1].sum() == 0


# --------------------------------------------------------------------------- #
# Fit on the new inputs                                                        #
# --------------------------------------------------------------------------- #

def test_fit_transform_smoke_and_node_accessors(spark):
    df = _df(spark)
    model = _est().fit(df)
    out = model.transform(df)
    theta = np.asarray(out.select("topicDistribution").head()[0].toArray())
    assert theta.shape == (2 + 6,)                 # K = n_bg + nodes * tpn
    np.testing.assert_allclose(theta.sum(), 1.0, atol=1e-6)
    assert model.result.metadata["model_class"] == "DagGatedOnlineLDA"
    assert model.dag() == {1: [], 2: [], 3: [1], 4: [1], 5: [2], 6: [2]}
    assert model.prunedNodes() == {} and model.splicedNodes() == []
    sup = model.nodeSupport()
    assert sup[1] == sup[3] + sup[4] and sum(sup[n] for n in (3, 4, 5, 6)) == 40
    assert model.backgroundTopics() == [0, 1]
    topics = sorted(t for n in model.dag() for t in model.nodeTopics(n))
    assert topics == list(range(2, 8))
    with pytest.raises(KeyError, match="99"):
        model.nodeTopics(99)
    rows = model.describeNodes(3).collect()
    assert len(rows) == 8 and [r.node for r in rows[:2]] == [None, None]
    assert {r.node for r in rows[2:]} == {1, 2, 3, 4, 5, 6}
    for r in rows[2:]:
        assert model.nodeTopics(r.node) == [r.topic]
    assert dict(model.describeNodes().dtypes)["node"] == "bigint"


def test_requires_the_dag_and_the_labels_column(spark):
    from spark_vi.mllib.topic.dag_gated_lda import DagGatedLDAEstimator
    df = _df(spark, n=6)
    with pytest.raises(ValueError, match="setDag"):
        DagGatedLDAEstimator(maxIter=1).fit(df)
    with pytest.raises(ValueError, match="labelsCol"):
        _est(labelsCol="nope", maxIter=1).fit(df)


def test_labels_element_type_must_match_the_dag_ids(spark):
    from pyspark.sql import functions as F
    df = _df(spark, n=6)
    as_str = df.withColumn("labels", F.col("labels").cast("array<string>"))
    with pytest.raises(ValueError, match="array<string>.*every DAG node id is an int"):
        _est(maxIter=1).fit(as_str)
    with pytest.raises(ValueError, match="every DAG node id is a str"):
        _est(maxIter=1, dag={str(k): (None if v is None else str(v))
                             for k, v in PARENT.items()}).fit(df)
    with pytest.raises(ValueError, match="array"):
        _est(maxIter=1).fit(df.withColumn("labels", F.lit(3)))


def test_unknown_labels_error_and_ignore(spark):
    from pyspark.sql import functions as F
    df = _df(spark, n=10).union(
        _df(spark, n=2, seed=9).withColumn("labels", F.array(F.lit(99))))
    with pytest.raises(ValueError, match="99"):
        _est(maxIter=1).fit(df)
    model = _est(maxIter=1, unknownLabels="ignore").fit(df)
    assert set(model.dag()) == {1, 2, 3, 4, 5, 6}
    with pytest.raises(ValueError, match="unknownLabels"):
        _est(maxIter=1, unknownLabels="drop").fit(df)


def test_node_set_prune_splice_multi_parent_and_background(spark, capsys):
    """The str-id DAG: "rare" (1 doc) is pruned onto 3, "1" (one kept child "c", no
    documents of its own) is spliced, 2 keeps both parents; background documents
    (empty and null labels) fit. Labels carry ancestors and duplicates."""
    df = _str_dag_df(spark)
    model = _str_est().fit(df)
    assert model.dag() == {"A": [], "B": [], "c": ["A"], "2": ["A", "B"], "3": ["2"]}
    assert model.prunedNodes() == {"rare": ["3"]}
    assert model.splicedNodes() == ["1"]
    sup = model.nodeSupport()
    assert sup["c"] == 17 and sup["B"] == sup["2"] + 8
    assert set(sup) == {"A", "B", "c", "2", "3"}
    assert len(model.trainedAlpha()) == 2 + 5
    with pytest.raises(KeyError, match="pruned"):
        model.nodeTopics("rare")
    with pytest.raises(KeyError, match="spliced"):
        model.nodeTopics("1")
    log = capsys.readouterr().out
    assert "nodes given 7 / kept 5 / pruned 1 / spliced 1" in log
    assert dict(model.describeNodes().dtypes)["node"] == "string"


def test_mixed_id_dag_with_int_labels(spark):
    """A DAG with str AND int ids, fed by an array<int> column: only the int nodes
    can be named from it (ids are matched exactly); the str nodes are reached
    through closure. describeNodes renders the mixed ids as strings."""
    rows = []
    rng = np.random.default_rng(0)
    for i in range(30):
        lab = [[3], [2], [3, 3, 2], [1], []][i % 5]
        idx = sorted(rng.choice(10, size=3, replace=False).tolist())
        rows.append((Vectors.sparse(10, idx, [1.0] * 3), lab))
    from pyspark.sql.types import ArrayType, IntegerType, StructField, StructType
    from pyspark.ml.linalg import VectorUDT
    df = spark.createDataFrame(rows, StructType([
        StructField("features", VectorUDT()),
        StructField("labels", ArrayType(IntegerType()))]))
    model = _est(dag=MIXED, maxIter=2).fit(df)
    # "c" and "rare" are reached by no document: pruned at minDocsPerNode=1 and
    # landed on their kept parents; "B" admits exactly 2's documents: spliced.
    assert model.dag() == {"A": [], 1: ["A"], 2: ["A"], 3: [2]}
    assert model.prunedNodes() == {"c": [1], "rare": [3]}
    assert model.splicedNodes() == ["B"]
    nodes = {r.node for r in model.describeNodes().collect() if r.node is not None}
    assert nodes == {"A", "1", "2", "3"}
    assert dict(model.describeNodes().dtypes)["node"] == "string"


def test_labels_with_ancestors_fit_like_the_most_specific_labels(spark):
    """Frontier equivalence: labels written with ancestors and duplicates give the
    identical fit to the most-specific labels alone (same seed): the gate is the
    closure of the labels either way, and so is the support."""
    kw = dict(gateWarm=1, gateSeedLeaves=True, maxIter=3)
    a = _str_est(**kw).fit(_str_dag_df(spark))
    b = _str_est(**kw).fit(_str_dag_df(spark, specific_only=True))
    assert a.dag() == b.dag() and a.nodeSupport() == b.nodeSupport()
    for m in a.result.global_params["lambda"]:
        np.testing.assert_allclose(a.result.global_params["lambda"][m],
                                   b.result.global_params["lambda"][m], rtol=1e-12)
    np.testing.assert_allclose(a.result.elbo_trace, b.result.elbo_trace, rtol=1e-12)


def test_multidomain_widths_vocabularies_and_seed_extras(spark, capsys):
    df = _str_dag_df(spark)
    est = _str_est(gateWarm=1, gateSeedLeaves=True, maxIter=3)
    est.setNodeTerms({"c": ["t1", ("f1", "t0")], "3": ["101"], "rare": ["t5"]})
    est.setGateSeedProfile({"2": {"t2": 2.0, "100": 1.0}})
    model = est.fit(df)
    lam = model.result.global_params["lambda"]
    assert isinstance(lam, dict) and lam[0].shape[1] == 8 and lam[1].shape[1] == 5
    assert model.result.metadata["domains"] == [8, 5]
    assert model.vocabSize() == 13
    out = model.transform(df).select("topicDistribution").head()[0]
    np.testing.assert_allclose(out.toArray().sum(), 1.0, atol=1e-6)
    log = capsys.readouterr().out
    assert "nodeTerms: 2 node(s) used, 1 not kept dropped" in log
    assert "gateSeedProfile: 1 node(s) used" in log
    with pytest.raises(ValueError, match="featureColumn"):
        _str_est(gateWarm=1, maxIter=1, nodeTerms={"c": ["t0"]}).fit(df)
    with pytest.raises(ValueError, match="vocabulary of 'f1' has 4 terms"):
        _str_est(maxIter=1, vocabularies=[V0_TERMS, ["1", "2", "3", "4"]]).fit(df)


def test_feature_width_from_column_metadata(spark):
    """A domain's width is read from the column's ML attribute size when present;
    a row whose vector disagrees is refused."""
    from pyspark.sql import functions as F
    df = _df(spark, n=10, V=30)
    ok = df.withColumn("features", F.col("features").alias(
        "features", metadata={"ml_attr": {"num_attrs": 30}}))
    est = _est(maxIter=1)
    assert est._domain_sizes(ok, ["features"]) == [30]
    bad = df.withColumn("features", F.col("features").alias(
        "features", metadata={"ml_attr": {"num_attrs": 31}}))
    assert est._domain_sizes(bad, ["features"]) == [31]
    with pytest.raises(Exception, match="vector size 30 != expected 31"):
        est.fit(bad)


def test_topic_concentration(spark):
    df = _df(spark)
    m = _est(maxIter=1).fit(df)
    np.testing.assert_allclose(float(m.result.global_params["eta"]), 1.0 / 8)
    m = _est(maxIter=1, topicConcentration=0.3).fit(df)
    np.testing.assert_allclose(float(m.result.global_params["eta"]), 0.3)
    with pytest.raises(ValueError, match="topicConcentration"):
        _est(maxIter=1, topicConcentration=0.0).fit(df)


def test_pipeline_fit(spark):
    from pyspark.ml import Pipeline
    df = _df(spark)
    pm = Pipeline(stages=[_est(maxIter=2)]).fit(df)
    model = pm.stages[0]
    assert model.dag()[3] == [1]
    assert "topicDistribution" in pm.transform(df).columns


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
    df = spark.createDataFrame(rows, ["features", "labels"])
    model = _est(dag={1: None, 2: None}, maxIter=8, learningOffset=1.0,
                 learningDecay=0.7, optimizeDocConcentration=True).fit(df)
    alpha = model.result.global_params["alpha"]
    assert alpha[model.nodeTopics(1)[0]] > alpha[model.nodeTopics(2)[0]]


def test_leaves_first_schedule_with_balanced_and_leaf_seeds(spark):
    df = _df(spark, n=60)
    est = _est(gateWarm=1, gateSeedLeaves=True, maxIter=4,
               vocabularies=[[f"w{i}" for i in range(30)]],
               nodeTerms={1: ["w0"], 2: ["w1"]})
    model = est.fit(df)
    gp = model.result.global_params
    assert "gate_iter" in gp and int(gp["gate_iter"]) == 4
    assert (np.asarray(gp["gate_open_at"]) >= 0).all()     # every block opened


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
# Persistence: save/load keeps the node set, resume (MLlib persistence: the   #
# persistence suite test_mllib_dag_persistence.py).                          #
# --------------------------------------------------------------------------- #

def test_model_save_load_round_trip_keeps_the_node_set(tmp_path, spark):
    from spark_vi.mllib.topic.dag_gated_lda import DagGatedLDAModel
    df = _str_dag_df(spark)
    model = _str_est().fit(df)
    model.save(str(tmp_path / "m"))
    loaded = DagGatedLDAModel.load(str(tmp_path / "m"))
    for m, arr in model.result.global_params["lambda"].items():
        np.testing.assert_array_equal(loaded.result.global_params["lambda"][m], arr)
    assert loaded.dag() == model.dag()
    assert loaded.prunedNodes() == model.prunedNodes()
    assert loaded.splicedNodes() == model.splicedNodes()
    assert loaded.nodeSupport() == model.nodeSupport()
    assert all(loaded.nodeTopics(n) == model.nodeTopics(n) for n in model.dag())
    assert loaded._frontier_fn() == model._frontier_fn()


def test_model_save_load_int_ids_and_transform(tmp_path, spark):
    from spark_vi.mllib.topic.dag_gated_lda import DagGatedLDAModel
    df = _df(spark)
    model = _est().fit(df)
    model.save(str(tmp_path / "m"))
    loaded = DagGatedLDAModel.load(str(tmp_path / "m"))
    assert loaded.dag() == model.dag() and 3 in loaded.dag()
    assert "topicDistribution" in loaded.transform(df).columns


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


def test_minibatch_resume_with_a_gate_schedule_is_the_continuous_fit(tmp_path, spark):
    """Checkpoint + resume through the shim reproduces the uninterrupted fit: the
    seed stream, the Robbins-Monro counter and the leaves-first gate state (which
    rides global_params) all continue across the resume."""
    df = _df(spark, n=60)
    kw = dict(seed=5, subsamplingRate=0.5, gateWarm=2, gateOpenSeed=False)
    want = _est(maxIter=6, **kw).fit(df).result
    a = _est(maxIter=3, **kw)
    a.setSaveDir(str(tmp_path / "ck"))
    a.fit(df)
    b = _est(maxIter=3, **kw)
    b.setResumeFrom(str(tmp_path / "ck"))
    got = b.fit(df).result
    assert got.n_iterations == 6
    np.testing.assert_allclose(got.global_params["lambda"], want.global_params["lambda"],
                               rtol=1e-10)
    np.testing.assert_allclose(got.elbo_trace, want.elbo_trace, rtol=1e-10)


# --------------------------------------------------------------------------- #
# Defaults, the recommended preset, help text (spec R10, R11, decision 1).     #
# --------------------------------------------------------------------------- #

def test_mllib_like_defaults():
    """Where a Param has an analogue on pyspark.ml.clustering.LDA, its default is
    LDA's, except optimizeDocConcentration (False: learned alpha starves deep blocks);
    gateNBg=10 makes the flat layout LDA's k=10."""
    from spark_vi.mllib.topic.dag_gated_lda import DagGatedLDAEstimator
    est = DagGatedLDAEstimator()
    want = dict(maxIter=20, learningOffset=1024.0, learningDecay=0.51,
                subsamplingRate=0.05, optimizeDocConcentration=False,
                featuresCol="features", topicDistributionCol="topicDistribution",
                gateNBg=10, gateTpn=1, gateWarm=0)
    assert {k: est.getOrDefault(k) for k in want} == want
    for unset in ("docConcentration", "topicConcentration"):
        assert not est.isSet(unset) and not est.hasDefault(unset), unset
    assert not est.isSet("seed")      # HasSeed's hash default is not used by fit


def test_help_text_states_defaults_and_is_domain_neutral():
    from spark_vi.mllib.topic.dag_gated_lda import DagGatedLDAEstimator
    est = DagGatedLDAEstimator()
    for p in est.params:
        assert "default" in p.doc.lower(), p.name
        for word in ("mondo", "hpo", "omop", "clinical", "patient", "condition",
                     "diagnos", "exp 0", "adr "):
            assert word not in p.doc.lower(), (p.name, word)


def test_plain_params_have_mllib_setters_and_getters():
    """Every plain Param has setX / getX; the featurizer's hand-off chain works."""
    from spark_vi.mllib.topic.dag_gated_lda import (_PLAIN_PARAMS,
                                                    DagGatedLDAEstimator,
                                                    DagGatedLDAModel)
    est = DagGatedLDAEstimator()
    for name in _PLAIN_PARAMS:
        cap = name[0].upper() + name[1:]
        assert callable(getattr(est, f"set{cap}")), name
        assert callable(getattr(est, f"get{cap}")), name
        assert hasattr(DagGatedLDAModel, f"set{cap}"), name
    assert not hasattr(est, "setDagEdges")          # only via setDag's validation
    out = (est.setFeaturesCols(["f0", "f1"]).setVocabularies([["a"], ["b"]])
           .setLabelsCol("nodes").setGateNBg(3).setMaxIter(7).setSeed(4))
    assert out is est
    assert est.getFeaturesCols() == ["f0", "f1"] and est.getLabelsCol() == "nodes"
    assert est.getGateNBg() == 3 and est.getMaxIter() == 7 and est.getSeed() == 4


def test_recommended_preset_and_overrides():
    from spark_vi.mllib.topic.dag_gated_lda import (RECOMMENDED_PARAMS,
                                                    DagGatedLDAEstimator)
    assert RECOMMENDED_PARAMS == dict(
        gateNBg=200, gateTpn=1, maxIter=60, learningOffset=64.0, learningDecay=0.51,
        subsamplingRate=0.1, docConcentration=[0.5], optimizeDocConcentration=False,
        gammaShape=100.0, caviMaxIter=100, caviTol=1e-3, gateWarm=2,
        gateOpenSeed=True, gateSeedMode="balanced", gateSeedLeaves=True)
    est = DagGatedLDAEstimator.recommended()
    for k, v in RECOMMENDED_PARAMS.items():
        assert est.isSet(k) and est.getOrDefault(k) == v, k
    est2 = DagGatedLDAEstimator.recommended(dag=PARENT, gateNBg=4, minDocsPerNode=3,
                                            labelsCol="nodes")
    assert est2.getGateNBg() == 4 and est2.getMinDocsPerNode() == 3
    assert est2.getLabelsCol() == "nodes" and est2.getDag()[3] == [1]
    assert est2.getLearningOffset() == 64.0          # the rest of the preset kept
    est2.getDocConcentration().append(9.0)            # no shared mutable state
    assert RECOMMENDED_PARAMS["docConcentration"] == [0.5]


def test_recommended_preset_fits(spark):
    from spark_vi.mllib.topic.dag_gated_lda import DagGatedLDAEstimator
    df = _df(spark, n=60)
    model = DagGatedLDAEstimator.recommended(
        dag=PARENT, gateNBg=2, maxIter=4, seed=0, subsamplingRate=1.0).fit(df)
    gp = model.result.global_params
    assert int(gp["gate_iter"]) == 4 and (np.asarray(gp["gate_open_at"]) >= 0).all()
    np.testing.assert_allclose(model.trainedAlpha(), 0.5)    # held fixed


# --------------------------------------------------------------------------- #
# resumeFrom checks the node set (WP5 follow-up).                              #
# --------------------------------------------------------------------------- #

def test_resume_refuses_a_different_node_set(tmp_path, spark):
    df = _df(spark)
    ck = str(tmp_path / "ck")
    a = _est(maxIter=1, seed=1)
    a.setSaveDir(ck)
    a.fit(df)

    def resume(data=df, **kw):
        b = _est(maxIter=1, seed=1, **kw)
        b.setResumeFrom(ck)
        return b.fit(data)

    # Fewer kept nodes: minDocsPerNode prunes leaves (~10 docs each).
    with pytest.raises(ValueError, match=r"0 kept node\(s\) added \(e\.g\. \[\]\), "
                                         r"[1-6] removed \(e\.g\. \[3, 4"):
        resume(minDocsPerNode=15)
    # More kept nodes: a new DAG node with documents of its own.
    df7 = df.union(spark.createDataFrame(
        [(Vectors.sparse(30, [0, 1], [1.0, 1.0]), [7])] * 3, df.schema))
    with pytest.raises(ValueError, match=r"1 kept node\(s\) added \(e\.g\. \[7\]\), "
                                         r"0 removed"):
        resume(data=df7, dag={**PARENT, 7: 2})
    # Same nodes, other parents (4 gains a second parent).
    with pytest.raises(ValueError, match="kept DAG differs"):
        resume(dag={**PARENT, 4: [1, 2]})
    # Same node set, other layout.
    with pytest.raises(ValueError, match="layout is gateNBg=2, gateTpn=1"):
        _est(maxIter=1, seed=1, resumeFrom=ck).setGateNBg(3).fit(df)
    # The matching node set resumes.
    assert resume().result.n_iterations == 2


def test_resume_refuses_a_checkpoint_without_a_node_set(tmp_path, spark):
    d = tmp_path / "other"
    (d / "params").mkdir(parents=True)
    np.save(d / "params" / "lambda.npy", np.zeros((8, 30)))
    (d / "manifest.json").write_text(json.dumps({
        "format_version": 1, "elbo_trace": [], "n_iterations": 0,
        "converged": False, "metadata": {"model_class": "DagGatedOnlineLDA"},
        "param_names": ["lambda"], "diagnostic_traces": {}}))
    with pytest.raises(ValueError, match="carries no node set"):
        _est(maxIter=1, resumeFrom=str(d)).fit(_df(spark))


# --------------------------------------------------------------------------- #
# describeNodes names the terms (WP5 follow-up).                               #
# --------------------------------------------------------------------------- #

def test_describe_nodes_terms_single_domain(spark):
    df = _df(spark)
    vocab = [f"w{i}" for i in range(30)]
    model = _est(maxIter=1, vocabularies=[vocab]).fit(df)
    assert dict(model.describeNodes().dtypes)["terms"] == "array<string>"
    for r in model.describeNodes(4).collect():
        assert r.terms == [vocab[i] for i in r.termIndices]
    plain = _est(maxIter=1).fit(df).describeNodes(4).collect()
    assert all(r.terms is None for r in plain)


def test_describe_nodes_terms_multi_domain(spark):
    df = _str_dag_df(spark)
    model = _str_est().fit(df)
    names = [f"f0:{t}" for t in V0_TERMS] + [f"f1:{t}" for t in V1_TERMS]
    rows = model.describeNodes(13).collect()
    for r in rows:
        assert len(r.terms) == 13
        assert r.terms == [names[i] for i in r.termIndices]
    assert {t.split(":", 1)[0] for r in rows for t in r.terms} == {"f0", "f1"}
