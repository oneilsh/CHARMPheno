"""DagGatedLDAModel.nodeStatistics (spec R14.1) and dropUnsupportedNodes (R14.2).

A small, well-separated fit (each leaf's documents use their own few terms, so every
fitted block is sharp), then a node's block is forced flat by writing the prior into
its λ rows in a COPY of the fitted result: the statistics must flag exactly that
node, and dropping must remove exactly its topics, keep every other fitted row as it
was, and leave a model that scores, describes, saves and feeds the heads like any
other.
"""
from __future__ import annotations

import numpy as np
import pytest

pyspark = pytest.importorskip("pyspark")
from pyspark.ml.linalg import Vectors  # noqa: E402

# Two top-level nodes with two leaves each; 1 also has a multi-parent grandchild 7
# (under 3 and 5), so dropping 3 re-attaches 7 under 1 (through 3) and 5.
PARENT = {1: None, 2: None, 3: 1, 4: 1, 5: 2, 6: 2, 7: [3, 5]}
LEAF_TERMS = {3: range(0, 5), 4: range(5, 10), 5: range(10, 15), 6: range(15, 20),
              7: range(20, 25)}
V = 40


def _rows(n=100, seed=0, labels_as=lambda x: x):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        leaf = [3, 4, 5, 6, 7][i % 5]
        terms = sorted(rng.choice(list(LEAF_TERMS[leaf]), size=4, replace=False).tolist())
        bg = int(rng.integers(30, V))
        idx = sorted(set(terms) | {bg})
        lab = [leaf] if i % 11 else [leaf, 1]           # some ancestors, harmless
        rows.append((Vectors.sparse(V, idx, [3.0] * len(idx)),
                     [labels_as(x) for x in lab]))
    return rows


def _df(spark, **kw):
    return spark.createDataFrame(_rows(**kw), ["features", "labels"])


def _est(dag=PARENT, **kw):
    from spark_vi.mllib.topic.dag_gated_lda import DagGatedLDAEstimator
    kw.setdefault("maxIter", 8)
    kw.setdefault("seed", 0)
    kw.setdefault("subsamplingRate", 1.0)
    kw.setdefault("learningOffset", 1.0)
    kw.setdefault("optimizeDocConcentration", False)
    # alpha 1: with the default 1/K one leaf's block gets no words here (starved)
    kw.setdefault("docConcentration", [1.0])
    kw.setdefault("gateNBg", 2)
    kw.setdefault("gateTpn", 1)
    return DagGatedLDAEstimator(dag=dag, **kw)


@pytest.fixture(scope="module")
def fitted(spark):
    df = _df(spark).cache()
    # gateWarm 1: the result carries the gate state (gate_iter, gate_open_at).
    return _est(gateWarm=1).fit(df), df


def _with_lambda(model, edit):
    """A copy of `model` whose λ is `edit(copy of λ)` (the original is untouched)."""
    from spark_vi.core.result import VIResult
    from spark_vi.mllib.topic.dag_gated_lda import DagGatedLDAModel
    r = model.result
    gp = dict(r.global_params)
    lam = gp["lambda"]
    gp["lambda"] = ({m: v.copy() for m, v in lam.items()} if isinstance(lam, dict)
                    else lam.copy())
    edit(gp["lambda"])
    res = VIResult(global_params=gp, elbo_trace=list(r.elbo_trace),
                   n_iterations=r.n_iterations, converged=r.converged,
                   metadata=dict(r.metadata), diagnostic_traces=dict(r.diagnostic_traces))
    return model._copyValues(DagGatedLDAModel(res))


def _flatten(model, *nodes):
    eta = float(model.result.global_params["eta"])

    def edit(lam):
        for n in nodes:
            for k in model.nodeTopics(n):
                if isinstance(lam, dict):
                    for m in lam:
                        lam[m][k, :] = eta * 50
                else:
                    lam[k, :] = eta * 50                 # uniform: the prior's shape
    return _with_lambda(model, edit)


# --------------------------------------------------------------------------- #
# nodeStatistics (R14.1)                                                       #
# --------------------------------------------------------------------------- #

def test_node_statistics_columns_types_and_values(fitted):
    from pyspark.sql.types import (BooleanType, DoubleType, IntegerType, LongType)
    from spark_vi.eval.topic.sharpness import support_fraction
    model, _ = fitted
    df = model.nodeStatistics()
    assert df.columns == ["node", "depth", "support", "supportFraction", "mass",
                          "massToParent", "flat"]
    types = {f.name: type(f.dataType) for f in df.schema.fields}
    assert types == {"node": LongType, "depth": IntegerType, "support": LongType,
                     "supportFraction": DoubleType, "mass": DoubleType,
                     "massToParent": DoubleType, "flat": BooleanType}
    rows = {r["node"]: r for r in df.collect()}
    assert list(rows) == list(model._ns().nodes)           # engine order
    assert {n: r["support"] for n, r in rows.items()} == model.nodeSupport()
    assert {n: r["depth"] for n, r in rows.items()} == {1: 1, 2: 1, 3: 2, 4: 2, 5: 2,
                                                         6: 2, 7: 3}
    sf, mass = support_fraction(model.result.global_params["lambda"])
    for n, r in rows.items():
        (k,) = model.nodeTopics(n)
        assert r["supportFraction"] == pytest.approx(sf[k])
        assert r["mass"] == pytest.approx(mass[k])
        assert r["flat"] is bool(sf[k] > 0.5)
        par = model.dag()[n]
        if not par:
            assert r["massToParent"] is None
        else:
            heavy = max(mass[model.nodeTopics(p)[0]] for p in par)
            assert r["massToParent"] == pytest.approx(mass[k] / heavy)
    assert not any(r["flat"] for r in rows.values())        # every block is sharp


def test_node_statistics_flags_a_hand_flattened_block(fitted):
    model, _ = fitted
    flat = _flatten(model, 3)
    rows = {r["node"]: r for r in flat.nodeStatistics().collect()}
    assert rows[3]["flat"] and rows[3]["supportFraction"] == pytest.approx(1.0)
    assert [n for n, r in rows.items() if r["flat"]] == [3]
    # the threshold is the caller's
    assert not any(r["flat"] for r in flat.nodeStatistics(flatThreshold=1.0).collect())
    # reporting only: nothing dropped
    assert flat.unsupportedNodes() == {} and 3 in flat._ns().engine_id


def test_node_statistics_string_ids_and_tpn_minimum(spark):
    from pyspark.sql.types import StringType
    sdag = {str(k): (None if v is None else
                     [str(x) for x in v] if isinstance(v, list) else str(v))
            for k, v in PARENT.items()}
    df = spark.createDataFrame(_rows(labels_as=str), ["features", "labels"])
    model = _est(dag=sdag, gateTpn=2, maxIter=4).fit(df)
    sdf = model.nodeStatistics()
    assert isinstance(sdf.schema["node"].dataType, StringType)
    eta = float(model.result.global_params["eta"])
    k0, k1 = model.nodeTopics("3")

    def one(lam):
        lam[k0, :] = eta

    def both(lam):
        lam[k0, :] = eta
        lam[k1, :] = eta
    half = {r["node"]: r for r in _with_lambda(model, one).nodeStatistics().collect()}
    full = {r["node"]: r for r in _with_lambda(model, both).nodeStatistics().collect()}
    assert not half["3"]["flat"]                 # the minimum over the block's topics
    assert full["3"]["flat"]


# --------------------------------------------------------------------------- #
# dropUnsupportedNodes (R14.2)                                                 #
# --------------------------------------------------------------------------- #

def test_drop_subsets_every_fitted_array_in_the_new_engine_order(fitted):
    model, _ = fitted
    flat = _flatten(model, 3)
    clean = flat.dropUnsupportedNodes()
    # the original is unchanged
    assert 3 in flat._ns().engine_id and flat.unsupportedNodes() == {}
    assert clean.unsupportedNodes() == {3: [1]}
    assert clean.dag() == {1: [], 2: [], 4: [1], 5: [2], 6: [2], 7: [1, 5]}
    assert clean.prunedNodes() == flat.prunedNodes()
    with pytest.raises(KeyError, match="dropped as unsupported"):
        clean.nodeTopics(3)
    old, new = flat.result.global_params, clean.result.global_params
    K = old["lambda"].shape[0]
    assert new["lambda"].shape == (K - 1, V)
    assert clean.result.metadata["K"] == K - 1
    pairs = [(k, k) for k in flat.backgroundTopics()]
    pairs += [(flat.nodeTopics(n)[0], clean.nodeTopics(n)[0]) for n in clean._ns().nodes]
    for ko, kn in pairs:
        np.testing.assert_array_equal(new["lambda"][kn], old["lambda"][ko])
        assert new["alpha"][kn] == old["alpha"][ko]
        assert new["gate_open_at"][kn] == old["gate_open_at"][ko]
    assert sorted(kn for _, kn in pairs) == list(range(K - 1))
    assert float(new["eta"]) == float(old["eta"])
    assert int(new["gate_iter"]) == int(old["gate_iter"])
    assert len(clean.result.diagnostic_traces["alpha"][-1]) == K - 1
    # Params carried
    assert clean.getOrDefault("gateNBg") == 2 and clean.getOrDefault("maxIter") == 8
    # the accessors work on the cleaned model
    assert clean.describeTopics().count() == K - 1
    nodes = {r["node"] for r in clean.describeNodes().collect()}
    assert nodes == {None, 1, 2, 4, 5, 6, 7}
    assert [r["node"] for r in clean.nodeStatistics().collect()] == list(clean._ns().nodes)


def test_drop_with_nothing_flat_is_an_equivalent_copy(fitted):
    model, _ = fitted
    same = model.dropUnsupportedNodes()
    assert same is not model and same.unsupportedNodes() == {}
    assert same._ns() == model._ns()
    np.testing.assert_array_equal(same.result.global_params["lambda"],
                                  model.result.global_params["lambda"])


def test_drop_a_top_level_node_with_kept_children(fitted):
    model, _ = fitted
    clean = _flatten(model, 1).dropUnsupportedNodes()
    assert clean.unsupportedNodes() == {1: []}
    assert clean.dag()[3] == [] and clean.dag()[4] == []
    assert clean._ns().frontier([1]) == frozenset()


def test_cleaned_transform_equals_a_model_built_from_the_subsetted_arrays(fitted):
    from spark_vi.core.result import VIResult
    from spark_vi.dag.nodeset import drop_nodes
    from spark_vi.mllib.topic.dag_gated_lda import DagGatedLDAModel
    model, df = fitted
    flat = _flatten(model, 3)
    clean = flat.dropUnsupportedNodes()
    gp = flat.result.global_params
    keep = [0, 1] + [flat.nodeTopics(n)[0] for n in drop_nodes(flat._ns(), [3]).nodes]
    direct_gp = {"lambda": gp["lambda"][keep], "alpha": gp["alpha"][keep],
                 "eta": gp["eta"]}
    md = dict(clean.result.metadata)
    direct = flat._copyValues(DagGatedLDAModel(VIResult(
        global_params=direct_gp, elbo_trace=[], n_iterations=1, converged=False,
        metadata=md)))
    a = [r[0].toArray() for r in clean.transform(df).select("topicDistribution").collect()]
    b = [r[0].toArray() for r in direct.transform(df).select("topicDistribution").collect()]
    np.testing.assert_allclose(np.stack(a), np.stack(b))


def test_cleaned_model_save_load_round_trip(fitted, tmp_path):
    from spark_vi.mllib.topic.dag_gated_lda import DagGatedLDAModel
    model, df = fitted
    clean = _flatten(model, 3).dropUnsupportedNodes()
    uri = f"file://{tmp_path}/clean"
    clean.save(uri)
    back = DagGatedLDAModel.load(uri)
    assert back.unsupportedNodes() == {3: [1]}
    assert back._ns() == clean._ns()
    np.testing.assert_array_equal(back.result.global_params["lambda"],
                                  clean.result.global_params["lambda"])
    a = clean.transform(df).select("topicDistribution").collect()
    b = back.transform(df).select("topicDistribution").collect()
    assert [r[0] for r in a] == [r[0] for r in b]


def test_heads_from_the_cleaned_model_have_no_head_for_the_dropped_node(fitted):
    from spark_vi.mllib import DagHeadsEstimator
    model, df = fitted
    clean = _flatten(model, 3).dropUnsupportedNodes()
    scored = clean.transform(df).cache()
    heads = DagHeadsEstimator(maxIter=50).setTopicModel(clean).fit(scored)
    assert heads.nodes() == list(clean._ns().nodes)
    assert 3 not in heads.nodes()
    assert heads.unsupportedNodes() == {3: [1]}
    with pytest.raises(KeyError, match="dropped as unsupported"):
        heads.nodeIndex(3)
    # a document labelled 3 is a positive of 3's landing (1) and its ancestors only
    lab = heads.attachLabels(scored.limit(1).select("labels").selectExpr(
        "array(cast(3 as bigint)) as labels")).head()
    assert lab["label"][heads.nodeIndex(1)] == 1.0
    assert sum(lab["label"]) == 2.0                        # root + 1
