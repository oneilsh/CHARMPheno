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
PARENT = {"1": None, "2": None, "3": "1", "4": "1", "5": "2", "6": "2",
          "7": ["3", "5"]}
LEAF_TERMS = {3: range(0, 5), 4: range(5, 10), 5: range(10, 15), 6: range(15, 20),
              7: range(20, 25)}
V = 40


def _rows(n=100, seed=0, labels_as=str):
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

def test_heads_from_the_cleaned_model_have_no_head_for_the_dropped_node(fitted):
    from charmpheno.readout.mllib import DagHeadsEstimator
    model, df = fitted
    clean = _flatten(model, "3").dropUnsupportedNodes()
    scored = clean.transform(df).cache()
    heads = DagHeadsEstimator(maxIter=50).setTopicModel(clean).fit(scored)
    assert heads.nodes() == list(clean._ns().nodes)
    assert "3" not in heads.nodes()
    assert heads.unsupportedNodes() == {"3": ["1"]}
    with pytest.raises(KeyError, match="dropped as unsupported"):
        heads.nodeIndex("3")
    # a document labelled 3 is a positive of 3's landing (1) and its ancestors only
    lab = heads.attachLabels(scored.limit(1).select("labels").selectExpr(
        "array('3') as labels")).head()
    assert lab["label"][heads.nodeIndex("1")] == 1.0
    assert sum(lab["label"]) == 2.0                        # root + 1
