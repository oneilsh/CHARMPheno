"""MLlib Estimator/Model shim for GatedOnlineLDA (hierarchical case-finding placement).

Mirrors mllib/topic/lda.py (ADR 0009): a translation layer over GatedOnlineLDA + VIRunner.
fit trains GATED (each row carries features + a frontier = set of DAG node ids); transform
folds held-out docs in UNGATED (full-K) and emits per-node affinity. v1 uses init="random"
(the validated default); block-aligned spectral init on Spark (distributed co-occurrence) is
deferred (the in-engine "spectral" strategy is validated in the no-Spark harness).
"""
from __future__ import annotations

import numpy as np
from pyspark import StorageLevel, keyword_only
from pyspark.ml.base import Estimator, Model
from pyspark.ml.param import Param, Params, TypeConverters
from pyspark.ml.param.shared import HasFeaturesCol, HasLabelCol, HasMaxIter, HasSeed

from spark_vi.core.config import VIConfig
from spark_vi.models.topic.dag_placement import DagLayout
from spark_vi.models.topic.gated_lda import GatedOnlineLDA, node_affinity
from spark_vi.models.topic.types import GatedBOWDocument
from spark_vi.mllib.topic._common import _vector_to_bow_document


class _GatedLDAParams(HasFeaturesCol, HasLabelCol, HasMaxIter, HasSeed):
    parent = Param(Params._dummy(), "parent",
                   "DAG parent map {child_int: parent_int or [parent_ints]}, anchor->0",
                   typeConverter=TypeConverters.identity)
    nBg = Param(Params._dummy(), "nBg", "number of shared background topics",
                typeConverter=TypeConverters.toInt)
    tpn = Param(Params._dummy(), "tpn", "topics per DAG node",
                typeConverter=TypeConverters.toInt)
    nodeAffinityCol = Param(Params._dummy(), "nodeAffinityCol",
                            "output column: per-node affinity Vector",
                            typeConverter=TypeConverters.toString)
    caviMaxIter = Param(Params._dummy(), "caviMaxIter", "inner CAVI max iters",
                        typeConverter=TypeConverters.toInt)
    caviTol = Param(Params._dummy(), "caviTol", "inner CAVI tolerance",
                    typeConverter=TypeConverters.toFloat)
    gammaShape = Param(Params._dummy(), "gammaShape", "Gamma init shape for gamma/lambda",
                       typeConverter=TypeConverters.toFloat)


def _layout(est_or_model) -> DagLayout:
    return DagLayout(est_or_model.getOrDefault("parent"),
                     n_bg=est_or_model.getOrDefault("nBg"),
                     tpn=est_or_model.getOrDefault("tpn"))


class GatedLDAEstimator(_GatedLDAParams, Estimator):
    @keyword_only
    def __init__(self, *, featuresCol="features", labelCol="frontier", parent=None,
                 nBg=2, tpn=1, maxIter=20, seed=None, caviMaxIter=100, caviTol=1e-3,
                 gammaShape=100.0):
        super().__init__()
        self._setDefault(featuresCol="features", labelCol="frontier", nBg=2, tpn=1,
                         maxIter=20, nodeAffinityCol="nodeAffinity",
                         caviMaxIter=100, caviTol=1e-3, gammaShape=100.0)
        self.setParams(**self._input_kwargs)

    @keyword_only
    def setParams(self, **kwargs):
        return self._set(**kwargs)

    def _fit(self, dataset) -> "GatedLDAModel":
        from spark_vi.core.runner import VIRunner
        if self.getOrDefault("parent") is None:
            raise ValueError("GatedLDAEstimator requires a `parent` DAG map.")
        lay = _layout(self)

        features_col = self.getOrDefault("featuresCol")
        label_col = self.getOrDefault("labelCol")
        first = dataset.select(features_col).head(1)
        if not first:
            raise ValueError("Cannot fit on an empty DataFrame.")
        V = first[0][0].size
        seed = self.getOrDefault("seed") if self.isSet("seed") else None

        model_obj = GatedOnlineLDA(
            lay, V, init="random",
            alpha=1.0 / lay.K, eta=1.0 / lay.K,
            gamma_shape=self.getOrDefault("gammaShape"),
            cavi_max_iter=self.getOrDefault("caviMaxIter"),
            cavi_tol=self.getOrDefault("caviTol"),
            random_seed=seed,
        )
        config = VIConfig(max_iterations=self.getOrDefault("maxIter"), random_seed=seed)

        def _to_gated(row):
            bow = _vector_to_bow_document(row[0])
            frontier = frozenset(int(x) for x in (row[1] or []))
            return GatedBOWDocument(indices=bow.indices, counts=bow.counts,
                                    length=bow.length, frontier=frontier)

        rdd = (dataset.select(features_col, label_col).rdd.map(_to_gated)
               .persist(StorageLevel.MEMORY_AND_DISK))
        rdd.count()
        try:
            result = VIRunner(model_obj, config=config).fit(rdd)
        finally:
            rdd.unpersist(blocking=False)

        out = GatedLDAModel(result, parent=self.getOrDefault("parent"),
                            nBg=self.getOrDefault("nBg"), tpn=self.getOrDefault("tpn"))
        for p in self.params:
            if self.isSet(p):
                out._set(**{p.name: self.getOrDefault(p)})
            elif self.hasDefault(p):
                out._setDefault(**{p.name: self.getOrDefault(p)})
        return out


class GatedLDAModel(_GatedLDAParams, Model):
    _expected_model_class = "GatedOnlineLDA"

    def __init__(self, result, *, parent, nBg, tpn):
        super().__init__()
        self._result = result
        self._setDefault(featuresCol="features", labelCol="frontier", nBg=nBg, tpn=tpn,
                         parent=parent, nodeAffinityCol="nodeAffinity",
                         caviMaxIter=100, caviTol=1e-3, gammaShape=100.0)

    @property
    def result(self):
        return self._result

    def _transform(self, dataset):
        from pyspark.ml.linalg import DenseVector, VectorUDT
        from pyspark.sql import functions as F
        from scipy.special import digamma
        from spark_vi.models.topic.lda import _cavi_doc_inference

        lay = _layout(self)
        lam = self._result.global_params["lambda"]
        expElogbeta = np.exp(digamma(lam) - digamma(lam.sum(axis=1, keepdims=True)))
        alpha = self._result.global_params["alpha"]
        gamma_shape = float(self.getOrDefault("gammaShape"))
        cavi_max_iter = int(self.getOrDefault("caviMaxIter"))
        cavi_tol = float(self.getOrDefault("caviTol"))
        K = expElogbeta.shape[0]
        nodes = list(lay.nodes)
        blocks = {u: lay.block[u] for u in nodes}

        sc = dataset.sparkSession.sparkContext
        bcast = sc.broadcast({
            "expElogbeta": expElogbeta, "alpha": alpha, "gamma_shape": gamma_shape,
            "cavi_max_iter": cavi_max_iter, "cavi_tol": cavi_tol, "K": K,
            "nodes": nodes, "blocks": blocks,
        })

        def _affinity(features):
            p = bcast.value
            doc = _vector_to_bow_document(features)
            rng = np.random.default_rng()
            gamma_init = rng.gamma(p["gamma_shape"], 1.0 / p["gamma_shape"], size=p["K"])
            gamma, _, _, _ = _cavi_doc_inference(
                indices=doc.indices, counts=doc.counts, expElogbeta=p["expElogbeta"],
                alpha=p["alpha"], gamma_init=gamma_init,
                max_iter=p["cavi_max_iter"], tol=p["cavi_tol"])
            theta = gamma / gamma.sum()
            return DenseVector([float(theta[p["blocks"][u]].sum()) for u in p["nodes"]])

        udf = F.udf(_affinity, returnType=VectorUDT())
        try:
            out_col = self.getOrDefault("nodeAffinityCol")
            return dataset.withColumn(out_col, udf(F.col(self.getOrDefault("featuresCol"))))
        finally:
            bcast.unpersist(blocking=False)
