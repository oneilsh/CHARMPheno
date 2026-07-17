"""MLlib Estimator/Model shim for GatedOnlineLDA (hierarchical case-finding placement).

Mirrors mllib/topic/lda.py (ADR 0009): a translation layer over GatedOnlineLDA + VIRunner.
fit trains GATED (each row carries features + a frontier = set of DAG node ids); transform
folds held-out docs in UNGATED (full-K) and emits per-node affinity. init="random" is the
validated default; init="spectral" seeds lambda from anchor-word spectral recovery (Arora et al.
2013), routed by spectralMethod between DENSE (collect the corpus to the driver, exact V×V,
mirroring the STM shim's dense spectral path, mllib/topic/stm.py) and SCALABLE (distributed
random-projection sketch over the RDD, ADR 0032 — the gated analogue, for large vocabularies).
"auto" picks dense below spectralMaxVocab and scalable at/above it (resolve_spectral_method).
"""
from __future__ import annotations

from typing import Callable

import numpy as np
from pyspark import StorageLevel, keyword_only
from pyspark.ml.base import Estimator, Model
from pyspark.ml.param import Param, Params, TypeConverters
from pyspark.ml.param.shared import HasFeaturesCol, HasLabelCol, HasMaxIter, HasSeed

from spark_vi.core.config import VIConfig
from spark_vi.models.topic.dag_placement import DagLayout
from spark_vi.models.topic.gated_lda import GatedOnlineLDA
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
    nodeAlphaScale = Param(Params._dummy(), "nodeAlphaScale",
                           "multiplier on the per-node-topic Dirichlet alpha "
                           "relative to the background alpha (1/K); 1.0 = symmetric "
                           "(default). <1 down-weights the node blocks so a document "
                           "needs stronger evidence to place mass on a disease node — "
                           "an asymmetric prior (Wallach et al. 2009) that reflects "
                           "the low prevalence of any single node and, at ungated "
                           "transform time, keeps background docs on the background "
                           "block.",
                           typeConverter=TypeConverters.toFloat)
    miniBatchFraction = Param(Params._dummy(), "miniBatchFraction",
                              "SVI mini-batch fraction in (0, 1]; each iteration "
                              "samples this fraction of the corpus. 0.0 = full-batch "
                              "(default). Mini-batching is what makes the decaying "
                              "Robbins-Monro step size appropriate (Hoffman et al. 2013).",
                              typeConverter=TypeConverters.toFloat)
    learningRateTau0 = Param(Params._dummy(), "learningRateTau0",
                             "SVI step-size delay tau0 in rho_t = (tau0 + t + 1)^-kappa "
                             "(Hoffman et al. 2013). Larger tau0 down-weights early "
                             "iterations (a gentler, less aggressive slow start). "
                             "Default 1.0.",
                             typeConverter=TypeConverters.toFloat)
    learningRateKappa = Param(Params._dummy(), "learningRateKappa",
                              "SVI step-size decay exponent kappa in rho_t = "
                              "(tau0 + t + 1)^-kappa. Robbins-Monro convergence holds "
                              "for kappa in (0.5, 1]; larger kappa decays faster. "
                              "Default 0.7.",
                              typeConverter=TypeConverters.toFloat)
    init = Param(Params._dummy(), "init",
                 "lambda init strategy: 'random' (default) or 'spectral' "
                 "(dense block-aligned anchor-word seed, Arora et al. 2013)",
                 typeConverter=TypeConverters.toString)
    spectralMaxVocab = Param(Params._dummy(), "spectralMaxVocab",
                             "max vocab for the DENSE spectral init (V x V driver "
                             "co-occurrence); the spectralMethod='auto' threshold "
                             "below which dense is used (scalable at/above)",
                             typeConverter=TypeConverters.toInt)
    spectralMethod = Param(Params._dummy(), "spectralMethod",
                           "spectral routing: 'auto'|'dense'|'scalable' "
                           "(auto -> dense if V < spectralMaxVocab else scalable)")
    spectralD = Param(Params._dummy(), "spectralD",
                      "random-projection dim for scalable init (0 = auto: "
                      "min(V, max(K, 1000)))")
    spectralMinDocFreq = Param(Params._dummy(), "spectralMinDocFreq",
                               "min within-group document frequency for a scalable "
                               "anchor candidate")


def _layout(est_or_model) -> DagLayout:
    return DagLayout(est_or_model.getOrDefault("parent"),
                     n_bg=est_or_model.getOrDefault("nBg"),
                     tpn=est_or_model.getOrDefault("tpn"))


class GatedLDAEstimator(_GatedLDAParams, Estimator):
    @keyword_only
    def __init__(self, *, featuresCol="features", labelCol="frontier", parent=None,
                 nBg=2, tpn=1, maxIter=20, seed=None, caviMaxIter=100, caviTol=1e-3,
                 gammaShape=100.0, init="random", spectralMaxVocab=8000,
                 spectralMethod="auto", spectralD=0, spectralMinDocFreq=5,
                 nodeAlphaScale=1.0, miniBatchFraction=0.0,
                 learningRateTau0=1.0, learningRateKappa=0.7):
        super().__init__()
        self._setDefault(featuresCol="features", labelCol="frontier", nBg=2, tpn=1,
                         maxIter=20, nodeAffinityCol="nodeAffinity",
                         caviMaxIter=100, caviTol=1e-3, gammaShape=100.0,
                         init="random", spectralMaxVocab=8000,
                         spectralMethod="auto", spectralD=0, spectralMinDocFreq=5,
                         nodeAlphaScale=1.0,
                         miniBatchFraction=0.0, learningRateTau0=1.0,
                         learningRateKappa=0.7)
        self.setParams(**self._input_kwargs)
        # Diagnostic-only per-iteration callback (mirrors OnlineLDAEstimator).
        # Stored as an instance attribute, not a Param — callables aren't
        # MLlib-serializable and persistence is deferred (ADR 0009).
        self._on_iteration = None

    @keyword_only
    def setParams(self, **kwargs):
        return self._set(**kwargs)

    def setOnIteration(
        self, fn: Callable[[int, dict, list[float]], None] | None,
    ) -> "GatedLDAEstimator":
        """Register a per-iteration diagnostic callback for the next fit.

        Signature: fn(iter_num, global_params, elbo_trace). Runs on the driver in
        the fit's hot path; throttle with a modulo if non-trivial. The callback
        must not mutate global_params (the same dict feeds the next iteration's
        broadcast). Not persisted (callables aren't MLlib-serializable; ADR 0009).
        """
        self._on_iteration = fn
        return self

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
        init = self.getOrDefault("init")

        # Validate init early (fail fast on the driver, not deep in a Spark task).
        from spark_vi.models.topic.gated_init import INIT_STRATEGIES
        if init != "random" and init not in INIT_STRATEGIES:
            raise ValueError(
                f"unknown init strategy {init!r}; "
                f"known: {['random'] + sorted(INIT_STRATEGIES)}"
            )
        # Dirichlet alpha over the K topics. Symmetric 1/K by default; when
        # nodeAlphaScale != 1, the per-node blocks (contiguous topics
        # [n_bg, K), background is [0, n_bg)) are scaled to make them a priori
        # rarer — a block-asymmetric prior (Wallach et al. 2009). OnlineLDA
        # accepts a length-K alpha vector; alpha is fixed (optimize_alpha is
        # disabled for the gated engine), so this vector holds through the fit
        # and into transform.
        node_alpha_scale = float(self.getOrDefault("nodeAlphaScale"))
        alpha_vec = np.full(lay.K, 1.0 / lay.K, dtype=np.float64)
        alpha_vec[lay.n_bg:] *= node_alpha_scale
        model_obj = GatedOnlineLDA(
            lay, V, init=init,
            alpha=alpha_vec, eta=1.0 / lay.K,
            gamma_shape=self.getOrDefault("gammaShape"),
            cavi_max_iter=self.getOrDefault("caviMaxIter"),
            cavi_tol=self.getOrDefault("caviTol"),
            random_seed=seed,
        )
        # SVI schedule: mini_batch_fraction 0.0 (the shim default) -> None ==
        # full-batch (every iteration sees the whole corpus). A value in (0, 1]
        # switches to mini-batch SVI, which is what makes the decaying step size
        # legitimate (a decaying rho over full batches stalls — VIConfig docs).
        mbf = float(self.getOrDefault("miniBatchFraction"))
        config = VIConfig(
            max_iterations=self.getOrDefault("maxIter"), random_seed=seed,
            mini_batch_fraction=(mbf if mbf and mbf > 0.0 else None),
            learning_rate_tau0=float(self.getOrDefault("learningRateTau0")),
            learning_rate_kappa=float(self.getOrDefault("learningRateKappa")),
        )

        def _to_gated(row):
            bow = _vector_to_bow_document(row[0])
            frontier = frozenset(int(x) for x in (row[1] or []))
            return GatedBOWDocument(indices=bow.indices, counts=bow.counts,
                                    length=bow.length, frontier=frontier)

        rdd = (dataset.select(features_col, label_col).rdd.map(_to_gated)
               .persist(StorageLevel.MEMORY_AND_DISK))
        rdd.count()

        # Non-random init: seed lambda from anchor-word spectral recovery. init
        # "spectral" routes dense (collect the corpus to the driver, exact V×V,
        # validated small-V default) vs scalable (distributed random-projection
        # sketch, ADR 0032 — the gated analogue), by spectralMethod. Passed to
        # initialize_global via data_summary; dense hands {train_docs,train_labels},
        # scalable hands a precomputed {spectral_lambda}.
        data_summary = None
        if init != "random":
            from spark_vi.mllib.topic.stm import resolve_spectral_method
            resolved = resolve_spectral_method(
                self.getOrDefault("spectralMethod"), V,
                threshold=self.getOrDefault("spectralMaxVocab"))
            if resolved == "scalable":
                from spark_vi.models.topic.gated_init import (
                    scalable_block_aligned_lambda,
                )
                sd = int(self.getOrDefault("spectralD"))
                lam0 = scalable_block_aligned_lambda(
                    rdd, lay, V,
                    d=(sd if sd > 0 else None),
                    seed=(seed or 0),
                    min_doc_freq=int(self.getOrDefault("spectralMinDocFreq")),
                )
                data_summary = {"spectral_lambda": lam0}
            else:  # dense — collect-to-driver exact path
                collected = dataset.select(features_col, label_col).collect()
                train_docs, train_labels = [], []
                for r in collected:
                    bow = _vector_to_bow_document(r[0])
                    train_docs.append(np.repeat(bow.indices, bow.counts.astype(int)))
                    train_labels.append(frozenset(int(x) for x in (r[1] or [])))
                data_summary = {"train_docs": train_docs, "train_labels": train_labels}

        try:
            result = VIRunner(model_obj, config=config).fit(
                rdd, data_summary=data_summary, on_iteration=self._on_iteration)
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
    # Documents intent for a future _PersistableModel checkpoint/persist mixin; the model
    # does not inherit _PersistableModel in v1, so this attribute is currently inert
    # (persistence is deferred to v2). Kept because it costs nothing and records the plan.
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
