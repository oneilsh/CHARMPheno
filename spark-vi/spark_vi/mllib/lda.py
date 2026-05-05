"""MLlib Estimator/Transformer shim for VanillaLDA.

Wraps spark_vi.models.lda.VanillaLDA + spark_vi.core.runner.VIRunner so the
model behaves like a pyspark.ml.clustering.LDA-shaped Estimator/Model pair.
The shim is a translation layer; all SVI logic lives in VanillaLDA. See
docs/superpowers/specs/2026-05-04-mllib-shim-design.md and ADR 0009.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
from pyspark import StorageLevel, keyword_only
from pyspark.ml.base import Estimator, Model
from pyspark.ml.linalg import DenseVector, SparseVector, Vector
from pyspark.ml.param import Param, Params, TypeConverters
from pyspark.ml.param.shared import HasFeaturesCol, HasMaxIter, HasSeed

from spark_vi.core.config import VIConfig
from spark_vi.core.types import BOWDocument
from spark_vi.models.lda import VanillaLDA


def _vector_to_bow_document(v: Vector) -> BOWDocument:
    """Convert a pyspark.ml.linalg Vector to a BOWDocument.

    SparseVector indices/values pass through. DenseVectors are sparsified
    (nonzero entries only) so the downstream CAVI loop sees the same shape
    of input regardless of the producer (CountVectorizer emits Sparse,
    user-constructed inputs may be Dense).
    """
    if isinstance(v, SparseVector):
        indices = np.asarray(v.indices, dtype=np.int32)
        counts = np.asarray(v.values, dtype=np.float64)
    elif isinstance(v, DenseVector):
        values = np.asarray(v.values, dtype=np.float64)
        nz = np.nonzero(values)[0].astype(np.int32)
        indices = nz
        counts = values[nz]
    else:
        raise TypeError(
            f"_vector_to_bow_document expected Sparse/DenseVector, got {type(v).__name__}"
        )
    return BOWDocument(indices=indices, counts=counts, length=int(counts.sum()))


def _validate_unsupported_params(estimator: "VanillaLDAEstimator") -> None:
    """Raise ValueError for any configuration the shim cannot honor.

    Per ADR 0010 the v0 rejections of `optimizeDocConcentration=True` and
    vector `docConcentration` are gone — both are now first-class. The
    only remaining rejections are the genuinely unsupported ones:

      * optimizer != "online" — we are SVI-only.
      * vector docConcentration with length != k — the model demands a
        length-k vector when asymmetric.

    Silent fallback would mislead users about what they are getting.
    """
    optimizer = estimator.getOrDefault("optimizer")
    if optimizer != "online":
        raise ValueError(
            f"VanillaLDAEstimator only supports optimizer='online', got {optimizer!r}. "
            f"The 'em' optimizer is not implemented in this shim."
        )

    if estimator.isSet("docConcentration"):
        doc_conc = estimator.getOrDefault("docConcentration")
        if doc_conc is not None and len(doc_conc) > 1:
            k = estimator.getOrDefault("k")
            if len(doc_conc) != k:
                raise ValueError(
                    f"docConcentration vector must have length k={k}, "
                    f"got length {len(doc_conc)}."
                )


def _build_model_and_config(
    estimator: "VanillaLDAEstimator",
    vocab_size: int,
) -> tuple[VanillaLDA, VIConfig]:
    """Translate Estimator Params into (VanillaLDA, VIConfig).

    Symmetric-alpha-only: docConcentration may be None or a length-1 list
    (the latter is what TypeConverters.toListFloat produces for a scalar
    input). Vector docConcentration is rejected by _validate_unsupported_params,
    not here.
    """
    k = estimator.getOrDefault("k")

    doc_conc = estimator.getOrDefault("docConcentration") if estimator.isSet("docConcentration") else None
    if doc_conc is None:
        alpha = 1.0 / k
    else:
        alpha = float(doc_conc[0])

    topic_conc = estimator.getOrDefault("topicConcentration") if estimator.isSet("topicConcentration") else None
    eta = 1.0 / k if topic_conc is None else float(topic_conc)

    model = VanillaLDA(
        K=k,
        vocab_size=vocab_size,
        alpha=alpha,
        eta=eta,
        gamma_shape=estimator.getOrDefault("gammaShape"),
        cavi_max_iter=estimator.getOrDefault("caviMaxIter"),
        cavi_tol=estimator.getOrDefault("caviTol"),
    )

    seed = estimator.getOrDefault("seed") if estimator.isSet("seed") else None
    config = VIConfig(
        max_iterations=estimator.getOrDefault("maxIter"),
        learning_rate_tau0=estimator.getOrDefault("learningOffset"),
        learning_rate_kappa=estimator.getOrDefault("learningDecay"),
        mini_batch_fraction=estimator.getOrDefault("subsamplingRate"),
        random_seed=seed,
    )
    return model, config


class _VanillaLDAParams(HasFeaturesCol, HasMaxIter, HasSeed):
    """Shared Param surface for VanillaLDAEstimator and VanillaLDAModel.

    Mirrors MLlib's `_LDAParams` mixin pattern: declare each Param once,
    inherit from both the Estimator and Model so they expose identical
    surfaces with no aliasing.
    """

    k = Param(
        Params._dummy(), "k",
        "number of topics (clusters) to infer; must be >= 1",
        typeConverter=TypeConverters.toInt,
    )
    topicDistributionCol = Param(
        Params._dummy(), "topicDistributionCol",
        "output column with estimates of topic mixture for each document",
        typeConverter=TypeConverters.toString,
    )
    optimizer = Param(
        Params._dummy(), "optimizer",
        "optimizer; only 'online' is supported by this shim",
        typeConverter=TypeConverters.toString,
    )
    learningOffset = Param(
        Params._dummy(), "learningOffset",
        "tau0 in the Robbins-Monro step rho_t = (tau0 + t)^-kappa",
        typeConverter=TypeConverters.toFloat,
    )
    learningDecay = Param(
        Params._dummy(), "learningDecay",
        "kappa in the Robbins-Monro step rho_t = (tau0 + t)^-kappa",
        typeConverter=TypeConverters.toFloat,
    )
    subsamplingRate = Param(
        Params._dummy(), "subsamplingRate",
        "fraction of corpus sampled per mini-batch",
        typeConverter=TypeConverters.toFloat,
    )
    docConcentration = Param(
        Params._dummy(), "docConcentration",
        "Dirichlet concentration alpha on theta; scalar (symmetric) only — vector raises",
        typeConverter=TypeConverters.toListFloat,
    )
    topicConcentration = Param(
        Params._dummy(), "topicConcentration",
        "Dirichlet concentration eta on beta; scalar (symmetric) only",
        typeConverter=TypeConverters.toFloat,
    )
    optimizeDocConcentration = Param(
        Params._dummy(), "optimizeDocConcentration",
        "whether to optimize alpha; True is rejected (see ADR 0008)",
        typeConverter=TypeConverters.toBoolean,
    )
    optimizeTopicConcentration = Param(
        Params._dummy(), "optimizeTopicConcentration",
        "whether to optimize η (symmetric scalar) via Newton-Raphson; "
        "see ADR 0010",
        typeConverter=TypeConverters.toBoolean,
    )
    gammaShape = Param(
        Params._dummy(), "gammaShape",
        "shape parameter for Gamma init of variational gamma; ADR 0008 default 100.0",
        typeConverter=TypeConverters.toFloat,
    )
    caviMaxIter = Param(
        Params._dummy(), "caviMaxIter",
        "max iterations for the inner CAVI loop per document",
        typeConverter=TypeConverters.toInt,
    )
    caviTol = Param(
        Params._dummy(), "caviTol",
        "relative tolerance on gamma for CAVI early stop",
        typeConverter=TypeConverters.toFloat,
    )


class VanillaLDAEstimator(_VanillaLDAParams, Estimator):
    """MLlib-shaped Estimator wrapping spark_vi.models.lda.VanillaLDA.

    Param defaults mirror pyspark.ml.clustering.LDA for the shared subset
    and ADR 0008 for our extras (gammaShape, caviMaxIter, caviTol).
    """

    @keyword_only
    def __init__(
        self,
        *,
        k: int = 10,
        maxIter: int = 20,
        seed: int | None = None,
        featuresCol: str = "features",
        topicDistributionCol: str = "topicDistribution",
        optimizer: str = "online",
        learningOffset: float = 1024.0,
        learningDecay: float = 0.51,
        subsamplingRate: float = 0.05,
        docConcentration: list[float] | None = None,
        topicConcentration: float | None = None,
        optimizeDocConcentration: bool = False,
        optimizeTopicConcentration: bool = False,
        gammaShape: float = 100.0,
        caviMaxIter: int = 100,
        caviTol: float = 1e-3,
    ) -> None:
        super().__init__()
        self._setDefault(
            k=10, maxIter=20,
            featuresCol="features", topicDistributionCol="topicDistribution",
            optimizer="online",
            learningOffset=1024.0, learningDecay=0.51, subsamplingRate=0.05,
            optimizeDocConcentration=False,
            optimizeTopicConcentration=False,
            gammaShape=100.0, caviMaxIter=100, caviTol=1e-3,
        )
        # Diagnostic-only iteration callback. Stored as an instance attribute
        # rather than a Param because callables aren't MLlib-serializable
        # (Pipeline.save persistence is deferred per ADR 0009 anyway).
        self._on_iteration = None
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, **kwargs) -> "VanillaLDAEstimator":
        """Standard MLlib pattern: set any subset of params after construction."""
        return self._set(**kwargs)

    def setOnIteration(
        self,
        fn: Callable[[int, dict, list[float]], None] | None,
    ) -> "VanillaLDAEstimator":
        """Register a per-iteration diagnostic callback for the next fit.

        Signature: fn(iter_num, global_params, elbo_trace). Runs on the driver
        in the fit's hot path; throttle with a modulo if non-trivial. The
        callback must not mutate global_params — the same dict feeds the next
        iteration's broadcast. Not persisted by Pipeline.save (callables
        aren't MLlib-serializable; persistence deferred per ADR 0009).
        """
        self._on_iteration = fn
        return self

    def _fit(self, dataset) -> "VanillaLDAModel":
        from spark_vi.core.runner import VIRunner

        _validate_unsupported_params(self)

        first_features = dataset.select(self.getOrDefault("featuresCol")).head(1)
        if not first_features:
            raise ValueError("Cannot fit on an empty DataFrame.")
        vocab_size = first_features[0][0].size

        model_obj, config = _build_model_and_config(self, vocab_size=vocab_size)

        features_col = self.getOrDefault("featuresCol")
        bow_rdd = (
            dataset.select(features_col).rdd
            .map(lambda row: _vector_to_bow_document(row[0]))
        )
        # `dataset.rdd.map(...)` builds a fresh, uncached RDD even when
        # `dataset` is DataFrame-cached upstream. The runner's strict
        # assert_persisted precondition requires *this* RDD to be in cache,
        # so persist + an action here. The action also pays the BoWDocument
        # conversion cost once instead of every iteration.
        bow_rdd = bow_rdd.persist(StorageLevel.MEMORY_AND_DISK)
        bow_rdd.count()

        runner = VIRunner(model_obj, config=config)
        try:
            result = runner.fit(bow_rdd, on_iteration=self._on_iteration)
        finally:
            bow_rdd.unpersist(blocking=False)

        out_model = VanillaLDAModel(result)
        # Copy every Param value the Estimator has set or has a default for, so
        # the Model's getters reflect the configuration that produced it.
        for param in self.params:
            if self.isSet(param):
                out_model._set(**{param.name: self.getOrDefault(param)})
            elif self.hasDefault(param):
                out_model._setDefault(**{param.name: self.getOrDefault(param)})
        return out_model


class VanillaLDAModel(_VanillaLDAParams, Model):
    """MLlib-shaped Model wrapping a trained spark_vi VIResult.

    Carries the trained global parameters plus a copy of every Param from
    the Estimator that produced it, so post-fit getters (model.getK(), ...)
    return the configuration that was actually used.
    """

    def __init__(self, result) -> None:  # result: VIResult
        super().__init__()
        self._result = result

    @property
    def result(self):
        """The trained VIResult (global_params, elbo_trace, n_iterations, ...)."""
        return self._result

    def vocabSize(self) -> int:
        """V dimension of the trained lambda."""
        return int(self._result.global_params["lambda"].shape[1])

    def topicsMatrix(self):
        """Topic-word distribution as an MLlib DenseMatrix of shape (V, K).

        Internally we keep lambda as (K, V); the transpose-and-normalize
        here matches MLlib's convention where `topicsMatrix` is indexed
        by (vocab term, topic).
        """
        from pyspark.ml.linalg import DenseMatrix

        lam = self._result.global_params["lambda"]
        beta = lam / lam.sum(axis=1, keepdims=True)  # (K, V), row-stochastic
        K, V = beta.shape
        # DenseMatrix expects column-major flattened values.
        return DenseMatrix(numRows=V, numCols=K, values=beta.T.flatten("F").tolist())

    def describeTopics(self, maxTermsPerTopic: int = 10):
        """DataFrame of (topic, termIndices, termWeights) — top terms per topic.

        Schema and orientation match pyspark.ml.clustering.LDAModel.describeTopics.
        """
        from pyspark.sql import SparkSession
        from pyspark.sql.types import (
            ArrayType, DoubleType, IntegerType, StructField, StructType,
        )

        if maxTermsPerTopic < 1:
            raise ValueError(f"maxTermsPerTopic must be >= 1, got {maxTermsPerTopic}")

        lam = self._result.global_params["lambda"]
        beta = lam / lam.sum(axis=1, keepdims=True)  # (K, V), row-stochastic
        K, V = beta.shape
        m = min(maxTermsPerTopic, V)

        rows = []
        for k in range(K):
            order = np.argsort(beta[k])[::-1][:m]
            rows.append((
                int(k),
                [int(i) for i in order],
                [float(beta[k, i]) for i in order],
            ))

        schema = StructType([
            StructField("topic", IntegerType(), False),
            StructField("termIndices", ArrayType(IntegerType(), False), False),
            StructField("termWeights", ArrayType(DoubleType(), False), False),
        ])
        return SparkSession.builder.getOrCreate().createDataFrame(rows, schema=schema)

    def _transform(self, dataset):
        from pyspark.ml.linalg import DenseVector, VectorUDT
        from pyspark.sql import functions as F
        from scipy.special import digamma

        from spark_vi.models.lda import _cavi_doc_inference

        lam = self._result.global_params["lambda"]
        expElogbeta = np.exp(digamma(lam) - digamma(lam.sum(axis=1, keepdims=True)))

        # docConcentration may be unset (None default) → resolve to 1/k.
        if self.isSet("docConcentration") and self.getOrDefault("docConcentration") is not None:
            alpha = float(self.getOrDefault("docConcentration")[0])
        else:
            alpha = 1.0 / self.getOrDefault("k")
        gamma_shape = float(self.getOrDefault("gammaShape"))
        cavi_max_iter = int(self.getOrDefault("caviMaxIter"))
        cavi_tol = float(self.getOrDefault("caviTol"))
        K = expElogbeta.shape[0]

        sc = dataset.sparkSession.sparkContext
        bcast = sc.broadcast({
            "expElogbeta": expElogbeta,
            "alpha": alpha,
            "gamma_shape": gamma_shape,
            "cavi_max_iter": cavi_max_iter,
            "cavi_tol": cavi_tol,
            "K": K,
        })

        def _infer(features):
            params = bcast.value
            doc = _vector_to_bow_document(features)
            rng = np.random.default_rng()
            gamma_init = rng.gamma(
                shape=params["gamma_shape"],
                scale=1.0 / params["gamma_shape"],
                size=params["K"],
            )
            gamma, _, _, _ = _cavi_doc_inference(
                indices=doc.indices,
                counts=doc.counts,
                expElogbeta=params["expElogbeta"],
                alpha=params["alpha"],
                gamma_init=gamma_init,
                max_iter=params["cavi_max_iter"],
                tol=params["cavi_tol"],
            )
            return DenseVector(gamma / gamma.sum())

        infer_udf = F.udf(_infer, returnType=VectorUDT())

        try:
            out_col = self.getOrDefault("topicDistributionCol")
            features_col = self.getOrDefault("featuresCol")
            return dataset.withColumn(out_col, infer_udf(F.col(features_col)))
        finally:
            bcast.unpersist(blocking=False)

    def logLikelihood(self, dataset):
        raise NotImplementedError(
            "logLikelihood is not implemented in this v1 shim. The training-time "
            "ELBO trace is available on the underlying VIResult via "
            "VanillaLDAModel.result.elbo_trace."
        )

    def logPerplexity(self, dataset):
        raise NotImplementedError(
            "logPerplexity is not implemented in this v1 shim. The training-time "
            "ELBO trace is available on the underlying VIResult via "
            "VanillaLDAModel.result.elbo_trace."
        )
