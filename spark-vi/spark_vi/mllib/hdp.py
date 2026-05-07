"""MLlib Estimator/Transformer shim for OnlineHDP.

Wraps spark_vi.models.online_hdp.OnlineHDP + spark_vi.core.runner.VIRunner so
the model behaves like a pyspark.ml.clustering.LDA-shaped Estimator/Model
pair. The shim is a translation layer; all SVI/CAVI logic lives in
OnlineHDP. See ADR 0012 (docs/decisions/0012-hdp-mllib-shim.md) for the
design rationale; ADR 0011 covers the underlying model.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
from pyspark import StorageLevel, keyword_only
from pyspark.ml.base import Estimator, Model
from pyspark.ml.param import Param, Params, TypeConverters
from pyspark.ml.param.shared import HasFeaturesCol, HasMaxIter, HasSeed

from spark_vi.core.config import VIConfig
from spark_vi.mllib._common import _vector_to_bow_document
from spark_vi.models.online_hdp import (
    OnlineHDP,
    expected_corpus_betas,
    topic_count_at_mass,
)


def _validate_unsupported_params(estimator: "OnlineHDPEstimator") -> None:
    """Raise ValueError for any configuration the shim cannot honor.

    Per ADR 0011, γ/α optimization is deferred — there are no optimize*
    flags on this shim, so the only rejections are genuinely-unsupported
    values:

      * optimizer != "online" — we are SVI-only.
      * vector docConcentration — HDP α is scalar (paper Eq 9, doc-stick
        concentration). Vector α has no derivation in v1.
      * vector topicConcentration — HDP η is scalar symmetric Dirichlet
        on the topic-word prior.

    Silent fallback would mislead users about what they are getting.
    """
    optimizer = estimator.getOrDefault("optimizer")
    if optimizer != "online":
        raise ValueError(
            f"OnlineHDPEstimator only supports optimizer='online', got {optimizer!r}. "
            f"The 'em' optimizer is not implemented in this shim."
        )

    if estimator.isSet("docConcentration"):
        doc_conc = estimator.getOrDefault("docConcentration")
        if doc_conc is not None and len(doc_conc) > 1:
            raise ValueError(
                "OnlineHDPEstimator only supports scalar docConcentration "
                f"(α); got vector of length {len(doc_conc)}. ADR 0011 keeps α "
                "scalar in v1 (γ/α optimization deferred)."
            )

    if estimator.isSet("topicConcentration"):
        topic_conc = estimator.getOrDefault("topicConcentration")
        # topicConcentration is typed as float, not list — but be defensive.
        if topic_conc is not None and isinstance(topic_conc, (list, tuple)):
            raise ValueError(
                "OnlineHDPEstimator only supports scalar topicConcentration (η)."
            )


def _build_model_and_config(
    estimator: "OnlineHDPEstimator",
    vocab_size: int,
) -> tuple[OnlineHDP, VIConfig]:
    """Translate Estimator Params into (OnlineHDP, VIConfig).

    Param mapping (see ADR 0012):
      k                       → T (corpus truncation)
      docTruncation           → K (doc truncation)
      docConcentration[0]     → α (scalar; default 1.0 if unset)
      corpusConcentration     → γ (scalar; default 1.0 if unset)
      topicConcentration      → η (scalar; default 0.01 if unset)
    """
    T = estimator.getOrDefault("k")
    K = estimator.getOrDefault("docTruncation")

    doc_conc = estimator.getOrDefault("docConcentration") if estimator.isSet("docConcentration") else None
    if doc_conc is None:
        alpha = 1.0
    else:
        # Length-1 list (validated by _validate_unsupported_params).
        alpha = float(doc_conc[0])

    gamma = (
        float(estimator.getOrDefault("corpusConcentration"))
        if estimator.isSet("corpusConcentration")
        else 1.0
    )
    eta = (
        float(estimator.getOrDefault("topicConcentration"))
        if estimator.isSet("topicConcentration")
        else 0.01
    )

    model = OnlineHDP(
        T=T,
        K=K,
        vocab_size=vocab_size,
        alpha=alpha,
        gamma=gamma,
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


class _OnlineHDPParams(HasFeaturesCol, HasMaxIter, HasSeed):
    """Shared Param surface for OnlineHDPEstimator and OnlineHDPModel.

    Mirrors MLlib's `_LDAParams` mixin pattern: declare each Param once,
    inherit from both the Estimator and Model so they expose identical
    surfaces with no aliasing.
    """

    k = Param(
        Params._dummy(), "k",
        "corpus-level truncation T (upper bound on discoverable topics); >= 2",
        typeConverter=TypeConverters.toInt,
    )
    docTruncation = Param(
        Params._dummy(), "docTruncation",
        "doc-level truncation K (upper bound on topics per document); >= 2",
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
        "doc-stick concentration α (scalar only); see ADR 0011 / 0012",
        typeConverter=TypeConverters.toListFloat,
    )
    corpusConcentration = Param(
        Params._dummy(), "corpusConcentration",
        "corpus-stick concentration γ (scalar); HDP-specific extra (no LDA analog)",
        typeConverter=TypeConverters.toFloat,
    )
    topicConcentration = Param(
        Params._dummy(), "topicConcentration",
        "Dirichlet concentration η on topic-word; scalar (symmetric) only",
        typeConverter=TypeConverters.toFloat,
    )
    gammaShape = Param(
        Params._dummy(), "gammaShape",
        "shape parameter for Gamma init of variational λ; ADR 0011 default 100.0",
        typeConverter=TypeConverters.toFloat,
    )
    caviMaxIter = Param(
        Params._dummy(), "caviMaxIter",
        "max iterations for the inner CAVI loop per document",
        typeConverter=TypeConverters.toInt,
    )
    caviTol = Param(
        Params._dummy(), "caviTol",
        "relative tolerance on per-iter ELBO for doc-CAVI early stop",
        typeConverter=TypeConverters.toFloat,
    )


class OnlineHDPEstimator(_OnlineHDPParams, Estimator):
    """MLlib-shaped Estimator wrapping spark_vi.models.online_hdp.OnlineHDP.

    Param defaults mirror pyspark.ml.clustering.LDA for the shared subset
    and ADR 0011 for HDP-specific extras (corpusConcentration, docTruncation,
    topicConcentration, gammaShape, caviMaxIter, caviTol).

    `k` here is the corpus truncation T (upper bound on discoverable topics);
    `docTruncation` is the doc truncation K (typically much smaller than T).
    """

    @keyword_only
    def __init__(
        self,
        *,
        k: int = 150,
        docTruncation: int = 15,
        maxIter: int = 20,
        seed: int | None = None,
        featuresCol: str = "features",
        topicDistributionCol: str = "topicDistribution",
        optimizer: str = "online",
        learningOffset: float = 1024.0,
        learningDecay: float = 0.51,
        subsamplingRate: float = 0.05,
        docConcentration: list[float] | None = None,
        corpusConcentration: float | None = None,
        topicConcentration: float | None = None,
        gammaShape: float = 100.0,
        caviMaxIter: int = 100,
        caviTol: float = 1e-4,
    ) -> None:
        super().__init__()
        self._setDefault(
            k=150, docTruncation=15, maxIter=20,
            featuresCol="features", topicDistributionCol="topicDistribution",
            optimizer="online",
            learningOffset=1024.0, learningDecay=0.51, subsamplingRate=0.05,
            gammaShape=100.0, caviMaxIter=100, caviTol=1e-4,
        )
        # Diagnostic-only iteration callback. Stored as an instance attribute
        # rather than a Param because callables aren't MLlib-serializable
        # (Pipeline.save persistence is deferred per ADR 0012 anyway).
        self._on_iteration = None
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, **kwargs) -> "OnlineHDPEstimator":
        """Standard MLlib pattern: set any subset of params after construction."""
        return self._set(**kwargs)

    def setOnIteration(
        self,
        fn: Callable[[int, dict, list[float]], None] | None,
    ) -> "OnlineHDPEstimator":
        """Register a per-iteration diagnostic callback for the next fit.

        Signature: fn(iter_num, global_params, elbo_trace). Runs on the driver
        in the fit's hot path; throttle with a modulo if non-trivial. The
        callback must not mutate global_params — the same dict feeds the next
        iteration's broadcast. Not persisted by Pipeline.save (callables
        aren't MLlib-serializable; persistence deferred per ADR 0012).
        """
        self._on_iteration = fn
        return self

    def _fit(self, dataset) -> "OnlineHDPModel":
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

        out_model = OnlineHDPModel(
            result,
            T=model_obj.T, K=model_obj.K,
            alpha=model_obj.alpha,
            gamma=model_obj.gamma,
            eta=model_obj.eta,
        )
        # Copy every Param value the Estimator has set or has a default for, so
        # the Model's getters reflect the configuration that produced it.
        for param in self.params:
            if self.isSet(param):
                out_model._set(**{param.name: self.getOrDefault(param)})
            elif self.hasDefault(param):
                out_model._setDefault(**{param.name: self.getOrDefault(param)})
        return out_model


class OnlineHDPModel(_OnlineHDPParams, Model):
    """MLlib-shaped Model wrapping a trained spark_vi VIResult.

    Carries the trained global parameters (λ, u, v) plus a copy of every
    Param from the Estimator that produced it, so post-fit getters
    (model.getK(), model.getDocTruncation(), ...) return the configuration
    that was actually used.
    """

    def __init__(
        self,
        result,                              # VIResult
        *,
        T: int,
        K: int,
        alpha: float,
        gamma: float,
        eta: float,
    ) -> None:
        super().__init__()
        self._result = result
        self._T = int(T)
        self._K = int(K)
        # Stash trained scalars as private attrs so the same-named
        # @property accessors below don't have to round-trip through the
        # MLlib Param machinery (which would recurse: property body calls
        # isSet → _resolveParam → getParam → getattr → property body...).
        # In v1 these equal the constructor inputs (no γ/α/η optimization
        # per ADR 0011); v3 will surface optimized values here.
        self._alpha = float(alpha)
        self._gamma = float(gamma)
        self._eta = float(eta)

    @property
    def result(self):
        """The trained VIResult (global_params, elbo_trace, n_iterations, ...)."""
        return self._result

    # Trained-scalar accessors. Methods rather than @property so the names
    # don't collide with the underlying Param descriptors (which would
    # break MLlib's `_set`/`_setDefault` resolution by name). This also
    # matches pyspark.ml.clustering.LDAModel which exposes
    # docConcentration() / topicConcentration() as methods. v1 returns the
    # constructor inputs unchanged (no optimization per ADR 0011).
    def trainedAlpha(self) -> float:
        """Trained α scalar (doc-stick concentration)."""
        return self._alpha

    def trainedCorpusConcentration(self) -> float:
        """Trained γ scalar (corpus-stick concentration). HDP-specific —
        no LDA analog."""
        return self._gamma

    def trainedTopicConcentration(self) -> float:
        """Trained η scalar (topic-word Dirichlet concentration)."""
        return self._eta

    def vocabSize(self) -> int:
        """V dimension of the trained lambda."""
        return int(self._result.global_params["lambda"].shape[1])

    def corpusStickWeights(self) -> np.ndarray:
        """E[β_t] vector (length T) under the mean-field variational posterior.

        Surfaces the effective topic prior so callers can rank/filter active
        topics. Exact mean under the mean-field q (see _expected_corpus_betas),
        not an approximation.
        """
        u = self._result.global_params["u"]
        v = self._result.global_params["v"]
        return expected_corpus_betas(u, v, T=self._T)

    def activeTopicCount(self, mass_threshold: float = 0.95) -> int:
        """Smallest count of topics whose top-ranked E[β_t] sum to ≥ mass_threshold.

        Truncation-independent: as long as T is ≥ the true effective topic
        count, the answer doesn't change with T. More robust than a fixed
        threshold like 1/(2T), which scales with the truncation knob.

        Default 0.95 is the "explained-variance" analog from PCA — count of
        topics needed to cover 95% of corpus-level prior probability.
        """
        return topic_count_at_mass(self.corpusStickWeights(), mass_threshold)

    def topicsMatrix(self):
        """Topic-word distribution as an MLlib DenseMatrix of shape (V, T).

        Internally we keep λ as (T, V); the transpose-and-normalize here
        matches MLlib's convention where `topicsMatrix` is indexed by
        (vocab term, topic). Returns the *full* T topics — filtering inactive
        ones via activeTopicCount() / corpusStickWeights() is a caller decision.
        """
        from pyspark.ml.linalg import DenseMatrix

        lam = self._result.global_params["lambda"]
        beta = lam / lam.sum(axis=1, keepdims=True)            # (T, V)
        T, V = beta.shape
        # DenseMatrix expects column-major flattened values.
        return DenseMatrix(numRows=V, numCols=T, values=beta.T.flatten("F").tolist())

    def describeTopics(self, maxTermsPerTopic: int = 10):
        """DataFrame of (topic, termIndices, termWeights) — top terms per topic.

        Schema and orientation match pyspark.ml.clustering.LDAModel.describeTopics.
        Reports all T topics; pair with corpusStickWeights() if the caller
        wants to filter to active topics only.
        """
        from pyspark.sql import SparkSession
        from pyspark.sql.types import (
            ArrayType, DoubleType, IntegerType, StructField, StructType,
        )

        if maxTermsPerTopic < 1:
            raise ValueError(f"maxTermsPerTopic must be >= 1, got {maxTermsPerTopic}")

        lam = self._result.global_params["lambda"]
        beta = lam / lam.sum(axis=1, keepdims=True)            # (T, V)
        T, V = beta.shape
        m = min(maxTermsPerTopic, V)

        rows = []
        for t in range(T):
            order = np.argsort(beta[t])[::-1][:m]
            rows.append((
                int(t),
                [int(i) for i in order],
                [float(beta[t, i]) for i in order],
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

        # Reconstruct the OnlineHDP instance for infer_local. We rebuild it
        # from the model's Params + trained globals rather than carrying the
        # original Python object across the fit boundary so the Model is
        # reconstructible from VIResult alone (matching the LDA shim's
        # transform-from-globals pattern).
        T = self._T
        K = self._K
        V = self.vocabSize()
        alpha = self.trainedAlpha()
        eta = self.trainedTopicConcentration()
        gamma = self.trainedCorpusConcentration()
        gamma_shape = float(self.getOrDefault("gammaShape"))
        cavi_max_iter = int(self.getOrDefault("caviMaxIter"))
        cavi_tol = float(self.getOrDefault("caviTol"))

        global_params = self._result.global_params

        sc = dataset.sparkSession.sparkContext
        bcast = sc.broadcast({
            "global_params": global_params,
            "T": T, "K": K, "V": V,
            "alpha": alpha, "eta": eta, "gamma": gamma,
            "gamma_shape": gamma_shape,
            "cavi_max_iter": cavi_max_iter,
            "cavi_tol": cavi_tol,
        })

        def _infer(features):
            params = bcast.value
            doc = _vector_to_bow_document(features)
            # Rebuild the model on the executor; cheap (just stores scalars
            # and references to the broadcast globals through closure).
            model = OnlineHDP(
                T=params["T"], K=params["K"], vocab_size=params["V"],
                alpha=params["alpha"], gamma=params["gamma"], eta=params["eta"],
                gamma_shape=params["gamma_shape"],
                cavi_max_iter=params["cavi_max_iter"],
                cavi_tol=params["cavi_tol"],
            )
            out = model.infer_local(doc, params["global_params"])
            return DenseVector(out["theta"])

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
            "OnlineHDPModel.result.elbo_trace."
        )

    def logPerplexity(self, dataset):
        raise NotImplementedError(
            "logPerplexity is not implemented in this v1 shim. The training-time "
            "ELBO trace is available on the underlying VIResult via "
            "OnlineHDPModel.result.elbo_trace."
        )
