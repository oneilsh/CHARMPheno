"""MLlib Estimator/Transformer shim for OnlineHDP.

Wraps spark_vi.models.online_hdp.OnlineHDP + spark_vi.core.runner.VIRunner so
the model behaves like a pyspark.ml.clustering.LDA-shaped Estimator/Model
pair. The shim is a translation layer; all SVI/CAVI logic lives in
OnlineHDP. See ADR 0012 (docs/decisions/0012-hdp-mllib-shim.md) for the
design rationale; ADR 0011 covers the underlying model.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path
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

    Per ADR 0013, γ/α/η optimization flags are now first-class. Remaining
    rejections are genuinely-unsupported values:

      * optimizer != "online" — we are SVI-only.
      * vector docConcentration — HDP α is scalar (paper Eq 9, doc-stick
        concentration). The closed-form M-step in ADR 0013 produces a
        single scalar; vector α has no derivation in HDP.
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

    Param mapping (see ADRs 0012, 0013):
      k                            → T (corpus truncation)
      docTruncation                → K (doc truncation)
      docConcentration[0]          → α (scalar; default 1.0 if unset)
      corpusConcentration          → γ (scalar; default 1.0 if unset)
      topicConcentration           → η (scalar; default 0.01 if unset)
      optimizeDocConcentration     → optimize_alpha   (bool; default True)
      optimizeCorpusConcentration  → optimize_gamma   (bool; default True)
      optimizeTopicConcentration   → optimize_eta     (bool; default False)
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
        optimize_gamma=bool(estimator.getOrDefault("optimizeCorpusConcentration")),
        optimize_alpha=bool(estimator.getOrDefault("optimizeDocConcentration")),
        optimize_eta=bool(estimator.getOrDefault("optimizeTopicConcentration")),
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
    optimizeDocConcentration = Param(
        Params._dummy(), "optimizeDocConcentration",
        "if True, run a closed-form M-step on doc-stick concentration α each "
        "iteration; default True. See ADR 0013.",
        typeConverter=TypeConverters.toBoolean,
    )
    optimizeCorpusConcentration = Param(
        Params._dummy(), "optimizeCorpusConcentration",
        "if True, run a closed-form M-step on corpus-stick concentration γ each "
        "iteration; default True. HDP-specific extra (no MLlib LDA analog). "
        "See ADR 0013.",
        typeConverter=TypeConverters.toBoolean,
    )
    optimizeTopicConcentration = Param(
        Params._dummy(), "optimizeTopicConcentration",
        "if True, run a scalar Newton step on topic-word concentration η each "
        "iteration; default False (matches LDA — least stable in SVI per "
        "Hoffman 2010 §3.4). See ADR 0013.",
        typeConverter=TypeConverters.toBoolean,
    )
    saveInterval = Param(
        Params._dummy(), "saveInterval",
        "Save every N iters during fit. -1 (default) = off. When > 0 and "
        "saveDir is set, the runner writes a VIResult checkpoint every N "
        "iterations. The directory is also written once at end-of-fit "
        "regardless of where the iteration count falls.",
        typeConverter=TypeConverters.toInt,
    )
    saveDir = Param(
        Params._dummy(), "saveDir",
        "Directory for auto-saves. Empty (default) = no auto-save. When "
        "set, fit writes a VIResult on completion (and at every "
        "saveInterval iters if that is also set). The directory is the "
        "authoritative post-fit artifact — load via OnlineHDPModel.load(...).",
        typeConverter=TypeConverters.toString,
    )
    resumeFrom = Param(
        Params._dummy(), "resumeFrom",
        "Path to a previously-written save dir. Empty (default) = fresh "
        "start. When set, fit loads the saved VIResult and continues from "
        "that iteration count, preserving Robbins-Monro continuity and "
        "ELBO trace.",
        typeConverter=TypeConverters.toString,
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
        optimizeDocConcentration: bool = True,
        optimizeCorpusConcentration: bool = True,
        optimizeTopicConcentration: bool = False,
    ) -> None:
        super().__init__()
        self._setDefault(
            k=150, docTruncation=15, maxIter=20,
            featuresCol="features", topicDistributionCol="topicDistribution",
            optimizer="online",
            learningOffset=1024.0, learningDecay=0.51, subsamplingRate=0.05,
            gammaShape=100.0, caviMaxIter=100, caviTol=1e-4,
            optimizeDocConcentration=True,
            optimizeCorpusConcentration=True,
            optimizeTopicConcentration=False,
            saveInterval=-1, saveDir="", resumeFrom="",
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

    def setSaveInterval(self, value: int) -> "OnlineHDPEstimator":
        """Set saveInterval (iterations between auto-saves; -1 disables)."""
        return self._set(saveInterval=value)

    def getSaveInterval(self) -> int:
        return int(self.getOrDefault("saveInterval"))

    def setSaveDir(self, value: str) -> "OnlineHDPEstimator":
        """Set saveDir (auto-save directory; empty disables)."""
        return self._set(saveDir=value)

    def getSaveDir(self) -> str:
        return str(self.getOrDefault("saveDir"))

    def setResumeFrom(self, value: str) -> "OnlineHDPEstimator":
        """Set resumeFrom (path to a previously-written save dir; empty = fresh start)."""
        return self._set(resumeFrom=value)

    def getResumeFrom(self) -> str:
        return str(self.getOrDefault("resumeFrom"))

    def _fit(self, dataset) -> "OnlineHDPModel":
        from spark_vi.core.runner import VIRunner

        _validate_unsupported_params(self)

        # Read & validate the persistence Params before any expensive setup.
        save_interval = int(self.getOrDefault("saveInterval"))
        save_dir = str(self.getOrDefault("saveDir"))
        resume_from = str(self.getOrDefault("resumeFrom"))
        if save_interval == 0:
            raise ValueError(
                "saveInterval=0 is not meaningful; use -1 to disable saves"
            )
        if save_interval > 0 and save_dir == "":
            raise ValueError(
                "saveInterval > 0 requires saveDir to be set"
            )
        if resume_from != "" and not (Path(resume_from) / "manifest.json").exists():
            raise FileNotFoundError(
                f"No manifest.json at resumeFrom path: {resume_from}"
            )

        first_features = dataset.select(self.getOrDefault("featuresCol")).head(1)
        if not first_features:
            raise ValueError("Cannot fit on an empty DataFrame.")
        vocab_size = first_features[0][0].size

        model_obj, config = _build_model_and_config(self, vocab_size=vocab_size)

        # Splice the save Params into VIConfig. VIConfig requires
        # checkpoint_dir and checkpoint_interval to be both-set or both-unset
        # (validation in core/config.py). When the caller wants saveDir
        # without a periodic interval (saveInterval=-1), we set
        # checkpoint_interval to max_iterations + 1 — the in-loop modulo
        # `(step+1) % interval == 0` then never fires while the runner's
        # final-save guarantee still writes the directory at end-of-fit.
        if save_dir != "":
            interval = save_interval if save_interval > 0 else config.max_iterations + 1
            config = dataclasses.replace(
                config,
                checkpoint_dir=Path(save_dir),
                checkpoint_interval=interval,
            )

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
            result = runner.fit(
                bow_rdd,
                resume_from=Path(resume_from) if resume_from else None,
                on_iteration=self._on_iteration,
            )
        finally:
            bow_rdd.unpersist(blocking=False)

        # T and K ride along in result.metadata via OnlineHDP.get_metadata,
        # so the Model constructor only needs the result.
        out_model = OnlineHDPModel(result)
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

    # Stamped into result.metadata by VIRunner as `model_class` (the runner
    # uses type(model).__name__ on the underlying VIModel). Used by
    # OnlineHDPModel.load to reject checkpoints from other model classes.
    _expected_model_class = "OnlineHDP"

    def __init__(self, result) -> None:  # result: VIResult
        super().__init__()
        self._result = result
        # Shape constants come from result.metadata (populated by
        # OnlineHDP.get_metadata + VIRunner). This makes the Model
        # reconstructible from a VIResult alone — necessary for load(path).
        self._T = int(result.metadata["T"])
        self._K = int(result.metadata["K"])
        # Seed default Param values on the Model so a freshly-constructed
        # Model (e.g. via OnlineHDPModel.load) has the values _transform
        # reads. The Estimator's _fit also runs a param-copy loop after
        # construction; that overwrites these with the Estimator's actual
        # configuration. Loaded Models keep these defaults.
        self._setDefault(
            k=150, docTruncation=15, maxIter=20,
            featuresCol="features", topicDistributionCol="topicDistribution",
            optimizer="online",
            learningOffset=1024.0, learningDecay=0.51, subsamplingRate=0.05,
            gammaShape=100.0, caviMaxIter=100, caviTol=1e-4,
            optimizeDocConcentration=True,
            optimizeCorpusConcentration=True,
            optimizeTopicConcentration=False,
            saveInterval=-1, saveDir="", resumeFrom="",
        )

    def save(self, path: str) -> None:
        """Persist this trained model to `path`.

        Wraps spark_vi.io.export.save_result. The directory contents
        round-trip through OnlineHDPModel.load(path).
        """
        from spark_vi.io.export import save_result
        save_result(self._result, path)

    @classmethod
    def load(cls, path: str) -> "OnlineHDPModel":
        """Load a previously-saved OnlineHDPModel from `path`.

        Validates that the saved metadata identifies an OnlineHDP fit;
        raises ValueError on type mismatch (e.g. trying to load an
        OnlineLDA checkpoint here).
        """
        from spark_vi.io.export import load_result
        result = load_result(path)
        saved_class = result.metadata.get("model_class")
        if saved_class is None:
            raise ValueError(
                f"Checkpoint at {path} has no 'model_class' in its metadata; "
                f"cannot determine model type. Was this saved by a recent "
                f"version of spark_vi?"
            )
        if saved_class != cls._expected_model_class:
            raise ValueError(
                f"Expected '{cls._expected_model_class}' checkpoint at "
                f"{path}, got {saved_class!r}. Did you mean a different "
                f"Model class (e.g. OnlineLDAModel.load)?"
            )
        # Reconstruct from VIResult; T and K come from metadata.
        # (No Param round-trip — Pipeline.save persistence is deferred per
        # ADR 0012.)
        return cls(result)

    @property
    def result(self):
        """The trained VIResult (global_params, elbo_trace, n_iterations, ...)."""
        return self._result

    # Trained-scalar accessors. Methods rather than @property so the names
    # don't collide with the underlying Param descriptors (which would
    # break MLlib's `_set`/`_setDefault` resolution by name). This also
    # matches pyspark.ml.clustering.LDAModel which exposes
    # docConcentration() / topicConcentration() as methods. Values are
    # read from the trained global_params dict so optimize_* flags
    # surface their post-fit values (ADR 0013); when optimization is off,
    # global_params still carries the original constructor inputs because
    # initialize_global seeds them there.
    def trainedAlpha(self) -> float:
        """Trained α scalar (doc-stick concentration)."""
        return float(self._result.global_params["alpha"])

    def trainedCorpusConcentration(self) -> float:
        """Trained γ scalar (corpus-stick concentration). HDP-specific —
        no LDA analog."""
        return float(self._result.global_params["gamma"])

    def trainedTopicConcentration(self) -> float:
        """Trained η scalar (topic-word Dirichlet concentration)."""
        return float(self._result.global_params["eta"])

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
