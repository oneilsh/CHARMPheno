"""MLlib Estimator/Transformer shim for OnlinePCLDA (Prediction-Constrained LDA).

Mirrors mllib/topic/lda.py (ADR 0009): a translation layer over
``spark_vi.models.topic.pc.OnlinePCLDA`` + ``VIRunner``. All SVI logic lives in
OnlinePCLDA; this shim only marshals DataFrame columns into ``PCDocument``s and
wraps the trained ``VIResult`` as an MLlib-shaped Model.

At ``weightY == 0`` (increment 1, the unsupervised SVI path) the label columns
(``labelCol``, ``labelMaskCol``) are THREADED into every ``PCDocument`` exactly
as the STM shim threads its covariate column, but tolerated-absent — the model
never reads y/label_mask, so a DataFrame with no label columns fits fine
(placeholder zeros are carried), and ``_transform`` appends only the label-free
``topicDistributionCol``.

At ``weightY > 0`` (increment 2, supervised) ``labelCol`` is REQUIRED — the
head trains on it — and ``_transform`` additionally appends a head-derived
``probabilityCol`` = ``sigmoid(w_CK · θ)`` (per-label P(y=1)); the label-free
``topicDistributionCol`` is unchanged (identical CAVI to train time).
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
from pyspark import StorageLevel, keyword_only
from pyspark.ml.base import Estimator, Model
from pyspark.ml.param import Param, Params, TypeConverters
from pyspark.ml.param.shared import HasFeaturesCol, HasMaxIter, HasSeed

from spark_vi.core.config import VIConfig
from spark_vi.mllib._common import (
    _PersistableModel,
    _PersistenceParams,
    apply_persistence_params,
)
from spark_vi.mllib.topic._common import _vector_to_bow_document
from spark_vi.models.topic.pc import OnlinePCLDA
from spark_vi.models.topic.types import PCDocument


def _row_to_pc_document(
    row,
    features_col: str,
    label_col: str | None,
    label_mask_col: str | None,
    C: int,
) -> PCDocument:
    """Build a PCDocument from a Spark row, tolerating absent label columns.

    ``features_col`` is a Sparse/Dense feature vector (sparsified to nonzero
    entries by ``_vector_to_bow_document``). ``label_col`` — when set — is the
    doc's (C,) outcome vector (a scalar for the single-label C == 1 case is
    promoted to a length-1 array); ``label_mask_col`` — when set — its (C,)
    observed mask. When a column is absent the corresponding field is a
    placeholder: ``y`` = zeros(C), and ``label_mask`` = zeros(C) (nothing
    observed) unless an explicit ``label_col`` is present, in which case a
    missing mask defaults to ones(C) (all observed). At weightY == 0 none of
    this is read — the placeholders exist only to keep the row type stable for
    increment 2.
    """
    bow = _vector_to_bow_document(row[features_col])

    if label_col is None:
        y = np.zeros(C, dtype=np.float64)
    else:
        raw = row[label_col]
        y = np.asarray(
            raw if isinstance(raw, (list, tuple, np.ndarray)) else [raw],
            dtype=np.float64,
        )

    if label_mask_col is not None:
        raw_m = row[label_mask_col]
        mask = np.asarray(
            raw_m if isinstance(raw_m, (list, tuple, np.ndarray)) else [raw_m],
            dtype=np.float64,
        )
    elif label_col is not None:
        mask = np.ones(C, dtype=np.float64)   # labelled doc, all cells observed
    else:
        mask = np.zeros(C, dtype=np.float64)  # no labels at all

    return PCDocument(
        indices=bow.indices,
        counts=bow.counts,
        length=bow.length,
        y=y,
        label_mask=mask,
    )


class _PCParams(HasFeaturesCol, HasMaxIter, HasSeed, _PersistenceParams):
    """Shared Param surface for PCEstimator and PCModel.

    The LDA param subset (k, topicDistributionCol, the SVI schedule knobs, the
    Dirichlet concentrations, the CAVI knobs) mirrors ``_OnlineLDAParams``; on
    top sit the Prediction-Constrained params: ``numLabels`` (C), ``labelCol``,
    ``labelMaskCol``, and ``weightY`` (default 0.0 for increment 1). Persistence
    Params (saveInterval, saveDir, resumeFrom) come from ``_PersistenceParams``
    (the same mixin the LDA/HDP shims use), so checkpoint + resume UX is
    identical across the topic-model shims.
    """

    k = Param(
        Params._dummy(), "k",
        "number of topics (clusters) to infer; must be >= 1",
        typeConverter=TypeConverters.toInt,
    )
    topicDistributionCol = Param(
        Params._dummy(), "topicDistributionCol",
        "output column with the label-free topic mixture (theta) for each document",
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
        "fraction of corpus sampled per mini-batch (None/1.0 = full batch)",
        typeConverter=TypeConverters.toFloat,
    )
    docConcentration = Param(
        Params._dummy(), "docConcentration",
        "Dirichlet concentration alpha on theta; scalar (symmetric) or length-k vector",
        typeConverter=TypeConverters.toListFloat,
    )
    topicConcentration = Param(
        Params._dummy(), "topicConcentration",
        "Dirichlet concentration eta on beta; scalar (symmetric)",
        typeConverter=TypeConverters.toFloat,
    )
    optimizeDocConcentration = Param(
        Params._dummy(), "optimizeDocConcentration",
        "whether to optimize alpha via Newton-Raphson (Blei 2003 App. A.4.2)",
        typeConverter=TypeConverters.toBoolean,
    )
    optimizeTopicConcentration = Param(
        Params._dummy(), "optimizeTopicConcentration",
        "whether to optimize eta (symmetric scalar) via Newton-Raphson",
        typeConverter=TypeConverters.toBoolean,
    )
    gammaShape = Param(
        Params._dummy(), "gammaShape",
        "shape parameter for Gamma init of variational gamma/lambda (ADR 0008 default 100.0)",
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
    # -- Prediction-Constrained params -------------------------------------
    numLabels = Param(
        Params._dummy(), "numLabels",
        "C = number of binary outcome heads (rows of the logistic head w_CK); >= 1",
        typeConverter=TypeConverters.toInt,
    )
    labelCol = Param(
        Params._dummy(), "labelCol",
        "column carrying the (C,) binary outcome vector; unused at weightY == 0",
        typeConverter=TypeConverters.toString,
    )
    labelMaskCol = Param(
        Params._dummy(), "labelMaskCol",
        "column carrying the (C,) per-cell observed mask; unused at weightY == 0",
        typeConverter=TypeConverters.toString,
    )
    weightY = Param(
        Params._dummy(), "weightY",
        "prediction-loss weight (the PC dial). 0.0 (default) = unsupervised "
        "LDA-MAP; > 0 turns on the supervised head + topic correction",
        typeConverter=TypeConverters.toFloat,
    )
    probabilityCol = Param(
        Params._dummy(), "probabilityCol",
        "output column with the head-derived per-label P(y=1) = sigmoid(w_CK . theta); "
        "appended by transform only when weightY > 0",
        typeConverter=TypeConverters.toString,
    )
    lambdaW = Param(
        Params._dummy(), "lambdaW",
        "L2 ridge on the head weights w_CK (scaled by weightY); authors' default 0.001",
        typeConverter=TypeConverters.toFloat,
    )
    gradCaviIters = Param(
        Params._dummy(), "gradCaviIters",
        "fixed CAVI unroll depth for the differentiated label-free pi (bounds the "
        "autograd tape); default 20",
        typeConverter=TypeConverters.toInt,
    )
    headLrScale = Param(
        Params._dummy(), "headLrScale",
        "extra multiplier on the head SGD step (RM <-> weightY decoupling knob); default 1.0",
        typeConverter=TypeConverters.toFloat,
    )
    topicTrust = Param(
        Params._dummy(), "topicTrust",
        "trust-region fraction capping the supervised topic correction on lambda to "
        "topicTrust * ||unsup lambda step|| (scale-invariant divergence guard); default 0.1",
        typeConverter=TypeConverters.toFloat,
    )
    weightYWarmupIters = Param(
        Params._dummy(), "weightYWarmupIters",
        "linearly ramp the effective weightY from 0 over this many global steps "
        "(0 = no warmup)",
        typeConverter=TypeConverters.toInt,
    )
    headOptimizer = Param(
        Params._dummy(), "headOptimizer",
        "head optimizer: 'sgd' (default; the RM-damped step rho*headLrScale*weightY*g) "
        "or 'adam' (per-parameter adaptive step DECOUPLED from rho/weightY, so the "
        "non-conjugate head runs on its own timescale — the two-timescale remedy for "
        "the coupled-objective mis-directed-head failure; headLrScale/weightYWarmup do "
        "NOT affect the adam step, headLr does)",
        typeConverter=TypeConverters.toString,
    )
    headLr = Param(
        Params._dummy(), "headLr",
        "base learning rate for the 'adam' head optimizer (ignored for 'sgd'); default 0.05",
        typeConverter=TypeConverters.toFloat,
    )
    warmStartFrom = Param(
        Params._dummy(), "warmStartFrom",
        "path to a previously-written save dir whose global params (topics/lambda) "
        "SEED this fit as its initial global params, with a FRESH Robbins-Monro "
        "iteration counter (rho restarts near rho_0). Empty (default) = cold start. "
        "DISTINCT from resumeFrom, which CONTINUES the counter (decayed rho) — a "
        "warm start needs the undecayed schedule so the head can move. The "
        "unsupervised-warm-start protocol (Hughes et al.): fit phase 1 at "
        "weightY == 0 (learns topics, leaves the head at its zero init), then "
        "warm-start a supervised phase 2 (weightY > 0) from it. Mutually exclusive "
        "with resumeFrom.",
        typeConverter=TypeConverters.toString,
    )

    def setWarmStartFrom(self, value: str):
        """Set warmStartFrom (path to a phase-1 checkpoint to warm-init from; empty = cold start)."""
        return self._set(warmStartFrom=value)

    def getWarmStartFrom(self) -> str:
        return str(self.getOrDefault(self.warmStartFrom))


def _build_model_and_config(
    estimator: "PCEstimator", vocab_size: int,
) -> tuple[OnlinePCLDA, VIConfig]:
    """Translate Estimator Params into (OnlinePCLDA, VIConfig).

    docConcentration follows the LDA-shim convention (unset -> 1/k symmetric;
    length-1 -> scalar; length-k -> asymmetric vector).
    """
    k = estimator.getOrDefault("k")

    doc_conc = (
        estimator.getOrDefault("docConcentration")
        if estimator.isSet("docConcentration") else None
    )
    if doc_conc is None:
        alpha = 1.0 / k
    elif len(doc_conc) == 1:
        alpha = float(doc_conc[0])
    else:
        if len(doc_conc) != k:
            raise ValueError(
                f"docConcentration vector must have length k={k}, got {len(doc_conc)}."
            )
        alpha = np.asarray(doc_conc, dtype=np.float64)

    topic_conc = (
        estimator.getOrDefault("topicConcentration")
        if estimator.isSet("topicConcentration") else None
    )
    eta = 1.0 / k if topic_conc is None else float(topic_conc)
    seed = estimator.getOrDefault("seed") if estimator.isSet("seed") else None

    model = OnlinePCLDA(
        K=k,
        vocab_size=vocab_size,
        C=estimator.getOrDefault("numLabels"),
        weight_y=float(estimator.getOrDefault("weightY")),
        lambda_w=float(estimator.getOrDefault("lambdaW")),
        grad_cavi_iters=int(estimator.getOrDefault("gradCaviIters")),
        head_lr_scale=float(estimator.getOrDefault("headLrScale")),
        topic_trust=float(estimator.getOrDefault("topicTrust")),
        weight_y_warmup_iters=int(estimator.getOrDefault("weightYWarmupIters")),
        head_optimizer=str(estimator.getOrDefault("headOptimizer")),
        head_lr=float(estimator.getOrDefault("headLr")),
        alpha=alpha,
        eta=eta,
        optimize_alpha=estimator.getOrDefault("optimizeDocConcentration"),
        optimize_eta=estimator.getOrDefault("optimizeTopicConcentration"),
        gamma_shape=estimator.getOrDefault("gammaShape"),
        cavi_max_iter=estimator.getOrDefault("caviMaxIter"),
        cavi_tol=estimator.getOrDefault("caviTol"),
        random_seed=seed,
    )

    mbf = float(estimator.getOrDefault("subsamplingRate"))
    config = VIConfig(
        max_iterations=estimator.getOrDefault("maxIter"),
        learning_rate_tau0=estimator.getOrDefault("learningOffset"),
        learning_rate_kappa=estimator.getOrDefault("learningDecay"),
        mini_batch_fraction=(mbf if 0.0 < mbf < 1.0 else None),
        random_seed=seed,
    )
    return model, config


_PC_DEFAULTS = dict(
    k=10, maxIter=20,
    featuresCol="features", topicDistributionCol="topicDistribution",
    learningOffset=1024.0, learningDecay=0.51, subsamplingRate=0.05,
    optimizeDocConcentration=True, optimizeTopicConcentration=False,
    gammaShape=100.0, caviMaxIter=100, caviTol=1e-3,
    numLabels=1, weightY=0.0, probabilityCol="probability",
    lambdaW=0.001, gradCaviIters=20, headLrScale=1.0, topicTrust=0.1,
    weightYWarmupIters=0, headOptimizer="sgd", headLr=0.05, warmStartFrom="",
)


class PCEstimator(_PCParams, Estimator):
    """MLlib-shaped Estimator wrapping ``spark_vi.models.topic.pc.OnlinePCLDA``.

    Param defaults mirror the LDA shim for the shared subset. ``weightY``
    defaults to 0.0 (increment 1 = unsupervised); ``labelCol``/``labelMaskCol``
    default unset and are tolerated-absent on that path.
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
        learningOffset: float = 1024.0,
        learningDecay: float = 0.51,
        subsamplingRate: float = 0.05,
        docConcentration: list[float] | None = None,
        topicConcentration: float | None = None,
        optimizeDocConcentration: bool = True,
        optimizeTopicConcentration: bool = False,
        gammaShape: float = 100.0,
        caviMaxIter: int = 100,
        caviTol: float = 1e-3,
        numLabels: int = 1,
        labelCol: str | None = None,
        labelMaskCol: str | None = None,
        weightY: float = 0.0,
        probabilityCol: str = "probability",
        lambdaW: float = 0.001,
        gradCaviIters: int = 20,
        headLrScale: float = 1.0,
        topicTrust: float = 0.1,
        weightYWarmupIters: int = 0,
        headOptimizer: str = "sgd",
        headLr: float = 0.05,
        warmStartFrom: str = "",
        # _PersistenceParams kwargs — see that mixin's docstring; these MUST
        # appear here explicitly (not just on the mixin) for kwarg-style
        # construction (the cloud driver builds via kwargs).
        # test_constructor_accepts_persistence_kwargs pins this.
        saveInterval: int = -1,
        saveDir: str = "",
        resumeFrom: str = "",
    ) -> None:
        super().__init__()
        self._setDefault(**_PC_DEFAULTS)
        self._set_persistence_defaults()
        # Diagnostic-only per-iteration callback (mirrors OnlineLDAEstimator).
        # Stored as an instance attribute — callables aren't MLlib-serializable
        # and persistence is deferred (ADR 0009).
        self._on_iteration = None
        self.setParams(**self._input_kwargs)

    @keyword_only
    def setParams(self, **kwargs) -> "PCEstimator":
        return self._set(**kwargs)

    def setOnIteration(
        self, fn: Callable[[int, dict, list[float]], None] | None,
    ) -> "PCEstimator":
        """Register a per-iteration diagnostic callback for the next fit.

        Signature fn(iter_num, global_params, elbo_trace); runs on the driver in
        the fit hot path. Must not mutate global_params. Not persisted.
        """
        self._on_iteration = fn
        return self

    def _fit(self, dataset) -> "PCModel":
        from spark_vi.core.runner import VIRunner

        weight_y = float(self.getOrDefault("weightY"))
        features_col = self.getOrDefault("featuresCol")
        first = dataset.select(features_col).head(1)
        if not first:
            raise ValueError("Cannot fit on an empty DataFrame.")
        vocab_size = first[0][0].size

        model_obj, config = _build_model_and_config(self, vocab_size=vocab_size)

        # Validate persistence Params and splice checkpoint_dir/interval into
        # VIConfig. Returns (config, resume_path) where resume_path is a Path
        # or None. Mirrors OnlineLDAEstimator._fit exactly — both weight_y == 0
        # and > 0 fits go through this one VIRunner.fit code path.
        config, resume_path = apply_persistence_params(self, config)

        # warmStartFrom is a PC-specific, fresh-counter warm-INIT (distinct from
        # resumeFrom's continue-counter). Resolve + validate it here rather than
        # in the shared apply_persistence_params so the LDA/HDP resume surface is
        # untouched. Empty (default) => cold start (warm_start_path stays None).
        warm_start = self.getOrDefault("warmStartFrom")
        warm_start_path = None
        if warm_start:
            if resume_path is not None:
                raise ValueError(
                    "resumeFrom and warmStartFrom are mutually exclusive: "
                    "resumeFrom continues the Robbins-Monro counter, "
                    "warmStartFrom resets it. Set at most one."
                )
            if not (Path(warm_start) / "manifest.json").exists():
                raise FileNotFoundError(
                    f"No manifest.json at warmStartFrom path: {warm_start}"
                )
            warm_start_path = Path(warm_start)

        label_col = self.getOrDefault("labelCol") if self.isSet("labelCol") else None
        label_mask_col = (
            self.getOrDefault("labelMaskCol") if self.isSet("labelMaskCol") else None
        )
        C = self.getOrDefault("numLabels")

        # Supervised training needs labels: without labelCol every doc carries an
        # all-zero observed mask, so the head/topic correction would see no
        # signal and the "supervised" fit would silently reduce to unsupervised.
        # Fail fast instead.
        if weight_y != 0.0 and label_col is None:
            raise ValueError(
                f"weightY={weight_y} > 0 requires labelCol to be set (the head "
                "trains on it); no labelCol was provided."
            )

        # Column set to pull: features always; label columns only when present
        # (tolerated-absent at weightY == 0). Threaded like the STM shim threads
        # its covariate column.
        select_cols = [features_col]
        if label_col is not None:
            select_cols.append(label_col)
        if label_mask_col is not None:
            select_cols.append(label_mask_col)

        def _to_pc(row, _fc=features_col, _lc=label_col, _mc=label_mask_col, _C=C):
            return _row_to_pc_document(row, _fc, _lc, _mc, _C)

        pc_rdd = (
            dataset.select(*select_cols).rdd
            .map(_to_pc)
            .persist(StorageLevel.MEMORY_AND_DISK)
        )
        pc_rdd.count()  # materialize for VIRunner's strict cache precondition

        try:
            result = VIRunner(model_obj, config=config).fit(
                pc_rdd,
                resume_from=resume_path,
                warm_start_from=warm_start_path,
                on_iteration=self._on_iteration,
            )
        finally:
            pc_rdd.unpersist(blocking=False)

        out_model = PCModel(result)
        for param in self.params:
            if self.isSet(param):
                out_model._set(**{param.name: self.getOrDefault(param)})
            elif self.hasDefault(param):
                out_model._setDefault(**{param.name: self.getOrDefault(param)})
        return out_model


class PCModel(_PCParams, _PersistableModel, Model):
    """MLlib-shaped Model wrapping a trained OnlinePCLDA VIResult.

    ``transform`` appends the label-free ``topicDistributionCol`` (theta) via
    the SAME CAVI used at train time — the faithfulness invariant. When the fit
    was supervised (``weightY > 0``) it ALSO appends a head-derived
    ``probabilityCol`` = ``sigmoid(w_CK . theta)`` (per-label P(y=1)); see
    ``predictProbability``.

    Persistable via ``_PersistableModel`` (save/load the wrapped VIResult),
    exactly as ``OnlineLDAModel`` is. ``_expected_model_class`` is the PC model
    tag the runner stamps (``OnlinePCLDA``), so ``load`` REJECTS a checkpoint
    from any other class — a PC checkpoint cannot load as LDA and vice-versa.
    """

    # Stamped into result.metadata by VIRunner as ``model_class`` (the runner
    # uses type(model).__name__ on the underlying VIModel). Used by
    # _PersistableModel.load to reject checkpoints from other model classes.
    _expected_model_class = "OnlinePCLDA"

    def __init__(self, result) -> None:  # result: VIResult
        super().__init__()
        self._result = result
        self._setDefault(**_PC_DEFAULTS)
        self._set_persistence_defaults()

    @property
    def result(self):
        """The trained VIResult (global_params, elbo_trace, n_iterations, ...)."""
        return self._result

    def vocabSize(self) -> int:
        """V dimension of the trained lambda."""
        return int(self._result.global_params["lambda"].shape[1])

    def topicsMatrix(self):
        """Topic-word distribution as an MLlib DenseMatrix of shape (V, K)."""
        from pyspark.ml.linalg import DenseMatrix

        lam = self._result.global_params["lambda"]
        beta = lam / lam.sum(axis=1, keepdims=True)
        K, V = beta.shape
        return DenseMatrix(numRows=V, numCols=K, values=beta.T.flatten("F").tolist())

    def headWeights(self) -> np.ndarray:
        """The logistic head w_CK (C x K). All-zero after an increment-1 fit
        (the head is seeded and left at init on the unsupervised path)."""
        return self._result.global_params["w_CK"]

    def _transform(self, dataset):
        import hashlib

        from pyspark.ml.linalg import DenseVector, VectorUDT
        from pyspark.sql import functions as F
        from scipy.special import digamma

        from spark_vi.models.topic.lda import _cavi_doc_inference

        lam = self._result.global_params["lambda"]
        expElogbeta = np.exp(digamma(lam) - digamma(lam.sum(axis=1, keepdims=True)))
        alpha = self._result.global_params["alpha"]
        gamma_shape = float(self.getOrDefault("gammaShape"))
        cavi_max_iter = int(self.getOrDefault("caviMaxIter"))
        cavi_tol = float(self.getOrDefault("caviTol"))
        K = expElogbeta.shape[0]

        sc = dataset.sparkSession.sparkContext
        bcast = sc.broadcast({
            "expElogbeta": expElogbeta, "alpha": alpha, "gamma_shape": gamma_shape,
            "cavi_max_iter": cavi_max_iter, "cavi_tol": cavi_tol, "K": K,
        })

        def _infer(features):
            p = bcast.value
            doc = _vector_to_bow_document(features)
            # Content-deterministic gamma_init (mirrors OnlinePCLDA/GatedOnlineLDA):
            # identical docs get identical init on every run, so a scoring path is
            # reproducible and independent of Spark partition/executor order.
            h = hashlib.blake2b(digest_size=8)
            h.update(np.ascontiguousarray(doc.indices, dtype=np.int32).tobytes())
            h.update(np.ascontiguousarray(doc.counts, dtype=np.float64).tobytes())
            rng = np.random.default_rng(int.from_bytes(h.digest(), "little"))
            gamma_init = rng.gamma(p["gamma_shape"], 1.0 / p["gamma_shape"], size=p["K"])
            gamma, _, _, _ = _cavi_doc_inference(
                indices=doc.indices, counts=doc.counts,
                expElogbeta=p["expElogbeta"], alpha=p["alpha"],
                gamma_init=gamma_init,
                max_iter=p["cavi_max_iter"], tol=p["cavi_tol"],
            )
            return DenseVector(gamma / gamma.sum())

        infer_udf = F.udf(_infer, returnType=VectorUDT())
        out_col = self.getOrDefault("topicDistributionCol")
        features_col = self.getOrDefault("featuresCol")
        # Broadcast lifetime is the returned DataFrame's (its UDF closure holds
        # bcast); ContextCleaner reclaims it on GC — do NOT eagerly unpersist.
        out = dataset.withColumn(out_col, infer_udf(F.col(features_col)))

        # Supervised fit -> also append the head-derived per-label probability.
        # The topicDistribution UDF already inferred theta from the SAME CAVI; the
        # head is a cheap sigmoid(w_CK . theta) on top of it (no second inference).
        if float(self.getOrDefault("weightY")) != 0.0:
            w_CK = self._result.global_params["w_CK"]
            wbcast = sc.broadcast(w_CK)

            def _proba(theta, _wb=wbcast):
                th = np.asarray(theta.toArray(), dtype=np.float64)
                logits = _wb.value @ th               # (C,)
                return DenseVector(1.0 / (1.0 + np.exp(-logits)))

            proba_udf = F.udf(_proba, returnType=VectorUDT())
            prob_col = self.getOrDefault("probabilityCol")
            out = out.withColumn(prob_col, proba_udf(F.col(out_col)))
        return out

    def predictProbability(self, dataset):
        """Append the head-derived ``probabilityCol`` = sigmoid(w_CK . theta).

        Per-label P(y=1) for every doc, from the trained logistic head on top of
        the label-free CAVI theta. Requires a supervised fit (``weightY > 0``): on
        the unsupervised path the head is at its zero seed (sigmoid(0) = 0.5 for
        every doc/label), so no meaningful probability exists.

        This is exactly the column ``transform`` already appends when
        ``weightY > 0``; provided as a named entry point for callers that only want
        the probability.
        """
        if float(self.getOrDefault("weightY")) == 0.0:
            raise NotImplementedError(
                "predictProbability requires a supervised fit (weightY > 0); the "
                "unsupervised head is at its zero seed (P == 0.5 everywhere)."
            )
        return self.transform(dataset)
