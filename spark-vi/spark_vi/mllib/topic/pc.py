"""MLlib Estimator/Transformer shim for OnlinePCLDA (Prediction-Constrained LDA).

Mirrors mllib/topic/lda.py (ADR 0009): a translation layer over
``spark_vi.models.topic.pc.OnlinePCLDA`` + ``VIRunner``. All SVI logic lives in
OnlinePCLDA; this shim only marshals DataFrame columns into ``PCDocument``s and
wraps the trained ``VIResult`` as an MLlib-shaped Model.

Increment 1 (``weightY == 0``): the unsupervised SVI path. The label columns
(``labelCol``, ``labelMaskCol``) are THREADED into every ``PCDocument`` exactly
as the STM shim threads its covariate column, but tolerated-absent — at
``weightY == 0`` the model never reads y/label_mask, so a DataFrame with no label
columns fits fine (placeholder zeros are carried). ``_transform`` appends the
label-free ``topicDistributionCol`` (identical CAVI to train time); a
head-derived ``probabilityCol`` (``sigmoid(w_CK · θ)``) is increment 2 and is
deliberately NOT emitted here.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
from pyspark import StorageLevel, keyword_only
from pyspark.ml.base import Estimator, Model
from pyspark.ml.param import Param, Params, TypeConverters
from pyspark.ml.param.shared import HasFeaturesCol, HasMaxIter, HasSeed

from spark_vi.core.config import VIConfig
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


class _PCParams(HasFeaturesCol, HasMaxIter, HasSeed):
    """Shared Param surface for PCEstimator and PCModel.

    The LDA param subset (k, topicDistributionCol, the SVI schedule knobs, the
    Dirichlet concentrations, the CAVI knobs) mirrors ``_OnlineLDAParams``; on
    top sit the Prediction-Constrained params: ``numLabels`` (C), ``labelCol``,
    ``labelMaskCol``, and ``weightY`` (default 0.0 for increment 1).
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
        "prediction-loss weight (the PC dial). 0.0 (increment-1 default) = "
        "unsupervised LDA-MAP; > 0 is the supervised increment 2 (not built)",
        typeConverter=TypeConverters.toFloat,
    )


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
    numLabels=1, weightY=0.0,
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
    ) -> None:
        super().__init__()
        self._setDefault(**_PC_DEFAULTS)
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
        if weight_y != 0.0:
            raise NotImplementedError(
                "PCEstimator increment 1 supports weightY == 0.0 (unsupervised "
                f"SVI) only; the supervised path (weightY={weight_y}) is "
                "increment 2 and is not built."
            )

        features_col = self.getOrDefault("featuresCol")
        first = dataset.select(features_col).head(1)
        if not first:
            raise ValueError("Cannot fit on an empty DataFrame.")
        vocab_size = first[0][0].size

        model_obj, config = _build_model_and_config(self, vocab_size=vocab_size)

        label_col = self.getOrDefault("labelCol") if self.isSet("labelCol") else None
        label_mask_col = (
            self.getOrDefault("labelMaskCol") if self.isSet("labelMaskCol") else None
        )
        C = self.getOrDefault("numLabels")

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
                pc_rdd, on_iteration=self._on_iteration,
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


class PCModel(_PCParams, Model):
    """MLlib-shaped Model wrapping a trained OnlinePCLDA VIResult.

    ``transform`` appends the label-free ``topicDistributionCol`` (theta) via
    the SAME CAVI used at train time — the faithfulness invariant. A
    head-derived ``probabilityCol`` is increment 2 (see ``predictProbability``).
    """

    # Stamped into result.metadata by VIRunner as ``model_class``.
    _expected_model_class = "OnlinePCLDA"

    def __init__(self, result) -> None:  # result: VIResult
        super().__init__()
        self._result = result
        self._setDefault(**_PC_DEFAULTS)

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
        return dataset.withColumn(out_col, infer_udf(F.col(features_col)))

    def predictProbability(self, dataset):
        """Per-label P(y=1) = sigmoid(w_CK . theta) — INCREMENT 2 (not built).

        The head is inert at weightY == 0 (all-zero w_CK -> sigmoid(0) = 0.5 for
        every doc/label), so a probability column carries no signal on the
        unsupervised path. Emitting a meaningful ``probabilityCol`` requires the
        trained head from increment 2; stubbed here to mark the seam.
        """
        raise NotImplementedError(
            "predictProbability (head-derived probabilityCol) is increment 2; "
            "increment 1 fits the unsupervised representation only, so the head "
            "w_CK stays at its zero seed."
        )
