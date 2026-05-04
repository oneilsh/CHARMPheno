"""MLlib Estimator/Transformer shim for VanillaLDA.

Wraps spark_vi.models.lda.VanillaLDA + spark_vi.core.runner.VIRunner so the
model behaves like a pyspark.ml.clustering.LDA-shaped Estimator/Model pair.
The shim is a translation layer; all SVI logic lives in VanillaLDA. See
docs/superpowers/specs/2026-05-04-mllib-shim-design.md and ADR 0009.
"""
from __future__ import annotations

import numpy as np
from pyspark import keyword_only
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

    Three cases (per ADR 0008 / ADR 0009):
      * optimizer != "online" — we are SVI-only.
      * optimizeDocConcentration=True — symmetric-alpha-only.
      * vector docConcentration (length > 1) — symmetric-alpha-only.

    Silent fallback would mislead users about what they are getting.
    """
    optimizer = estimator.getOrDefault("optimizer")
    if optimizer != "online":
        raise ValueError(
            f"VanillaLDAEstimator only supports optimizer='online', got {optimizer!r}. "
            f"The 'em' optimizer is not implemented in this shim."
        )

    if estimator.getOrDefault("optimizeDocConcentration"):
        raise ValueError(
            "VanillaLDAEstimator does not support optimizeDocConcentration=True. "
            "Empirical-Bayes alpha optimization is deferred per ADR 0008 'Future work'. "
            "Set optimizeDocConcentration=False (the default) and pass a fixed alpha "
            "via docConcentration."
        )

    if estimator.isSet("docConcentration"):
        doc_conc = estimator.getOrDefault("docConcentration")
        if doc_conc is not None and len(doc_conc) > 1:
            raise ValueError(
                f"VanillaLDAEstimator only supports symmetric (scalar) docConcentration, "
                f"got vector of length {len(doc_conc)}. Asymmetric alpha is deferred per "
                f"ADR 0008 'Future work'."
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
            gammaShape=100.0, caviMaxIter=100, caviTol=1e-3,
        )
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, **kwargs) -> "VanillaLDAEstimator":
        """Standard MLlib pattern: set any subset of params after construction."""
        return self._set(**kwargs)

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

        runner = VIRunner(model_obj, config=config)
        result = runner.fit(bow_rdd)

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
        raise NotImplementedError("Implemented in a later task.")
