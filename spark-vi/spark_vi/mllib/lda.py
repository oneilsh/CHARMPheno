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

from spark_vi.core.types import BOWDocument


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


class VanillaLDAEstimator(Estimator, HasFeaturesCol, HasMaxIter, HasSeed):
    """MLlib-shaped Estimator wrapping spark_vi.models.lda.VanillaLDA.

    Param defaults mirror pyspark.ml.clustering.LDA for the shared subset
    and ADR 0008 for our extras (gammaShape, caviMaxIter, caviTol).
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
        "whether to optimize alpha; MLlib default is True, but this shim rejects True (see ADR 0008)",
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

    def _fit(self, dataset):
        raise NotImplementedError("Implemented in a later task.")


class VanillaLDAModel(Model):
    """Stub — state and methods added in subsequent tasks."""
