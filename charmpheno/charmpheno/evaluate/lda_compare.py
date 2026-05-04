"""Head-to-head orchestration: VanillaLDA vs Spark MLlib LDA on the same input.

Pure functions; no plotting, no driver concerns. Drivers in analysis/local/
compose these with topic_alignment.alignment_biplot_data to produce figures.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from pyspark import RDD
from pyspark.ml.clustering import LDA as MLlibLDA
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from spark_vi.core import VIConfig


@dataclass
class LDARunArtifacts:
    """Common artifact bundle for both implementations.

    Asymmetric on ELBO-trace availability: ours records every iter; MLlib
    only exposes a final log-likelihood. See spec for details.

    per_iter_seconds is approximated as wall_time / n_iter for both
    implementations; true per-iteration timing is not instrumented.
    """
    topics_matrix: np.ndarray
    topic_prevalence: np.ndarray
    elbo_trace: list[float] | None
    per_iter_seconds: list[float]
    wall_time_seconds: float
    final_log_likelihood: float | None


def run_ours(
    rdd: RDD,
    vocab_size: int,
    K: int,
    config: VIConfig,
) -> LDARunArtifacts:
    """Fit VanillaLDA via the MLlib-shaped shim; collect artifacts.

    Wraps the input RDD[BOWDocument] back into a DataFrame[Vector] so the
    shim's DataFrame-shaped API accepts it. Net effect: this function is
    now structurally symmetric with run_mllib (both fit MLlib-shaped
    Estimators on a DataFrame), tightening the head-to-head comparison.
    """
    from pyspark.ml.linalg import Vectors, VectorUDT
    from pyspark.sql.types import StructField, StructType
    from pyspark.sql import SparkSession

    from spark_vi.mllib.lda import VanillaLDAEstimator

    spark = SparkSession.builder.getOrCreate()

    def _bow_to_row(doc):
        return (Vectors.sparse(
            vocab_size,
            [int(i) for i in doc.indices],
            [float(c) for c in doc.counts],
        ),)

    schema = StructType([StructField("features", VectorUDT(), False)])
    df = spark.createDataFrame(rdd.map(_bow_to_row), schema=schema)

    estimator = VanillaLDAEstimator(
        k=K,
        maxIter=config.max_iterations,
        seed=config.random_seed,
        learningOffset=config.learning_rate_tau0,
        learningDecay=config.learning_rate_kappa,
        subsamplingRate=config.mini_batch_fraction or 1.0,
    )

    t0 = time.perf_counter()
    model = estimator.fit(df)
    t1 = time.perf_counter()
    wall = t1 - t0

    n_iter = max(1, model.result.n_iterations)
    per_iter = [wall / n_iter] * n_iter

    tm = model.topicsMatrix().toArray().T  # (K, V), row-stochastic
    tm = tm / tm.sum(axis=1, keepdims=True)

    transformed = model.transform(df).select("topicDistribution").collect()
    prev = np.zeros(K)
    for r in transformed:
        prev += np.asarray(r["topicDistribution"].toArray())

    return LDARunArtifacts(
        topics_matrix=tm,
        topic_prevalence=prev,
        elbo_trace=list(model.result.elbo_trace),
        per_iter_seconds=per_iter,
        wall_time_seconds=wall,
        final_log_likelihood=None,
    )


def run_mllib(
    df: DataFrame,
    vocab_size: int,
    K: int,
    max_iter: int = 100,
    seed: int = 0,
    optimizer: str = "online",
    subsampling_rate: float = 0.05,
    optimize_doc_concentration: bool = True,
) -> LDARunArtifacts:
    """Fit pyspark.ml.clustering.LDA on the BOW DataFrame; collect artifacts.

    optimize_doc_concentration: if True (MLlib default), alpha is adapted
    during training. Set to False for a head-to-head parity test against
    VanillaLDA, which holds alpha fixed.
    """
    lda = (
        MLlibLDA()
        .setK(K)
        .setMaxIter(max_iter)
        .setOptimizer(optimizer)
        .setSeed(seed)
        .setSubsamplingRate(subsampling_rate)
        .setOptimizeDocConcentration(optimize_doc_concentration)
        .setFeaturesCol("features")
    )

    t0 = time.perf_counter()
    model = lda.fit(df)
    t1 = time.perf_counter()
    wall = t1 - t0
    per_iter = [wall / max(1, max_iter)] * max(1, max_iter)

    tm = model.topicsMatrix().toArray().T
    tm = tm / tm.sum(axis=1, keepdims=True)

    transformed = model.transform(df).select("topicDistribution")
    rows = transformed.collect()
    prev = np.zeros(K)
    for r in rows:
        prev += np.asarray(r["topicDistribution"].toArray())

    final_ll = float(model.logLikelihood(df))

    return LDARunArtifacts(
        topics_matrix=tm,
        topic_prevalence=prev,
        elbo_trace=None,
        per_iter_seconds=per_iter,
        wall_time_seconds=wall,
        final_log_likelihood=final_ll,
    )
