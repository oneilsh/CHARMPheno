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

from spark_vi.core import VIConfig, VIRunner, BOWDocument
from spark_vi.models.lda import VanillaLDA


@dataclass
class LDARunArtifacts:
    """Common artifact bundle for both implementations.

    Asymmetric on ELBO-trace availability: ours records every iter; MLlib
    only exposes a final log-likelihood. See spec for details.
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
    """Fit VanillaLDA via VIRunner; collect artifacts."""
    model = VanillaLDA(K=K, vocab_size=vocab_size)

    t0 = time.perf_counter()
    result = VIRunner(model, config=config).fit(rdd)
    t1 = time.perf_counter()
    wall = t1 - t0
    n_iter = max(1, result.n_iterations)
    per_iter = [wall / n_iter] * n_iter

    lam = result.global_params["lambda"]
    topics_matrix = lam / lam.sum(axis=1, keepdims=True)

    runner = VIRunner(model, config=config)
    inferred = runner.transform(rdd, global_params=result.global_params).collect()
    prev = np.zeros(K)
    for d in inferred:
        prev += np.asarray(d["theta"])

    return LDARunArtifacts(
        topics_matrix=topics_matrix,
        topic_prevalence=prev,
        elbo_trace=list(result.elbo_trace),
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
