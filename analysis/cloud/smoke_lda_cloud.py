"""Cluster smoke test for the OnlineLDA MLlib shim.

Exercises the minimal end-to-end path on a real Spark cluster:
  1. driver-side import of spark_vi from the --py-files zip,
  2. fit OnlineLDAEstimator (broadcast/aggregate across executors),
  3. transform via the executor-side UDF (the load-bearing
     --py-files test — UDF closure must rehydrate spark_vi on workers),
  4. fit pyspark.ml.clustering.LDA on the same input as a baseline
     that depends only on Spark's built-in MLlib (no --py-files).

Hardcoded synthetic corpus, no I/O. The point is the cluster, not the data.

Submit (from this directory on a Dataproc master node):
    make smoke              # spark-submit --master yarn --deploy-mode client
    make smoke-dataproc \\
        CLUSTER=... REGION=... BUCKET=...   # gcloud dataproc jobs submit pyspark
"""
from __future__ import annotations

import logging
import sys

import numpy as np
from pyspark.ml.clustering import LDA as MLlibLDA
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql import SparkSession
from pyspark.sql.types import StructField, StructType


def _synthetic_bow(n_docs: int = 300, vocab_size: int = 30,
                   k_true: int = 3, doc_len: int = 25,
                   seed: int = 0) -> list[tuple]:
    """Mixture-of-multinomials draws over `k_true` block-supported topics.

    Each topic places mass on a contiguous vocab block; documents pick a
    topic uniformly then draw `doc_len` tokens i.i.d. — enough structure
    that LDA can recover something topic-shaped, but trivial to generate.
    """
    rng = np.random.default_rng(seed)
    block = vocab_size // k_true
    rows = []
    for _ in range(n_docs):
        t = rng.integers(k_true)
        probs = np.full(vocab_size, 0.01)
        probs[t * block:(t + 1) * block] = 1.0
        probs /= probs.sum()
        idx = rng.choice(vocab_size, size=doc_len, p=probs)
        uniq, counts = np.unique(idx, return_counts=True)
        rows.append((Vectors.sparse(vocab_size,
                                    [int(i) for i in uniq],
                                    [float(c) for c in counts]),))
    return rows


def main() -> int:
    from spark_vi.mllib.topic.lda import OnlineLDAEstimator
    print(f"[driver] spark_vi.mllib.topic.lda loaded from {OnlineLDAEstimator.__module__}",
          flush=True)

    # Surface spark_vi.core.runner per-iter INFO lines as [driver] output.
    logging.basicConfig(level=logging.WARNING,
                         format="[driver]   %(message)s",
                         stream=sys.stdout, force=True)
    logging.getLogger("spark_vi").setLevel(logging.INFO)

    spark = SparkSession.builder.appName("smoke_lda_cloud").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")  # silence GCS connector chatter
    sc = spark.sparkContext
    print(f"[driver] Spark {sc.version}, master={sc.master}, "
          f"defaultParallelism={sc.defaultParallelism}", flush=True)

    vocab_size, k_true, k_fit = 30, 3, 3
    rows = _synthetic_bow(n_docs=300, vocab_size=vocab_size, k_true=k_true)
    schema = StructType([StructField("features", VectorUDT(), False)])
    df = spark.createDataFrame(rows, schema=schema).repartition(4).cache()
    print(f"[driver] corpus: {df.count()} docs, {df.rdd.getNumPartitions()} partitions",
          flush=True)

    print("[driver] fitting OnlineLDAEstimator...", flush=True)
    ours = OnlineLDAEstimator(k=k_fit, maxIter=20, seed=0).fit(df)
    print(f"[driver] ours.topicsMatrix shape: {ours.topicsMatrix().toArray().shape}",
          flush=True)
    print(f"[driver] ours.elbo_trace tail: {ours.result.elbo_trace[-3:]}",
          flush=True)

    # transform → UDF runs on executors. This is the real --py-files smoke.
    print("[driver] transform via executor UDF...", flush=True)
    out = ours.transform(df).select("topicDistribution")
    out.show(3, truncate=False)
    print(f"[driver] transformed row count: {out.count()}", flush=True)

    print("[driver] fitting pyspark.ml.clustering.LDA (built-in MLlib baseline)...",
          flush=True)
    ml = (MLlibLDA().setK(k_fit).setMaxIter(20).setOptimizer("online")
          .setSeed(0).setFeaturesCol("features")
          .setOptimizeDocConcentration(False)).fit(df)
    print(f"[driver] mllib.topicsMatrix shape: {ml.topicsMatrix().toArray().shape}",
          flush=True)
    print(f"[driver] mllib.logLikelihood: {ml.logLikelihood(df):.3f}", flush=True)

    print("[driver] SMOKE TEST PASSED", flush=True)
    spark.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
