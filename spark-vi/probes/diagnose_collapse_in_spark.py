"""Run seed=23 collapse case under Spark with verbose per-iter dumps.

Captures α at every iter, and also the warning context for any RuntimeWarning
that fires. Goal: find which iter the collapse happens, so we can re-target
the offline diagnostic.
"""
import os
import sys
import warnings

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
from pyspark.sql import SparkSession
from spark_vi.core import BOWDocument, VIConfig, VIRunner
from spark_vi.models.lda import OnlineLDA

K, V, D = 3, 100, 10_000
docs_avg_len = 100
true_alpha = np.array([0.1, 0.5, 0.9])
SEED = 23
MAX_ITER = 5


def gen_corpus():
    rng_b = np.random.default_rng(11)
    true_beta = rng_b.dirichlet(np.full(V, 0.05), size=K)
    rng = np.random.default_rng(11)
    docs = []
    for d in range(D):
        theta_d = rng.dirichlet(true_alpha)
        N_d = max(1, rng.poisson(docs_avg_len))
        zs = rng.choice(K, size=N_d, p=theta_d)
        ws = np.array([rng.choice(V, p=true_beta[z]) for z in zs])
        unique, counts = np.unique(ws, return_counts=True)
        docs.append(BOWDocument(
            indices=unique.astype(np.int32),
            counts=counts.astype(np.float64),
            length=int(counts.sum()),
        ))
    return docs


def main():
    docs = gen_corpus()
    print(f"Generated {len(docs)} docs.")

    spark = (
        SparkSession.builder
            .master("local[4]")
            .appName("diagnose-collapse")
            .config("spark.sql.shuffle.partitions", "8")
            .config("spark.driver.memory", "4g")
            .config("spark.ui.enabled", "false")
            .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    np.random.seed(SEED)
    rdd = spark.sparkContext.parallelize(docs, numSlices=8).persist()
    rdd.count()

    cfg = VIConfig(
        max_iterations=MAX_ITER,
        mini_batch_fraction=0.05,
        random_seed=SEED,
        convergence_tol=1e-9,
    )
    model = OnlineLDA(K=K, vocab_size=V, optimize_alpha=True)

    # Force warnings to be raised so we can catch them with stack info.
    captured = []
    def _showwarning(message, category, filename, lineno, file=None, line=None):
        captured.append((str(message), filename, lineno))
        print(f"WARNING@{filename}:{lineno}: {message}", flush=True)
    warnings.showwarning = _showwarning
    warnings.simplefilter("always")

    print(f"\nFitting with seed={SEED}, max_iter={MAX_ITER} ...")

    history = []
    def _on_iter(iter_num, gp, _):
        a = np.array(gp["alpha"])
        lam = np.array(gp["lambda"])
        history.append(a)
        print(f"[iter {iter_num}] α={a}, Σα={a.sum():.4g}, "
              f"Σλ_k={lam.sum(axis=1)}", flush=True)

    result = VIRunner(model, config=cfg).fit(rdd, on_iteration=_on_iter)

    print()
    print(f"Final α: {result.global_params['alpha']}")
    print(f"ELBO trace: {[f'{e:.2f}' for e in result.elbo_trace]}")
    print(f"\nWarnings captured: {len(captured)}")
    for msg, fn, ln in captured[:5]:
        print(f"  {fn}:{ln}  {msg}")

    spark.stop()


if __name__ == "__main__":
    main()
