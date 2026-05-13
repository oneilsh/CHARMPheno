"""One-off probe: does α drift toward synthetic ground truth at D=10k?

Background. The originally-planned integration test
`test_alpha_optimization_drifts_toward_corpus_truth` (see ADR 0010) was
cut because at the small synthetic-corpus scales used in unit tests
(D=200), topic collapse routes one true-topic's mass to the α floor
independent of optimization quality, making truth-recovery untestable.
Hoffman 2010 §4 used D=100K-352K to validate recovery; D=200 is three
orders of magnitude shy.

This probe sits in the middle: D=10K, large enough that we expect
recovery to work-if-it-works, small enough to run in a few minutes on
Spark local. Run as:

    cd spark-vi
    PYTHONPATH=$PWD python probes/alpha_drift_probe.py

What the probe does:
  1. Generate a synthetic LDA corpus with known asymmetric true_α
     and known true_β.
  2. Fit OnlineLDA with optimize_alpha=True for many iterations.
  3. Match fitted topics to true topics via Hungarian assignment on β
     cosine similarity (LDA topics have arbitrary ordering — without
     this matching, "α distance to truth" is meaningless).
  4. Report L1 distance from 1/K-init to truth (baseline) and from
     fitted-aligned to truth (the actual recovery signal).
  5. Flag floor-clip events on the fitted α, since those indicate
     topic collapse rather than a recovery failure of the helper math.

Output is a single readable report; no test assertions. The user reads
it to decide whether the recovery is empirically sufficient at this
scale to justify a corresponding integration test.
"""
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
from pyspark.sql import SparkSession
from scipy.optimize import linear_sum_assignment

from spark_vi.core import VIConfig, VIRunner
from spark_vi.models.topic import BOWDocument
from spark_vi.models.topic.lda import OnlineLDA


def generate_synthetic_corpus(D, V, K, docs_avg_len, true_alpha, true_beta,
                              seed):
    rng = np.random.default_rng(seed)
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


def hungarian_topic_alignment(fitted_beta, true_beta):
    """Return permutation P such that fitted_beta[P[k]] best matches
    true_beta[k] in the cosine sense. Standard symmetric-LDA tool.
    """
    K = true_beta.shape[0]
    norm_t = true_beta / np.linalg.norm(true_beta, axis=1, keepdims=True)
    norm_f = fitted_beta / np.linalg.norm(fitted_beta, axis=1, keepdims=True)
    cosine = norm_f @ norm_t.T  # (fitted_K, true_K)
    fitted_idx, true_idx = linear_sum_assignment(-cosine)
    perm = np.empty(K, dtype=int)
    for fi, ti in zip(fitted_idx, true_idx):
        perm[ti] = fi
    return perm, cosine[fitted_idx, true_idx]


def fit_and_score(spark, docs, true_beta, true_alpha, K, V, max_iter,
                  mini_batch_fraction, seed_global, vi_seed):
    """Run one fit, return (l1_init→truth, l1_fitted→truth, fitted_alpha,
    fitted_alpha_aligned, alpha_history, elbo_trace, n_iters, hit_floor)."""
    np.random.seed(seed_global)  # seed global RNG for lambda/gamma init
    rdd = spark.sparkContext.parallelize(docs, numSlices=8).persist()
    rdd.count()
    cfg = VIConfig(
        max_iterations=max_iter,
        mini_batch_fraction=mini_batch_fraction,
        random_seed=vi_seed,
        convergence_tol=1e-9,
    )
    model = OnlineLDA(K=K, vocab_size=V, optimize_alpha=True)
    history = []
    def _on_iter(it, gp, _):
        history.append(np.array(gp["alpha"]))
    result = VIRunner(model, config=cfg).fit(rdd, on_iteration=_on_iter)
    rdd.unpersist()
    fitted_alpha = result.global_params["alpha"]
    fitted_lambda = result.global_params["lambda"]
    fitted_beta = fitted_lambda / fitted_lambda.sum(axis=1, keepdims=True)
    perm, cosines = hungarian_topic_alignment(fitted_beta, true_beta)
    fitted_aligned = fitted_alpha[perm]
    init_alpha = np.full(K, 1.0 / K)
    l1_init = np.abs(init_alpha - true_alpha).sum()
    l1_fit = np.abs(fitted_aligned - true_alpha).sum()
    return dict(
        l1_init=l1_init,
        l1_fit=l1_fit,
        drift_pct=100.0 * (1.0 - l1_fit / l1_init),
        fitted_alpha=fitted_alpha,
        fitted_aligned=fitted_aligned,
        history=history,
        elbo=np.asarray(result.elbo_trace),
        n_iters=result.n_iterations,
        hit_floor=bool((fitted_alpha <= 1e-3 + 1e-9).any()),
        cosines=cosines,
        perm=perm,
    )


def main():
    K = 3
    V = 100
    D = 10_000
    docs_avg_len = 100
    true_alpha = np.array([0.1, 0.5, 0.9])
    max_iter_long = 2000
    mini_batch_fraction = 0.05

    print(f"--- α-drift probe ---")
    print(f"D={D}, V={V}, K={K}, docs_avg_len={docs_avg_len}")
    print(f"true_alpha = {true_alpha} (sum = {true_alpha.sum():.4f})")
    print(f"true_alpha / sum = {true_alpha / true_alpha.sum()}")
    print(f"mini_batch_fraction={mini_batch_fraction}")
    print()

    rng = np.random.default_rng(11)
    true_beta = rng.dirichlet(np.full(V, 0.05), size=K)

    print("[probe] generating synthetic corpus...", flush=True)
    t0 = time.perf_counter()
    docs = generate_synthetic_corpus(
        D=D, V=V, K=K, docs_avg_len=docs_avg_len,
        true_alpha=true_alpha, true_beta=true_beta, seed=11,
    )
    print(f"[probe] generated {len(docs)} docs in {time.perf_counter()-t0:.1f}s",
          flush=True)

    spark = (
        SparkSession.builder
            .master("local[4]")
            .appName("alpha-drift-probe")
            .config("spark.sql.shuffle.partitions", "8")
            .config("spark.driver.memory", "4g")
            .config("spark.ui.enabled", "false")
            .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    # ──── Step 1: seed sweep at 300 iters to characterize variance ────
    print("\n=== Seed sweep @ max_iter=300 ===")
    print(f"  {'global_seed':>11}  {'vi_seed':>7}  "
          f"{'l1_init':>8}  {'l1_fit':>8}  {'drift%':>7}  "
          f"{'min_α':>8}  {'floor?':>6}  "
          f"{'βcos_min':>8}")
    sweep = []
    for s in [11, 23, 42, 7, 99]:
        out = fit_and_score(
            spark, docs, true_beta, true_alpha, K, V,
            max_iter=300, mini_batch_fraction=mini_batch_fraction,
            seed_global=s, vi_seed=s,
        )
        sweep.append((s, out))
        print(f"  {s:>11d}  {s:>7d}  "
              f"{out['l1_init']:>8.4f}  {out['l1_fit']:>8.4f}  "
              f"{out['drift_pct']:>+7.1f}  "
              f"{out['fitted_alpha'].min():>8.4f}  "
              f"{str(out['hit_floor']):>6}  "
              f"{out['cosines'].min():>8.4f}")

    # Pick the seed with the highest drift% AND no floor hit for the long run.
    healthy = [(s, o) for (s, o) in sweep if not o['hit_floor']]
    if not healthy:
        print("\n[probe] all seeds collapsed to floor; using least-bad")
        chosen_seed, _ = max(sweep, key=lambda x: x[1]['drift_pct'])
    else:
        chosen_seed, _ = max(healthy, key=lambda x: x[1]['drift_pct'])
    print(f"\n[probe] chosen seed for long run: {chosen_seed}")

    # ──── Step 2: long fit at chosen seed ────
    print(f"\n=== Long fit @ max_iter={max_iter_long}, seed={chosen_seed} ===")
    t0 = time.perf_counter()
    out = fit_and_score(
        spark, docs, true_beta, true_alpha, K, V,
        max_iter=max_iter_long, mini_batch_fraction=mini_batch_fraction,
        seed_global=chosen_seed, vi_seed=chosen_seed,
    )
    print(f"[probe] long fit completed in {time.perf_counter()-t0:.1f}s "
          f"({out['n_iters']} iters)")

    print()
    print("=" * 60)
    print("LONG-RUN RESULTS")
    print("=" * 60)
    print(f"true_alpha               = {true_alpha}")
    print(f"init_alpha (1/K)         = {np.full(K, 1.0 / K)}")
    print(f"fitted_alpha (raw)       = {out['fitted_alpha']}")
    print(f"fitted_alpha (aligned)   = {out['fitted_aligned']}")
    print(f"hungarian permutation    = {out['perm'].tolist()}")
    print(f"β cosine similarities    = {out['cosines']}")
    print()
    print(f"L1 dist (init  → truth)  = {out['l1_init']:.4f}")
    print(f"L1 dist (fitted→ truth)  = {out['l1_fit']:.4f}")
    print(f"drift toward truth       = {out['drift_pct']:+.1f}%")
    print()
    print(f"fitted_alpha at floor?   = {out['hit_floor']} "
          f"(min component = {out['fitted_alpha'].min():.6f})")
    print()

    print("α evolution (raw, unaligned):")
    print(f"  {'iter':>5}  {'α[0]':>10}  {'α[1]':>10}  {'α[2]':>10}  Σα")
    n = len(out['history'])
    cadence = max(1, n // 40)
    for i, a in enumerate(out['history']):
        if i == 0 or (i + 1) % cadence == 0 or i == n - 1:
            print(f"  {i+1:>5d}  {a[0]:>10.4g}  {a[1]:>10.4g}  "
                  f"{a[2]:>10.4g}  {a.sum():.4g}")

    elbo = out['elbo']
    print()
    print(f"ELBO trace: first={elbo[0]:.2f}, last={elbo[-1]:.2f}, "
          f"min={elbo.min():.2f}, max={elbo.max():.2f}")

    spark.stop()


if __name__ == "__main__":
    main()
