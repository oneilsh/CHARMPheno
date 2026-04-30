"""End-to-end Spark-local integration tests for VanillaLDA.

Hermetic by construction: each test builds its own synthetic LDA dataset
inside the test, no dependency on simulate_lda_omop.py.

Scope: these tests verify that the VIRunner <-> VanillaLDA integration
ends up in a sensible place — fitting actually drives the ELBO up over
iterations, and a fit run produces a well-formed VIResult. Recovery
quality (does the fitted beta match the synthetic ground truth?) is
intentionally NOT tested here; topic collapse at small synthetic-corpus
scales is a known SVI characteristic (MLlib has the same behavior;
Hoffman 2010 §4 uses corpora of 100K-352K documents). Recovery is
verified by the MLlib parity test in charmpheno/tests/test_lda_compare.py
which runs both VanillaLDA and pyspark.ml.clustering.LDA on the same
data and asserts the two implementations agree — any math regression on
our side will diverge from the reference.
"""
import numpy as np
import pytest


def _generate_synthetic_corpus(D: int, V: int, K: int,
                               docs_avg_len: int, seed: int):
    """Generate (true_beta, docs_as_BOWDocuments) under standard LDA."""
    from spark_vi.core import BOWDocument
    rng = np.random.default_rng(seed)

    true_beta = rng.dirichlet(np.full(V, 0.05), size=K)
    docs = []
    for d in range(D):
        theta_d = rng.dirichlet(np.full(K, 0.3))
        N_d = max(1, rng.poisson(docs_avg_len))
        zs = rng.choice(K, size=N_d, p=theta_d)
        ws = np.array([rng.choice(V, p=true_beta[z]) for z in zs])
        unique, counts = np.unique(ws, return_counts=True)
        docs.append(BOWDocument(
            indices=unique.astype(np.int32),
            counts=counts.astype(np.float64),
            length=int(counts.sum()),
        ))
    return true_beta, docs


@pytest.mark.slow
def test_vanilla_lda_fit_produces_well_formed_result(spark):
    """A short Spark-local fit returns a VIResult with positive lambda and a finite ELBO trace."""
    from spark_vi.core import VIConfig, VIRunner
    from spark_vi.models.lda import VanillaLDA

    K, V, D = 3, 30, 100
    np.random.seed(0)
    _, docs = _generate_synthetic_corpus(D=D, V=V, K=K, docs_avg_len=40, seed=0)
    rdd = spark.sparkContext.parallelize(docs, numSlices=2)

    cfg = VIConfig(
        max_iterations=20,
        mini_batch_fraction=0.3,
        random_seed=0,
        convergence_tol=1e-9,
    )
    result = VIRunner(VanillaLDA(K=K, vocab_size=V), config=cfg).fit(rdd)

    assert result.global_params["lambda"].shape == (K, V)
    assert (result.global_params["lambda"] > 0).all()
    assert len(result.elbo_trace) >= 1
    assert all(np.isfinite(v) for v in result.elbo_trace)


@pytest.mark.slow
def test_vanilla_lda_elbo_smoothed_trend_is_non_decreasing(spark):
    """A 10-iter moving average of the ELBO trace improves over training.

    Hard monotonicity is too strict for stochastic VI; trend is the right
    gate. This catches regressions where the math is wrong in a direction
    that drives the bound the wrong way (e.g., a sign error on the global
    KL term in compute_elbo, or update_global pushing away from the
    natural gradient).
    """
    from spark_vi.core import VIConfig, VIRunner
    from spark_vi.models.lda import VanillaLDA

    np.random.seed(1)
    _, docs = _generate_synthetic_corpus(D=100, V=30, K=3, docs_avg_len=40, seed=7)
    rdd = spark.sparkContext.parallelize(docs, numSlices=2)
    cfg = VIConfig(max_iterations=40, mini_batch_fraction=0.3,
                    random_seed=7, convergence_tol=1e-9)
    result = VIRunner(VanillaLDA(K=3, vocab_size=30), config=cfg).fit(rdd)

    trace = np.asarray(result.elbo_trace)
    window = 10
    assert len(trace) >= window, f"need at least {window} iterations for smoothing"
    smooth = np.convolve(trace, np.ones(window) / window, mode="valid")
    assert smooth[-1] > smooth[0], (
        f"Smoothed ELBO did not improve: start={smooth[0]:.3f}, end={smooth[-1]:.3f}. "
        f"This usually indicates a sign error or wrong-direction update upstream."
    )
