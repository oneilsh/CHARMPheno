"""End-to-end Spark-local integration tests for OnlineHDP.

Hermetic by construction: each test builds its own synthetic LDA/HDP
dataset inside the test, no external data dependencies.

Scope: verify the VIRunner ↔ OnlineHDP integration converges sensibly.
Recovery quality (does fitted β match ground truth?) is in
`test_online_hdp_synthetic_recovery_top_topics`. ELBO trend is here.
"""
import numpy as np
import pytest


def _generate_synthetic_corpus(D, V, K, docs_avg_len, seed):
    """Generate (true_beta, docs_as_BOWDocuments) under standard LDA.

    LDA-shaped data is fine for testing HDP fits; the HDP will simply
    learn that K topics are active and the rest of the truncation is
    unused. Same generator pattern as test_lda_integration.py.
    """
    from spark_vi.core import BOWDocument
    rng = np.random.default_rng(seed)

    true_beta = rng.dirichlet(np.full(V, 0.05), size=K)
    docs = []
    for d in range(D):
        doc_len = max(2, int(rng.poisson(docs_avg_len)))
        theta_d = rng.dirichlet(np.full(K, 0.3))
        topics = rng.choice(K, size=doc_len, p=theta_d)
        words = np.array([rng.choice(V, p=true_beta[t]) for t in topics])
        unique, counts = np.unique(words, return_counts=True)
        docs.append(BOWDocument(
            indices=unique.astype(np.int32),
            counts=counts.astype(np.float64),
            length=int(counts.sum()),
        ))
    return true_beta, docs


@pytest.mark.slow
def test_online_hdp_short_fit_returns_finite_elbo_trace(spark):
    from spark_vi.core import VIConfig, VIRunner
    from spark_vi.models import OnlineHDP

    _, docs = _generate_synthetic_corpus(D=200, V=50, K=5,
                                         docs_avg_len=15, seed=0)
    rdd = spark.sparkContext.parallelize(docs, numSlices=2).persist()
    rdd.count()  # materialize for VIRunner's strict cache precondition

    np.random.seed(0)
    model = OnlineHDP(T=10, K=5, vocab_size=50)
    runner = VIRunner(model, config=VIConfig(max_iterations=10))
    result = runner.fit(rdd)

    assert result.elbo_trace is not None
    assert len(result.elbo_trace) >= 10
    assert all(np.isfinite(v) for v in result.elbo_trace)
    assert np.all(result.global_params["lambda"] > 0)
    assert np.all(result.global_params["u"] > 0)
    assert np.all(result.global_params["v"] > 0)


@pytest.mark.slow
def test_online_hdp_elbo_smoothed_endpoints_show_overall_improvement(spark):
    """Smoothed-endpoint ELBO trend must improve over a 30+iter fit.

    Mirrors test_lda_integration.py:test_vanilla_lda_elbo_smoothed_*.
    NOT a monotonicity check — SVI noise produces 100+ ELBO-unit drops
    mid-trace even on healthy fits. Endpoint trend on the smoothed
    series catches sign errors and runaway divergence.
    """
    from spark_vi.core import VIConfig, VIRunner
    from spark_vi.models import OnlineHDP

    _, docs = _generate_synthetic_corpus(D=200, V=50, K=5,
                                         docs_avg_len=15, seed=42)
    rdd = spark.sparkContext.parallelize(docs, numSlices=2).persist()
    rdd.count()  # materialize for VIRunner's strict cache precondition

    np.random.seed(42)
    model = OnlineHDP(T=10, K=5, vocab_size=50)
    runner = VIRunner(model, config=VIConfig(max_iterations=30))
    result = runner.fit(rdd)

    trace = np.asarray(result.elbo_trace)
    window = 10
    assert len(trace) >= window, (
        f"need at least {window} iterations for smoothing, got {len(trace)}"
    )
    smooth = np.convolve(trace, np.ones(window) / window, mode="valid")
    assert smooth[-1] > smooth[0], (
        f"Smoothed ELBO endpoints went backward: "
        f"start={smooth[0]:.3f}, end={smooth[-1]:.3f}. "
        f"Indicates a sign error or wrong-direction update."
    )
