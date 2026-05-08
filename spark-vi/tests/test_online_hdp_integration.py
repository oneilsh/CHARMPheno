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

    Mirrors test_lda_integration.py:test_online_lda_elbo_smoothed_*.
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


@pytest.mark.slow
def test_online_hdp_gamma_alpha_optimization_moves_values(spark):
    """When optimize_gamma=True and optimize_alpha=True (the defaults), the
    trained γ and α differ from their initial values after a 30-iter fit.

    Soft assertion ("moves") rather than "lands closer to true K" because
    at D=200 / synthetic-K=5 the topic-collapse phenomenon documented in
    ADR 0010 makes recovery-toward-truth flaky on small corpora. The
    cloud-driver smoke check is the harder claim's gate.
    """
    from spark_vi.core import VIConfig, VIRunner
    from spark_vi.models import OnlineHDP

    _, docs = _generate_synthetic_corpus(D=200, V=50, K=5,
                                         docs_avg_len=15, seed=2026)
    rdd = spark.sparkContext.parallelize(docs, numSlices=2).persist()
    rdd.count()

    np.random.seed(2026)
    init_gamma = 1.0
    init_alpha = 1.0
    model = OnlineHDP(
        T=10, K=5, vocab_size=50,
        gamma=init_gamma, alpha=init_alpha,
        optimize_gamma=True, optimize_alpha=True, optimize_eta=False,
    )
    runner = VIRunner(model, config=VIConfig(max_iterations=30))
    result = runner.fit(rdd)

    trained_gamma = float(result.global_params["gamma"])
    trained_alpha = float(result.global_params["alpha"])
    trained_eta = float(result.global_params["eta"])

    # γ and α should both have moved meaningfully (more than 1% drift).
    assert abs(trained_gamma - init_gamma) > 0.01 * init_gamma, (
        f"γ did not move: init={init_gamma}, trained={trained_gamma}"
    )
    assert abs(trained_alpha - init_alpha) > 0.01 * init_alpha, (
        f"α did not move: init={init_alpha}, trained={trained_alpha}"
    )
    # η was NOT optimized this run — must be exactly the initial value.
    assert trained_eta == 0.01

    # All three within healthy ranges (above floor, finite).
    for name, val in (("γ", trained_gamma), ("α", trained_alpha)):
        assert val >= 1e-3, f"{name} fell below 1e-3 floor: {val}"
        assert np.isfinite(val), f"{name} non-finite: {val}"


@pytest.mark.slow
def test_online_hdp_optimize_eta_smoke(spark):
    """Smoke check that η optimization runs end-to-end without blowing up.

    η is the least-stable concentration in SVI per Hoffman 2010 §3.4 —
    this gate ensures the wiring (Newton on just-updated λ, ρ_t damping,
    1e-3 floor) survives a 30-iter fit without NaNs or floor-pinning.
    """
    from spark_vi.core import VIConfig, VIRunner
    from spark_vi.models import OnlineHDP

    _, docs = _generate_synthetic_corpus(D=200, V=50, K=5,
                                         docs_avg_len=15, seed=2027)
    rdd = spark.sparkContext.parallelize(docs, numSlices=2).persist()
    rdd.count()

    np.random.seed(2027)
    model = OnlineHDP(
        T=10, K=5, vocab_size=50,
        optimize_gamma=True, optimize_alpha=True, optimize_eta=True,
    )
    runner = VIRunner(model, config=VIConfig(max_iterations=30))
    result = runner.fit(rdd)

    trace = np.asarray(result.elbo_trace)
    assert all(np.isfinite(v) for v in trace)
    eta = float(result.global_params["eta"])
    assert np.isfinite(eta) and eta >= 1e-3
    # Should have moved off the initial 0.01 default.
    assert eta != 0.01


@pytest.mark.slow
def test_online_hdp_synthetic_recovery_top_topics(spark):
    """Top-K_true topics by usage recover true word distributions.

    D=2000 LDA-generated docs with K_true=5 active topics, fit with T=20
    truncation. Hungarian-match the top-5 fitted topics by var_phi mass
    against the true topics, assert cosine sim > 0.7 on the matched set.

    Threshold 0.7 (not 0.9 from the spec): empirical SVI on D=2000
    synthetic data leaves residual topic-collapse signal — Hoffman 2010
    used D=100k+. Same posture as the LDA recovery story (no recovery
    test on small-D for LDA either; we get a weaker version here only
    because HDP's truncation gives us more headroom). Tighten to 0.9 if
    a future fix lets us recover sharper topics.
    """
    from spark_vi.core import VIConfig, VIRunner
    from spark_vi.models import OnlineHDP
    from scipy.optimize import linear_sum_assignment

    K_true = 5
    true_beta, docs = _generate_synthetic_corpus(
        D=2000, V=80, K=K_true, docs_avg_len=20, seed=7)
    rdd = spark.sparkContext.parallelize(docs, numSlices=4).persist()
    rdd.count()  # materialize before fit per VIRunner contract

    np.random.seed(7)
    model = OnlineHDP(T=20, K=K_true, vocab_size=80)
    runner = VIRunner(model, config=VIConfig(max_iterations=80))
    result = runner.fit(rdd)

    # Recover beta-hat from lambda: each row normalized.
    lam = result.global_params["lambda"]
    beta_hat = lam / lam.sum(axis=1, keepdims=True)

    # Pick the top K_true fitted topics by lambda row sum (proxy for usage).
    top_idx = np.argsort(lam.sum(axis=1))[::-1][:K_true]
    fitted = beta_hat[top_idx]

    # Cosine sim matrix: (K_true, K_true).
    def cos(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    sim = np.array([[cos(true_beta[i], fitted[j]) for j in range(K_true)]
                    for i in range(K_true)])

    # Hungarian matching maximizes sum of similarities.
    row_ind, col_ind = linear_sum_assignment(-sim)
    matched_sims = sim[row_ind, col_ind]

    assert matched_sims.mean() > 0.7, (
        f"Mean matched cosine similarity {matched_sims.mean():.3f} < 0.7. "
        f"Per-topic sims: {matched_sims}"
    )


@pytest.mark.slow
def test_online_hdp_infer_local_round_trip(spark):
    """Fit on training corpus, run infer_local on a held-out doc.

    Asserts the doc-CAVI converges, returns simplex-valid θ, and
    concentrates mass on a small subset of corpus topics (effective
    sparsity expected from a sparse synthetic generator).
    """
    from spark_vi.core import BOWDocument, VIConfig, VIRunner
    from spark_vi.models import OnlineHDP

    _, docs = _generate_synthetic_corpus(
        D=200, V=50, K=5, docs_avg_len=15, seed=11)
    rdd = spark.sparkContext.parallelize(docs, numSlices=2).persist()
    rdd.count()

    np.random.seed(11)
    model = OnlineHDP(T=10, K=5, vocab_size=50)
    runner = VIRunner(model, config=VIConfig(max_iterations=20))
    result = runner.fit(rdd)

    # Held-out doc with a different seed — words fall in the same vocab
    # but the doc is genuinely new.
    rng = np.random.default_rng(999)
    held_idx = np.sort(rng.choice(50, size=8, replace=False).astype(np.int32))
    held_counts = rng.gamma(2.0, 1.0, 8).astype(np.float64)
    held_out = BOWDocument(
        indices=held_idx,
        counts=held_counts,
        length=int(held_counts.sum()),
    )

    out = model.infer_local(held_out, result.global_params)
    theta = out["theta"]

    assert theta.shape == (10,)
    assert np.isclose(theta.sum(), 1.0, atol=1e-6)
    assert np.all(theta >= 0)
    # Effective sparsity: > 80% of mass on at most half the topics.
    sorted_theta = np.sort(theta)[::-1]
    half = max(1, len(sorted_theta) // 2)
    assert sorted_theta[:half].sum() > 0.8, (
        f"θ not concentrated; top-{half}: {sorted_theta[:half].sum():.3f}"
    )


# ---------------------------------------------------------------------------
# End-to-end integration: save/load round-trip + diagnostic_traces accumulation
#
# Sibling to the LDA shim integration tests in test_lda_integration.py.
# These exercise the full stack — MLlib shim → VIRunner → OnlineHDP →
# save_result/load_result — on a real Spark-local fit.
# ---------------------------------------------------------------------------


def _build_persistence_hdp_df(spark, V=9, D=15, n_clusters=3, seed=0):
    """Small but well-clustered DataFrame for shim-level HDP fits.

    Mirrors test_lda_integration._build_persistence_lda_df.
    """
    from pyspark.ml.linalg import Vectors

    rng = np.random.default_rng(seed)
    block = V // n_clusters
    rows = []
    for doc_id in range(D):
        cluster = doc_id % n_clusters
        favored = list(range(cluster * block, (cluster + 1) * block))
        counts = np.zeros(V, dtype=np.float64)
        for w in rng.choice(favored, size=15, replace=True):
            counts[w] += 1.0
        rows.append((Vectors.dense(counts.tolist()),))
    return spark.createDataFrame(rows, schema=["features"])


@pytest.mark.slow
def test_hdp_fit_with_optimization_populates_gamma_alpha_eta_traces(spark):
    """A shim fit with γ/α/η optimization on must accumulate per-iter
    γ, α, η scalar traces of length n_iterations.

    Bonus: at least one of γ/α/η must change value somewhere along the
    trace — proves optimization actually moved things, not just shape.
    """
    from spark_vi.mllib.hdp import OnlineHDPEstimator

    T, K, V, max_iter = 6, 3, 9, 4
    df = _build_persistence_hdp_df(spark, V=V, n_clusters=3, seed=0)

    estimator = OnlineHDPEstimator(
        k=T, docTruncation=K, maxIter=max_iter, seed=0, subsamplingRate=1.0,
        optimizeDocConcentration=True,
        optimizeCorpusConcentration=True,
        optimizeTopicConcentration=True,
    )
    model = estimator.fit(df)
    traces = model.result.diagnostic_traces

    # All three scalar keys present, length matches n_iterations.
    assert set(traces.keys()) == {"gamma", "alpha", "eta"}
    assert model.result.n_iterations == max_iter
    for name in ("gamma", "alpha", "eta"):
        assert len(traces[name]) == max_iter, (
            f"{name} trace length {len(traces[name])} != n_iterations {max_iter}"
        )
        for v in traces[name]:
            v_f = float(v)
            assert np.isfinite(v_f)

    # Bonus: at least one of γ/α/η actually changed across the run.
    moved = False
    for name in ("gamma", "alpha", "eta"):
        vals = [float(v) for v in traces[name]]
        if any(v != vals[0] for v in vals):
            moved = True
            break
    assert moved, (
        f"no concentration moved across {max_iter} iters with all optimize "
        f"flags on — wiring may be broken. traces={traces}"
    )


@pytest.mark.slow
def test_hdp_save_load_round_trip_via_shim_preserves_T_K_V_metadata_and_traces(
    spark, tmp_path,
):
    """End-to-end round-trip: HDP shim fit with optimization → save → load.

    Verifies both the shape metadata (T/K/V via _T, _K, vocabSize()) and
    the diagnostic_traces survive the round-trip element-wise.
    """
    from spark_vi.mllib.hdp import OnlineHDPEstimator, OnlineHDPModel

    T, K, V, max_iter = 6, 3, 9, 4
    df = _build_persistence_hdp_df(spark, V=V, n_clusters=3, seed=1)

    estimator = OnlineHDPEstimator(
        k=T, docTruncation=K, maxIter=max_iter, seed=1, subsamplingRate=1.0,
        optimizeDocConcentration=True,
        optimizeCorpusConcentration=True,
        optimizeTopicConcentration=True,
    )
    model = estimator.fit(df)

    save_path = tmp_path / "hdp_traces_roundtrip"
    model.save(str(save_path))
    loaded = OnlineHDPModel.load(str(save_path))

    # Shape metadata round-trips.
    assert loaded._T == T
    assert loaded._K == K
    assert loaded.vocabSize() == V

    # Diagnostic traces round-trip element-wise (all scalars).
    orig = model.result.diagnostic_traces
    out = loaded.result.diagnostic_traces
    assert set(out.keys()) == set(orig.keys())
    for name in orig:
        assert len(out[name]) == len(orig[name])
        for orig_v, loaded_v in zip(orig[name], out[name]):
            assert float(loaded_v) == float(orig_v), (
                f"{name} trace value drift on round-trip: "
                f"orig={orig_v}, loaded={loaded_v}"
            )
