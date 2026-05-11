"""End-to-end Spark-local integration tests for OnlineLDA.

Hermetic by construction: each test builds its own synthetic LDA dataset
inside the test, no dependency on simulate_lda_omop.py.

Scope: these tests verify that the VIRunner <-> OnlineLDA integration
ends up in a sensible place — fitting actually drives the ELBO up over
iterations, and a fit run produces a well-formed VIResult. Recovery
quality (does the fitted beta match the synthetic ground truth?) is
intentionally NOT tested here; topic collapse at small synthetic-corpus
scales is a known SVI characteristic (MLlib has the same behavior;
Hoffman 2010 §4 uses corpora of 100K-352K documents). Recovery is
verified by the MLlib parity test in charmpheno/tests/test_lda_compare.py
which runs both OnlineLDA and pyspark.ml.clustering.LDA on the same
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
def test_online_lda_fit_produces_well_formed_result(spark):
    """A short Spark-local fit returns a VIResult with positive lambda and a finite ELBO trace."""
    from spark_vi.core import VIConfig, VIRunner
    from spark_vi.models.lda import OnlineLDA

    K, V, D = 3, 30, 100
    np.random.seed(0)
    _, docs = _generate_synthetic_corpus(D=D, V=V, K=K, docs_avg_len=40, seed=0)
    rdd = spark.sparkContext.parallelize(docs, numSlices=2).persist()
    rdd.count()  # materialize for VIRunner's strict cache precondition

    cfg = VIConfig(
        max_iterations=20,
        mini_batch_fraction=0.3,
        random_seed=0,
        convergence_tol=1e-9,
    )
    result = VIRunner(OnlineLDA(K=K, vocab_size=V), config=cfg).fit(rdd)

    assert result.global_params["lambda"].shape == (K, V)
    assert (result.global_params["lambda"] > 0).all()
    assert len(result.elbo_trace) >= 1
    assert all(np.isfinite(v) for v in result.elbo_trace)


@pytest.mark.slow
def test_online_lda_elbo_smoothed_endpoints_show_overall_improvement(spark):
    """Endpoint check on a 10-iter moving average: smoothed[-1] > smoothed[0].

    NOT a monotonicity check, despite what the surface intuition might
    suggest. Empirically, even with a healthy fit, the 10-window smoothed
    trace has ~50% positive consecutive differences — within-trace local
    drops of 100+ ELBO units are normal for stochastic VI on this corpus
    size. Hard monotonicity is too strict; smoothed monotonicity is also
    too strict. What survives is the *endpoint trend*: the very early
    iterations climb hard from a random init, and tail noise can't undo
    those gains in any healthy run.

    What this catches:
      * gross sign-flip regressions in any ELBO-driven update
        (a Newton step pushing α away from the optimum, or a global-KL
        term added with wrong sign in compute_elbo) — the trace ends
        below where it started.
      * runaway divergence — `smooth[-1] > smooth[0]` fails when a fit
        actually moves backward from random init.

    What this DOES NOT catch:
      * a regression that hurts the bound *after* the easy early gains,
        as long as smooth[-1] is still above smooth[0]. A bug that drags
        the bound down by 50 ELBO units per iteration in mid-trace would
        pass this test if the early gain is large enough.

    For a stronger gate (mid-run trend + largest-local-drop bound), see
    the follow-up notes on the lda-concentration-opt branch. As of this
    writing the endpoint check is the only gate; it remains the load-
    bearing regression test for the SVI machinery, modulo that gap.
    """
    from spark_vi.core import VIConfig, VIRunner
    from spark_vi.models.lda import OnlineLDA

    np.random.seed(1)
    _, docs = _generate_synthetic_corpus(D=100, V=30, K=3, docs_avg_len=40, seed=7)
    rdd = spark.sparkContext.parallelize(docs, numSlices=2).persist()
    rdd.count()  # materialize for VIRunner's strict cache precondition
    cfg = VIConfig(max_iterations=40, mini_batch_fraction=0.3,
                    random_seed=7, convergence_tol=1e-9)
    result = VIRunner(OnlineLDA(K=3, vocab_size=30), config=cfg).fit(rdd)

    trace = np.asarray(result.elbo_trace)
    window = 10
    assert len(trace) >= window, f"need at least {window} iterations for smoothing"
    smooth = np.convolve(trace, np.ones(window) / window, mode="valid")
    assert smooth[-1] > smooth[0], (
        f"Smoothed ELBO endpoints went backward: start={smooth[0]:.3f}, "
        f"end={smooth[-1]:.3f}. This indicates a sign error or wrong-direction "
        f"update large enough to overwhelm the early random-init gains."
    )


@pytest.mark.slow
def test_alpha_optimization_runs_end_to_end_without_regression(spark):
    """End-to-end smoke gate for asymmetric α optimization.

    A 40-iter fit with optimize_alpha=True on a synthetic LDA corpus must:
      * complete without raising,
      * produce a length-K α with all finite components,
      * respect the 1e-3 floor on every component,
      * not blow up to absurd magnitudes,
      * actually move from the 1/K initialization (proves the wiring fires).

    Why this signal rather than "α drifts toward synthetic truth": LDA
    topic ordering is arbitrary, and at small synthetic-corpus scales
    (D≪10K) one true topic's mass routes to the α floor under topic
    collapse — independent of optimization quality. The module docstring
    already spells this out for β; the same applies to α. Hoffman 2010 §4
    used corpora of 100K-352K documents to verify recovery; we cannot
    replicate that scale in a unit-test-runtime budget.

    The faithful-math gate that this test does NOT replicate lives upstream:
      * test_alpha_newton_step_recovers_known_alpha_on_synthetic
        (in tests/test_lda_math.py) verifies the closed-form Newton step
        against an idealized (asymptotically concentrated) E[log θ_d] sum.
      * test_online_lda_elbo_smoothed_endpoints_show_overall_improvement
        (this file) catches sign or wrong-direction regressions in any
        ELBO-driven update by gating on the smoothed trace trend.

    What this test does catch:
      * wiring failure: optimize_alpha=True silently no-ops → α stays at 1/K.
      * NaN propagation in the Newton step → fails finite check.
      * floor mis-applied: α goes negative or to zero → fails floor check.
      * gross scale errors: α blows up to runaway values → fails range check.
    """
    import numpy as np
    from spark_vi.core import VIConfig, VIRunner, BOWDocument
    from spark_vi.models.lda import OnlineLDA

    K, V, D = 3, 30, 200
    rng = np.random.default_rng(2)
    true_alpha = np.array([0.1, 0.5, 0.9])
    true_beta = rng.dirichlet(np.full(V, 0.05), size=K)
    docs = []
    for d in range(D):
        theta_d = rng.dirichlet(true_alpha)
        N_d = max(1, rng.poisson(40))
        zs = rng.choice(K, size=N_d, p=theta_d)
        ws = np.array([rng.choice(V, p=true_beta[z]) for z in zs])
        unique, counts = np.unique(ws, return_counts=True)
        docs.append(BOWDocument(
            indices=unique.astype(np.int32),
            counts=counts.astype(np.float64),
            length=int(counts.sum()),
        ))

    rdd = spark.sparkContext.parallelize(docs, numSlices=2).persist()
    rdd.count()

    cfg = VIConfig(max_iterations=40, mini_batch_fraction=0.3,
                   random_seed=2, convergence_tol=1e-9)
    model = OnlineLDA(K=K, vocab_size=V, optimize_alpha=True)
    result = VIRunner(model, config=cfg).fit(rdd)

    final_alpha = result.global_params["alpha"]
    init_alpha = np.full(K, 1.0 / K)

    # Shape and finiteness.
    assert final_alpha.shape == (K,), f"wrong shape: {final_alpha.shape}"
    assert np.isfinite(final_alpha).all(), f"non-finite α: {final_alpha}"

    # Floor honored on every component.
    assert (final_alpha >= 1e-3).all(), (
        f"floor violated: {final_alpha}"
    )

    # No blow-up. α should stay in a reasonable Dirichlet range; >100 is a
    # signal that the Newton step has lost its damping or scaling.
    assert (final_alpha < 100.0).all(), f"α blew up: {final_alpha}"

    # Optimization actually fired (proves wiring is live).
    assert not np.allclose(final_alpha, init_alpha, atol=1e-6), (
        f"α stayed at 1/K = {init_alpha} — optimize_alpha may not be wired"
    )


@pytest.mark.slow
def test_alpha_optimization_drifts_toward_corpus_truth_at_D10k(spark):
    """At D=10k synthetic docs with asymmetric true α, α moves measurably
    toward the truth.

    Why this works at D=10k but not at the D=200 used in the smoke test
    above: at small (D, K) the random λ initialization can starve a fitted
    topic of corpus mass, and the resulting α component pins to the floor
    independent of optimization quality. With D=10k, every fitted topic
    accrues enough mass for the Newton step to differentiate α
    components meaningfully — empirical sweep over five seeds at this
    configuration showed +43.6% to +62.5% L1 reduction toward truth
    after 300 iterations, with no all-floor collapses (post the
    `e_log_theta_sum` digamma-underflow fix at models/lda.py).

    Threshold: ≥30% L1 reduction. Worst seed empirically observed +43.6%,
    so 30% leaves margin for cross-platform numerical drift.

    Topic-permutation handling: LDA topic ordering is arbitrary. We
    Hungarian-match fitted topics to true topics by β cosine similarity
    before comparing α — without this step, "α distance to truth" is
    dominated by permutation noise rather than recovery quality.

    Runtime: ~30s on a small Spark-local cluster.
    """
    from scipy.optimize import linear_sum_assignment

    from spark_vi.core import BOWDocument, VIConfig, VIRunner
    from spark_vi.models.lda import OnlineLDA

    K, V, D = 3, 100, 10_000
    docs_avg_len = 100
    true_alpha = np.array([0.1, 0.5, 0.9])

    rng_b = np.random.default_rng(11)
    true_beta = rng_b.dirichlet(np.full(V, 0.05), size=K)

    rng = np.random.default_rng(11)
    docs = []
    for _ in range(D):
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

    # The model uses np.random globally for λ and γ initialization (see the
    # TODO at models/lda.py near gamma_init). Until that's plumbed through
    # cfg.random_seed, we seed the global RNG explicitly so the test is
    # reproducible. Empirical sweep showed +43.6%–+62.5% drift across five
    # seeds; 7 is a representative healthy seed (+62.5% in the sweep).
    np.random.seed(7)
    rdd = spark.sparkContext.parallelize(docs, numSlices=4).persist()
    rdd.count()

    cfg = VIConfig(
        max_iterations=300,
        mini_batch_fraction=0.05,
        random_seed=7,
        convergence_tol=1e-9,
    )
    model = OnlineLDA(K=K, vocab_size=V, optimize_alpha=True)
    result = VIRunner(model, config=cfg).fit(rdd)

    fitted_alpha = result.global_params["alpha"]
    fitted_lambda = result.global_params["lambda"]
    fitted_beta = fitted_lambda / fitted_lambda.sum(axis=1, keepdims=True)

    # Hungarian alignment by β cosine.
    norm_t = true_beta / np.linalg.norm(true_beta, axis=1, keepdims=True)
    norm_f = fitted_beta / np.linalg.norm(fitted_beta, axis=1, keepdims=True)
    cosine = norm_f @ norm_t.T
    fitted_idx, true_idx = linear_sum_assignment(-cosine)
    perm = np.empty(K, dtype=int)
    for fi, ti in zip(fitted_idx, true_idx):
        perm[ti] = fi
    fitted_alpha_aligned = fitted_alpha[perm]

    init_alpha = np.full(K, 1.0 / K)
    l1_init = float(np.abs(init_alpha - true_alpha).sum())
    l1_fit = float(np.abs(fitted_alpha_aligned - true_alpha).sum())
    drift_ratio = 1.0 - l1_fit / l1_init

    assert drift_ratio >= 0.30, (
        f"Expected ≥30% L1 reduction toward true α; got {100 * drift_ratio:.1f}%. "
        f"true α = {true_alpha}, fitted α (aligned) = {fitted_alpha_aligned}, "
        f"L1(init→truth) = {l1_init:.4f}, L1(fitted→truth) = {l1_fit:.4f}, "
        f"β cosines = {cosine[fitted_idx, true_idx]}."
    )


# ---------------------------------------------------------------------------
# End-to-end integration: save/load round-trip + diagnostic_traces accumulation
#
# These tests exercise the full stack — MLlib shim → VIRunner → OnlineLDA →
# save_result/load_result — on a real Spark-local fit. They close the gap
# between the unit-level Param tests in test_mllib_lda_persistence.py (which
# don't drive iteration_diagnostics on the hot path) and the per-method
# tests in test_runner.py / test_export.py (which use stub models, not
# OnlineLDA).
# ---------------------------------------------------------------------------


def _build_persistence_lda_df(spark, K=3, V=9, D=15, seed=0):
    """Small, well-clustered DataFrame for shim-level fits.

    Each doc favors a contiguous V/K-sized vocab block, giving the optimizer
    enough signal to move α and η in 3-5 iterations without paying for a
    300-doc synthetic corpus on every test.
    """
    from pyspark.ml.linalg import Vectors

    rng = np.random.default_rng(seed)
    block = V // K
    rows = []
    for doc_id in range(D):
        topic = doc_id % K
        favored = list(range(topic * block, (topic + 1) * block))
        counts = np.zeros(V, dtype=np.float64)
        for w in rng.choice(favored, size=15, replace=True):
            counts[w] += 1.0
        rows.append((Vectors.dense(counts.tolist()),))
    return spark.createDataFrame(rows, schema=["features"])


@pytest.mark.slow
def test_lda_fit_with_optimization_populates_alpha_and_eta_traces(spark):
    """A shim fit with both α and η optimization on must accumulate per-iter
    α (length-K vector) and η (scalar) traces of length n_iterations.

    Closes the gap left by test_mllib_lda_persistence.py: those tests verify
    Estimator Params and Model save/load round-trip global_params, but never
    run iteration_diagnostics through the runner's accumulation path.
    """
    from spark_vi.mllib.lda import OnlineLDAEstimator

    K, V, max_iter = 3, 9, 4
    df = _build_persistence_lda_df(spark, K=K, V=V, seed=0)

    estimator = OnlineLDAEstimator(
        k=K, maxIter=max_iter, seed=0, subsamplingRate=1.0,
        optimizeDocConcentration=True, optimizeTopicConcentration=True,
    )
    model = estimator.fit(df)
    traces = model.result.diagnostic_traces

    # Both keys present, both with length n_iterations.
    assert set(traces.keys()) == {"alpha", "eta"}
    assert model.result.n_iterations == max_iter
    assert len(traces["alpha"]) == max_iter
    assert len(traces["eta"]) == max_iter

    # Per-iter α is a K-vector; per-iter η is a scalar.
    for entry in traces["alpha"]:
        arr = np.asarray(entry)
        assert arr.shape == (K,), f"expected α shape ({K},), got {arr.shape}"
        assert np.isfinite(arr).all()
    for entry in traces["eta"]:
        assert np.isscalar(entry) or np.ndim(entry) == 0
        assert np.isfinite(float(entry))


@pytest.mark.slow
def test_lda_save_load_round_trip_via_shim_preserves_alpha_eta_traces(
    spark, tmp_path,
):
    """End-to-end round-trip: shim fit with optimization → save → load.

    The loaded Model's diagnostic_traces must match the pre-save values
    element-wise — α arrays via assert_array_equal, η scalars via ==.
    """
    from spark_vi.mllib.lda import OnlineLDAEstimator, OnlineLDAModel

    K, V, max_iter = 3, 9, 4
    df = _build_persistence_lda_df(spark, K=K, V=V, seed=1)

    estimator = OnlineLDAEstimator(
        k=K, maxIter=max_iter, seed=1, subsamplingRate=1.0,
        optimizeDocConcentration=True, optimizeTopicConcentration=True,
    )
    model = estimator.fit(df)

    save_path = tmp_path / "lda_traces_roundtrip"
    model.save(str(save_path))
    loaded = OnlineLDAModel.load(str(save_path))

    orig = model.result.diagnostic_traces
    out = loaded.result.diagnostic_traces

    assert set(out.keys()) == set(orig.keys())
    assert len(out["alpha"]) == len(orig["alpha"])
    assert len(out["eta"]) == len(orig["eta"])

    for orig_alpha, loaded_alpha in zip(orig["alpha"], out["alpha"]):
        np.testing.assert_array_equal(np.asarray(loaded_alpha),
                                      np.asarray(orig_alpha))
    for orig_eta, loaded_eta in zip(orig["eta"], out["eta"]):
        assert float(loaded_eta) == float(orig_eta)


@pytest.mark.slow
def test_lda_resume_from_continues_iteration_count_and_elbo_trace(
    spark, tmp_path,
):
    """Resume from a saved fit: iteration counter and ELBO trace continue
    from where the prior run left off, not restart at zero.

    Step 1: fit for 3 iters with saveDir set → run1 directory.
    Step 2: fit a fresh Estimator for 3 more iters with saveDir + resumeFrom
            both pointing at run1.

    Asserts:
      * run1 has n_iterations == 3, len(elbo_trace) == 3.
      * run2 has n_iterations == 6 (counter continues, not restarts).
      * run2.elbo_trace has length 6 and its first 3 elements match run1's
        elbo_trace exactly (history is preserved on resume).
    """
    from spark_vi.mllib.lda import OnlineLDAEstimator

    K, V = 3, 9
    df = _build_persistence_lda_df(spark, K=K, V=V, seed=2)

    save_dir = tmp_path / "run1"

    estimator_a = OnlineLDAEstimator(
        k=K, maxIter=3, seed=2, subsamplingRate=1.0,
        optimizeDocConcentration=True, optimizeTopicConcentration=False,
    )
    estimator_a.setSaveDir(str(save_dir))
    model_a = estimator_a.fit(df)
    result_a = model_a.result

    assert result_a.n_iterations == 3
    assert len(result_a.elbo_trace) == 3

    estimator_b = OnlineLDAEstimator(
        k=K, maxIter=3, seed=2, subsamplingRate=1.0,
        optimizeDocConcentration=True, optimizeTopicConcentration=False,
    )
    estimator_b.setSaveDir(str(save_dir))
    estimator_b.setResumeFrom(str(save_dir))
    model_b = estimator_b.fit(df)
    result_b = model_b.result

    assert result_b.n_iterations == 6, (
        f"resume should continue iteration counter; got {result_b.n_iterations}"
    )
    assert len(result_b.elbo_trace) == 6, (
        f"resumed elbo_trace should be length 6; got {len(result_b.elbo_trace)}"
    )
    # First 3 entries match the pre-resume trace exactly (resume preserves
    # history).
    for i, (a, b) in enumerate(zip(result_a.elbo_trace, result_b.elbo_trace[:3])):
        assert a == b, (
            f"resume mutated history at index {i}: pre={a}, post={b}"
        )
