"""Pure-numpy tests for OnlineHDP module-level math helpers and CAVI.

No Spark — these test the math in isolation. Single document, hand-checked
shapes and values where possible.
"""
import numpy as np
import pytest
from scipy.special import digamma


def test_log_normalize_rows_simplex_invariant():
    """exp(out) rows sum to 1; out and input differ by a constant per row."""
    from spark_vi.models.online_hdp import _log_normalize_rows

    rng = np.random.default_rng(0)
    M = rng.normal(size=(5, 7))
    out = _log_normalize_rows(M)

    assert out.shape == M.shape
    assert np.allclose(np.exp(out).sum(axis=1), 1.0)

    # Each row of (M - out) should be a single repeated constant (the log-norm).
    diff = M - out
    assert np.allclose(diff, diff[:, [0]])


def test_log_normalize_rows_handles_large_magnitudes():
    """Numerical stability under large positive entries (no inf/nan)."""
    from spark_vi.models.online_hdp import _log_normalize_rows

    M = np.array([[1000.0, 1001.0, 999.0],
                  [-1000.0, -1001.0, -999.0]])
    out = _log_normalize_rows(M)

    assert np.all(np.isfinite(out))
    assert np.allclose(np.exp(out).sum(axis=1), 1.0)


def test_expect_log_sticks_uniform_prior_known_values():
    """For Beta(1, 1) sticks, the math reduces to a closed form we can hand-check.

    With a = b = 1 (uniform Beta), digamma(1) = -gamma_euler ≈ -0.5772 and
    digamma(2) = 1 - gamma_euler ≈ 0.4228. So:
      Elog_W   = digamma(1) - digamma(2) = -1.0
      Elog_1mW = digamma(1) - digamma(2) = -1.0
    For T=3 (a, b each length 2):
      out[0] = Elog_W[0]                                = -1.0
      out[1] = Elog_W[1] + Elog_1mW[0]                  = -2.0
      out[2] =             Elog_1mW[0] + Elog_1mW[1]    = -2.0
    """
    from spark_vi.models.online_hdp import _expect_log_sticks

    a = np.array([1.0, 1.0])
    b = np.array([1.0, 1.0])
    out = _expect_log_sticks(a, b)

    assert out.shape == (3,)
    assert np.allclose(out, [-1.0, -2.0, -2.0])


def test_expect_log_sticks_truncation_handles_last_atom():
    """For T atoms with (T-1) sticks, the trailing entry receives only the
    cumulative E[log(1-W)] sum (q(W_T = 1) = 1 ⇒ E[log W_T] = 0)."""
    from spark_vi.models.online_hdp import _expect_log_sticks

    rng = np.random.default_rng(42)
    T_minus_1 = 5
    a = 1.0 + rng.gamma(2.0, 1.0, T_minus_1)
    b = 1.0 + rng.gamma(2.0, 1.0, T_minus_1)
    out = _expect_log_sticks(a, b)

    dig_sum = digamma(a + b)
    Elog_1mW = digamma(b) - dig_sum
    expected_last = Elog_1mW.sum()

    assert out.shape == (T_minus_1 + 1,)
    assert np.isclose(out[-1], expected_last)

    # Spot-check intermediate positions to catch cumsum off-by-one.
    assert np.isclose(out[0], digamma(a[0]) - dig_sum[0])
    expected_2 = (digamma(a[2]) - dig_sum[2]) + (
        (digamma(b[0]) - dig_sum[0]) + (digamma(b[1]) - dig_sum[1])
    )
    assert np.isclose(out[2], expected_2)


def test_beta_kl_zero_when_posterior_matches_prior():
    """KL(Beta(a, b) ‖ Beta(a, b)) = 0 element-wise."""
    from spark_vi.models.online_hdp import _beta_kl

    a = np.array([1.0, 2.0, 3.0])
    b = np.array([1.0, 2.0, 5.0])
    kl = _beta_kl(a, b, prior_a=np.array([1.0, 2.0, 3.0]),
                  prior_b=np.array([1.0, 2.0, 5.0]))

    assert np.allclose(kl, 0.0, atol=1e-12)


def test_beta_kl_zero_for_matched_corpus_prior():
    """For corpus sticks the prior is Beta(1, gamma); when (u, v) = (1, gamma)
    KL is zero."""
    from spark_vi.models.online_hdp import _beta_kl

    gamma = 1.5
    T_minus_1 = 4
    u = np.ones(T_minus_1)
    v = np.full(T_minus_1, gamma)
    kl = _beta_kl(u, v, prior_a=1.0, prior_b=gamma)

    assert np.allclose(kl, 0.0, atol=1e-12)


def test_beta_kl_positive_when_posterior_differs():
    """Concentrate the variational posterior away from the prior; KL > 0."""
    from spark_vi.models.online_hdp import _beta_kl

    u = np.array([10.0, 10.0])
    v = np.array([1.0, 1.0])
    kl = _beta_kl(u, v, prior_a=1.0, prior_b=1.0)

    assert np.all(kl > 0)


def test_beta_kl_positive_when_prior_more_concentrated():
    """KL[Beta(1,1) ‖ Beta(5,1)] > 0 — also catches a sign-flip on term 3."""
    from spark_vi.models.online_hdp import _beta_kl

    kl = _beta_kl(
        np.array([1.0, 1.0]),
        np.array([1.0, 1.0]),
        prior_a=5.0,
        prior_b=1.0,
    )
    assert np.all(kl > 0)


def _peaked_elogbeta(T: int, V: int, sharpness: float = 5.0) -> np.ndarray:
    """Stylized E[log beta] where each topic peaks on a single word.

    Topic t peaks on word t (mod V). Used to make doc-CAVI tests
    deterministic and visually inspectable. Returns shape (T, V).
    """
    eb = np.full((T, V), -sharpness, dtype=np.float64)
    for t in range(T):
        eb[t, t % V] = 0.0
    return eb


def test_doc_e_step_shape_and_simplex_contract():
    """Run one doc through CAVI; output arrays have right shapes and are valid."""
    from spark_vi.models.online_hdp import _doc_e_step, _expect_log_sticks

    T, K, V = 10, 5, 20
    Elogbeta = _peaked_elogbeta(T, V)
    u = np.ones(T - 1)
    v = np.full(T - 1, 1.0)  # gamma=1.0
    Elog_sticks_corpus = _expect_log_sticks(u, v)

    indices = np.array([0, 1, 2, 3], dtype=np.int32)
    counts = np.array([3.0, 2.0, 1.0, 4.0], dtype=np.float64)
    Elogbeta_doc = Elogbeta[:, indices]

    result = _doc_e_step(
        indices=indices,
        counts=counts,
        Elogbeta_doc=Elogbeta_doc,
        Elog_sticks_corpus=Elog_sticks_corpus,
        alpha=1.0,
        K=K,
        max_iter=20,
        tol=1e-4,
        warmup=3,
    )

    a = result["a"]
    b = result["b"]
    phi = result["phi"]
    var_phi = result["var_phi"]

    assert a.shape == (K - 1,)
    assert b.shape == (K - 1,)
    assert phi.shape == (len(indices), K)
    assert var_phi.shape == (K, T)

    assert np.all(a > 0)
    assert np.all(b > 0)
    assert np.all(np.isfinite(phi))
    assert np.all(np.isfinite(var_phi))

    # Simplex contracts.
    assert np.allclose(phi.sum(axis=1), 1.0)
    assert np.allclose(var_phi.sum(axis=1), 1.0)


def test_doc_e_step_returns_doc_elbo_terms():
    """The returned dict must include the four ELBO contributions used by
    local_update for sufficient-stat aggregation."""
    from spark_vi.models.online_hdp import _doc_e_step, _expect_log_sticks

    T, K, V = 10, 5, 20
    Elogbeta = _peaked_elogbeta(T, V)
    u = np.ones(T - 1)
    v = np.full(T - 1, 1.0)
    Elog_sticks_corpus = _expect_log_sticks(u, v)

    indices = np.array([0, 1, 2], dtype=np.int32)
    counts = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    Elogbeta_doc = Elogbeta[:, indices]

    result = _doc_e_step(
        indices=indices, counts=counts,
        Elogbeta_doc=Elogbeta_doc,
        Elog_sticks_corpus=Elog_sticks_corpus,
        alpha=1.0, K=K, max_iter=20, tol=1e-4, warmup=3,
    )

    for key in ("doc_loglik", "doc_z_term", "doc_c_term", "doc_stick_kl"):
        assert key in result, f"missing {key}"
        assert np.isfinite(result[key]), f"{key} is not finite"
    assert result["doc_stick_kl"] >= 0  # KL divergence is non-negative


def test_doc_e_step_per_iter_elbo_nondecreasing():
    """Coordinate ascent must monotonically increase the doc ELBO.

    Patch _doc_e_step to record the per-iter ELBO trace and assert no drop
    larger than numerical noise. This is the regression gate for any change
    to the var_phi / phi / (a, b) update logic.
    """
    from spark_vi.models import online_hdp as hdp

    T, K, V = 10, 5, 20
    Elogbeta = _peaked_elogbeta(T, V)
    u = np.ones(T - 1)
    v = np.full(T - 1, 1.0)
    Elog_sticks_corpus = hdp._expect_log_sticks(u, v)

    indices = np.array([0, 1, 2, 3, 4], dtype=np.int32)
    counts = np.array([5.0, 3.0, 2.0, 4.0, 1.0], dtype=np.float64)
    Elogbeta_doc = Elogbeta[:, indices]

    # Track per-iter ELBO by re-running the inner update with max_iter=i for
    # increasing i, capturing the ELBO at each completed iter count. This
    # avoids modifying _doc_e_step itself with debug instrumentation.
    elbo_trace = []
    for n_iters in range(1, 25):
        result = hdp._doc_e_step(
            indices=indices, counts=counts,
            Elogbeta_doc=Elogbeta_doc,
            Elog_sticks_corpus=Elog_sticks_corpus,
            alpha=1.0, K=K,
            max_iter=n_iters, tol=0.0,  # tol=0 => never early-break
            warmup=3,
        )
        elbo = (
            result["doc_loglik"] + result["doc_z_term"]
            + result["doc_c_term"] - result["doc_stick_kl"]
        )
        elbo_trace.append(elbo)

    # Coordinate ascent guarantees monotone ELBO increase per iter — but ONLY
    # post-warmup. During warmup the var_phi/phi updates maximize a reduced
    # objective (no prior corrections in the updates) while the ELBO eval
    # always includes the priors, so the warmup→post-warmup transition is not
    # a CA step. Test only the post-warmup tail.
    warmup = 3
    post_warmup = elbo_trace[warmup:]   # entries from max_iter=warmup+1 onward
    diffs = np.diff(post_warmup)
    assert np.all(diffs > -1e-9), (
        f"doc ELBO decreased mid-trace (post-warmup): {post_warmup}\n"
        f"diffs: {diffs}\n"
        f"first violation at step {int(np.argmin(diffs)) + warmup + 1} "
        f"(diff={float(diffs.min()):.3e})\n"
        f"This indicates a coordinate-ascent regression."
    )

    # Sanity: warmup-phase trace should at least be finite.
    assert np.all(np.isfinite(elbo_trace[: warmup])), (
        f"warmup-phase ELBO non-finite: {elbo_trace[: warmup]}"
    )


def test_online_hdp_init_validates_inputs():
    from spark_vi.models.online_hdp import OnlineHDP

    # Valid construction.
    m = OnlineHDP(T=20, K=5, vocab_size=100)
    assert m.T == 20
    assert m.K == 5
    assert m.V == 100
    assert m.alpha == 1.0
    assert m.gamma == 1.0
    assert m.eta == 0.01

    # T must be at least 2 (we need T-1 sticks at corpus level).
    with pytest.raises(ValueError, match="T"):
        OnlineHDP(T=1, K=5, vocab_size=100)

    # K must be at least 2 (we need K-1 sticks at doc level).
    with pytest.raises(ValueError, match="K"):
        OnlineHDP(T=20, K=1, vocab_size=100)

    # vocab_size must be >= 1.
    with pytest.raises(ValueError, match="vocab_size"):
        OnlineHDP(T=20, K=5, vocab_size=0)

    # Concentrations must be > 0.
    with pytest.raises(ValueError, match="alpha"):
        OnlineHDP(T=20, K=5, vocab_size=100, alpha=0.0)
    with pytest.raises(ValueError, match="gamma"):
        OnlineHDP(T=20, K=5, vocab_size=100, gamma=-1.0)
    with pytest.raises(ValueError, match="eta"):
        OnlineHDP(T=20, K=5, vocab_size=100, eta=0.0)


def test_online_hdp_init_accepts_all_optional_args():
    from spark_vi.models.online_hdp import OnlineHDP

    m = OnlineHDP(
        T=30, K=10, vocab_size=500,
        alpha=2.0, gamma=1.5, eta=0.05,
        gamma_shape=50.0,
        cavi_max_iter=50, cavi_tol=1e-3,
    )
    assert m.gamma_shape == 50.0
    assert m.cavi_max_iter == 50
    assert m.cavi_tol == 1e-3


def test_initialize_global_shapes_and_validity():
    from spark_vi.models.online_hdp import OnlineHDP

    m = OnlineHDP(T=10, K=5, vocab_size=50, gamma=1.5, gamma_shape=100.0)
    np.random.seed(0)
    g = m.initialize_global(data_summary=None)

    assert set(g.keys()) == {"lambda", "u", "v"}
    assert g["lambda"].shape == (10, 50)
    assert g["u"].shape == (9,)
    assert g["v"].shape == (9,)
    # Match VanillaLDA: positive Gamma init for lambda.
    assert np.all(g["lambda"] > 0)
    # Paper-following init: u = 1, v = gamma.
    assert np.allclose(g["u"], 1.0)
    assert np.allclose(g["v"], 1.5)


def test_local_update_returns_expected_keys_and_shapes():
    """Run a tiny partition through local_update; check stats dict shape."""
    from spark_vi.core import BOWDocument
    from spark_vi.models.online_hdp import OnlineHDP

    m = OnlineHDP(T=10, K=5, vocab_size=50, gamma_shape=100.0)
    np.random.seed(0)
    g = m.initialize_global(data_summary=None)

    rows = [
        BOWDocument(
            indices=np.array([0, 1, 2], dtype=np.int32),
            counts=np.array([2.0, 1.0, 3.0], dtype=np.float64),
            length=6,
        ),
        BOWDocument(
            indices=np.array([10, 11], dtype=np.int32),
            counts=np.array([1.0, 1.0], dtype=np.float64),
            length=2,
        ),
    ]

    stats = m.local_update(rows, g)

    expected_keys = {
        "lambda_stats", "var_phi_sum_stats",
        "doc_loglik_sum", "doc_z_term_sum", "doc_c_term_sum",
        "doc_stick_kl_sum", "n_docs",
    }
    assert set(stats.keys()) == expected_keys
    assert stats["lambda_stats"].shape == (10, 50)
    assert stats["var_phi_sum_stats"].shape == (10,)
    assert float(stats["n_docs"]) == 2.0
    # All scalar accumulators must be finite.
    for k in ("doc_loglik_sum", "doc_z_term_sum",
              "doc_c_term_sum", "doc_stick_kl_sum"):
        assert np.isfinite(stats[k])
    # Suff-stat columns we touched should be non-zero; columns we didn't
    # touch should be exactly zero.
    touched = np.array([0, 1, 2, 10, 11])
    untouched = np.setdiff1d(np.arange(50), touched)
    assert np.any(stats["lambda_stats"][:, touched] > 0)
    assert np.allclose(stats["lambda_stats"][:, untouched], 0.0)


def test_local_update_combine_stats_is_elementwise_sum():
    """Default VIModel.combine_stats should sum HDP suff-stats correctly."""
    from spark_vi.core import BOWDocument
    from spark_vi.models.online_hdp import OnlineHDP

    m = OnlineHDP(T=10, K=5, vocab_size=50)
    np.random.seed(0)
    g = m.initialize_global(data_summary=None)

    docs = [
        BOWDocument(indices=np.array([0, 1], dtype=np.int32),
                    counts=np.array([1.0, 1.0]), length=2),
        BOWDocument(indices=np.array([2, 3], dtype=np.int32),
                    counts=np.array([1.0, 1.0]), length=2),
    ]
    a_stats = m.local_update(docs[:1], g)
    b_stats = m.local_update(docs[1:], g)
    combined = m.combine_stats(a_stats, b_stats)

    assert np.allclose(combined["lambda_stats"],
                       a_stats["lambda_stats"] + b_stats["lambda_stats"])
    assert np.allclose(combined["var_phi_sum_stats"],
                       a_stats["var_phi_sum_stats"] + b_stats["var_phi_sum_stats"])
    assert float(combined["n_docs"]) == 2.0


def test_update_global_rho_zero_is_identity():
    """rho=0 ⇒ globals unchanged."""
    from spark_vi.models.online_hdp import OnlineHDP

    m = OnlineHDP(T=10, K=5, vocab_size=50, gamma=1.0)
    np.random.seed(0)
    g = m.initialize_global(data_summary=None)

    fake_stats = {
        "lambda_stats": np.full((10, 50), 7.0),
        "var_phi_sum_stats": np.full(10, 3.0),
    }
    new_g = m.update_global(g, fake_stats, learning_rate=0.0)

    assert np.allclose(new_g["lambda"], g["lambda"])
    assert np.allclose(new_g["u"], g["u"])
    assert np.allclose(new_g["v"], g["v"])


def test_update_global_rho_one_replaces_with_target():
    """rho=1 ⇒ globals become eta + target / (1 + s) / (gamma + s_tail)."""
    from spark_vi.models.online_hdp import OnlineHDP

    T, K, V = 5, 3, 10
    m = OnlineHDP(T=T, K=K, vocab_size=V, eta=0.01, gamma=2.0)
    np.random.seed(0)
    g = m.initialize_global(data_summary=None)

    s = np.array([10.0, 5.0, 2.0, 1.0, 0.5])
    fake_stats = {
        "lambda_stats": np.full((T, V), 4.0),
        "var_phi_sum_stats": s,
    }
    new_g = m.update_global(g, fake_stats, learning_rate=1.0)

    assert np.allclose(new_g["lambda"], 0.01 + 4.0)
    # u_k = 1 + s[k] for k = 0..T-2.
    assert np.allclose(new_g["u"], 1.0 + s[:T - 1])
    # v_k = gamma + cumsum(s[1:].reverse).reverse:
    # s_tail[0] = s[1] + s[2] + s[3] + s[4]
    # s_tail[1] =        s[2] + s[3] + s[4]
    # s_tail[2] =               s[3] + s[4]
    # s_tail[3] =                      s[4]
    expected_tail = np.cumsum(s[1:][::-1])[::-1]
    assert np.allclose(new_g["v"], 2.0 + expected_tail)


def test_compute_elbo_finite_on_initial_state():
    """ELBO is finite when called on init globals + zero stats (no docs)."""
    from spark_vi.models.online_hdp import OnlineHDP

    m = OnlineHDP(T=10, K=5, vocab_size=50, gamma=1.0)
    np.random.seed(0)
    g = m.initialize_global(data_summary=None)

    zero_stats = {
        "lambda_stats": np.zeros((10, 50)),
        "var_phi_sum_stats": np.zeros(10),
        "doc_loglik_sum": np.array(0.0),
        "doc_z_term_sum": np.array(0.0),
        "doc_c_term_sum": np.array(0.0),
        "doc_stick_kl_sum": np.array(0.0),
        "n_docs": np.array(0.0),
    }
    elbo = m.compute_elbo(g, zero_stats)
    assert np.isfinite(elbo)


def test_compute_elbo_corpus_kl_zero_at_prior():
    """When (u, v) == (1, gamma) and lambda is set to the eta prior, the
    corpus-level KL terms are exactly zero. Per-doc terms are zero with no
    docs. Therefore ELBO == 0 in that case."""
    from spark_vi.models.online_hdp import OnlineHDP

    T, V = 5, 8
    eta = 0.01
    gamma = 1.0
    m = OnlineHDP(T=T, K=3, vocab_size=V, eta=eta, gamma=gamma)

    g = {
        "lambda": np.full((T, V), eta, dtype=np.float64),  # equals prior
        "u": np.ones(T - 1),
        "v": np.full(T - 1, gamma),
    }
    zero_stats = {
        "lambda_stats": np.zeros((T, V)),
        "var_phi_sum_stats": np.zeros(T),
        "doc_loglik_sum": np.array(0.0),
        "doc_z_term_sum": np.array(0.0),
        "doc_c_term_sum": np.array(0.0),
        "doc_stick_kl_sum": np.array(0.0),
        "n_docs": np.array(0.0),
    }
    elbo = m.compute_elbo(g, zero_stats)
    assert np.isclose(elbo, 0.0, atol=1e-9)


def test_infer_local_returns_simplex_theta():
    """infer_local returns the doc variational posterior + a θ derived from it."""
    from spark_vi.core import BOWDocument
    from spark_vi.models.online_hdp import OnlineHDP

    T, K, V = 10, 5, 50
    m = OnlineHDP(T=T, K=K, vocab_size=V)
    np.random.seed(0)
    g = m.initialize_global(data_summary=None)

    doc = BOWDocument(
        indices=np.array([0, 1, 2], dtype=np.int32),
        counts=np.array([2.0, 1.0, 3.0], dtype=np.float64),
        length=6,
    )
    out = m.infer_local(doc, g)

    expected = {"a", "b", "phi", "var_phi", "theta"}
    assert set(out.keys()) == expected
    assert out["theta"].shape == (T,)
    assert np.isclose(out["theta"].sum(), 1.0)


def test_iteration_summary_returns_string():
    """iteration_summary returns a short non-empty diagnostic string."""
    from spark_vi.models.online_hdp import OnlineHDP

    m = OnlineHDP(T=10, K=5, vocab_size=50)
    np.random.seed(0)
    g = m.initialize_global(data_summary=None)
    s = m.iteration_summary(g)

    assert isinstance(s, str)
    assert len(s) > 0
    # Must reference active topic count somewhere in the line.
    assert "active" in s.lower() or "topics" in s.lower()
