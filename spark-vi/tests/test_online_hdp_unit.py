"""Pure-numpy tests for OnlineHDP module-level math helpers and CAVI.

No Spark — these test the math in isolation. Single document, hand-checked
shapes and values where possible.
"""
import numpy as np
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
