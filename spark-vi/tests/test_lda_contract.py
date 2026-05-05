"""VIModel contract tests for the optional infer_local capability."""
import numpy as np
import pytest


def test_vimodel_default_infer_local_raises_clear_error():
    """A VIModel that doesn't override infer_local must raise NotImplementedError
    with a message naming the concrete class — no silent fallback to None/NaN.
    """
    from spark_vi.models.counting import CountingModel

    m = CountingModel()
    with pytest.raises(NotImplementedError) as exc:
        m.infer_local(row=1, global_params={"alpha": 1.0, "beta": 1.0})
    msg = str(exc.value)
    assert "CountingModel" in msg
    assert "transform" in msg.lower() or "inference" in msg.lower()


def test_vanilla_lda_is_a_vimodel():
    from spark_vi.core import VIModel
    from spark_vi.models.lda import VanillaLDA
    assert issubclass(VanillaLDA, VIModel)


def test_vanilla_lda_default_alpha_eta_match_one_over_k():
    """Default symmetric α and η both default to 1/K, matching MLlib.

    α is stored on self as a length-K vector (constructor broadcasts a
    scalar). η is scalar.
    """
    from spark_vi.models.lda import VanillaLDA

    m = VanillaLDA(K=4, vocab_size=10)
    np.testing.assert_allclose(m.alpha, 0.25)
    assert m.alpha.shape == (4,)
    assert m.eta == pytest.approx(0.25)


def test_vanilla_lda_explicit_alpha_eta_respected():
    from spark_vi.models.lda import VanillaLDA

    m = VanillaLDA(K=10, vocab_size=100, alpha=0.1, eta=0.2)
    np.testing.assert_allclose(m.alpha, 0.1)
    assert m.alpha.shape == (10,)
    assert m.eta == pytest.approx(0.2)


def test_vanilla_lda_accepts_vector_alpha():
    """A length-K alpha is accepted and stored verbatim (no broadcast)."""
    from spark_vi.models.lda import VanillaLDA

    m = VanillaLDA(K=3, vocab_size=10, alpha=np.array([0.1, 0.5, 0.9]))
    np.testing.assert_allclose(m.alpha, [0.1, 0.5, 0.9])

    # Wrong shape rejected.
    with pytest.raises(ValueError, match="length-3 1-D array"):
        VanillaLDA(K=3, vocab_size=10, alpha=np.array([0.1, 0.5]))


def test_vanilla_lda_rejects_invalid_hyperparams():
    from spark_vi.models.lda import VanillaLDA
    with pytest.raises(ValueError):
        VanillaLDA(K=0, vocab_size=10)
    with pytest.raises(ValueError):
        VanillaLDA(K=2, vocab_size=0)
    with pytest.raises(ValueError):
        VanillaLDA(K=2, vocab_size=10, alpha=-1.0)
    with pytest.raises(ValueError):
        VanillaLDA(K=2, vocab_size=10, eta=0.0)
    with pytest.raises(ValueError):
        VanillaLDA(K=2, vocab_size=10, cavi_max_iter=0)
    with pytest.raises(ValueError):
        VanillaLDA(K=2, vocab_size=10, cavi_tol=0.0)


def test_vanilla_lda_initialize_global_returns_lambda_of_correct_shape():
    import numpy as np
    from spark_vi.models.lda import VanillaLDA

    m = VanillaLDA(K=5, vocab_size=20, gamma_shape=100.0)
    g = m.initialize_global(data_summary=None)
    assert "lambda" in g
    assert g["lambda"].shape == (5, 20)
    # Gamma(100, 1/100) draws are positive with mean ~1; sanity-check positivity.
    assert (g["lambda"] > 0).all()


def test_vanilla_lda_initialize_global_is_seedable_via_numpy():
    """Seeding numpy.random produces reproducible lambda init.

    The model's lambda init draws from numpy's default Gamma RNG; tests can
    pin reproducibility by seeding np.random before construction.
    """
    import numpy as np
    from spark_vi.models.lda import VanillaLDA

    np.random.seed(42)
    g1 = VanillaLDA(K=3, vocab_size=10).initialize_global(None)
    np.random.seed(42)
    g2 = VanillaLDA(K=3, vocab_size=10).initialize_global(None)
    np.testing.assert_array_equal(g1["lambda"], g2["lambda"])


def test_vanilla_lda_local_update_returns_expected_keys():
    """local_update returns the four keys the runner + ELBO need."""
    import numpy as np
    from spark_vi.core import BOWDocument
    from spark_vi.models.lda import VanillaLDA

    np.random.seed(0)
    m = VanillaLDA(K=3, vocab_size=5)
    g = m.initialize_global(None)
    docs = [
        BOWDocument(indices=np.array([0, 2], dtype=np.int32),
                    counts=np.array([1.0, 2.0]), length=3),
        BOWDocument(indices=np.array([1, 4], dtype=np.int32),
                    counts=np.array([3.0, 1.0]), length=4),
    ]
    stats = m.local_update(rows=iter(docs), global_params=g)
    # When optimize_alpha=False (the default), e_log_theta_sum is NOT emitted.
    assert set(stats.keys()) == {"lambda_stats", "doc_loglik_sum", "doc_theta_kl_sum", "n_docs"}
    assert stats["lambda_stats"].shape == (3, 5)
    assert isinstance(float(stats["doc_loglik_sum"]), float)
    assert isinstance(float(stats["doc_theta_kl_sum"]), float)
    assert int(stats["n_docs"]) == 2


def test_vanilla_lda_local_update_lambda_stats_is_nonzero_only_on_seen_columns():
    """Lambda stats accumulate only on columns whose token indices appeared."""
    import numpy as np
    from spark_vi.core import BOWDocument
    from spark_vi.models.lda import VanillaLDA

    np.random.seed(0)
    m = VanillaLDA(K=2, vocab_size=6)
    g = m.initialize_global(None)
    # Only indices 1 and 3 ever appear.
    docs = [BOWDocument(indices=np.array([1, 3], dtype=np.int32),
                         counts=np.array([2.0, 1.0]), length=3)]
    stats = m.local_update(rows=iter(docs), global_params=g)
    untouched_cols = [0, 2, 4, 5]
    np.testing.assert_array_equal(stats["lambda_stats"][:, untouched_cols], 0.0)
    # The seen columns received some mass.
    assert (stats["lambda_stats"][:, [1, 3]] > 0).any()


def test_vanilla_lda_local_update_handles_empty_partition():
    """Empty rows iterator returns zero stats and n_docs=0."""
    import numpy as np
    from spark_vi.models.lda import VanillaLDA

    m = VanillaLDA(K=2, vocab_size=4)
    g = m.initialize_global(None)
    stats = m.local_update(rows=iter([]), global_params=g)
    np.testing.assert_array_equal(stats["lambda_stats"], np.zeros((2, 4)))
    assert int(stats["n_docs"]) == 0
    assert float(stats["doc_loglik_sum"]) == 0.0
    assert float(stats["doc_theta_kl_sum"]) == 0.0


def test_vanilla_lda_update_global_at_lr_zero_is_identity():
    import numpy as np
    from spark_vi.models.lda import VanillaLDA
    np.random.seed(0)
    m = VanillaLDA(K=2, vocab_size=4)
    g = m.initialize_global(None)
    target = {"lambda_stats": np.ones((2, 4)) * 5.0}
    new_g = m.update_global(g, target_stats=target, learning_rate=0.0)
    np.testing.assert_array_equal(new_g["lambda"], g["lambda"])


def test_vanilla_lda_update_global_at_lr_one_jumps_to_target():
    """At rho=1.0, lambda becomes (eta + expElogbeta * lambda_stats).

    The expElogbeta factor is the per-topic-per-vocab term factored out of
    local_update's per-doc accumulation. update_global multiplies it back in
    at the driver, matching MLlib's "statsSum *:* expElogbeta.t".
    """
    import numpy as np
    from scipy.special import digamma
    from spark_vi.models.lda import VanillaLDA
    np.random.seed(0)
    m = VanillaLDA(K=2, vocab_size=4, eta=0.05)
    g = m.initialize_global(None)
    target = {"lambda_stats": np.full((2, 4), 7.0)}
    new_g = m.update_global(g, target_stats=target, learning_rate=1.0)

    lam = g["lambda"]
    expElogbeta = np.exp(digamma(lam) - digamma(lam.sum(axis=1, keepdims=True)))
    np.testing.assert_allclose(new_g["lambda"], 0.05 + expElogbeta * 7.0)


def test_vanilla_lda_update_global_applies_expElogbeta_factor():
    """Regression test for the MLlib-equivalent expElogbeta multiplication.

    A non-uniform target_stats fed to update_global must produce a result
    that reflects expElogbeta multiplication. Computing the buggy formula
    (eta + target_stats, no expElogbeta factor) would give a different
    answer; this test pins the corrected behavior.
    """
    import numpy as np
    from scipy.special import digamma
    from spark_vi.models.lda import VanillaLDA

    K, V = 2, 3
    eta = 0.1
    m = VanillaLDA(K=K, vocab_size=V, eta=eta)
    # Hand-chosen lam: row 0 peaks on column 0, row 1 peaks on column 2.
    lam = np.array([[10.0, 1.0, 1.0], [1.0, 1.0, 10.0]])
    target_stats = {"lambda_stats": np.array([[3.0, 3.0, 3.0], [3.0, 3.0, 3.0]])}

    g = {"lambda": lam, "alpha": m.alpha.copy(), "eta": np.array(m.eta)}
    new_g = m.update_global(g, target_stats=target_stats,
                             learning_rate=1.0)

    expElogbeta = np.exp(digamma(lam) - digamma(lam.sum(axis=1, keepdims=True)))
    expected = eta + expElogbeta * 3.0
    np.testing.assert_allclose(new_g["lambda"], expected)
    # Sanity: the buggy formula would have given eta + 3.0 = 3.1 uniformly,
    # which is materially different from the corrected result on row 0.
    buggy = np.full((K, V), eta + 3.0)
    assert not np.allclose(new_g["lambda"], buggy, atol=0.5)


def test_vanilla_lda_update_global_uses_input_lambda_for_expElogbeta():
    """update_global must compute expElogbeta from the *input* lambda — the
    one local_update saw at the start of the iteration — not from any
    intermediate or post-update lambda candidate.

    This invariant matters because local_update's per-doc accumulator
    deliberately omits the expElogbeta factor (Lee/Seung implicit-phi
    deferred multiplication); update_global must put it back using the
    *same* lambda local_update used, otherwise the natural-gradient
    direction is computed in a mixed reference frame.

    Tested at lr=0.5 with non-uniform lam: the convex combination
    (1-rho)*lam + rho*lambda_hat exposes the dependence of the result on
    which lambda generated expElogbeta; computing expElogbeta from a
    different lambda would shift the answer in a hand-checkable way.
    """
    import numpy as np
    from scipy.special import digamma
    from spark_vi.models.lda import VanillaLDA

    K, V = 2, 3
    eta = 0.05
    m = VanillaLDA(K=K, vocab_size=V, eta=eta)
    # Non-uniform lam: row 0 peaked on col 0, row 1 peaked on col 2.
    lam_in = np.array([[5.0, 0.5, 0.5], [0.5, 0.5, 5.0]])
    lambda_stats = np.array([[2.0, 1.0, 0.5], [0.5, 1.0, 2.0]])
    target_stats = {"lambda_stats": lambda_stats}

    rho = 0.5
    g_in = {"lambda": lam_in, "alpha": m.alpha.copy(), "eta": np.array(m.eta)}
    new_lam = m.update_global(
        g_in, target_stats=target_stats, learning_rate=rho,
    )["lambda"]

    # The contract: expElogbeta computed from lam_in.
    expElogbeta_in = np.exp(digamma(lam_in) - digamma(lam_in.sum(axis=1, keepdims=True)))
    lambda_hat = eta + expElogbeta_in * lambda_stats
    expected = (1 - rho) * lam_in + rho * lambda_hat
    np.testing.assert_allclose(new_lam, expected)

    # Counter-case: if expElogbeta were computed from the prior (eta * 1)
    # instead of lam_in, the result would differ noticeably. Pinning this
    # rules out a class of refactoring bugs where the reference lambda
    # silently drifts away from the input.
    eta_lam = np.full_like(lam_in, eta)
    expElogbeta_wrong = np.exp(
        digamma(eta_lam) - digamma(eta_lam.sum(axis=1, keepdims=True))
    )
    wrong = (1 - rho) * lam_in + rho * (eta + expElogbeta_wrong * lambda_stats)
    assert not np.allclose(new_lam, wrong, atol=0.05), (
        "update_global appears to use a lambda other than the input for "
        "expElogbeta — natural-gradient reference frame would be inconsistent."
    )


def test_vanilla_lda_combine_stats_is_associative():
    """treeReduce relies on associativity: combine(a, combine(b, c)) == combine(combine(a, b), c)."""
    import numpy as np
    from spark_vi.models.lda import VanillaLDA
    rng = np.random.default_rng(0)
    m = VanillaLDA(K=2, vocab_size=3)
    def _stats():
        return {
            "lambda_stats": rng.normal(size=(2, 3)),
            "doc_loglik_sum": np.array(rng.normal()),
            "doc_theta_kl_sum": np.array(rng.normal()),
            "n_docs": np.array(float(rng.integers(0, 100))),
        }
    a, b, c = _stats(), _stats(), _stats()
    left = m.combine_stats(a, m.combine_stats(b, c))
    right = m.combine_stats(m.combine_stats(a, b), c)
    for k in left:
        np.testing.assert_allclose(left[k], right[k])


def test_vanilla_lda_infer_local_returns_gamma_and_theta():
    """infer_local returns dict with K-vector gamma and normalized theta."""
    import numpy as np
    from spark_vi.core import BOWDocument
    from spark_vi.models.lda import VanillaLDA

    np.random.seed(0)
    m = VanillaLDA(K=4, vocab_size=10)
    g = m.initialize_global(None)
    doc = BOWDocument(indices=np.array([2, 5], dtype=np.int32),
                      counts=np.array([1.0, 1.0]), length=2)

    out = m.infer_local(doc, g)
    assert set(out.keys()) == {"gamma", "theta"}
    assert out["gamma"].shape == (4,)
    assert out["theta"].shape == (4,)
    np.testing.assert_allclose(out["theta"].sum(), 1.0, atol=1e-12)
    np.testing.assert_allclose(out["theta"], out["gamma"] / out["gamma"].sum())


def test_vanilla_lda_infer_local_is_pure_function_of_inputs():
    """Same row + same global_params + same RNG state => identical output."""
    import numpy as np
    from spark_vi.core import BOWDocument
    from spark_vi.models.lda import VanillaLDA

    np.random.seed(7)
    m = VanillaLDA(K=3, vocab_size=8)
    g = m.initialize_global(None)
    doc = BOWDocument(indices=np.array([0, 4, 7], dtype=np.int32),
                      counts=np.array([2.0, 1.0, 1.0]), length=4)

    np.random.seed(123)
    out_a = m.infer_local(doc, g)
    np.random.seed(123)
    out_b = m.infer_local(doc, g)
    np.testing.assert_array_equal(out_a["gamma"], out_b["gamma"])
    np.testing.assert_array_equal(out_a["theta"], out_b["theta"])


def test_vanilla_lda_optimize_flags_default_false():
    from spark_vi.models.lda import VanillaLDA

    m = VanillaLDA(K=3, vocab_size=10)
    assert m.optimize_alpha is False
    assert m.optimize_eta is False


def test_vanilla_lda_optimize_flags_can_be_set():
    from spark_vi.models.lda import VanillaLDA

    m = VanillaLDA(K=3, vocab_size=10, optimize_alpha=True, optimize_eta=True)
    assert m.optimize_alpha is True
    assert m.optimize_eta is True


def test_update_global_with_optimize_alpha_runs_newton_and_floors():
    """At ρ=1.0, optimize_alpha=True applies a full Newton step plus floor.

    The verification is structural: with synthetic e_log_theta_sum drawn from
    Dir([0.1, 0.5, 0.9]), one full Newton step from 1/K starting α moves
    α measurably toward the truth. Convergence (closed-form) is covered by
    test_alpha_newton_step_recovers_known_alpha_on_synthetic; this test
    confirms wiring.
    """
    from spark_vi.models.lda import VanillaLDA
    import numpy as np

    K, V = 3, 5
    rng = np.random.default_rng(0)
    true_alpha = np.array([0.1, 0.5, 0.9])
    thetas = rng.dirichlet(true_alpha, size=10000)
    e_log_theta_sum = np.log(thetas).sum(axis=0)

    m = VanillaLDA(K=K, vocab_size=V, optimize_alpha=True)
    g = {
        "lambda": np.ones((K, V)),
        "alpha": np.full(K, 1.0 / K),
        "eta": np.array(0.1),
    }
    target_stats = {
        "lambda_stats": np.zeros((K, V)),
        "e_log_theta_sum": e_log_theta_sum,  # already corpus-scaled (D=10000)
        "doc_loglik_sum": np.array(0.0),
        "doc_theta_kl_sum": np.array(0.0),
        "n_docs": np.array(10000.0),
    }
    new_g = m.update_global(g, target_stats, learning_rate=1.0)

    # α moved toward truth.
    assert np.argmax(new_g["alpha"]) == 2  # largest component is index 2 (0.9)
    assert np.argmin(new_g["alpha"]) == 0  # smallest is index 0 (0.1)
    # Floor respected.
    assert (new_g["alpha"] >= 1e-3).all()

    # Wiring tightness: the result must match what _alpha_newton_step + floor
    # produces directly (catches any wiring bug that the argmax/argmin sign
    # check above would miss).
    from spark_vi.models.lda import _alpha_newton_step
    direct_delta = _alpha_newton_step(
        alpha=g["alpha"],
        e_log_theta_sum_scaled=e_log_theta_sum,
        D=10000.0,
    )
    expected_alpha = np.maximum(g["alpha"] + 1.0 * direct_delta, 1e-3)
    np.testing.assert_allclose(new_g["alpha"], expected_alpha, rtol=1e-12)


def test_local_update_emits_e_log_theta_sum_when_optimize_alpha():
    """The new stat key is present iff optimize_alpha=True (avoids paying
    the digamma cost when off)."""
    import numpy as np
    from spark_vi.core.types import BOWDocument
    from spark_vi.models.lda import VanillaLDA

    K, V = 3, 5
    docs = [BOWDocument(
        indices=np.array([0, 2], dtype=np.int32),
        counts=np.array([1.0, 2.0]),
        length=3,
    )]
    g = {
        "lambda": np.ones((K, V)),
        "alpha": np.full(K, 1.0 / K),
        "eta": np.array(0.1),
    }

    m_off = VanillaLDA(K=K, vocab_size=V, optimize_alpha=False)
    stats_off = m_off.local_update(rows=iter(docs), global_params=g)
    assert "e_log_theta_sum" not in stats_off

    m_on = VanillaLDA(K=K, vocab_size=V, optimize_alpha=True)
    stats_on = m_on.local_update(rows=iter(docs), global_params=g)
    assert "e_log_theta_sum" in stats_on
    assert stats_on["e_log_theta_sum"].shape == (K,)
