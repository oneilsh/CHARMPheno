"""Pure-numpy tests for VanillaLDA's CAVI inner loop and ELBO machinery.

No Spark — these test the math at module level. Single document, hand-checked
shapes and values where possible.
"""
import numpy as np
import pytest


def _peaked_expElogbeta(K: int, V: int, sharpness: float = 5.0) -> np.ndarray:
    """Build a stylized expElogbeta where each topic peaks on a single word.

    Returns (K, V) array. Topic k peaks on word k (mod V). Used to make CAVI
    tests deterministic and visually inspectable.
    """
    eb = np.full((K, V), -sharpness, dtype=np.float64)
    for k in range(K):
        eb[k, k % V] = 0.0
    return np.exp(eb)


def test_cavi_doc_inference_converges_to_dominant_topic():
    """A document whose tokens are all word-0 should drive gamma toward topic 0."""
    from spark_vi.models.lda import _cavi_doc_inference

    K, V = 3, 5
    expElogbeta = _peaked_expElogbeta(K, V)  # topic 0 peaks on word 0
    indices = np.array([0], dtype=np.int32)
    counts = np.array([20.0], dtype=np.float64)
    alpha = 0.1
    gamma_init = np.full(K, 1.0)  # symmetric init

    gamma, expElogthetad, phi_norm, n_iter = _cavi_doc_inference(
        indices, counts, expElogbeta, alpha, gamma_init,
        max_iter=100, tol=1e-4,
    )
    # Topic 0 should dominate the posterior.
    assert np.argmax(gamma) == 0
    assert gamma[0] > gamma[1] + 5.0
    assert gamma[0] > gamma[2] + 5.0
    # Phi normalizer is per-unique-token; one unique token here.
    assert phi_norm.shape == (1,)
    # expElogthetad is K-vector, all positive.
    assert expElogthetad.shape == (K,)
    assert (expElogthetad > 0).all()
    # Should converge well under the iteration cap.
    assert n_iter < 100


def test_cavi_doc_inference_respects_max_iter():
    """If tol is impossibly tight, max_iter is the hard ceiling."""
    from spark_vi.models.lda import _cavi_doc_inference

    K, V = 2, 3
    expElogbeta = _peaked_expElogbeta(K, V)
    indices = np.array([0, 1], dtype=np.int32)
    counts = np.array([1.0, 1.0])
    gamma_init = np.full(K, 1.0)

    _, _, _, n_iter = _cavi_doc_inference(
        indices, counts, expElogbeta, alpha=0.1, gamma_init=gamma_init,
        max_iter=3, tol=1e-100,
    )
    assert n_iter == 3


def test_cavi_doc_inference_matches_explicit_phi_implementation():
    """Lee/Seung trick (implicit phi) must agree numerically with the
    explicit-phi formulation on a small fixture. The production path uses
    the implicit form for memory efficiency; this test pins them to the
    same answer.
    """
    from spark_vi.models.lda import _cavi_doc_inference

    K, V = 3, 6
    rng = np.random.default_rng(0)
    expElogbeta = np.exp(rng.normal(size=(K, V)) * 0.3)
    indices = np.array([0, 2, 5], dtype=np.int32)
    counts = np.array([3.0, 1.0, 2.0])
    alpha = 0.5
    gamma_init = np.ones(K) * 1.0
    max_iter = 50
    tol = 1e-8

    # Implicit (production):
    gamma_impl, _, _, _ = _cavi_doc_inference(
        indices, counts, expElogbeta, alpha, gamma_init.copy(),
        max_iter=max_iter, tol=tol,
    )

    # Explicit phi reference, same recurrence:
    eb_d = expElogbeta[:, indices]                 # (K, n_unique)
    gamma_exp = gamma_init.copy()
    from scipy.special import digamma
    for _ in range(max_iter):
        prev = gamma_exp.copy()
        eEt = np.exp(digamma(gamma_exp) - digamma(gamma_exp.sum()))
        # phi_dnk ∝ eEt[k] * eb_d[k, n]; normalize over k per token n.
        unnorm = eb_d * eEt[:, None]               # (K, n_unique)
        phi_explicit = unnorm / unnorm.sum(axis=0, keepdims=True)
        gamma_exp = alpha + (phi_explicit * counts[None, :]).sum(axis=1)
        if np.mean(np.abs(gamma_exp - prev)) < tol:
            break

    np.testing.assert_allclose(gamma_impl, gamma_exp, atol=1e-6)


def test_compute_elbo_returns_finite_float_on_realistic_inputs():
    import numpy as np
    from spark_vi.models.lda import VanillaLDA
    np.random.seed(0)
    m = VanillaLDA(K=3, vocab_size=5)
    g = m.initialize_global(None)
    agg = {
        "lambda_stats": np.ones((3, 5)),
        "doc_loglik_sum": np.array(-12.0),
        "doc_theta_kl_sum": np.array(0.4),
        "n_docs": np.array(7.0),
    }
    val = m.compute_elbo(g, agg)
    assert isinstance(val, float)
    assert np.isfinite(val)


def test_compute_elbo_lambda_kl_zero_when_lambda_equals_eta():
    """When lambda equals the prior eta·1, the global Dirichlet KL term is 0,
    so the ELBO equals just the data-likelihood + (-doc-theta-KL).
    """
    import numpy as np
    from spark_vi.models.lda import VanillaLDA
    K, V = 2, 3
    eta = 0.1
    m = VanillaLDA(K=K, vocab_size=V, eta=eta)
    g = {
        "lambda": np.full((K, V), eta),
        "alpha": np.full(K, 1.0 / K),
        "eta": np.array(eta),
    }
    agg = {
        "lambda_stats": np.zeros((K, V)),  # not used directly in ELBO, but realistic
        "doc_loglik_sum": np.array(-3.0),
        "doc_theta_kl_sum": np.array(0.5),
        "n_docs": np.array(2.0),
    }
    val = m.compute_elbo(g, agg)
    # ELBO = doc_loglik_sum - doc_theta_kl_sum - global_kl
    # global_kl = 0 (lambda == eta * 1_V row-wise per topic)
    np.testing.assert_allclose(val, -3.0 - 0.5)


def test_alpha_newton_step_recovers_known_alpha_on_synthetic():
    """Newton iterations on _alpha_newton_step recover the true α from
    samples of Dir(α). Sanity check on the closed-form Sherman-Morrison step.
    """
    from spark_vi.models.lda import _alpha_newton_step

    rng = np.random.default_rng(42)
    true_alpha = np.array([0.1, 0.5, 0.9])
    K = 3
    D = 10000

    # Sample θ_d ~ Dir(true_alpha), gather Σ_d log θ_dk. Under a perfectly
    # concentrated variational posterior q(θ_d) = δ(θ_d - true_θ_d), this is
    # exactly the corpus-scaled e_log_theta_sum the helper expects.
    thetas = rng.dirichlet(true_alpha, size=D)
    e_log_theta_sum = np.log(thetas).sum(axis=0)  # shape (K,)

    # Initialize from the symmetric prior 1/K, iterate full Newton steps.
    alpha = np.full(K, 1.0 / K, dtype=np.float64)
    for _ in range(50):
        delta = _alpha_newton_step(alpha, e_log_theta_sum, D=float(D))
        alpha = alpha + delta
        alpha = np.maximum(alpha, 1e-3)

    np.testing.assert_allclose(alpha, true_alpha, atol=0.05)


def test_eta_newton_step_recovers_known_eta_on_synthetic():
    """Newton iterations on _eta_newton_step recover the true η from
    samples of Dir(η · 1_V). Symmetric scalar version of the α test.
    """
    from spark_vi.models.lda import _eta_newton_step

    rng = np.random.default_rng(7)
    true_eta = 0.5
    K = 50
    V = 100

    # Sample K topics φ_t ~ Dir(η · 1_V); compute Σ_t Σ_v log φ_tv.
    # As in the α test, this is the asymptotic E[log φ] under a sharply
    # concentrated variational q(φ_t) = δ(φ_t − true_φ_t).
    phis = rng.dirichlet(np.full(V, true_eta), size=K)
    e_log_phi_sum = float(np.log(phis).sum())

    eta = 0.1
    for _ in range(50):
        delta = _eta_newton_step(eta, e_log_phi_sum, K=K, V=V)
        eta = max(eta + delta, 1e-3)

    assert abs(eta - true_eta) < 0.05, f"got {eta}, expected ~{true_eta}"
