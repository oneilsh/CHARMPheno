import numpy as np
from spark_vi.models.topic.stm import (
    _stm_neg_log_joint, _stm_neg_log_joint_grad, _stm_neg_log_joint_hessian,
)

def _toy():
    rng = np.random.default_rng(0)
    K, Vn = 3, 4
    eta = rng.normal(size=K)
    indices = np.array([0, 1, 2, 3], dtype=np.int32)
    counts = np.array([2.0, 1.0, 3.0, 1.0])
    expElogbeta = np.abs(rng.normal(size=(K, Vn))) + 0.1
    Gamma = rng.normal(size=(2, K))
    x = np.array([1.0, 0.5])
    return dict(eta=eta, indices=indices, counts=counts,
               expElogbeta=expElogbeta, Gamma=Gamma, x=x), K

def test_full_precision_prior_matches_diagonal_special_case():
    common, K = _toy()
    eta = common.pop("eta")
    sigma_diag = np.array([1.5, 2.0, 0.5])
    Sigma_inv = np.diag(1.0 / sigma_diag)
    # Prior term with full precision == diagonal hand-computation.
    diff = eta - common["Gamma"].T @ common["x"]
    expected_prior = 0.5 * float(diff @ Sigma_inv @ diff)
    f = _stm_neg_log_joint(eta, Sigma_inv=Sigma_inv, **common)
    # Recompute the data term alone by zeroing the prior (Sigma_inv = 0).
    data_only = _stm_neg_log_joint(eta, Sigma_inv=np.zeros((K, K)), **common)
    assert np.isclose(f - data_only, expected_prior)

def test_full_precision_grad_and_hessian_use_Sigma_inv():
    common, K = _toy()
    eta = common.pop("eta")
    Sigma_inv = np.array([[2.0, 0.3, 0.0],
                          [0.3, 1.0, 0.1],
                          [0.0, 0.1, 4.0]])
    diff = eta - common["Gamma"].T @ common["x"]
    g = _stm_neg_log_joint_grad(eta, Sigma_inv=Sigma_inv, **common)
    g_data = _stm_neg_log_joint_grad(eta, Sigma_inv=np.zeros((K, K)), **common)
    assert np.allclose(g - g_data, Sigma_inv @ diff)
    H = _stm_neg_log_joint_hessian(eta, Sigma_inv=Sigma_inv, **common)
    H_data = _stm_neg_log_joint_hessian(eta, Sigma_inv=np.zeros((K, K)), **common)
    assert np.allclose(H - H_data, Sigma_inv)


def test_initialize_global_sigma_is_matrix():
    from spark_vi.models.topic.stm import OnlineSTM
    m = OnlineSTM(K=4, vocab_size=10, P=2, reference_topic=False)
    gp = m.initialize_global(None)
    assert gp["Sigma"].shape == (4, 4)
    assert np.allclose(gp["Sigma"], np.eye(4) * m.sigma_init)


def test_nongated_recovers_planted_correlated_sigma():
    """One global M-step from planted (eta, Gamma) recovers off-diagonal Sigma.

    The full-covariance M-step is the unit under test, so this plants the η_d
    points directly and accumulates the residual outer-product scatter EXACTLY
    as local_update does (Gamma.T @ x = 0, nu_d = 0 -> contrib = outer(resid)),
    then drives the real update_global. This isolates the M-step from the E-step
    Laplace/softmax-simplex shrinkage that attenuates correlation when η is
    reconstructed from token evidence (see test_nongated_e2e_offdiagonal_sign
    for the end-to-end directional check, and the task-3 report for why a
    one-step token-driven recovery threshold is not achievable at K=3 without a
    reference topic).
    """
    from spark_vi.models.topic.stm import OnlineSTM
    rng = np.random.default_rng(3)
    K, V, P, D = 3, 8, 1, 8000
    Sigma_true = np.array([[1.0, 0.6, 0.0],
                           [0.6, 1.0, 0.0],
                           [0.0, 0.0, 1.0]])
    etas = rng.multivariate_normal(np.zeros(K), Sigma_true, size=D)
    m = OnlineSTM(K=K, vocab_size=V, P=P, reference_topic=False)
    gp = m.initialize_global(None)
    gp["Gamma"] = np.zeros((P, K))   # prior mean 0 -> resid = eta
    # Build the K×K scatter + per-pair support the way local_update accumulates
    # them (every topic free in every doc, nu_d = 0 here).
    residual_outer = etas.T @ etas              # Σ_d outer(eta_d, eta_d)
    n_pairs = np.full((K, K), float(D))
    stats = {
        "lambda_stats": np.zeros((K, V)),
        "XtX": np.full((P, P), float(D)),
        "XtX_groups": np.zeros((0, P, P)),
        "XtMu": np.zeros((P, K)),
        "residual_outer_stat": residual_outer,
        "n_pairs_stat": n_pairs,
        "n_docs_per_topic": np.full(K, float(D)),
        "doc_loglik_sum": np.array(0.0),
        "doc_eta_kl_sum": np.array(0.0),
        "n_docs": np.array(float(D)),
    }
    gp2 = m.update_global(gp, stats, learning_rate=1.0)
    R = gp2["Sigma"] / np.sqrt(np.outer(np.diag(gp2["Sigma"]), np.diag(gp2["Sigma"])))
    assert R[0, 1] > 0.3          # recovers the planted positive correlation
    assert abs(R[0, 2]) < 0.2     # uncorrelated pair stays near zero
    assert np.min(np.linalg.eigvalsh(gp2["Sigma"])) > 0  # SPD


def _planted_docs(rng, K=3, V=8, D=2000):
    from spark_vi.models.topic.types import STMDocument
    Sigma_true = np.array([[1.0, 0.6, 0.0], [0.6, 1.0, 0.0], [0.0, 0.0, 1.0]])
    etas = rng.multivariate_normal(np.zeros(K), Sigma_true, size=D)
    beta = np.abs(rng.normal(size=(K, V))) + 0.1
    beta /= beta.sum(1, keepdims=True)
    docs = []
    for d in range(D):
        th = np.exp(etas[d]); th /= th.sum()
        cf = rng.multinomial(60, th @ beta)
        idx = np.nonzero(cf)[0].astype(np.int32)
        docs.append(STMDocument(indices=idx, counts=cf[idx].astype(float),
                    length=int(cf.sum()), x=np.array([1.0]), groups=frozenset()))
    return docs, beta


def test_iw_prior_off_equals_mle():
    """IW prior off (default) produces the same Sigma as the plain MLE."""
    from spark_vi.models.topic.stm import OnlineSTM
    rng = np.random.default_rng(4)
    docs, beta = _planted_docs(rng)

    def run(**kw):
        m = OnlineSTM(K=3, vocab_size=8, P=1, reference_topic=False, **kw)
        gp = m.initialize_global(None)
        gp["lambda"] = beta * 200.0
        gp["Gamma"] = np.zeros((1, 3))
        return m.update_global(gp, m.local_update(docs, gp), 1.0)["Sigma"]

    assert np.allclose(run(), run(sigma_prior_scale=None, sigma_prior_count=0.0))


def test_iw_prior_shrinks_toward_scale():
    """With huge pseudo-count, diagonal is pulled to sigma_prior_scale and
    off-diagonal is pulled to 0 (inverse-Wishart MAP: Psi = nu * scale * I).

    Blei & Lafferty 2007 conjugate prior on Sigma; diagonal inverse-gamma
    is the Psi = psi*I special case (Roberts et al.)."""
    from spark_vi.models.topic.stm import OnlineSTM
    rng = np.random.default_rng(5)
    docs, beta = _planted_docs(rng)
    m = OnlineSTM(K=3, vocab_size=8, P=1, reference_topic=False,
                  sigma_prior_scale=5.0, sigma_prior_count=1e6)
    gp = m.initialize_global(None)
    gp["lambda"] = beta * 200.0
    gp["Gamma"] = np.zeros((1, 3))
    Sig = m.update_global(gp, m.local_update(docs, gp), 1.0)["Sigma"]
    assert np.allclose(np.diag(Sig), 5.0, atol=0.2)               # diagonal pulled to scale
    assert np.allclose(Sig - np.diag(np.diag(Sig)), 0.0, atol=0.2)  # off-diag pulled to 0


def test_diag_shrink_one_diagonalizes():
    """sigma_diag_shrink=1 forces Sigma to be exactly diagonal after the step
    (Roberts et al. stm sigma.prior lever).

    After blending toward diagonal, off-diagonal entries should be ~0 to float
    precision. SPD must still hold."""
    from spark_vi.models.topic.stm import OnlineSTM
    rng = np.random.default_rng(6)
    docs, beta = _planted_docs(rng)
    m = OnlineSTM(K=3, vocab_size=8, P=1, reference_topic=False, sigma_diag_shrink=1.0)
    gp = m.initialize_global(None)
    gp["lambda"] = beta * 200.0
    gp["Gamma"] = np.zeros((1, 3))
    Sig = m.update_global(gp, m.local_update(docs, gp), 1.0)["Sigma"]
    assert np.allclose(Sig - np.diag(np.diag(Sig)), 0.0, atol=1e-9)  # fully diagonal


def test_sigma_diag_shrink_validation():
    """sigma_diag_shrink outside [0, 1] raises ValueError."""
    from spark_vi.models.topic.stm import OnlineSTM
    import pytest
    with pytest.raises(ValueError):
        OnlineSTM(K=3, vocab_size=8, P=1, sigma_diag_shrink=1.5)


def test_unsupported_entry_exactly_preserved():
    """An entry (i,j) with zero support keeps its value BIT-IDENTICAL across
    a step when sigma_ridge is NOT added to the SPD matrix.

    The lazy rule keeps Sigma_target[i,j] = current Sigma[i,j], and the
    rho-blend keeps new_Sigma[i,j] = current Sigma[i,j]. nearest_spd is
    identity on an already-SPD matrix, so the entry is exact -- not merely
    approx -- after the update."""
    from spark_vi.models.topic.stm import OnlineSTM
    K, V, P = 3, 6, 1
    m = OnlineSTM(K=K, vocab_size=V, P=P, reference_topic=False)
    gp = m.initialize_global(None)
    # SPD Sigma with a distinctive value at (0,1) = (1,0).
    Sigma0 = np.array([[2.0, 0.3, 0.0],
                       [0.3, 1.5, 0.0],
                       [0.0, 0.0, 0.8]])
    gp["Sigma"] = Sigma0.copy()
    # Stats: topic pair (0,1) has zero support; topic 2 has some support.
    # n_pairs_stat[0,1] = n_pairs_stat[1,0] = 0 means the lazy rule fires.
    n_pairs = np.array([[5.0, 0.0, 0.0],
                        [0.0, 5.0, 0.0],
                        [0.0, 0.0, 5.0]])
    S = np.array([[3.0, 0.0, 0.0],
                  [0.0, 2.0, 0.0],
                  [0.0, 0.0, 1.0]])
    stats = {
        "lambda_stats": np.zeros((K, V)),
        "XtX": np.eye(P),
        "XtX_groups": np.zeros((0, P, P)),
        "XtMu": np.zeros((P, K)),
        "residual_outer_stat": S,
        "n_pairs_stat": n_pairs,
        "n_docs_per_topic": np.array([5.0, 5.0, 5.0]),
        "doc_loglik_sum": np.array(0.0),
        "doc_eta_kl_sum": np.array(0.0),
        "n_docs": np.array(5.0),
    }
    out = m.update_global(gp, stats, learning_rate=1.0)
    # (0,1) was unsupported -- must be bit-identical to the original.
    assert out["Sigma"][0, 1] == Sigma0[0, 1]
    assert out["Sigma"][1, 0] == Sigma0[1, 0]


def test_nongated_e2e_offdiagonal_sign():
    """End-to-end (token-driven E-step + M-step), the recovered off-diagonal Σ
    has the planted POSITIVE sign for the correlated pair. The magnitude is
    attenuated by per-doc Laplace shrinkage + the softmax simplex degeneracy at
    small K (reference_topic=False), so this asserts only the sign, not a
    threshold — the M-step magnitude is pinned by the planted-scatter test
    above.
    """
    from spark_vi.models.topic.stm import OnlineSTM
    from spark_vi.models.topic.types import STMDocument
    rng = np.random.default_rng(3)
    K, V, P, D = 3, 8, 1, 4000
    Sigma_true = np.array([[1.0, 0.6, 0.0],
                           [0.6, 1.0, 0.0],
                           [0.0, 0.0, 1.0]])
    etas = rng.multivariate_normal(np.zeros(K), Sigma_true, size=D)
    beta = np.abs(rng.normal(size=(K, V))) + 0.1
    beta /= beta.sum(1, keepdims=True)
    docs = []
    for d in range(D):
        theta = np.exp(etas[d]); theta /= theta.sum()
        counts_full = rng.multinomial(60, theta @ beta)
        idx = np.nonzero(counts_full)[0].astype(np.int32)
        docs.append(STMDocument(indices=idx, counts=counts_full[idx].astype(float),
                                length=int(counts_full.sum()),
                                x=np.array([1.0]), groups=frozenset()))
    m = OnlineSTM(K=K, vocab_size=V, P=P, reference_topic=False)
    gp = m.initialize_global(None)
    gp["lambda"] = beta * 200.0  # seed beta near truth so the E-step is informative
    gp["Gamma"] = np.zeros((P, K))
    gp2 = m.update_global(gp, m.local_update(docs, gp), learning_rate=1.0)
    R = gp2["Sigma"] / np.sqrt(np.outer(np.diag(gp2["Sigma"]), np.diag(gp2["Sigma"])))
    assert R[0, 1] > 0.0          # correct sign of the planted positive correlation
    assert R[0, 1] > abs(R[0, 2])  # correlated pair stronger than the null pair
    assert np.min(np.linalg.eigvalsh(gp2["Sigma"])) > 0  # SPD
