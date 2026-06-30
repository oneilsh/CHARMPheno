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


def test_gated_prior_uses_marginal_subblock_not_conditional():
    """Gated doc uses the MARGINAL sub-block precision inv(Sigma_AA), not the
    conditional (sub-block of the inverse). They differ whenever Sigma has
    off-diagonal structure. This test patches the Hessian builder to capture
    the actual precision passed in, then asserts it equals the marginal form."""
    from spark_vi.models.topic import stm as stm_mod
    from spark_vi.models.topic.stm import _stm_doc_inference
    from spark_vi.models.topic._linalg import safe_inverse
    Sigma = np.array([[2.0, 0.8, 0.5],
                      [0.8, 1.5, 0.3],
                      [0.5, 0.3, 1.0]])
    allowed = np.array([0, 2], dtype=np.int64)  # drop topic 1
    marginal = safe_inverse(Sigma[np.ix_(allowed, allowed)])
    conditional = safe_inverse(Sigma)[np.ix_(allowed, allowed)]
    assert not np.allclose(marginal, conditional)  # they genuinely differ
    # capture the precision actually used by patching the hessian builder
    seen = {}
    orig = stm_mod._stm_neg_log_joint_hessian
    def spy(eta, **kw): seen["Sigma_inv"] = kw["Sigma_inv"]; return orig(eta, **kw)
    stm_mod._stm_neg_log_joint_hessian = spy
    try:
        _stm_doc_inference(
            indices=np.array([0,1],dtype=np.int32), counts=np.array([1.0,1.0]),
            expElogbeta=np.ones((3,2))*0.5, Gamma=np.zeros((1,3)),
            Sigma_inv_allowed=marginal, x=np.array([1.0]),
            allowed=allowed, reference=None, max_iter=5)
    finally:
        stm_mod._stm_neg_log_joint_hessian = orig
    assert np.allclose(seen["Sigma_inv"], marginal)


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


# --- Task 6: min_pair_support floor + multi-group cross-covariance -----------

def _gated_multigroup_docs(rng, n_comorbid):
    """background topics {0,1}; group A foreground {2}; group B foreground {3}.
    n_comorbid docs belong to BOTH A and B (co-activate topics 2 and 3)."""
    from spark_vi.models.topic.types import STMDocument
    docs = []
    def mk(groups, idx):
        cf = np.zeros(6)
        for i in idx: cf[i] = 5.0
        ix = np.nonzero(cf)[0].astype(np.int32)
        return STMDocument(indices=ix, counts=cf[ix], length=int(cf.sum()),
                           x=np.array([1.0]), groups=frozenset(groups))
    for _ in range(400): docs.append(mk(["A"], [0, 1, 4]))   # vocab 4 ~ topic2 word
    for _ in range(400): docs.append(mk(["B"], [0, 1, 5]))   # vocab 5 ~ topic3 word
    for _ in range(n_comorbid): docs.append(mk(["A", "B"], [0, 1, 4, 5]))
    return docs


def _fit_block(min_pair_support, n_comorbid):
    from spark_vi.models.topic.stm import OnlineSTM
    from spark_vi.models.topic.partition import TopicBlockPartition
    rng = np.random.default_rng(7)
    part = TopicBlockPartition(group_var="g", background_k=2,
                               foreground=(("A", 1), ("B", 1)))  # K=4
    m = OnlineSTM(K=4, vocab_size=6, P=1, reference_topic=False,
                  topic_blocks=part, min_pair_support=min_pair_support)
    gp = m.initialize_global(None); gp["Gamma"] = np.zeros((1, 4))
    docs = _gated_multigroup_docs(rng, n_comorbid)
    stats = m.local_update(docs, gp)
    return m.update_global(gp, stats, 1.0)["Sigma"], part


def test_cross_group_covariance_from_comorbid_docs():
    Sig, part = _fit_block(min_pair_support=10, n_comorbid=300)
    a = part.block_indices("A")[0]; b = part.block_indices("B")[0]
    assert Sig[a, b] != 0.0                     # informed cross-group entry
    assert np.min(np.linalg.eigvalsh(Sig)) > 0  # SPD


def test_thin_cross_group_falls_back_to_prior():
    Sig, part = _fit_block(min_pair_support=50, n_comorbid=5)  # below floor
    a = part.block_indices("A")[0]; b = part.block_indices("B")[0]
    assert Sig[a, b] == 0.0    # 5 comorbid docs < floor 50 -> not estimated
    assert np.min(np.linalg.eigvalsh(Sig)) > 0


def test_min_pair_support_validation():
    import pytest
    from spark_vi.models.topic.stm import OnlineSTM
    with pytest.raises(ValueError):
        OnlineSTM(K=4, vocab_size=6, P=1, reference_topic=False, min_pair_support=0)


# --- Task 7: topic_correlation_matrix method ----------------------------------

def test_topic_correlation_matrix_from_sigma():
    from spark_vi.models.topic.stm import OnlineSTM
    m = OnlineSTM(K=3, vocab_size=8, P=1, reference_topic=False)
    gp = m.initialize_global(None)
    gp["Sigma"] = np.array([[4.0, 2.0, 0.0], [2.0, 9.0, 0.0], [0.0, 0.0, 1.0]])
    R = m.topic_correlation_matrix(gp)
    assert np.allclose(np.diag(R), 1.0)
    assert np.isclose(R[0, 1], 2.0 / np.sqrt(36.0))


# --- Task 10: adversarial SPD-assembly + full-Sigma local end-to-end ----------

def test_assembled_sigma_is_spd_when_cross_block_inconsistent():
    """Adversarial gated-assembly: background topic 0 correlates strongly with
    BOTH foreground 1 and 2, but no doc co-activates 1 and 2, so entry (1,2)
    stays at the prior 0. The raw per-pair assembly is then INDEFINITE; the
    M-step's nearest_spd repair must return a valid SPD covariance. (Design spec
    C2 — the central numerical risk; this is a deliberate characterization test
    of the repair, not a red-first TDD test.)"""
    from spark_vi.models.topic.stm import OnlineSTM
    K, D = 3, 1000
    # Target raw assembly: strong bg<->fg1 and bg<->fg2, fg1<->fg2 pinned 0.
    # det = 1 - 0.9^2 - 0.9^2 = -0.62 < 0  => genuinely indefinite.
    M = np.array([[1.0, 0.9, 0.9],
                  [0.9, 1.0, 0.0],
                  [0.9, 0.0, 1.0]])
    m = OnlineSTM(K=K, vocab_size=4, P=1, reference_topic=False)
    gp = m.initialize_global(None)        # Sigma0 = eye(3); its (1,2) entry is 0
    gp["Gamma"] = np.zeros((1, K))
    # Plant S and N so the per-pair MLE S/N = M on supported pairs; pair (1,2)
    # has zero support -> stays at Sigma0[1,2] = 0 (the inconsistent cross cell).
    N = np.full((K, K), float(D))
    N[1, 2] = N[2, 1] = 0.0
    S = M * float(D)
    S[1, 2] = S[2, 1] = 0.0
    stats = {
        "lambda_stats": np.zeros((K, 4)),
        "XtX": np.full((1, 1), float(D)),
        "XtX_groups": np.zeros((0, 1, 1)),
        "XtMu": np.zeros((1, K)),
        "residual_outer_stat": S,
        "n_pairs_stat": N,
        "n_docs_per_topic": np.full(K, float(D)),
        "doc_loglik_sum": np.array(0.0),
        "doc_eta_kl_sum": np.array(0.0),
        "n_docs": np.array(float(D)),
    }
    # (a) the RAW assembly is indefinite -> the scenario truly triggers the risk
    raw = np.where(N > 0, S / np.where(N > 0, N, 1.0), gp["Sigma"])
    assert np.min(np.linalg.eigvalsh(raw)) < 0, "scenario must produce an indefinite raw assembly"
    # (b) update_global's nearest_spd repair returns a valid SPD covariance
    Sig = m.update_global(gp, stats, learning_rate=1.0)["Sigma"]
    assert np.allclose(Sig, Sig.T)
    assert np.min(np.linalg.eigvalsh(Sig)) > 0
