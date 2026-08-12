"""DagClosureHead — the label-side HIERARCHY flavor of the PC supervised-head seam.

Mondo is a DAG (not a tree), so HSLDA's parent-gating (Perotte 2011, ICD-9 tree)
generalizes to a closure PRODUCT that counts each ancestor once (diamond-safe):

    log P(node_l = 1) = Σ_{a ∈ closure(l)} log σ(w_a · π) .

These tests pin (i) the closure matrix incl. a diamond, (ii) the monotone is-a
consistency P(child) ≤ P(parent) that the product buys by construction, (iii) that
the base autograd accumulators differentiate the DAG loss correctly (finite-
difference, both the topic-correction and the head gradient), and (iv) that the
model accepts the injected head and — since the DAG head has no closed-form Fisher —
'newton' gracefully degrades (no head_hess_stat emitted). Pure numpy except (iv).
"""
import numpy as np
import pytest

from spark_vi.models.topic.pc import DagClosureHead, SupervisedHead


# 0=root; 1,2 under root; 3 under BOTH 1 and 2 (a diamond). closure(3)={3,1,2,0}.
DIAMOND_PARENTS = [(), (0,), (0,), (1, 2)]


def _dag_docs(seed, C, V, n=6):
    """A tiny fixed PCDocument batch with C labels and a non-trivial observed mask,
    guaranteeing one fully-observed doc so the batch is non-degenerate."""
    from spark_vi.models.topic.types import PCDocument
    rng = np.random.default_rng(seed)
    docs = []
    for _ in range(n):
        nnz = int(rng.integers(3, 7))
        idx = np.sort(rng.choice(V, size=nnz, replace=False)).astype(np.int32)
        cnt = rng.integers(1, 6, size=nnz).astype(np.float64)
        y = rng.integers(0, 2, size=C).astype(np.float64)
        mask = rng.integers(0, 2, size=C).astype(np.float64)
        docs.append(PCDocument(indices=idx, counts=cnt, length=int(cnt.sum()),
                               y=y, label_mask=mask))
    docs[0] = PCDocument(indices=docs[0].indices, counts=docs[0].counts,
                         length=docs[0].length, y=np.ones(C), label_mask=np.ones(C))
    return docs


def test_closure_matrix_is_diamond_safe():
    """closure(l) = l ∪ all ancestors, each counted ONCE even under a diamond."""
    h = DagClosureHead(DIAMOND_PARENTS)
    assert isinstance(h, SupervisedHead)
    assert h.C == 4
    M = h._closure_matrix
    assert M[0].tolist() == [1, 0, 0, 0]            # root: just itself
    assert M[1].tolist() == [1, 1, 0, 0]            # {1, 0}
    assert M[2].tolist() == [1, 0, 1, 0]            # {2, 0}
    assert M[3].tolist() == [1, 1, 1, 1]            # {3,1,2,0} — shared ancestor 0 once
    assert M[3].sum() == 4                          # not 5 (0 not double-counted)


def test_monotone_is_a_consistency():
    """P(child) ≤ P(parent) for ANY head weights — the product of ≤1 sigmoids over a
    growing closure can only shrink. This is the hierarchy baked into the model."""
    h = DagClosureHead(DIAMOND_PARENTS)
    rng = np.random.default_rng(0)
    K = 3
    for _ in range(50):
        W = rng.standard_normal((4, K)) * 2.0
        theta = rng.dirichlet(np.ones(K))
        ls = np.log(1.0 / (1.0 + np.exp(-(W @ theta))))
        P = np.exp(h._closure_matrix @ ls)
        assert P[1] <= P[0] + 1e-12                 # child ≤ parent along every edge
        assert P[2] <= P[0] + 1e-12
        assert P[3] <= P[1] + 1e-12 and P[3] <= P[2] + 1e-12


def test_dag_head_gradient_matches_finite_difference():
    """The base autograd accumulators differentiate the DAG-closure loss correctly:
    the accumulated (∂/∂topics_repr, ∂/∂w_CK) match a central finite-difference of
    the same batch NLL to max rel err <= 1e-5 — so a new flavor needs NO hand-derived
    gradient, only its per-doc loss."""
    h = DagClosureHead(DIAMOND_PARENTS)
    rng = np.random.default_rng(1)
    K, V, C = 4, 12, 4
    n_iters = 20
    alpha = np.full(K, 1.1)
    topics_repr = rng.random((K, V)) * 0.3 + 0.01      # expElogbeta-like, positive
    w_CK = rng.standard_normal((C, K)) * 0.4           # modest -> logP away from 0
    docs = _dag_docs(seed=0, C=C, V=V)

    _loss, grad_topics, grad_wCK = h.batch_value_and_grad(
        topics_repr, w_CK, docs, alpha, K, n_iters)

    def _fd(param, grad, evaluate, eps=1e-6):
        rel = []
        for (i, j) in zip(*np.nonzero(grad)):
            pp = param.copy(); pp[i, j] += eps
            pm = param.copy(); pm[i, j] -= eps
            num = (evaluate(pp) - evaluate(pm)) / (2 * eps)
            ana = grad[i, j]
            rel.append(abs(num - ana) / max(abs(ana), abs(num), 1e-8))
        return max(rel)

    err_topics = _fd(topics_repr, grad_topics,
                     lambda p: h.batch_value(p, w_CK, docs, alpha, K, n_iters))
    err_head = _fd(w_CK, grad_wCK,
                   lambda p: h.batch_value(topics_repr, p, docs, alpha, K, n_iters))
    assert err_topics <= 1e-5, f"DAG topic-gradient rel err {err_topics:.2e} > 1e-5"
    assert err_head <= 1e-5, f"DAG head-gradient rel err {err_head:.2e} > 1e-5"


# --- Exact block Fisher: z-space reference (decoupled from theta/CAVI) ----------
# Per-doc node loss as a pure function of the local logits z = w_CK @ theta:
#   P_l = prod_{a in closure(l)} sigmoid(z_a);  nll_l = -y_l log P_l - (1-y_l) log(1-P_l)
def _Lz(z, y, obs, M):
    ls = np.minimum(z, 0.0) - np.log1p(np.exp(-np.abs(z)))
    P = np.clip(np.exp(M @ ls), 1e-12, 1 - 1e-9)
    return float(np.sum(obs * (-y * np.log(P) - (1 - y) * np.log(1 - P))))


def _gz(z, y, obs, M):                                   # analytic z-gradient
    p = 1.0 / (1.0 + np.exp(-z))
    ls = np.minimum(z, 0.0) - np.log1p(np.exp(-np.abs(z)))
    P = np.clip(np.exp(M @ ls), 1e-12, 1 - 1e-9)
    r = -y + (1 - y) * P / (1 - P)
    return (1 - p) * (M.T @ (obs * r))


def _Iz(z, obs, M):                                      # analytic block FISHER (C,C)
    # I_fisher_{ab} = sum_l obs_l (1-p_a)(1-p_b) [a,b in cl(l)] P_l/(1-P_l)
    p = 1.0 / (1.0 + np.exp(-z))
    ls = np.minimum(z, 0.0) - np.log1p(np.exp(-np.abs(z)))
    P = np.clip(np.exp(M @ ls), 1e-12, 1 - 1e-9)
    ratio = obs * P / (1 - P)
    W = M.T @ (ratio[:, None] * M)                       # sum_l ratio_l M[l,a] M[l,b]
    return np.outer(1 - p, 1 - p) * W


def _Ifull(z, y, obs, M):                                # PSD 'kept' part of the ACTUAL Hessian
    # differs from the Fisher by carrying (1-y_l)·P_l/(1-P_l)^2 (not P_l/(1-P_l));
    # E_y[_Ifull] == _Iz since E[1-y_l] = 1-P_l.
    p = 1.0 / (1.0 + np.exp(-z))
    ls = np.minimum(z, 0.0) - np.log1p(np.exp(-np.abs(z)))
    P = np.clip(np.exp(M @ ls), 1e-12, 1 - 1e-9)
    w = obs * (1 - y) * P / (1 - P) ** 2
    W = M.T @ (w[:, None] * M)
    return np.outer(1 - p, 1 - p) * W


def _Dz(z, y, obs, M):                                   # dropped indefinite diagonal (E_y = 0)
    p = 1.0 / (1.0 + np.exp(-z))
    ls = np.minimum(z, 0.0) - np.log1p(np.exp(-np.abs(z)))
    P = np.clip(np.exp(M @ ls), 1e-12, 1 - 1e-9)
    r = -y + (1 - y) * P / (1 - P)
    return -p * (1 - p) * (M.T @ (obs * r))


def test_dag_block_fisher_formula_matches_finite_difference():
    """Two exact checks tying the block Fisher to the loss: (i) the FULL Hessian
    _Ifull + diag(_Dz) matches central FD of the analytic z-gradient (which itself
    matches FD of the loss); (ii) the Fisher _Iz == E_y[full Hessian] with y ~ ∏
    Bernoulli(P_l), by EXACT enumeration over all 2^C labels (E[1-y]=1-P collapses
    _Ifull->_Iz, E[r]=0 kills _Dz). So H_aa = I_a·θθᵀ is the true expected curvature."""
    import itertools
    M = DagClosureHead(DIAMOND_PARENTS)._closure_matrix
    C = 4
    rng = np.random.default_rng(7)
    z = rng.standard_normal(C) * 0.6
    obs = np.ones(C)
    eps = 1e-6

    # (i) analytic gradient matches FD of the loss.
    y = rng.integers(0, 2, C).astype(float)
    g = _gz(z, y, obs, M)
    g_fd = np.array([(_Lz(z + eps * e, y, obs, M) - _Lz(z - eps * e, y, obs, M)) / (2 * eps)
                     for e in np.eye(C)])
    assert np.max(np.abs(g - g_fd)) < 1e-6

    # ... and the full Hessian _Ifull + diag(_Dz) matches FD of that gradient.
    Hz = _Ifull(z, y, obs, M) + np.diag(_Dz(z, y, obs, M))
    Hz_fd = np.zeros((C, C))
    for j in range(C):
        ej = np.zeros(C); ej[j] = eps
        Hz_fd[:, j] = (_gz(z + ej, y, obs, M) - _gz(z - ej, y, obs, M)) / (2 * eps)
    assert np.max(np.abs(Hz - Hz_fd)) < 1e-5

    # (ii) the implementation's Fisher == E_y[full Hessian], exactly.
    ls = np.minimum(z, 0.0) - np.log1p(np.exp(-np.abs(z)))
    P = np.clip(np.exp(M @ ls), 1e-12, 1 - 1e-9)
    EH = np.zeros((C, C))
    for bits in itertools.product([0, 1], repeat=C):
        yb = np.array(bits, dtype=np.float64)
        wgt = float(np.prod(P ** yb * (1 - P) ** (1 - yb)))
        EH += wgt * (_Ifull(z, yb, obs, M) + np.diag(_Dz(z, yb, obs, M)))
    assert np.max(np.abs(EH - _Iz(z, obs, M))) < 1e-9
    assert np.min(np.linalg.eigvalsh(_Iz(z, obs, M))) >= -1e-9      # Fisher is PSD


def test_dag_block_fisher_impl_matches_formula():
    """batch_hessian emits exactly H[a] = diag(Fisher)_a · θθᵀ per doc, with θ the same
    CAVI mean, tying the (C,K,K) implementation to the FD-verified z-space Fisher."""
    from spark_vi.models.topic.pc import _cavi_theta_anp
    h = DagClosureHead(DIAMOND_PARENTS)
    M = h._closure_matrix
    K, V, C = 4, 12, 4
    alpha = np.full(K, 1.1)
    rng = np.random.default_rng(5)
    topics_repr = rng.random((K, V)) * 0.3 + 0.01
    w_CK = rng.standard_normal((C, K)) * 0.4
    docs = _dag_docs(seed=1, C=C, V=V)

    H = h.batch_hessian(topics_repr, w_CK, docs, alpha, K, 10)
    assert H.shape == (C, K, K)
    ref = np.zeros((C, K, K))
    for doc in docs:
        obs = np.asarray(doc.label_mask, dtype=np.float64)
        if obs.sum() == 0.0:
            continue
        theta = _cavi_theta_anp(topics_repr[:, doc.indices],
                                np.asarray(doc.counts, np.float64), alpha, K, 10)
        z = np.clip(w_CK @ theta, -50, 50)
        ref += np.diag(_Iz(z, obs, M))[:, None, None] * np.outer(theta, theta)[None, :, :]
    assert np.allclose(H, ref)
    for c in range(C):
        assert np.min(np.linalg.eigvalsh(H[c])) >= -1e-9           # each block PSD


def test_dag_block_fisher_reduces_to_flat_on_star():
    """On a STAR DAG (every node its own root, closure(l)={l}) the block Fisher must
    equal the flat logistic Fisher exactly — the DAG head's curvature is a strict
    generalization of the flat head's."""
    from spark_vi.models.topic.pc import _supervised_head_hessian
    K, V, C = 4, 12, 4
    alpha = np.full(K, 1.1)
    rng = np.random.default_rng(5)
    topics_repr = rng.random((K, V)) * 0.3 + 0.01
    w_CK = rng.standard_normal((C, K)) * 0.4
    docs = _dag_docs(seed=1, C=C, V=V)
    h = DagClosureHead([[] for _ in range(C)])           # no edges -> M = I
    H = h.batch_hessian(topics_repr, w_CK, docs, alpha, K, 10)
    H_flat = _supervised_head_hessian(topics_repr, w_CK, docs, alpha, K, 10)
    assert np.allclose(H, H_flat)


def test_model_dag_head_runs_newton():
    """OnlinePCLDA(head=DagClosureHead(...)) with head_optimizer='newton' emits the
    (quasi-Newton) head_hess_stat and takes the Newton solve — NOT the SGD step —
    moving the head off zero without error."""
    from spark_vi.models.topic.pc import OnlinePCLDA
    K, V, C = 4, 12, 4
    head = DagClosureHead(DIAMOND_PARENTS)
    model = OnlinePCLDA(K=K, vocab_size=V, C=C, weight_y=1.0, grad_cavi_iters=10,
                        head_optimizer="newton", head=head)
    gp = model.initialize_global(None)
    docs = _dag_docs(seed=3, C=C, V=V)
    stats = model.local_update(docs, gp)
    assert "grad_wCK_stat" in stats and "grad_topics_stat" in stats
    assert "head_hess_stat" in stats                # exact block Fisher emitted
    new_gp = model.update_global(gp, stats, learning_rate=0.5)
    assert new_gp["w_CK"].shape == (C, K)
    assert not np.allclose(new_gp["w_CK"], 0.0)


def test_newton_falls_back_to_sgd_when_flavor_has_no_hessian():
    """The defensive guard: a flavor whose batch_hessian is None (no PD curvature)
    emits no head_hess_stat, so 'newton' degrades to the SGD head step in
    update_global rather than KeyError-ing."""
    from spark_vi.models.topic.pc import OnlinePCLDA

    class _NoHessDagHead(DagClosureHead):
        def batch_hessian(self, *a, **k):
            return None

    K, V, C = 4, 12, 4
    model = OnlinePCLDA(K=K, vocab_size=V, C=C, weight_y=1.0, grad_cavi_iters=10,
                        head_optimizer="newton", head=_NoHessDagHead(DIAMOND_PARENTS))
    gp = model.initialize_global(None)
    stats = model.local_update(_dag_docs(seed=3, C=C, V=V), gp)
    assert "head_hess_stat" not in stats
    new_gp = model.update_global(gp, stats, learning_rate=0.5)     # no KeyError
    assert not np.allclose(new_gp["w_CK"], 0.0)


def test_model_rejects_head_label_count_mismatch():
    """A head whose C disagrees with the model's C is a wiring error — fail fast."""
    from spark_vi.models.topic.pc import OnlinePCLDA
    with pytest.raises(ValueError, match="label count must match"):
        OnlinePCLDA(K=4, vocab_size=12, C=3, weight_y=1.0,
                    head=DagClosureHead(DIAMOND_PARENTS))   # head C=4 != model C=3
