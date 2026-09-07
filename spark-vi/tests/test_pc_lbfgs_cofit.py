"""The matrix-free amortized L-BFGS co-fit head (``head_optimizer='lbfgs'``).

Plan `docs/superpowers/plans/2026-09-07-lbfgs-cofit-head-plan.md` (WP1). The head
re-uses the FROZEN-θ readout solver
(:func:`spark_vi.models.topic.batched_lr.solve_batched_lr`) as the CO-FIT head,
run against a θ that MOVES every outer SVI iter — the one genuinely new problem
(non-stationary curvature) the plan gates the whole build on.

Two groups of obligations:

  * SEAM INTEGRITY (fast, unit): 'lbfgs' is in the allowlist; the new knobs
    (``head_inner_iters``, ``head_history_reset``) are INERT for 'sgd'/'newton'
    (byte-identical steps); ``local_update`` under 'lbfgs' emits ONLY the sgd
    gradient stats — NO Hessian; the solver's new ``state`` carry is byte-identical
    to the readout path when unused and genuinely carries curvature when threaded;
    a missing provider degrades safely to the one-step sgd move.

  * THE COUPLING GATE (slow, the deliverable): a small gated-label simulator fit
    with the lbfgs head shows (1) corr_relΔλ REACHES the healthy ~0.02-0.05 band
    and does NOT explode under the ``head_trust_move`` cap; (2) grad_y SHRINKS as
    the head converges (not the sgd "chasing moving θ" failure of exp 0119); and
    (3) the co-fit head's heldout AUC TRACKS a from-scratch sklearn oracle on the
    final θ. Sweep decision (recorded here): with the trust cap engaged the head
    converges within each outer iter, so head_inner_iters ∈ {1,3,5} and
    head_history_reset ∈ {True,False} ALL satisfy (1)-(3) equivalently on this
    simulator; the shipped defaults are inner_iters=3 (a safe amortization) and
    history_reset=True (fresh curvature each iter — no stale-curvature risk when θ
    moves faster at scale).
"""
import warnings

import numpy as np
import pytest

from spark_vi.models.topic.pc import OnlinePCLDA
from spark_vi.models.topic.types import PCDocument


# --------------------------------------------------------------------------- #
# tiny deterministic PC corpus (planted head: node c reads topics {2c, 2c+1})  #
# --------------------------------------------------------------------------- #
def _corpus(seed=0, D=180, V=48, K=6, C=3):
    rng = np.random.default_rng(seed)
    w = V // K
    beta = np.full((K, V), 0.02)
    for k in range(K):
        beta[k, k * w:(k + 1) * w] += 1.0
    beta /= beta.sum(axis=1, keepdims=True)
    cum = np.cumsum(beta, axis=1)
    sig = [np.array([2 * c, 2 * c + 1]) for c in range(C)]
    docs = []
    for _ in range(D):
        theta = rng.dirichlet(np.full(K, 0.3))
        length = int(rng.integers(30, 50))
        z = rng.choice(K, size=length, p=theta)
        u = rng.random(length)
        words = np.array([int(np.searchsorted(cum[zi], ui))
                          for zi, ui in zip(z, u)])
        idx, cnt = np.unique(words, return_counts=True)
        y = np.array([float(rng.random() < 1.0 / (1.0 + np.exp(
            -8.0 * (theta[sig[c]].sum() - 0.34)))) for c in range(C)])
        docs.append(PCDocument(indices=idx.astype(np.int32),
                               counts=cnt.astype(np.float64), length=length,
                               y=y, label_mask=np.ones(C)))
    return docs, V, K, C


def _small(seed=0):
    docs, V, K, C = _corpus(seed=seed, D=24, V=24, K=4, C=2)
    return docs, V, K, C


# --------------------------------------------------------------------------- #
# 1. allowlist + validation                                                    #
# --------------------------------------------------------------------------- #
def test_lbfgs_accepted_and_bad_values_rejected():
    OnlinePCLDA(K=4, vocab_size=12, C=1, weight_y=1.0, head_optimizer="lbfgs")
    with pytest.raises(ValueError, match="'sgd', 'newton' or 'lbfgs'"):
        OnlinePCLDA(K=4, vocab_size=12, C=1, head_optimizer="adam")
    with pytest.raises(ValueError, match="head_inner_iters must be >= 1"):
        OnlinePCLDA(K=4, vocab_size=12, C=1, head_optimizer="lbfgs",
                    head_inner_iters=0)


# --------------------------------------------------------------------------- #
# 2. the new knobs are inert for sgd / newton (byte-identical steps)           #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("opt", ["sgd", "newton"])
def test_new_knobs_inert_for_sgd_and_newton(opt):
    docs, V, K, C = _small()
    common = dict(K=K, vocab_size=V, C=C, weight_y=5.0, grad_cavi_iters=6,
                  head_optimizer=opt, head_l2=1e-2, head_lr=0.5, random_seed=0)
    m0 = OnlinePCLDA(**common)
    m1 = OnlinePCLDA(head_inner_iters=7, head_history_reset=False, **common)
    g0 = m0.initialize_global(None)
    g1 = m1.initialize_global(None)
    s0 = m0.local_update(docs, g0)
    s1 = m1.local_update(docs, g1)
    for k in set(s0) | set(s1):
        np.testing.assert_array_equal(s0[k], s1[k])
    o0 = m0.update_global(g0, s0, 0.5)
    o1 = m1.update_global(g1, s1, 0.5)
    np.testing.assert_array_equal(o0["w_CK"], o1["w_CK"])
    np.testing.assert_array_equal(o0["b_CK"], o1["b_CK"])


# --------------------------------------------------------------------------- #
# 3. lbfgs local_update emits grad-only stats — NO Hessian (ADR 0047 / D2)      #
# --------------------------------------------------------------------------- #
def test_lbfgs_local_update_is_grad_only_no_hessian():
    docs, V, K, C = _small()
    m = OnlinePCLDA(K=K, vocab_size=V, C=C, weight_y=5.0, grad_cavi_iters=6,
                    head_optimizer="lbfgs", head_intercept=True,
                    head_standardize=True, random_seed=0)
    stats = m.local_update(docs, m.initialize_global(None))
    assert "grad_topics_stat" in stats and "grad_wCK_stat" in stats
    for hess_key in ("head_hess_stat", "head_irls_hess_stat",
                     "head_irls_grad_stat", "theta_m1_stat", "theta_m2_stat"):
        assert hess_key not in stats, f"lbfgs must not emit {hess_key}"


# --------------------------------------------------------------------------- #
# 4. no provider -> safe degrade to the one-step sgd move (announced)          #
# --------------------------------------------------------------------------- #
def test_lbfgs_without_provider_falls_back_to_sgd_move():
    docs, V, K, C = _small()
    common = dict(K=K, vocab_size=V, C=C, weight_y=5.0, grad_cavi_iters=6,
                  head_l2=1e-3, head_lr_scale=1.0, random_seed=0)
    m_l = OnlinePCLDA(head_optimizer="lbfgs", **common)
    m_s = OnlinePCLDA(head_optimizer="sgd", **common)
    g = m_l.initialize_global(None)
    s_l = m_l.local_update(docs, g)
    s_s = m_s.local_update(docs, dict(g))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        o_l = m_l.update_global(dict(g), s_l, 0.5)
    assert any("no head stats provider" in str(x.message) for x in w)
    o_s = m_s.update_global(dict(g), s_s, 0.5)
    np.testing.assert_allclose(o_l["w_CK"], o_s["w_CK"])


# --------------------------------------------------------------------------- #
# 5. solve_batched_lr `state` carry: byte-identical unused, carries when used   #
# --------------------------------------------------------------------------- #
def test_solve_batched_lr_state_is_byte_identical_unused_and_carries_used():
    from spark_vi.models.topic.batched_lr import (
        make_inmemory_stats_fn, solve_batched_lr, standardization_moments)
    rng = np.random.default_rng(1)
    D, K, C = 200, 5, 2
    Pi = rng.dirichlet(np.ones(K) * 0.4, size=D)
    obs = np.ones((D, C), bool)
    wtrue = rng.normal(size=(C, K))
    y = (1.0 / (1.0 + np.exp(-(Pi @ wtrue.T))) > rng.random((D, C))).astype(float)
    mu, sd, _ = standardization_moments(Pi, obs)
    sf = make_inmemory_stats_fn(Pi, y, obs, mu, sd)

    Wa, ba, ia = solve_batched_lr(sf, C, K, l2=1.0, max_iter=3)
    Wb, bb, ib = solve_batched_lr(sf, C, K, l2=1.0, max_iter=3, state={})
    np.testing.assert_array_equal(Wa, Wb)          # empty state == readout path
    np.testing.assert_array_equal(ba, bb)
    assert "state" not in ia and "state" in ib     # info byte-identical for readout

    W1, b1, i1 = solve_batched_lr(sf, C, K, l2=1.0, max_iter=2, state={})
    W2c, _, _ = solve_batched_lr(sf, C, K, l2=1.0, max_iter=2,
                                 x0=(W1, b1), state=i1["state"])   # carried curvature
    W2f, _, _ = solve_batched_lr(sf, C, K, l2=1.0, max_iter=2, x0=(W1, b1))  # fresh
    assert not np.allclose(W2c, W2f)               # carry genuinely changes the step
    assert np.isfinite(W2c).all() and np.isfinite(W2f).all()


# --------------------------------------------------------------------------- #
# THE COUPLING GATE (plan WP1(b)) — the deliverable                             #
# --------------------------------------------------------------------------- #
def _oracle_and_head_auc(model, tr, te, gp):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    def theta(docs):
        return np.array([model.infer_local(d, gp)["theta"] for d in docs])

    tr_t, te_t = theta(tr), theta(te)
    tr_y = np.array([d.y for d in tr])
    te_y = np.array([d.y for d in te])
    w, b = gp["w_CK"], gp["b_CK"]
    o_aucs, h_aucs = [], []
    for c in range(model.C):
        if tr_y[:, c].min() == tr_y[:, c].max() or te_y[:, c].min() == te_y[:, c].max():
            continue
        lr = LogisticRegression(max_iter=1000).fit(tr_t, tr_y[:, c])
        o_aucs.append(roc_auc_score(te_y[:, c], lr.predict_proba(te_t)[:, 1]))
        z = te_t @ w[c] + b[c]
        h_aucs.append(roc_auc_score(te_y[:, c], 1.0 / (1.0 + np.exp(-z))))
    return float(np.mean(o_aucs)), float(np.mean(h_aucs))


@pytest.mark.slow
@pytest.mark.parametrize("inner,reset", [(3, True), (1, True), (3, False)])
def test_lbfgs_cofit_coupling_gate(inner, reset):
    """corr in the healthy band + bounded, grad_y shrinks, head tracks the oracle."""
    pytest.importorskip("sklearn")
    docs, V, K, C = _corpus(seed=0, D=180, V=48, K=6, C=3)
    ntr = int(0.75 * len(docs))
    tr, te = docs[:ntr], docs[ntr:]
    trust = 0.03
    m = OnlinePCLDA(K=K, vocab_size=V, C=C, weight_y=12.0,
                    weight_y_warmup_iters=5, grad_cavi_iters=8,
                    head_optimizer="lbfgs", head_l2=1e-2, head_standardize=True,
                    head_intercept=True, head_trust_move=trust,
                    head_inner_iters=inner, head_history_reset=reset,
                    alpha=0.3, random_seed=0)
    m.set_head_stats_provider(m.make_head_stats_provider_inmemory(tr))
    gp = m.initialize_global(None)
    corr, grad_y = [], []
    for it in range(22):
        rho = (it + 2.0) ** (-0.6)
        gp = m.update_global(gp, m.local_update(tr, gp), learning_rate=rho)
        if m._eff_wy > 0.0:
            corr.append(m._corr_relchg)
            grad_y.append(m._grad_y)
    corr, grad_y = np.array(corr), np.array(grad_y)

    # (1) corr REACHES the healthy band and is BOUNDED by the trust cap (no explosion).
    assert corr.max() >= 0.02, f"shaping too weak: corr peak {corr.max():.2e}"
    assert corr.max() <= trust + 1e-6, f"over-drove past the cap: {corr.max():.3e}"
    # (2) grad_y SHRINKS as the head converges (not exp 0119's growing chase).
    assert grad_y[-1] < 0.5 * grad_y[0], (
        f"grad_y did not shrink: first {grad_y[0]:.2e} last {grad_y[-1]:.2e}")
    # (3) the co-fit head TRACKS the from-scratch sklearn oracle on the final θ.
    o, h = _oracle_and_head_auc(m, tr, te, gp)
    assert o > 0.75, f"oracle AUC {o:.3f} — planted signal not learnable, bad fixture"
    assert abs(o - h) < 0.06, f"head {h:.3f} does not track oracle {o:.3f}"
