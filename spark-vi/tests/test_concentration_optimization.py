"""Unit tests for spark_vi.inference.concentration_optimization.

The Newton helpers (alpha_newton_step, eta_newton_step) have existing
recovery-test coverage in test_lda_math.py via the back-compat aliases
in lda.py. This file owns coverage for the new beta_concentration_closed_form
helper introduced for HDP's γ and α optimization.
"""
from __future__ import annotations

import numpy as np
import pytest

from spark_vi.inference.concentration_optimization import (
    alpha_newton_step,
    beta_concentration_closed_form,
    eta_newton_step,
)


def test_beta_concentration_closed_form_recovers_known_beta_on_synthetic():
    """β* = -N/S recovers the true β from i.i.d. Beta(1, β_true) draws.

    Under a degenerate variational posterior q(W) = δ(W − W_t) (i.e.
    treat each sampled stick break as the variational mean), the closed
    form's input S = Σ E_q[log(1 − W)] = Σ log(1 − W_t) exactly. The
    recovered β* therefore matches β_true up to Monte-Carlo noise. The
    SVI runtime feeds in S = Σ [ψ(b) − ψ(a + b)] from non-degenerate
    Beta(a, b) posteriors, but the closed-form math being tested is
    identical.
    """
    rng = np.random.default_rng(123)
    true_beta = 3.0
    n = 5000

    sticks = rng.beta(1.0, true_beta, size=n)
    s = float(np.log(1.0 - sticks).sum())

    beta_star = beta_concentration_closed_form(n=n, s_log_one_minus=s)
    assert abs(beta_star - true_beta) < 0.1, f"got {beta_star}, expected ~{true_beta}"


def test_beta_concentration_closed_form_recovers_small_beta():
    """Small β (~0.5) — concentrated weight on a few sticks; recovery still holds."""
    rng = np.random.default_rng(7)
    true_beta = 0.5
    n = 5000

    sticks = rng.beta(1.0, true_beta, size=n)
    s = float(np.log(1.0 - sticks).sum())

    beta_star = beta_concentration_closed_form(n=n, s_log_one_minus=s)
    assert abs(beta_star - true_beta) < 0.05


def test_beta_concentration_closed_form_rejects_invalid_n():
    with pytest.raises(ValueError, match="n must be > 0"):
        beta_concentration_closed_form(n=0, s_log_one_minus=-1.0)
    with pytest.raises(ValueError, match="n must be > 0"):
        beta_concentration_closed_form(n=-5.0, s_log_one_minus=-1.0)


def test_beta_concentration_closed_form_rejects_nonnegative_s():
    """S = Σ log(1−W) is always negative for W ∈ (0,1); guard against
    misuse (e.g. caller swapped sign or passed E[log W] instead).
    """
    with pytest.raises(ValueError, match="s_log_one_minus must be < 0"):
        beta_concentration_closed_form(n=10, s_log_one_minus=0.0)
    with pytest.raises(ValueError, match="s_log_one_minus must be < 0"):
        beta_concentration_closed_form(n=10, s_log_one_minus=1.5)


def test_alpha_newton_step_importable_from_inference_module():
    """The lifted helper must be reachable at its new home (the lda.py
    back-compat alias is tested in test_lda_math.py).
    """
    K = 3
    alpha = np.full(K, 1.0 / K, dtype=np.float64)
    e_log_theta_sum = np.array([-1.0, -2.0, -1.5])
    delta = alpha_newton_step(alpha, e_log_theta_sum, D=100.0)
    assert delta.shape == (K,)
    assert np.all(np.isfinite(delta))


def test_eta_newton_step_importable_from_inference_module():
    delta = eta_newton_step(eta=0.1, e_log_phi_sum=-200.0, K=5, V=20)
    assert np.isfinite(delta)


def test_gated_alpha_newton_step_matches_finite_difference_gradient():
    # The raw Newton step is -H^-1 g. We validate the assembled gradient g and
    # Hessian H against numerical differentiation of the exact gated ELBO-in-alpha
    #   L(a) = Σ_g N_g[logΓ(Σ_{b in g} m_b a_b) − Σ_{b in g} m_b logΓ(a_b)]
    #        + Σ_b e_b a_b            (data term is linear in a_b; constant drops)
    # on a tiny system (2 groups, 3 tied blocks).
    import numpy as np
    from scipy.special import gammaln
    from spark_vi.inference.concentration_optimization import gated_alpha_newton_step

    m = np.array([2.0, 1.0, 1.0])                    # block sizes: bg=2, two nodes tpn=1
    a = np.array([0.30, 0.05, 0.12])                 # current tied alpha
    e = np.array([-3.1, -0.7, -1.4])                 # data term (scaled)
    Ng = np.array([40.0, 15.0])
    memb = np.array([[True, True, False],            # group 0: bg + node1
                     [True, False, True]])           # group 1: bg + node2

    def L(av):
        val = float(np.dot(e, av))
        for g in range(len(Ng)):
            idx = np.where(memb[g])[0]
            s = np.sum(m[idx] * av[idx])
            val += Ng[g] * (gammaln(s) - np.sum(m[idx] * gammaln(av[idx])))
        return val

    # numerical gradient and Hessian of L at a
    eps = 1e-6
    B = a.shape[0]
    grad = np.zeros(B)
    for b in range(B):
        ap = a.copy(); ap[b] += eps
        am = a.copy(); am[b] -= eps
        grad[b] = (L(ap) - L(am)) / (2 * eps)
    H = np.zeros((B, B))
    for b in range(B):
        for c in range(B):
            app = a.copy(); app[b] += eps; app[c] += eps
            apm = a.copy(); apm[b] += eps; apm[c] -= eps
            amp = a.copy(); amp[b] -= eps; amp[c] += eps
            amm = a.copy(); amm[b] -= eps; amm[c] -= eps
            H[b, c] = (L(app) - L(apm) - L(amp) + L(amm)) / (4 * eps * eps)

    expected_delta = -np.linalg.solve(H, grad)       # the Newton step L should produce
    got = gated_alpha_newton_step(a, m, e, Ng, memb)
    assert np.allclose(got, expected_delta, rtol=1e-3, atol=1e-5), (got, expected_delta)


def test_gated_alpha_newton_step_iterates_toward_optimum():
    # Repeated damped Newton steps on a fixed synthetic system should climb L
    # (monotone non-decreasing) and converge (‖Δα‖ shrinks). Guards the sign.
    #
    # e magnitude note: each group's Dirichlet log-partition term
    # N_g[logΓ(Σ m_b a_b) − Σ m_b logΓ(a_b)] is convex-in-reverse (its negative
    # is the Dirichlet cumulant function, which is convex), so along any ray
    # where a group's blocks scale up together it grows like +N_g·log(M_g)·t
    # (M_g = Σ_{b in g} m_b), unboundedly — this is the standard fact that the
    # asymmetric-Dirichlet Newton objective is concave but need not have a
    # finite maximum. Since e is linear in the same directions, L(a) is
    # bounded above only if |e_b| is large enough, in the SAME units as N_g,
    # to out-pace that growth (realistic e_log_theta_node_sum values are sums
    # over ~N_g documents, so they are O(N_g), not O(1)). We scale the raw
    # per-doc data term (-4.0, -0.5, -2.0) by the group size so the synthetic
    # system has a genuine finite optimum instead of diverging to +infinity.
    import numpy as np
    from scipy.special import gammaln
    from spark_vi.inference.concentration_optimization import gated_alpha_newton_step

    m = np.array([2.0, 1.0, 1.0])
    e = np.array([-4.0, -0.5, -2.0]) * 50.0
    Ng = np.array([50.0, 20.0])
    memb = np.array([[True, True, False], [True, False, True]])

    def L(av):
        val = float(np.dot(e, av))
        for g in range(len(Ng)):
            idx = np.where(memb[g])[0]
            s = np.sum(m[idx] * av[idx])
            val += Ng[g] * (gammaln(s) - np.sum(m[idx] * gammaln(av[idx])))
        return val

    a = np.array([0.2, 0.2, 0.2])
    prev = L(a)
    last_step = None
    for _ in range(50):
        d = gated_alpha_newton_step(a, m, e, Ng, memb)
        a = np.maximum(a + 0.5 * d, 1e-3)          # ρ damping + floor
        cur = L(a)
        assert cur >= prev - 1e-6                   # monotone ascent
        prev = cur
        last_step = np.abs(d).max()
    assert last_step < 1e-3                          # converged


def test_gated_alpha_newton_step_handles_uncovered_block():
    # A tied block never present in any group (a DAG node with no labeled coverage)
    # must not make the dense solve singular; it stays put (Delta = 0) while the
    # covered blocks still move. Guards the crash the full-space solve would raise.
    import numpy as np
    from spark_vi.inference.concentration_optimization import gated_alpha_newton_step
    m = np.array([2.0, 1.0, 1.0])
    a = np.array([0.3, 0.1, 0.2])
    e = np.array([-3.0, -0.5, 0.0])          # uncovered block has no data (e=0)
    Ng = np.array([25.0])
    memb = np.array([[True, True, False]])   # block index 2 is uncovered
    d = gated_alpha_newton_step(a, m, e, Ng, memb)
    assert np.isfinite(d).all()              # no LinAlgError / NaN
    assert d[2] == 0.0                        # uncovered block does not move
    assert d[0] != 0.0 or d[1] != 0.0         # covered blocks still move
