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
