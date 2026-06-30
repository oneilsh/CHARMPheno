"""Σ M-step PD-completion contract for OnlineSTM (Task 2).

The gated full-covariance Σ M-step assembles K×K covariance from per-pair
scatter over inconsistent document subsets, so some cross-group pairs are
unobserved (support N below min_pair_support). The completion fills those free
entries with the maximum-determinant PD completion (zero PRECISION on free
entries) rather than the old zero-COVARIANCE pin / inverse-Wishart MAP blend.

Spec: docs/superpowers/specs/2026-06-30-stm-gated-sigma-pd-completion-design.md
"""
from __future__ import annotations

import numpy as np
import pytest

from spark_vi.models.topic.stm import OnlineSTM


def _is_pd(M):
    try:
        np.linalg.cholesky(M)
        return True
    except np.linalg.LinAlgError:
        return False


def _make_target_stats(stm, S, N):
    """Full valid target_stats dict that update_global consumes.

    Keys mirror the fixture in test_stm_contract.py (TestUpdateGlobal). Only
    residual_outer_stat (=S) and n_pairs_stat (=N) drive the Σ M-step; the
    other keys keep the λ / Γ / ELBO branches well-formed. XtX_groups is empty
    (no foreground groups) and XtX is PD so the Γ ridge solve is well-posed.
    """
    K, V, P = stm.K, stm.V, stm.P
    return {
        "lambda_stats": np.zeros((K, V)),
        "XtX": np.eye(P) * 100.0,
        "XtX_groups": np.zeros((0, P, P)),
        "XtMu": np.zeros((P, K)),
        "residual_outer_stat": S,
        "n_pairs_stat": N,
        "n_docs_per_topic": np.full(K, float(N.max())),
        "doc_loglik_sum": np.array(-1.0),
        "doc_eta_kl_sum": np.array(0.0),
        "n_docs": np.array(float(N.max())),
    }


def test_removed_params_rejected():
    for bad in ("sigma_prior_scale", "sigma_prior_count", "sigma_diag_shrink"):
        with pytest.raises(TypeError):
            OnlineSTM(K=5, vocab_size=10, P=2, **{bad: 1.0})


def test_completed_sigma_is_pd_and_preserves_observed():
    # Build target_stats so a cross-pair (3,4) is unsupported (N=0) -> free.
    K = 5
    stm = OnlineSTM(K=K, vocab_size=8, P=2, min_pair_support=2)
    rng = np.random.default_rng(0)
    eta = rng.standard_normal((40, K))
    S = eta.T @ eta
    N = np.full((K, K), 40.0)
    N[3, 4] = N[4, 3] = 0.0                       # thin cross-pair -> free
    gp = stm.initialize_global({"vocab_size": 8})
    out = stm.update_global(gp, _make_target_stats(stm, S, N), learning_rate=1.0)
    Sigma = out["Sigma"]
    assert _is_pd(Sigma)
    # observed diagonal/cross entries carried through (ρ=1 -> Σ_target on observed).
    assert abs(np.linalg.inv(Sigma)[3, 4]) < 1e-5    # free pair -> zero precision
