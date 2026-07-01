"""Σ M-step contract for OnlineSTM.

The gated Σ M-step is block-wise unit-diagonal (ADR 0034): it standardizes the
observed per-pair scatter to correlations, lazy-keeps unsupported pairs (support
N below min_pair_support) at their prior Σ value, and pins the diagonal to 1 —
no PD completion. (The earlier full-covariance-with-max-determinant-completion
M-step was retired from the fit path; pd_complete remains a tested linalg
utility.) The removed inverse-Wishart / diag-shrink knobs stay rejected.

Spec: docs/superpowers/specs/2026-07-01-stm-blockwise-unit-diagonal-correlation-design.md
"""
from __future__ import annotations

import numpy as np
import pytest

from spark_vi.models.topic.stm import OnlineSTM


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


def test_mstep_unit_diagonal_and_lazy_keeps_unsupported_pair():
    # Cross-pair (3,4) is unsupported (N=0) -> lazy-kept at its prior Σ value
    # (0 from the identity init); no completion. Diagonal pinned to 1 (block-wise).
    K = 5
    stm = OnlineSTM(K=K, vocab_size=8, P=2, min_pair_support=2)
    rng = np.random.default_rng(0)
    eta = rng.standard_normal((40, K))
    S = eta.T @ eta
    N = np.full((K, K), 40.0)
    N[3, 4] = N[4, 3] = 0.0                       # thin cross-pair -> unsupported
    gp = stm.initialize_global({"vocab_size": 8})
    prior_34 = gp["Sigma"][3, 4]                  # 0 from the identity init
    out = stm.update_global(gp, _make_target_stats(stm, S, N), learning_rate=1.0)
    Sigma = out["Sigma"]
    np.testing.assert_allclose(np.diag(Sigma), 1.0, atol=1e-12)   # unit diagonal
    assert Sigma[3, 4] == prior_34                # lazy-kept, not completed
    assert abs(Sigma[3, 4]) <= 1.0               # valid correlation entry
