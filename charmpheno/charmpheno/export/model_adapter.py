"""Model-class adapters for the dashboard export.

Each supported VIResult normalizes to a DashboardExport, so the export
builder and the dashboard contract are model-class-agnostic.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DashboardExport:
    """Uniform shape consumed by the dashboard bundle builder."""
    beta: np.ndarray              # K_display × V (row-stochastic)
    alpha: np.ndarray             # K_display
    corpus_prevalence: np.ndarray # K_display
    topic_indices: np.ndarray     # K_display (original model-side topic ids)


def _global_params(result) -> dict[str, np.ndarray]:
    return result.global_params


def _model_class(result) -> str:
    return str(result.metadata.get("model_class", "lda")).lower()


def adapt_lda(result) -> DashboardExport:
    """LDA → DashboardExport. Identity on β; corpus_prevalence from gamma."""
    gp = _global_params(result)
    lambda_ = np.asarray(gp["lambda"], dtype=np.float64)
    K = lambda_.shape[0]
    beta = lambda_ / lambda_.sum(axis=1, keepdims=True)
    alpha = np.asarray(gp.get("alpha", np.full(K, 1.0 / K)), dtype=np.float64)
    gamma = gp.get("gamma")
    if gamma is not None:
        gamma = np.asarray(gamma, dtype=np.float64)
        if gamma.ndim == 2 and gamma.shape[1] == K:
            theta = gamma / gamma.sum(axis=1, keepdims=True)
            corpus_prev = theta.mean(axis=0)
        else:
            corpus_prev = alpha / alpha.sum()
    else:
        corpus_prev = alpha / alpha.sum()
    return DashboardExport(
        beta=beta,
        alpha=alpha,
        corpus_prevalence=corpus_prev,
        topic_indices=np.arange(K, dtype=np.int64),
    )


def adapt_hdp(result, *, top_k: int = 50) -> DashboardExport:
    """HDP → DashboardExport. Filters to top-K used topics; computes
    effective Dirichlet α from the corpus-level GEM sticks."""
    gp = _global_params(result)
    lambda_ = np.asarray(gp["lambda"], dtype=np.float64)
    u = np.asarray(gp["u"], dtype=np.float64)
    v = np.asarray(gp["v"], dtype=np.float64)
    # E[stick weights] from Beta(u, v) and stick-breaking remainder
    stick_means = u / (u + v)
    remainder = np.cumprod(np.concatenate([[1.0], 1 - stick_means[:-1]]))
    e_beta = stick_means * remainder  # corpus-level mass per truncation index
    # top_k by E[beta]
    K_use = min(top_k, len(e_beta))
    order = np.argsort(-e_beta)[:K_use]
    order = np.sort(order)  # stable order by original index
    beta_filt = lambda_[order] / lambda_[order].sum(axis=1, keepdims=True)
    # Effective Dirichlet alpha: renormalize the selected sticks to sum to 1
    sel = e_beta[order]
    alpha_eff = sel / sel.sum() if sel.sum() > 0 else np.full(K_use, 1.0 / K_use)
    return DashboardExport(
        beta=beta_filt,
        alpha=alpha_eff,
        corpus_prevalence=alpha_eff.copy(),
        topic_indices=order.astype(np.int64),
    )


_LDA_ALIASES = {"lda", "onlinelda", "onlineldamodel", "onlineldaestimator"}
_HDP_ALIASES = {"hdp", "onlinehdp", "onlinehdpmodel", "onlinehdpestimator"}


def adapt(result, *, hdp_top_k: int = 50) -> DashboardExport:
    mc = _model_class(result)
    if mc in _LDA_ALIASES:
        return adapt_lda(result)
    if mc in _HDP_ALIASES:
        return adapt_hdp(result, top_k=hdp_top_k)
    raise ValueError(f"unsupported model class: {mc}")
