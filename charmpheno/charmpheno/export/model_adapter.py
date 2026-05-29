"""Model-class adapters for the dashboard export.

Each supported VIResult normalizes to a DashboardExport, so the export
builder and the dashboard contract are model-class-agnostic.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class DashboardExport:
    """Uniform shape consumed by the dashboard bundle builder."""
    beta: np.ndarray              # K_display × V (row-stochastic)
    alpha: np.ndarray             # K_display
    corpus_prevalence: np.ndarray # K_display
    topic_indices: np.ndarray     # K_display (original model-side topic ids)
    theta_histogram: np.ndarray | None = field(default=None)   # K_display × n_bins; np.nan = suppressed (use np.nansum to aggregate); None for HDP/legacy
    theta_percentiles: np.ndarray | None = field(default=None)  # K_display × 5 in [p5, p25, p50, p75, p95] column order; None for HDP/legacy


def _global_params(result) -> dict[str, np.ndarray]:
    return result.global_params


def _model_class(result) -> str:
    return str(result.metadata.get("model_class", "lda")).lower()


def _parse_theta_histogram(raw: list[list[float | None]]) -> np.ndarray:
    """Convert list-of-lists (with None for suppressed) to (K, n_bins) ndarray.

    None entries are mapped to np.nan explicitly before constructing the array.
    """
    rows = []
    for row in raw:
        rows.append([np.nan if v is None else float(v) for v in row])
    return np.asarray(rows, dtype=np.float64)


def _parse_theta_percentiles(raw: list[dict[str, float]]) -> np.ndarray:
    """Convert list of dicts to (K, 5) ndarray with column order [p5, p25, p50, p75, p95]."""
    cols = ["p5", "p25", "p50", "p75", "p95"]
    rows = [[d[c] for c in cols] for d in raw]
    return np.asarray(rows, dtype=np.float64)


def adapt_lda(result) -> DashboardExport:
    """LDA → DashboardExport. Reads corpus_prevalence, theta_histogram, and
    theta_percentiles from checkpoint metadata. Raises ValueError if
    corpus_prevalence is absent."""
    gp = _global_params(result)
    meta = result.metadata

    # --- beta from lambda (row-stochastic) ---
    lambda_ = np.asarray(gp["lambda"], dtype=np.float64)
    K = lambda_.shape[0]
    beta = lambda_ / lambda_.sum(axis=1, keepdims=True)

    # --- alpha from global_params ---
    alpha = np.asarray(gp.get("alpha", np.full(K, 1.0 / K)), dtype=np.float64)

    # --- corpus_prevalence from metadata (required) ---
    if "corpus_prevalence" not in meta:
        raise ValueError(
            "checkpoint metadata missing 'corpus_prevalence' (LDA aggregates "
            "were not computed at fit time). This checkpoint predates the "
            "in-driver aggregates change; re-fit with the current LDA driver."
        )
    corpus_prev = np.asarray(meta["corpus_prevalence"], dtype=np.float64)

    # --- theta_histogram from metadata (optional) ---
    raw_hist = meta.get("theta_histogram")
    theta_histogram: np.ndarray | None
    if raw_hist is not None:
        theta_histogram = _parse_theta_histogram(raw_hist)
    else:
        theta_histogram = None

    # --- theta_percentiles from metadata (optional) ---
    raw_pct = meta.get("theta_percentiles")
    theta_percentiles: np.ndarray | None
    if raw_pct is not None:
        theta_percentiles = _parse_theta_percentiles(raw_pct)
    else:
        theta_percentiles = None

    return DashboardExport(
        beta=beta,
        alpha=alpha,
        corpus_prevalence=corpus_prev,
        topic_indices=np.arange(K, dtype=np.int64),
        theta_histogram=theta_histogram,
        theta_percentiles=theta_percentiles,
    )


def adapt_hdp(result, *, top_k: int = 50) -> DashboardExport:
    """HDP → DashboardExport. Filters to top-K used topics; computes
    effective Dirichlet α from the corpus-level GEM sticks.
    theta_histogram and theta_percentiles are always None (no per-doc θ in HDP)."""
    gp = _global_params(result)
    meta = result.metadata
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

    # corpus_prevalence: prefer metadata slice (after top-K selection) if present;
    # fall back to stick-derived alpha_eff for checkpoints without metadata aggregates.
    raw_cp = meta.get("corpus_prevalence")
    if raw_cp is not None:
        full_cp = np.asarray(raw_cp, dtype=np.float64)
        corpus_prev = full_cp[order]
    else:
        # Fallback: stick-derived alpha_eff (legacy checkpoints without metadata)
        corpus_prev = alpha_eff.copy()

    return DashboardExport(
        beta=beta_filt,
        alpha=alpha_eff,
        corpus_prevalence=corpus_prev,
        topic_indices=order.astype(np.int64),
        theta_histogram=None,
        theta_percentiles=None,
    )


def adapt_stm(result) -> DashboardExport:
    """STM → DashboardExport. α-equivalent derived from softmax(Γ[intercept])."""
    gp = _global_params(result)
    meta = result.metadata
    lambda_ = np.asarray(gp["lambda"], dtype=np.float64)
    K = lambda_.shape[0]
    beta = lambda_ / lambda_.sum(axis=1, keepdims=True)
    Gamma = np.asarray(gp["Gamma"], dtype=np.float64)  # (P, K)
    covariate_names = meta["covariate_manifest"]["covariate_names"]
    intercept_idx = next(
        (i for i, n in enumerate(covariate_names) if "intercept" in str(n).lower()),
        None,
    )
    if intercept_idx is not None:
        eta_bar = Gamma[intercept_idx]
        alpha_eq = np.exp(eta_bar - eta_bar.max())
        alpha_eq /= alpha_eq.sum()
    else:
        alpha_eq = np.full(K, 1.0 / K)
    # v1: corpus_prevalence stand-in equals α_eq. v1.x would compute
    # (1/D) Σ_d softmax(Γ x_d) over the actual corpus covariate distribution.
    # Follow-up: track in docs/insights/ when the empirical version ships;
    # affects only the dashboard "default topic proportion" display.
    corpus_prev = alpha_eq.copy()
    # theta_histogram / theta_percentiles: optional pass-through if metadata has them.
    raw_hist = meta.get("theta_histogram")
    theta_histogram = _parse_theta_histogram(raw_hist) if raw_hist is not None else None
    raw_pct = meta.get("theta_percentiles")
    theta_percentiles = _parse_theta_percentiles(raw_pct) if raw_pct is not None else None
    return DashboardExport(
        beta=beta, alpha=alpha_eq, corpus_prevalence=corpus_prev,
        topic_indices=np.arange(K, dtype=np.int64),
        theta_histogram=theta_histogram, theta_percentiles=theta_percentiles,
    )


_LDA_ALIASES = {"lda", "onlinelda", "onlineldamodel", "onlineldaestimator"}
_HDP_ALIASES = {"hdp", "onlinehdp", "onlinehdpmodel", "onlinehdpestimator"}
_STM_ALIASES = {"stm", "onlinestm"}


def adapt(result, *, hdp_top_k: int = 50) -> DashboardExport:
    mc = _model_class(result)
    if mc in _LDA_ALIASES:
        return adapt_lda(result)
    if mc in _HDP_ALIASES:
        return adapt_hdp(result, top_k=hdp_top_k)
    if mc in _STM_ALIASES:
        return adapt_stm(result)
    raise ValueError(f"unsupported model class: {mc}")
