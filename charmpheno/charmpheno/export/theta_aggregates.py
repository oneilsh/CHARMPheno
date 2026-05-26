"""Per-topic empirical θ aggregates for the dashboard export.

Given LDA's γ matrix (N_docs × K), computes a histogram of θ_k across
patients per topic, plus a fixed set of percentiles. Used by the fit
driver's training-termination block (to write aggregates into the
checkpoint before γ is dropped) and by the checkpoint migration script.
"""
from __future__ import annotations

import numpy as np


def compute_theta_aggregates(
    gamma: np.ndarray,
    *,
    n_bins: int = 50,
    min_count: int = 20,
) -> dict:
    """Compute per-topic θ histogram + percentiles + mean from γ.

    Parameters
    ----------
    gamma : np.ndarray, shape (N, K)
        LDA's per-document Dirichlet parameters. Rows are normalized
        internally to obtain θ = γ / γ.sum(axis=1, keepdims=True).
        All rows must have positive sum; rows with ``gamma[i].sum() == 0``
        produce NaN that this function does not guard against.
    n_bins : int, default 50
        Number of equal-width histogram bins on [0, 1].
    min_count : int, default 20
        Small-cell suppression threshold. Bins with patient counts in
        [1, min_count) are set to None in the output. Bins with count
        0 are preserved as 0.0 (no patients = no privacy concern).

    Returns
    -------
    dict with keys:
        'theta_histogram': list[list[float | None]]
            Shape (K, n_bins). Each entry is the fraction of patients
            in that bin (bin_count / N), or None if suppressed.
        'theta_percentiles': list[dict[str, float]]
            Length K. Each is {'p5': ..., 'p25': ..., 'p50': ..., 'p75': ..., 'p95': ...}.
        'corpus_prevalence': list[float]
            Length K. Exact θ.mean(axis=0).
        'n_patients': int
            N (number of documents/patients).
    """
    gamma = np.asarray(gamma, dtype=np.float64)
    n, k = gamma.shape

    # Normalize rows to obtain θ
    theta = gamma / gamma.sum(axis=1, keepdims=True)

    # corpus_prevalence is computed directly from theta (not from binned data)
    # to preserve float64 precision — binning would introduce discretization error.
    corpus_prevalence = theta.mean(axis=0).tolist()

    bin_edges = np.linspace(0, 1, n_bins + 1)

    theta_histogram: list[list[float | None]] = []
    theta_percentiles: list[dict[str, float]] = []

    for topic_idx in range(k):
        col = theta[:, topic_idx]

        # Clip values to [0, 1-ε). numpy's last bin is right-inclusive, so exact
        # 1.0 is fine; this defensively guards against floating-point values
        # marginally above 1.0 due to normalization noise.
        col_clipped = np.clip(col, 0.0, 1.0 - 1e-12)
        counts, _ = np.histogram(col_clipped, bins=bin_edges)

        row: list[float | None] = []
        for c in counts:
            if c == 0:
                row.append(0.0)
            elif 1 <= c < min_count:
                row.append(None)
            else:
                row.append(float(c) / n)
        theta_histogram.append(row)

        pcts = np.percentile(col, [5, 25, 50, 75, 95])
        theta_percentiles.append({
            "p5": float(pcts[0]),
            "p25": float(pcts[1]),
            "p50": float(pcts[2]),
            "p75": float(pcts[3]),
            "p95": float(pcts[4]),
        })

    return {
        "theta_histogram": theta_histogram,
        "theta_percentiles": theta_percentiles,
        "corpus_prevalence": corpus_prevalence,
        "n_patients": n,
    }
