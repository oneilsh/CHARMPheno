"""Shared eval-driver utilities for both local and cloud NPMI coherence drivers.

Houses ``print_ranked_report``, the per-topic NPMI + descriptive-stats
printout shared by ``analysis/local/eval_coherence.py`` and
``analysis/cloud/eval_coherence_cloud.py``. Lives at this seam so the
local and cloud drivers stay symmetric without one importing private
state from the other.
"""
from __future__ import annotations

import sys

import numpy as np


def stm_sigma_diagnostic(Sigma, labels=None, top_k: int = 8):
    """Per-topic eta-variance ranking + spectrum summary for an STM covariance.

    Surfaces the largest-variance ("runaway") topics at eval time, so a
    saved model can be inspected via ``make eval-exp ID=N`` without a refit.

    Parameters
    ----------
    Sigma : array-like
        The (K, K) topic covariance over global topic ids
        (``global_params["Sigma"]``). Returns ``None`` for anything that is not
        a square 2-D matrix (e.g. a legacy diagonal-Σ K-vector, or a non-STM
        model with no Σ) so callers can guard on the return value.
    labels : list | dict | None
        Optional topic-id -> block-name map (``None`` block -> "background"),
        e.g. ``foreground_reference_groups(topic_block_spec)``.
    top_k : int
        How many largest-variance topics to list.

    Returns
    -------
    str | None
        A multi-line report naming the largest-variance ("runaway") topics and
        the eigenvalue range (min/max eigenvalue of Σ, a reporting statistic
        only -- the fit only ever uses within-allowed-set marginal sub-blocks
        of Σ, not the full assembled matrix), or ``None`` if Sigma is not a
        square matrix.
    """
    if Sigma is None:
        return None
    Sigma = np.asarray(Sigma, dtype=float)
    if Sigma.ndim != 2 or Sigma.shape[0] != Sigma.shape[1]:
        return None
    K = Sigma.shape[0]
    diag = np.diag(Sigma)
    order = np.argsort(diag)[::-1]

    def _blk(k: int) -> str:
        if labels is None:
            return "?"
        lab = labels.get(k) if isinstance(labels, dict) else labels[k]
        return "background" if lab is None else str(lab)

    # Symmetrized eigenvalue range (Σ is symmetric up to fp; guard anyway).
    # Reporting statistic only: the full-matrix eigenvalue range spans
    # cross-block entries that never enter the fit.
    w = np.linalg.eigvalsh(0.5 * (Sigma + Sigma.T))
    wmin, wmax = float(w[0]), float(w[-1])

    n = min(top_k, K)
    lines = [
        f"Sigma spectrum (reporting statistic): eig[min={wmin:.3g} max={wmax:.3g}]",
        f"top-{n} topics by eta-variance Sigma_ii:",
    ]
    for k in order[:n]:
        lines.append(f"  topic {int(k):3d} [{_blk(int(k))}]  Sigma_ii={diag[k]:.3e}")
    amax = int(order[0])
    lines.append(
        f"runaway = topic {amax} [{_blk(amax)}]  Sigma_ii={diag[amax]:.3e}")
    return "\n".join(lines)


def _resolve_use_color(mode: str) -> bool:
    """Map a CLI --color {auto, always, never} value to a bool."""
    if mode == "always":
        return True
    if mode == "never":
        return False
    return sys.stdout.isatty()


def print_ranked_report(
    report,
    name_by_idx: dict[int, str],
    lambda_: np.ndarray,
    *,
    alpha: np.ndarray | None = None,
    color: str = "auto",
) -> None:
    """Print per-topic NPMI ranked by usage (Σλ desc) with descriptive stats.

    For each scored topic prints NPMI, coverage, E[β]=Σλ_k/Σλ_total,
    Σλ_k, peak topic-word mass, and (for LDA only) α_k. Topics whose
    NPMI is NaN (zero scored pairs after the min_pair_count threshold)
    are dimmed via ANSI escape codes so the "gracefully unused" tail
    visually recedes — see [docs/insights/0019](docs/insights/0019-lda-large-k-with-full-convergence-gracefully-unused-slots.md)
    for what that pattern looks like and why distinguishing it matters.

    E[β] is normalized by the FULL model's λ sum, so values sum to 1
    across ALL topics (not just the scored subset). For HDP this means
    the E[β]s of the scored top-K subset will not sum to 1 — they sum
    to the corpus-mass fraction the top-K covers. That's a feature:
    callers can read the residual `1 - Σ E[β]_scored` as how much
    corpus mass landed outside the scored window.

    Args:
        report: CoherenceReport from `compute_npmi_coherence`.
        name_by_idx: mapping vocab-index -> display label (e.g.
            "320128 (Essential hypertension)"). Passed by the caller
            since the two drivers reach concept names differently
            (cloud: BQ; local: checkpoint metadata).
        lambda_: full (K, V) variational Dirichlet topic-word matrix
            from `result.global_params['lambda']`. Used to derive the
            per-topic stats — we don't take pre-computed arrays
            because callers need lambda_ regardless (to build the
            scored topic_term matrix), and re-deriving here keeps
            the stats and the report aligned.
        alpha: LDA per-topic asymmetric α array (shape (K,)). Pass
            None for HDP (α is a scalar there and not per-topic
            meaningful for this view). When provided, an `α=...`
            column is appended to each row.
        color: 'auto' (TTY -> on, otherwise off), 'always', or 'never'.
    """
    use_color = _resolve_use_color(color)
    DIM = "\033[2m" if use_color else ""
    RESET = "\033[0m" if use_color else ""

    topic_indices = report.topic_indices
    npmi_arr = report.per_topic_npmi
    scored = report.per_topic_scored_pairs
    top_terms = report.top_term_indices

    lam_rows = lambda_[topic_indices]
    lam_row_sums = lam_rows.sum(axis=1)
    total_lam = float(lambda_.sum())
    E_beta = lam_row_sums / max(total_lam, 1e-12)
    peak = (lam_rows / np.clip(lam_row_sums, 1e-12, None)[:, None]).max(axis=1)

    has_alpha = (
        alpha is not None
        and hasattr(alpha, "__len__")
        and len(alpha) == lambda_.shape[0]
    )
    alpha_per_topic = alpha[topic_indices] if has_alpha else None

    order = sorted(range(len(topic_indices)), key=lambda i: -float(lam_row_sums[i]))

    total_pairs = report.per_topic_total_pairs
    print(
        f"\n  per-topic stats (reference=fit corpus, "
        f"reference_size={report.reference_size}, top_n={report.top_n}, "
        f"min_pair_count={report.min_pair_count}, "
        f"unrated={report.n_topics_unrated}/{len(npmi_arr)}, "
        f"sorted by Σλ desc):"
    )
    print(
        f"  mean={report.mean:+.4f}  median={report.median:+.4f}  "
        f"stdev={report.stdev:.4f}  min={report.min:+.4f}  "
        f"max={report.max:+.4f}\n"
    )

    # Two-char leading marker so the unused tail is visible even without
    # color support (some terminals strip or invert dim). '·' = unrated
    # (cov=0%, NPMI NaN); ' ' = scored.
    for i in order:
        topic_idx = int(topic_indices[i])
        npmi_v = float(npmi_arr[i])
        scored_pairs = int(scored[i])
        labels = [name_by_idx.get(int(t), f"#{int(t)}") for t in top_terms[i]]
        cov_pct = int(round(100 * scored_pairs / total_pairs)) if total_pairs else 0
        is_unused = np.isnan(npmi_v)
        marker = "·" if is_unused else " "
        npmi_str = "  NaN  " if is_unused else f"{npmi_v:+.4f}"
        alpha_str = (
            f"  α={float(alpha_per_topic[i]):.3g}"
            if alpha_per_topic is not None else ""
        )
        line = (
            f" {marker} topic {topic_idx:3d}  "
            f"NPMI={npmi_str}  cov={cov_pct:>3d}%  "
            f"E[β]={float(E_beta[i]):.4f}  Σλ={float(lam_row_sums[i]):.3g}  "
            f"peak={float(peak[i]):.3f}{alpha_str}  "
            f"top: {', '.join(labels[:8])}"
        )
        if is_unused:
            line = f"{DIM}{line}{RESET}"
        print(line)
