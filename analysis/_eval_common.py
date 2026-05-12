"""Shared eval-driver utilities for both local and cloud NPMI coherence drivers.

Houses (1) `verify_split_contract`, factored out of
`analysis/local/eval_coherence.py` so the cloud eval driver
(`analysis/cloud/eval_coherence_cloud.py`) can reuse the exact same
fit/eval split-provenance check without duplicating logic, and (2)
`print_ranked_report`, the per-topic NPMI+stats printout shared by
both drivers.
"""
from __future__ import annotations

import logging
import sys

import numpy as np

log = logging.getLogger(__name__)


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
    npmi_reference: str = "full",
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
        npmi_reference: 'full' or 'holdout', echoed in the banner so
            the reader knows what reference corpus produced these
            scores.
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
        f"\n  per-topic stats (reference={npmi_reference}, "
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


def verify_split_contract(result, *, holdout_fraction: float, seed: int) -> None:
    """Verify the eval CLI args match the split provenance stamped at fit time.

    The fit driver (fit_lda_local.py / fit_hdp_local.py / lda_bigquery_cloud.py)
    stamps split parameters under VIResult.metadata['split']. If absent, the
    model was fit on the full corpus and the eval is optimistically biased
    (the held-out patients were seen during training); we warn loudly. If
    present but mismatched, we abort.
    """
    split_meta = result.metadata.get("split")
    if split_meta is None or not split_meta.get("applied", False):
        log.warning(
            "checkpoint has no split provenance; the fit driver was likely run "
            "without --holdout-fraction. NPMI on the hashed holdout will be "
            "OPTIMISTICALLY BIASED because the model saw those patients during "
            "fitting. Re-fit with matching --holdout-fraction and --holdout-seed "
            "for an honest benchmark."
        )
        return
    fit_frac = float(split_meta.get("holdout_fraction", -1.0))
    fit_seed = int(split_meta.get("holdout_seed", -1))
    if (abs(fit_frac - holdout_fraction) > 1e-9) or (fit_seed != seed):
        raise SystemExit(
            "split mismatch: checkpoint was fit with "
            f"holdout_fraction={fit_frac}, seed={fit_seed} but eval was invoked "
            f"with holdout_fraction={holdout_fraction}, seed={seed}. "
            "Re-run with matching values (the eval holdout must be the held-out "
            "portion the model did NOT see)."
        )
