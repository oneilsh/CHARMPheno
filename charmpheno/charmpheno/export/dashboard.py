"""Dashboard bundle export. Writes a four-file JSON bundle consumed by
the static Svelte dashboard. Schema defined in
docs/superpowers/specs/2026-05-13-dashboard-design.md.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


def _none_if_nan(x: float) -> float | None:
    """Convert NaN to None for JSON serialization (json.dumps emits
    'NaN' literally otherwise, which fails JSON.parse)."""
    x = float(x)
    return None if math.isnan(x) else x


def _round_floats(arr: np.ndarray, *, decimals: int = 6) -> list:
    return np.round(arr.astype(np.float64), decimals=decimals).tolist()


def select_top_n_with_min_cell(
    code_marginals: list[float],
    *,
    top_n: int,
) -> list[int]:
    """Pick the top-N codes by marginal token frequency.

    Small-cell suppression is enforced upstream via CountVectorizer.minDF
    at vocab-build time, so the codes reaching this function have already
    cleared the privacy threshold structurally. This function is therefore
    a pure ranking step: sort by marginal descending, take the first top_n.
    """
    if top_n <= 0:
        return []
    marginals = np.asarray(code_marginals, dtype=float)
    order = np.argsort(-marginals)
    return order[:top_n].tolist()


def write_model_and_vocab_bundles(
    *,
    out_dir: Path,
    beta: np.ndarray,           # K × V_full (row-stochastic)
    alpha: np.ndarray,          # length K
    vocab_ids: list[int],       # length V_full; vocab_ids[i] = concept_id at index i
    descriptions: dict[int, str],
    domains: dict[int, str],
    code_marginals: list[float],
    top_n: int,
) -> int:
    """Write model.json and vocab.json. Returns the displayed-vocab width.

    Accepts a row-stochastic β matrix (K × V_full) where each row sums to 1.
    Callers must normalize before passing; this function raises ValueError
    if any row sum deviates from 1.0 by more than 1e-6.

    Trims β columns and vocab metadata to the top-N codes ranked by
    corpus frequency (token marginal). After column trimming (which breaks
    row-stochasticity), β rows are renormalized so each row sums to 1 over
    the surviving trimmed columns.

    Small-cell suppression is enforced upstream via CountVectorizer.minDF
    at vocab-build time; this function applies only top-N ranking.
    """
    beta = np.asarray(beta, dtype=float)
    row_sums = beta.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-6):
        raise ValueError(
            f"write_model_and_vocab_bundles: beta must be row-stochastic; "
            f"got row sums in [{row_sums.min():.6f}, {row_sums.max():.6f}]"
        )

    K, V_full = beta.shape
    keep = select_top_n_with_min_cell(
        code_marginals,
        top_n=top_n,
    )
    V_disp = len(keep)
    beta_trimmed = beta[:, keep]
    # Column trim breaks row-stochasticity; renormalize the surviving columns.
    row_sums = beta_trimmed.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    beta_trimmed = beta_trimmed / row_sums

    model_payload = {
        "K": int(K),
        "V": int(V_disp),
        "alpha": _round_floats(np.asarray(alpha)),
        "beta": _round_floats(beta_trimmed),
    }
    (out_dir / "model.json").write_text(json.dumps(model_payload, allow_nan=False))

    codes = []
    for new_idx, orig_idx in enumerate(keep):
        cid = vocab_ids[orig_idx]
        codes.append({
            "id": new_idx,
            "code": str(cid),
            "description": descriptions.get(cid, ""),
            "domain": domains.get(cid, "unknown"),
            "corpus_freq": float(code_marginals[orig_idx]),
        })
    (out_dir / "vocab.json").write_text(json.dumps({"codes": codes}, allow_nan=False))

    return V_disp


def write_phenotypes_bundle(
    out_path: Path,
    *,
    npmi: list[float],
    pair_coverage: list[float],
    corpus_prevalence: list[float],
    theta_histogram: list[list[float | None]] | None = None,
    theta_percentiles: list[dict[str, float]] | None = None,
    n_bins: int = 50,
    min_count: int = 20,
    topic_indices: list[int] | None = None,
    labels: list[str] | None = None,
) -> None:
    """Write phenotypes.json.

    pair_coverage[k] is the fraction of top-N pairs that contributed to
    the NPMI calculation for topic k (cleared the joint-count threshold
    in the reference corpus). NaN-valued NPMI topics — those with zero
    scored pairs — should be passed in as NaN and pair_coverage as 0.0;
    downstream readers can use the pair_coverage=0 case to distinguish
    "unrated" from "rated and incoherent".

    topic_indices[k] is the original model-side topic id for displayed
    phenotype k. For LDA the adapter passes 0..K-1; for HDP it passes
    the mask-filtered truncation indices so the advanced view can
    surface them.

    Per-phenotype `label`, `description`, and `quality` start empty and
    are populated by the post-fit labeling step
    (scripts/label_phenotypes.py).

    theta_histogram[k] is the empirical θ histogram for topic k: a list
    of length n_bins where each entry is either a float (bin count) or
    None (bin suppressed due to small-cell, i.e., fewer than min_count
    documents). theta_percentiles[k] is a dict with keys p5, p25, p50,
    p75, p95 giving the corresponding θ percentiles for topic k. When
    theta_histogram is provided, the top-level payload includes
    theta_histogram_bin_edges (n_bins+1 evenly-spaced values from 0.0
    to 1.0) and theta_histogram_min_count (the suppression threshold
    used during histogram construction). None entries in histogram rows
    serialize to JSON null and round-trip cleanly.
    """
    K = len(npmi)
    if theta_percentiles is not None and theta_histogram is None:
        raise ValueError(
            "theta_percentiles requires theta_histogram to be provided",
        )
    if len(pair_coverage) != K:
        raise ValueError(
            f"pair_coverage length {len(pair_coverage)} != npmi length {K}",
        )
    if len(corpus_prevalence) != K:
        raise ValueError(
            f"corpus_prevalence length {len(corpus_prevalence)} != npmi length {K}",
        )
    if theta_histogram is not None:
        if len(theta_histogram) != K:
            raise ValueError(
                f"theta_histogram length {len(theta_histogram)} != npmi length {K}",
            )
        for row_idx, row in enumerate(theta_histogram):
            if len(row) != n_bins:
                raise ValueError(
                    f"theta_histogram row {row_idx} length {len(row)} != n_bins {n_bins}",
                )
            for entry in row:
                if entry is not None and not isinstance(entry, float):
                    raise ValueError(
                        f"theta_histogram row {row_idx} contains non-float, non-None entry: {entry!r}",
                    )
    if theta_percentiles is not None:
        if len(theta_percentiles) != K:
            raise ValueError(
                f"theta_percentiles length {len(theta_percentiles)} != npmi length {K}",
            )
    labels = labels or [""] * K
    if topic_indices is None:
        topic_indices = list(range(K))
    phenotypes = []
    for k in range(K):
        entry: dict = {
            "id": k,
            "label": labels[k],
            "description": "",
            "quality": None,
            "npmi": _none_if_nan(npmi[k]),
            "pair_coverage": _none_if_nan(pair_coverage[k]),
            "corpus_prevalence": float(corpus_prevalence[k]),
            "original_topic_id": int(topic_indices[k]),
        }
        if theta_histogram is not None:
            entry["theta_histogram"] = theta_histogram[k]
        if theta_percentiles is not None:
            entry["theta_percentiles"] = theta_percentiles[k]
        phenotypes.append(entry)
    payload: dict = {"phenotypes": phenotypes}
    if theta_histogram is not None:
        payload["theta_histogram_bin_edges"] = np.linspace(0, 1, n_bins + 1).tolist()
        payload["theta_histogram_min_count"] = int(min_count)
    out_path.write_text(json.dumps(payload, allow_nan=False))
