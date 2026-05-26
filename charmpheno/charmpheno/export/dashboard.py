"""Dashboard bundle export. Writes a four-file JSON bundle consumed by
the static Svelte dashboard. Schema defined in
docs/superpowers/specs/2026-05-13-dashboard-design.md.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _round_floats(arr: np.ndarray, *, decimals: int = 6) -> list:
    return np.round(arr.astype(np.float64), decimals=decimals).tolist()


def select_top_n_with_min_cell(
    code_marginals: list[float],
    *,
    code_doc_counts: list[int],
    top_n: int,
    min_doc_count: int,
) -> list[int]:
    """Top-N selection with a group-size guard.

    Drops codes whose distinct-document count (``code_doc_counts[i]``) is
    below ``min_doc_count`` BEFORE the top-N cut. Codes appearing in
    1..min_doc_count-1 docs are suppressed to prevent small-cell
    disclosure of which OMOP concepts appear in a tiny number of
    patients/documents. Codes with zero doc count are implicitly excluded
    (no document contains them).

    Filtering on real doc counts is independent of cohort size and
    mean_codes_per_doc; ranking by ``code_marginals`` (token frequency)
    keeps the displayed vocab focused on codes with heavy weight in
    topic-word distributions.

    Returned in descending marginal order, length up to ``top_n``.
    """
    marginals = np.asarray(code_marginals, dtype=np.float64)
    doc_counts = np.asarray(code_doc_counts, dtype=np.int64)
    if doc_counts.shape != marginals.shape:
        raise ValueError(
            f"code_doc_counts length {doc_counts.shape[0]} != "
            f"code_marginals length {marginals.shape[0]}",
        )
    eligible = doc_counts >= min_doc_count
    eligible_idx = np.where(eligible)[0]
    if len(eligible_idx) == 0:
        raise ValueError(
            f"no codes have >= {min_doc_count} documents; "
            f"cannot satisfy minimum cell-size guard",
        )
    order = np.argsort(-marginals[eligible_idx])
    return eligible_idx[order][:top_n].tolist()


def write_model_and_vocab_bundles(
    *,
    out_dir: Path,
    lambda_: np.ndarray,        # K × V_full
    alpha: np.ndarray,          # length K
    vocab_ids: list[int],       # length V_full; vocab_ids[i] = concept_id at index i
    descriptions: dict[int, str],
    domains: dict[int, str],
    code_marginals: list[float],
    code_doc_counts: list[int],
    top_n: int,
    min_doc_count: int = 20,
) -> int:
    """Write model.json and vocab.json. Returns the displayed-vocab width.

    Trims β columns and vocab metadata to the top-N codes ranked by
    corpus frequency (token), with a small-cell guard on distinct doc
    count: codes appearing in fewer than ``min_doc_count`` documents are
    dropped before the top-N cut. Default 20 matches AoU-style group-size
    suppression. β rows are renormalized so each row sums to 1 over the
    trimmed columns.

    Ranking by ``code_marginals`` (token frequency) prioritizes codes
    with heavy weight in topic-word distributions; filtering by
    ``code_doc_counts`` (distinct doc count) enforces the patient-count
    privacy threshold. These are two different per-code measurements and
    must not be substituted for one another: a code can have low
    token-frequency but high doc count (appears once in many docs), or
    high token-frequency but low doc count (appears many times in few
    docs).
    """
    K, V_full = lambda_.shape
    n_before = int((np.asarray(code_doc_counts) > 0).sum())
    keep = select_top_n_with_min_cell(
        code_marginals,
        code_doc_counts=code_doc_counts,
        top_n=top_n,
        min_doc_count=min_doc_count,
    )
    V_disp = len(keep)
    n_eligible = int((np.asarray(code_doc_counts) >= min_doc_count).sum())
    n_suppressed = n_before - n_eligible
    if n_suppressed > 0:
        print(
            f"[export] suppressed {n_suppressed} codes with < {min_doc_count} "
            f"documents (small-cell guard); displaying top {V_disp} of "
            f"{n_eligible} eligible codes.",
            flush=True,
        )
    beta_full = lambda_ / lambda_.sum(axis=1, keepdims=True)
    beta = beta_full[:, keep]
    # renormalize
    row_sums = beta.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    beta = beta / row_sums

    model_payload = {
        "K": int(K),
        "V": int(V_disp),
        "alpha": _round_floats(np.asarray(alpha)),
        "beta": _round_floats(beta),
    }
    (out_dir / "model.json").write_text(json.dumps(model_payload))

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
    (out_dir / "vocab.json").write_text(json.dumps({"codes": codes}))

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
    if len(pair_coverage) != K:
        raise ValueError(
            f"pair_coverage length {len(pair_coverage)} != npmi length {K}",
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
            "npmi": float(npmi[k]),
            "pair_coverage": float(pair_coverage[k]),
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
    out_path.write_text(json.dumps(payload))
