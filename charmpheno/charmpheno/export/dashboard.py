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


def select_top_n_indices(code_marginals: list[float], top_n: int) -> list[int]:
    """Return original-vocab indices for the top-N codes by marginal, sorted descending."""
    marg = np.asarray(code_marginals, dtype=np.float64)
    if top_n >= len(marg):
        return list(np.argsort(-marg))
    idx = np.argpartition(-marg, kth=top_n - 1)[:top_n]
    return list(idx[np.argsort(-marg[idx])])


def select_top_n_with_min_cell(
    code_marginals: list[float],
    *,
    top_n: int,
    corpus_size_docs: int,
    min_doc_count: int,
) -> list[int]:
    """Top-N selection with a group-size guard.

    Drops codes whose empirical doc count (marginal * corpus_size_docs) is
    below ``min_doc_count`` BEFORE the top-N cut. Codes with doc count
    exactly zero are not displayed by construction (their marginal is 0
    and they would never rank). Codes with 1..min_doc_count-1 docs are
    suppressed to prevent small-cell disclosure of which OMOP concepts
    appear in a tiny number of patients/documents.

    Returned in descending marginal order, length up to ``top_n``.
    """
    marginals = np.asarray(code_marginals, dtype=np.float64)
    eligible = marginals * corpus_size_docs >= min_doc_count
    eligible_idx = np.where(eligible)[0]
    if len(eligible_idx) == 0:
        raise ValueError(
            f"no codes have >= {min_doc_count} documents "
            f"(corpus_size_docs={corpus_size_docs}); "
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
    corpus_size_docs: int,
    top_n: int,
    min_doc_count: int = 20,
) -> int:
    """Write model.json and vocab.json. Returns the displayed-vocab width.

    Trims β columns and vocab metadata to the top-N codes by corpus
    frequency, with a small-cell guard: codes whose empirical doc count
    (corpus_freq * corpus_size_docs) is below ``min_doc_count`` are
    dropped before the top-N cut. Default 20 matches AoU-style group-size
    suppression. Zero-count codes are implicitly excluded; the guard
    targets the 1..min_doc_count-1 range. β rows are renormalized so each
    row sums to 1 over the trimmed columns.
    """
    K, V_full = lambda_.shape
    n_before = int((np.asarray(code_marginals) > 0).sum())
    keep = select_top_n_with_min_cell(
        code_marginals,
        top_n=top_n,
        corpus_size_docs=corpus_size_docs,
        min_doc_count=min_doc_count,
    )
    V_disp = len(keep)
    n_eligible = int(
        (np.asarray(code_marginals) * corpus_size_docs >= min_doc_count).sum(),
    )
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
    """
    K = len(npmi)
    if len(pair_coverage) != K:
        raise ValueError(
            f"pair_coverage length {len(pair_coverage)} != npmi length {K}",
        )
    labels = labels or [""] * K
    if topic_indices is None:
        topic_indices = list(range(K))
    phenotypes = [
        {
            "id": k,
            "label": labels[k],
            "description": "",
            "quality": None,
            "npmi": float(npmi[k]),
            "pair_coverage": float(pair_coverage[k]),
            "corpus_prevalence": float(corpus_prevalence[k]),
            "original_topic_id": int(topic_indices[k]),
        }
        for k in range(K)
    ]
    out_path.write_text(json.dumps({"phenotypes": phenotypes}))
