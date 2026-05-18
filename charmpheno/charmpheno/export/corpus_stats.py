"""Aggregate corpus statistics for the dashboard bundle.

The driver computes these once from a held-out (or full) BOW and uses them
in three ways: (1) the small scalars get written to corpus_stats.json;
(2) code_marginals drive the top-N vocab ranking in the vocab.json writer;
(3) code_doc_counts drive the small-cell suppression guard before that
ranking. Marginals are NOT exported on their own — vocab.json carries the
surviving codes' corpus_freq (token-frequency) per row.

Two distinct per-code measurements are tracked because they answer
different questions:

- ``code_marginals[i]`` = P(any token in corpus is code i) = TOKEN
  frequency. Used for ranking codes by display importance, since the
  dashboard cares about how heavily codes appear in topic-word
  distributions.
- ``code_doc_counts[i]`` = number of distinct documents containing code
  i (≥1 occurrence). Used for the AoU-style small-cell guard which is a
  patient-count privacy threshold (suppress codes appearing in fewer
  than N distinct patients/documents).

Mixing the two — e.g. using token frequency to back-compute a doc count —
is unsafe because it scales with ``mean_codes_per_doc`` rather than with
patient count. The bug that motivated splitting the two: a small cohort
with mean_codes_per_doc=130 made the implicit per-token-rate-based
threshold ~6× harsher than intended, suppressing nearly all phenotype-
specific codes from the displayed vocab.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from pyspark.sql import DataFrame


@dataclass(frozen=True)
class CorpusStats:
    corpus_size_docs: int
    mean_codes_per_doc: float
    k: int
    v_full: int
    code_marginals: list[float]   # length V_full — token frequency
    code_doc_counts: list[int]    # length V_full — distinct-doc count


def compute_corpus_stats(*, docs: Iterator[dict], vocab_size: int, k: int) -> CorpusStats:
    """Compute CorpusStats from an iterator of BOW dict rows.

    Each row must have keys 'indices' (list[int]) and 'counts' (list[int]).
    The BOW convention is that each index in ``indices`` is a distinct
    code present in the document (CountVectorizer output dedups by
    construction), so per-code doc count is incremented once per row per
    distinct index.
    """
    n_docs = 0
    n_codes_sum = 0
    code_total = [0] * vocab_size
    code_doc_count = [0] * vocab_size
    total_tokens = 0
    for row in docs:
        n_docs += 1
        n_codes_sum += sum(row["counts"])
        for idx, cnt in zip(row["indices"], row["counts"]):
            code_total[idx] += cnt
            total_tokens += cnt
            code_doc_count[idx] += 1
    mean_codes = n_codes_sum / max(n_docs, 1)
    marginals = [c / max(total_tokens, 1) for c in code_total]
    return CorpusStats(
        corpus_size_docs=n_docs,
        mean_codes_per_doc=mean_codes,
        k=k,
        v_full=vocab_size,
        code_marginals=marginals,
        code_doc_counts=code_doc_count,
    )


def write_corpus_stats_sidecar(
    stats: CorpusStats,
    out_path: Path,
    *,
    v_displayed: int,
    cohort: dict[str, str] | None = None,
) -> None:
    """Write the small-scalars sidecar. v_displayed is the trimmed-vocab width.

    ``cohort`` is an optional ``{id, label, description}`` dict (from
    ``charmpheno.omop.cohorts.cohort_metadata``) describing which cohort
    filter the corpus was fit on. Embedded in the sidecar so the dashboard
    bundle is self-describing — the UI's cohort selector can use these
    inline values without re-fetching every bundle's metadata.
    """
    payload: dict[str, object] = {
        "corpus_size_docs": stats.corpus_size_docs,
        "mean_codes_per_doc": stats.mean_codes_per_doc,
        "k": stats.k,
        "v": int(v_displayed),
        "v_full": stats.v_full,
    }
    if cohort is not None:
        payload["cohort"] = cohort
    out_path.write_text(json.dumps(payload))


def compute_corpus_stats_from_bow_df(bow_df: DataFrame, *, vocab_size: int, k: int) -> CorpusStats:
    """PySpark wrapper. Input bow_df must have 'indices' and 'counts' columns (array<int>).
    Streams rows to the driver via toLocalIterator."""
    rows = bow_df.select("indices", "counts").toLocalIterator()
    return compute_corpus_stats(
        docs=({"indices": list(r.indices), "counts": list(r.counts)} for r in rows),
        vocab_size=vocab_size,
        k=k,
    )
