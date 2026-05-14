"""Aggregate corpus statistics for the dashboard bundle.

The driver computes these once from a held-out (or full) BOW and uses them
in two ways: (1) the small scalars get written to corpus_stats.json;
(2) code_marginals drive the top-N vocab trim in the vocab.json writer.
Marginals are NOT exported to the bundle on their own — vocab.json carries
the surviving codes' corpus_freq per row.
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
    code_marginals: list[float]  # length V_full


def compute_corpus_stats(*, docs: Iterator[dict], vocab_size: int, k: int) -> CorpusStats:
    """Compute CorpusStats from an iterator of BOW dict rows.

    Each row must have keys 'indices' (list[int]) and 'counts' (list[int]).
    """
    n_docs = 0
    n_codes_sum = 0
    code_total = [0] * vocab_size
    total_tokens = 0
    for row in docs:
        n_docs += 1
        n_codes_sum += sum(row["counts"])
        for idx, cnt in zip(row["indices"], row["counts"]):
            code_total[idx] += cnt
            total_tokens += cnt
    mean_codes = n_codes_sum / max(n_docs, 1)
    marginals = [c / max(total_tokens, 1) for c in code_total]
    return CorpusStats(
        corpus_size_docs=n_docs,
        mean_codes_per_doc=mean_codes,
        k=k,
        v_full=vocab_size,
        code_marginals=marginals,
    )


def write_corpus_stats_sidecar(stats: CorpusStats, out_path: Path, *, v_displayed: int) -> None:
    """Write the small-scalars sidecar. v_displayed is the trimmed-vocab width."""
    out_path.write_text(json.dumps({
        "corpus_size_docs": stats.corpus_size_docs,
        "mean_codes_per_doc": stats.mean_codes_per_doc,
        "k": stats.k,
        "v": int(v_displayed),
        "v_full": stats.v_full,
    }))


def compute_corpus_stats_from_bow_df(bow_df: DataFrame, *, vocab_size: int, k: int) -> CorpusStats:
    """PySpark wrapper. Input bow_df must have 'indices' and 'counts' columns (array<int>).
    Streams rows to the driver via toLocalIterator."""
    rows = bow_df.select("indices", "counts").toLocalIterator()
    return compute_corpus_stats(
        docs=({"indices": list(r.indices), "counts": list(r.counts)} for r in rows),
        vocab_size=vocab_size,
        k=k,
    )
