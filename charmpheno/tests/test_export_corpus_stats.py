from __future__ import annotations
import json
from pathlib import Path

import pytest

from charmpheno.export.corpus_stats import (
    CorpusStats,
    compute_corpus_stats,
    write_corpus_stats_sidecar,
)


def test_compute_corpus_stats_basic():
    docs = [
        {"indices": [0, 1], "counts": [2, 1]},
        {"indices": [1, 2], "counts": [1, 1]},
        {"indices": [0, 2], "counts": [1, 1]},
    ]
    stats = compute_corpus_stats(docs=iter(docs), vocab_size=3, k=4)
    assert stats.corpus_size_docs == 3
    assert stats.mean_codes_per_doc == pytest.approx((3 + 2 + 2) / 3)
    assert stats.k == 4
    assert stats.v_full == 3
    # code 0: 2+1=3 tokens across 2 docs (rows 0 and 2)
    # code 1: 1+1=2 tokens across 2 docs (rows 0 and 1)
    # code 2: 1+1=2 tokens across 2 docs (rows 1 and 2)
    assert stats.code_marginals[0] == pytest.approx(3 / 7)
    assert stats.code_marginals[1] == pytest.approx(2 / 7)
    assert stats.code_marginals[2] == pytest.approx(2 / 7)
    assert stats.code_doc_counts == [2, 2, 2]


def test_compute_corpus_stats_doc_count_differs_from_token_count():
    """A code appearing many times in one doc still counts as one doc.
    This is exactly the producer/consumer mismatch the doc-count field
    is meant to disambiguate."""
    docs = [
        {"indices": [0], "counts": [100]},  # code 0 dominates one doc
        {"indices": [1], "counts": [1]},
        {"indices": [1], "counts": [1]},
        {"indices": [1], "counts": [1]},
    ]
    stats = compute_corpus_stats(docs=iter(docs), vocab_size=2, k=1)
    # code 0 has 100/103 ≈ 0.97 token mass but appears in 1 doc
    # code 1 has 3/103 ≈ 0.03 token mass but appears in 3 docs
    assert stats.code_marginals[0] == pytest.approx(100 / 103)
    assert stats.code_marginals[1] == pytest.approx(3 / 103)
    assert stats.code_doc_counts == [1, 3]


def test_write_corpus_stats_sidecar_omits_marginals(tmp_path: Path):
    # marginals and doc_counts are intermediates, not in the sidecar file
    stats = CorpusStats(
        corpus_size_docs=10, mean_codes_per_doc=18.4,
        k=80, v_full=10000,
        code_marginals=[0.0001] * 10000,
        code_doc_counts=[1] * 10000,
    )
    out = tmp_path / "corpus_stats.json"
    write_corpus_stats_sidecar(stats, out, v_displayed=5000)
    payload = json.loads(out.read_text())
    assert set(payload.keys()) == {"corpus_size_docs", "mean_codes_per_doc", "k", "v", "v_full"}
    assert payload["v"] == 5000
    assert payload["v_full"] == 10000
