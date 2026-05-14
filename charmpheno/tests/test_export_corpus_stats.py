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
    assert stats.code_marginals[0] == pytest.approx(3 / 7)
    assert stats.code_marginals[1] == pytest.approx(2 / 7)
    assert stats.code_marginals[2] == pytest.approx(2 / 7)


def test_write_corpus_stats_sidecar_omits_marginals(tmp_path: Path):
    # marginals are an intermediate, not in the sidecar file
    stats = CorpusStats(
        corpus_size_docs=10, mean_codes_per_doc=18.4,
        k=80, v_full=10000, code_marginals=[0.0001] * 10000,
    )
    out = tmp_path / "corpus_stats.json"
    write_corpus_stats_sidecar(stats, out, v_displayed=5000)
    payload = json.loads(out.read_text())
    assert set(payload.keys()) == {"corpus_size_docs", "mean_codes_per_doc", "k", "v", "v_full"}
    assert payload["v"] == 5000
    assert payload["v_full"] == 10000
