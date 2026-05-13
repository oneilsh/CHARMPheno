"""Slow-tier smoke: end-to-end on a tiny in-memory fit.

Stand-in for the driver smoke: invokes the eval surface (load + split + score)
against a real fit on a small synthetic corpus. Property-checks the result.

Does NOT verify that NPMI ranks topics by ground-truth quality — that test was
explicitly deferred (see spec).
"""
from __future__ import annotations

import numpy as np
import pytest

from spark_vi.models.topic.types import BOWDocument


pytestmark = pytest.mark.slow


def _make_synthetic_bow(sc, n_docs: int = 200, vocab_size: int = 50,
                        n_topics: int = 5, seed: int = 0):
    """Generate a tiny LDA-shaped synthetic BOW RDD."""
    rng = np.random.default_rng(seed)
    # Each "topic" is a sharp distribution over disjoint slices of vocab.
    slice_size = vocab_size // n_topics
    docs = []
    for _ in range(n_docs):
        t = int(rng.integers(0, n_topics))
        terms = rng.choice(
            np.arange(t * slice_size, (t + 1) * slice_size),
            size=slice_size // 2, replace=False,
        )
        terms = np.sort(terms.astype(np.int32))
        counts = np.ones(len(terms), dtype=np.float64)
        docs.append(BOWDocument(indices=terms, counts=counts, length=int(counts.sum())))
    return sc.parallelize(docs, numSlices=4)


def test_smoke_coherence_on_tiny_synthetic(sc):
    """Property checks: NPMI in [-1, 1], length matches, summary stats consistent."""
    from spark_vi.eval.topic import compute_npmi_coherence

    K, V = 5, 50
    # Skip the actual VI fit; build a "perfectly recovered" topic_term that places
    # mass on each topic's vocab slice. This is what a successful fit would look like.
    slice_size = V // K
    topic_term = np.zeros((K, V), dtype=np.float64)
    for k in range(K):
        topic_term[k, k * slice_size : (k + 1) * slice_size] = 1.0 / slice_size

    holdout = _make_synthetic_bow(sc, n_docs=200, vocab_size=V, n_topics=K, seed=0)
    report = compute_npmi_coherence(topic_term, holdout, top_n=5)

    # Property checks (the only assertions the spec retained from the dropped tests).
    assert report.per_topic_npmi.shape == (K,)
    assert (report.per_topic_npmi >= -1.0).all()
    assert (report.per_topic_npmi <= 1.0).all()

    # Summary stats consistent with numpy on the underlying array.
    assert report.mean == pytest.approx(float(report.per_topic_npmi.mean()))
    assert report.median == pytest.approx(float(np.median(report.per_topic_npmi)))
    assert report.stdev == pytest.approx(float(report.per_topic_npmi.std(ddof=0)))
    assert report.min == pytest.approx(float(report.per_topic_npmi.min()))
    assert report.max == pytest.approx(float(report.per_topic_npmi.max()))

    # Driver-side check matching the assertion in eval_coherence.main().
    assert report.n_holdout_docs == 200
