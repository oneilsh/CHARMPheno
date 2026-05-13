"""Tests for spark_vi.eval.topic.coherence."""
from __future__ import annotations

import math

import numpy as np
import pytest


def test_npmi_pair_independent_returns_zero():
    """If p(i,j) == p(i)*p(j), NPMI = 0."""
    from spark_vi.eval.topic.coherence import _npmi_pair
    # p_i = p_j = 0.5, p_ij = 0.25 -> log(1)/-log(0.25) = 0
    assert _npmi_pair(p_i=0.5, p_j=0.5, p_ij=0.25) == pytest.approx(0.0)


def test_npmi_pair_perfect_cooccurrence_returns_one():
    """If p(i,j) == p(i) == p(j), NPMI = 1 (always co-occur)."""
    from spark_vi.eval.topic.coherence import _npmi_pair
    # p_i = p_j = p_ij = 0.5
    # numerator: log(0.5 / 0.25) = log(2)
    # denominator: -log(0.5) = log(2)
    # ratio = 1
    assert _npmi_pair(p_i=0.5, p_j=0.5, p_ij=0.5) == pytest.approx(1.0)


def test_npmi_pair_zero_cooccurrence_returns_minus_one():
    """Roder et al. 2015 convention: NPMI = -1 when p(i,j) = 0."""
    from spark_vi.eval.topic.coherence import _npmi_pair
    assert _npmi_pair(p_i=0.5, p_j=0.5, p_ij=0.0) == -1.0


def test_npmi_pair_anti_correlated_is_negative():
    """If pair appears less than independence would predict, NPMI < 0."""
    from spark_vi.eval.topic.coherence import _npmi_pair
    # p_i = p_j = 0.5, p_ij = 0.1 << 0.25 (independence baseline)
    result = _npmi_pair(p_i=0.5, p_j=0.5, p_ij=0.1)
    assert result < 0.0
    assert result > -1.0  # not the zero-cooccur sentinel


def test_npmi_pair_handles_small_probabilities():
    """No log-of-zero NaN/inf for tiny but non-zero p_ij."""
    from spark_vi.eval.topic.coherence import _npmi_pair
    result = _npmi_pair(p_i=0.01, p_j=0.01, p_ij=1e-6)
    # Independence baseline: 0.01 * 0.01 = 1e-4; p_ij = 1e-6 << baseline.
    # Anti-correlated; bounded above by 1 and below by -1; not NaN.
    assert math.isfinite(result)
    assert -1.0 <= result <= 1.0


def test_top_n_terms_per_topic_picks_argmax_n():
    """For each topic row, return the indices of the N largest entries, sorted descending."""
    from spark_vi.eval.topic.coherence import _top_n_terms_per_topic
    topic_term = np.array([
        [0.1, 0.4, 0.2, 0.3],  # topic 0: top-2 by value -> indices [1, 3]
        [0.5, 0.05, 0.4, 0.05],  # topic 1: top-2 -> indices [0, 2]
    ])
    out = _top_n_terms_per_topic(topic_term, top_n=2)
    assert out.shape == (2, 2)
    assert list(out[0]) == [1, 3]
    assert list(out[1]) == [0, 2]


def test_top_n_terms_per_topic_breaks_ties_by_index():
    """Ties broken by ascending index, deterministic across runs."""
    from spark_vi.eval.topic.coherence import _top_n_terms_per_topic
    topic_term = np.array([
        [0.25, 0.25, 0.25, 0.25],  # all tied
    ])
    out = _top_n_terms_per_topic(topic_term, top_n=3)
    # With all ties, smaller indices win.
    assert list(out[0]) == [0, 1, 2]


def test_top_n_terms_per_topic_top_n_must_not_exceed_vocab():
    from spark_vi.eval.topic.coherence import _top_n_terms_per_topic
    topic_term = np.array([[0.5, 0.5]])
    with pytest.raises(ValueError, match="top_n"):
        _top_n_terms_per_topic(topic_term, top_n=3)


def test_aggregate_topic_coherence_tiny_example():
    """Hand-computed: 2 topics, 4 terms, 5 docs. min_pair_count=1 keeps
    every non-zero pair so the math here is the pre-threshold formula."""
    from spark_vi.eval.topic.coherence import _aggregate_topic_coherence

    # Top-N terms per topic:
    #   topic 0: terms [0, 1]
    #   topic 1: terms [2, 3]
    top_n_indices = np.array([[0, 1], [2, 3]])

    # Doc frequencies (out of 5 docs):
    #   term 0: 4 docs; term 1: 3 docs; term 2: 5 docs; term 3: 1 doc
    doc_freqs = {0: 4, 1: 3, 2: 5, 3: 1}

    # Pairwise co-occurrence:
    #   (0,1): 2 docs
    #   (2,3): 1 doc  (term 3 appears only in docs that also have term 2)
    pair_freqs = {(0, 1): 2, (2, 3): 1}

    n_docs = 5
    out, scored = _aggregate_topic_coherence(
        top_n_indices=top_n_indices,
        doc_freqs=doc_freqs,
        pair_freqs=pair_freqs,
        n_docs=n_docs,
        min_pair_count=1,
    )
    assert out.shape == (2,)
    assert list(scored) == [1, 1]  # each topic has one scored pair

    # Topic 0: NPMI((0, 1)) with p_0=4/5, p_1=3/5, p_01=2/5
    p_0, p_1, p_01 = 4/5, 3/5, 2/5
    expected_t0 = math.log(p_01 / (p_0 * p_1)) / -math.log(p_01)
    assert out[0] == pytest.approx(expected_t0)

    # Topic 1: NPMI((2, 3)) with p_2=5/5=1.0, p_3=1/5, p_23=1/5
    # PMI = log(0.2 / (1.0 * 0.2)) = log(1) = 0; NPMI = 0 / -log(0.2) = 0.
    assert out[1] == pytest.approx(0.0)


def test_aggregate_topic_coherence_missing_pair_is_skipped_not_floored():
    """Post-2026-05-12 behavior: a pair with joint count below threshold
    is *skipped* (contributes nothing to the mean), not floored at -1.

    Previously, the Roder convention would have floored a missing pair
    at NPMI = -1 and dragged the topic mean down — see
    docs/insights/0007-npmi-zero-pair-floor-penalizes-rare-phenotypes.md.
    """
    from spark_vi.eval.topic.coherence import _aggregate_topic_coherence

    top_n_indices = np.array([[0, 1]])
    doc_freqs = {0: 3, 1: 3}
    pair_freqs: dict[tuple[int, int], int] = {}  # never co-occurred
    out, scored = _aggregate_topic_coherence(
        top_n_indices=top_n_indices,
        doc_freqs=doc_freqs,
        pair_freqs=pair_freqs,
        n_docs=10,
        min_pair_count=1,
    )
    # No pair cleared the threshold → topic is unrated (NaN), with
    # zero scored-pair count. The old behavior (floor at -1) is gone.
    assert math.isnan(out[0])
    assert scored[0] == 0


def test_aggregate_topic_coherence_threshold_skips_low_counts():
    """A pair with joint count below min_pair_count contributes nothing
    even when nonzero. Mean is over scored pairs only."""
    from spark_vi.eval.topic.coherence import _aggregate_topic_coherence

    # One topic with two pairs. Pair (0,1) has count 2 (below threshold 3);
    # pair (0,2) has count 5 (above). Only (0,2) should be scored.
    top_n_indices = np.array([[0, 1, 2]])
    doc_freqs = {0: 5, 1: 4, 2: 5}
    pair_freqs = {(0, 1): 2, (0, 2): 5, (1, 2): 5}
    n_docs = 10
    out, scored = _aggregate_topic_coherence(
        top_n_indices=top_n_indices,
        doc_freqs=doc_freqs,
        pair_freqs=pair_freqs,
        n_docs=n_docs,
        min_pair_count=3,
    )
    # Two scored pairs: (0, 2) and (1, 2). (0, 1) skipped.
    assert scored[0] == 2
    # Topic NPMI is mean of those two pair-NPMIs (both nonzero,
    # data-driven; what matters is that scored count is 2, not 3).
    assert not math.isnan(out[0])


def test_aggregate_topic_coherence_default_threshold_matches_three():
    """Confirm the documented default of min_pair_count=3 for the inner
    function — pairs with counts in {0, 1, 2} are skipped, count >= 3
    is kept. (The inner function defaults to 1 to preserve back-compat
    in callsites; the production default of 3 lives on the public
    `compute_npmi_coherence`.)"""
    from spark_vi.eval.topic.coherence import _aggregate_topic_coherence

    top_n_indices = np.array([[0, 1, 2, 3]])
    doc_freqs = {0: 10, 1: 10, 2: 10, 3: 10}
    # Pair counts: 0, 1, 2, 3, 4, 5 across the 6 pairs.
    pair_freqs = {
        (0, 1): 0, (0, 2): 1, (0, 3): 2,
        (1, 2): 3, (1, 3): 4, (2, 3): 5,
    }
    _out, scored = _aggregate_topic_coherence(
        top_n_indices=top_n_indices,
        doc_freqs=doc_freqs,
        pair_freqs=pair_freqs,
        n_docs=20,
        min_pair_count=3,
    )
    # Counts >= 3: (1,2), (1,3), (2,3) → 3 scored pairs.
    assert scored[0] == 3


def test_aggregate_topic_coherence_zero_coverage_returns_nan():
    """Topic where every pair is below threshold yields NaN (unrated)."""
    from spark_vi.eval.topic.coherence import _aggregate_topic_coherence

    top_n_indices = np.array([[0, 1, 2]])
    doc_freqs = {0: 5, 1: 5, 2: 5}
    # All pair counts below threshold 5.
    pair_freqs = {(0, 1): 1, (0, 2): 2, (1, 2): 3}
    out, scored = _aggregate_topic_coherence(
        top_n_indices=top_n_indices,
        doc_freqs=doc_freqs,
        pair_freqs=pair_freqs,
        n_docs=10,
        min_pair_count=5,
    )
    assert math.isnan(out[0])
    assert scored[0] == 0


def test_aggregate_topic_coherence_averages_over_pairs():
    """Topic with top-N=3 averages over its 3 pairs of perfectly-co-occurring terms (NPMI = 1.0 each, mean = 1.0)."""
    from spark_vi.eval.topic.coherence import _aggregate_topic_coherence

    top_n_indices = np.array([[0, 1, 2]])
    # All three pairs perfectly co-occur: NPMI = 1 for each.
    doc_freqs = {0: 5, 1: 5, 2: 5}
    pair_freqs = {(0, 1): 5, (0, 2): 5, (1, 2): 5}
    out, scored = _aggregate_topic_coherence(
        top_n_indices=top_n_indices,
        doc_freqs=doc_freqs,
        pair_freqs=pair_freqs,
        n_docs=5,
        min_pair_count=1,
    )
    assert out[0] == pytest.approx(1.0)
    assert scored[0] == 3


def test_compute_doc_freqs_counts_distinct_doc_membership(sc):
    """Each term gets the number of distinct held-out docs it appears in.

    Counts are doc-level (binary): a doc with term 0 appearing 5 times still
    contributes 1 to doc_freqs[0].
    """
    from spark_vi.models.topic.types import BOWDocument
    from spark_vi.eval.topic.coherence import _compute_doc_freqs

    docs = [
        BOWDocument(indices=np.array([0, 1], dtype=np.int32), counts=np.array([5.0, 2.0]), length=7),
        BOWDocument(indices=np.array([0, 2], dtype=np.int32), counts=np.array([1.0, 1.0]), length=2),
        BOWDocument(indices=np.array([1], dtype=np.int32), counts=np.array([3.0]), length=3),
    ]
    rdd = sc.parallelize(docs, numSlices=2)
    interest = {0, 1, 2}

    out = _compute_doc_freqs(rdd, interest)
    assert out == {0: 2, 1: 2, 2: 1}


def test_compute_doc_freqs_ignores_terms_outside_interest_set(sc):
    from spark_vi.models.topic.types import BOWDocument
    from spark_vi.eval.topic.coherence import _compute_doc_freqs

    docs = [
        BOWDocument(indices=np.array([0, 99], dtype=np.int32), counts=np.array([1.0, 1.0]), length=2),
    ]
    rdd = sc.parallelize(docs)
    interest = {0}
    out = _compute_doc_freqs(rdd, interest)
    assert out == {0: 1}  # 99 absent


def test_compute_pair_freqs_emits_only_interest_set_pairs(sc):
    """Pair (i, j) with both i < j and both in interest set."""
    from spark_vi.models.topic.types import BOWDocument
    from spark_vi.eval.topic.coherence import _compute_pair_freqs

    docs = [
        # Intentionally feed an unsorted indices array to exercise the
        # canonical-ordering contract: keys must come back as (min, max).
        BOWDocument(indices=np.array([2, 0, 1], dtype=np.int32), counts=np.array([1.0, 1.0, 1.0]), length=3),
        BOWDocument(indices=np.array([0, 2], dtype=np.int32), counts=np.array([1.0, 1.0]), length=2),
        BOWDocument(indices=np.array([1, 99], dtype=np.int32), counts=np.array([1.0, 1.0]), length=2),
    ]
    rdd = sc.parallelize(docs, numSlices=2)
    interest = {0, 1, 2}

    out = _compute_pair_freqs(rdd, interest)
    # Doc 0: pairs (0,1), (0,2), (1,2). Doc 1: pair (0,2). Doc 2: no interest pairs.
    assert out == {(0, 1): 1, (0, 2): 2, (1, 2): 1}
    # Lock the canonical (min, max) ordering contract documented on
    # _aggregate_topic_coherence's pair_freqs arg.
    for (a, b) in out.keys():
        assert a < b, f"pair key ({a}, {b}) is not in (min, max) order"


def test_compute_npmi_coherence_lda_path(sc):
    """End-to-end on a tiny synthetic corpus, no HDP mask."""
    from spark_vi.models.topic.types import BOWDocument
    from spark_vi.eval.topic.coherence import compute_npmi_coherence
    from spark_vi.eval.topic.types import CoherenceReport

    # 2 topics over 4 terms. Topic 0 places mass on terms 0 and 1; topic 1 on 2 and 3.
    topic_term = np.array([
        [0.45, 0.45, 0.05, 0.05],
        [0.05, 0.05, 0.45, 0.45],
    ])
    docs = [
        BOWDocument(indices=np.array([0, 1], dtype=np.int32), counts=np.array([1.0, 1.0]), length=2),
        BOWDocument(indices=np.array([0, 1], dtype=np.int32), counts=np.array([1.0, 1.0]), length=2),
        BOWDocument(indices=np.array([2, 3], dtype=np.int32), counts=np.array([1.0, 1.0]), length=2),
        BOWDocument(indices=np.array([2, 3], dtype=np.int32), counts=np.array([1.0, 1.0]), length=2),
    ]
    rdd = sc.parallelize(docs, numSlices=2)

    # Pass min_pair_count=1 so the tiny synthetic fixture (each pair
    # co-occurs in exactly 2 docs) isn't filtered by the production
    # default of 3.
    report = compute_npmi_coherence(topic_term, rdd, top_n=2, min_pair_count=1)
    assert isinstance(report, CoherenceReport)
    assert report.per_topic_npmi.shape == (2,)
    assert report.top_term_indices.shape == (2, 2)
    assert list(report.topic_indices) == [0, 1]
    assert report.reference_size == 4
    assert report.n_holdout_docs == 4  # legacy alias
    assert report.top_n == 2
    assert report.min_pair_count == 1
    assert report.per_topic_total_pairs == 1  # C(2, 2) = 1
    assert list(report.per_topic_scored_pairs) == [1, 1]
    assert report.n_topics_unrated == 0
    # Each topic's top-N pair always co-occurs => NPMI = 1.0 per pair => mean = 1.0.
    assert report.per_topic_npmi[0] == pytest.approx(1.0)
    assert report.per_topic_npmi[1] == pytest.approx(1.0)
    assert report.mean == pytest.approx(1.0)
    assert report.min == pytest.approx(1.0)
    assert report.max == pytest.approx(1.0)
    # All values bounded in [-1, 1]
    assert (report.per_topic_npmi >= -1.0).all()
    assert (report.per_topic_npmi <= 1.0).all()


def test_compute_npmi_coherence_hdp_mask_path(sc):
    """HDP-style: T=4 topics in topic_term, mask selects 2 of them."""
    from spark_vi.models.topic.types import BOWDocument
    from spark_vi.eval.topic.coherence import compute_npmi_coherence

    # 4 topics, 4 terms. Topics 0 and 2 are "useful" (mass on 0,1 and 2,3).
    # Topics 1 and 3 are flat (should not be scored).
    topic_term = np.array([
        [0.45, 0.45, 0.05, 0.05],
        [0.25, 0.25, 0.25, 0.25],
        [0.05, 0.05, 0.45, 0.45],
        [0.25, 0.25, 0.25, 0.25],
    ])
    mask = np.array([True, False, True, False])

    docs = [
        BOWDocument(indices=np.array([0, 1], dtype=np.int32), counts=np.array([1.0, 1.0]), length=2),
        BOWDocument(indices=np.array([0, 1], dtype=np.int32), counts=np.array([1.0, 1.0]), length=2),
        BOWDocument(indices=np.array([2, 3], dtype=np.int32), counts=np.array([1.0, 1.0]), length=2),
        BOWDocument(indices=np.array([2, 3], dtype=np.int32), counts=np.array([1.0, 1.0]), length=2),
    ]
    rdd = sc.parallelize(docs, numSlices=2)

    report = compute_npmi_coherence(
        topic_term, rdd, top_n=2, hdp_topic_mask=mask, min_pair_count=1,
    )
    assert report.per_topic_npmi.shape == (2,)
    assert list(report.topic_indices) == [0, 2]
    assert report.per_topic_npmi[0] == pytest.approx(1.0)
    assert report.per_topic_npmi[1] == pytest.approx(1.0)


def test_compute_npmi_coherence_empty_mask_raises(sc):
    from spark_vi.models.topic.types import BOWDocument
    from spark_vi.eval.topic.coherence import compute_npmi_coherence

    topic_term = np.array([[0.5, 0.5]])
    mask = np.array([False])
    rdd = sc.parallelize(
        [BOWDocument(indices=np.array([0, 1], dtype=np.int32), counts=np.array([1.0, 1.0]), length=2)]
    )
    with pytest.raises(ValueError, match="zero topics"):
        compute_npmi_coherence(topic_term, rdd, top_n=2, hdp_topic_mask=mask)


def test_compute_npmi_coherence_threshold_produces_unrated_topic(sc):
    """A topic whose top-N pairs all fall below the threshold reports
    NaN NPMI, scored_pairs=0, and lifts n_topics_unrated by one."""
    from spark_vi.models.topic.types import BOWDocument
    from spark_vi.eval.topic.coherence import compute_npmi_coherence

    # Two topics. Topic 0 has top-2 (0, 1) — co-occurs in 5 docs (above any
    # reasonable threshold). Topic 1 has top-2 (2, 3) — co-occurs in only
    # 1 doc; with min_pair_count=3 it's unrated.
    topic_term = np.array([
        [0.45, 0.45, 0.05, 0.05],
        [0.05, 0.05, 0.45, 0.45],
    ])
    docs = (
        # Five docs containing (0, 1) — pair count 5.
        [BOWDocument(indices=np.array([0, 1], dtype=np.int32),
                     counts=np.array([1.0, 1.0]), length=2)] * 5
        # One doc containing (2, 3) — pair count 1.
        + [BOWDocument(indices=np.array([2, 3], dtype=np.int32),
                       counts=np.array([1.0, 1.0]), length=2)]
    )
    rdd = sc.parallelize(docs, numSlices=2)

    report = compute_npmi_coherence(topic_term, rdd, top_n=2, min_pair_count=3)
    # Topic 0 rated; topic 1 unrated (only 1 < 3).
    assert not math.isnan(report.per_topic_npmi[0])
    assert math.isnan(report.per_topic_npmi[1])
    assert list(report.per_topic_scored_pairs) == [1, 0]
    assert report.n_topics_unrated == 1
    # Summary stats use nan-aware aggregates: mean is topic-0's NPMI alone.
    assert not math.isnan(report.mean)
    assert report.mean == pytest.approx(report.per_topic_npmi[0])


def test_compute_npmi_coherence_all_topics_unrated_yields_nan_summary(sc):
    """If no topic clears the threshold, summary stats are NaN, not
    raised, so the driver can surface the situation in its banner."""
    from spark_vi.models.topic.types import BOWDocument
    from spark_vi.eval.topic.coherence import compute_npmi_coherence

    topic_term = np.array([
        [0.5, 0.5, 0.0],
        [0.0, 0.5, 0.5],
    ])
    # Each pair co-occurs in exactly 1 doc; threshold of 3 skips all.
    docs = [
        BOWDocument(indices=np.array([0, 1], dtype=np.int32),
                    counts=np.array([1.0, 1.0]), length=2),
        BOWDocument(indices=np.array([1, 2], dtype=np.int32),
                    counts=np.array([1.0, 1.0]), length=2),
    ]
    rdd = sc.parallelize(docs)

    report = compute_npmi_coherence(topic_term, rdd, top_n=2, min_pair_count=3)
    assert report.n_topics_unrated == 2
    assert math.isnan(report.mean)
    assert math.isnan(report.median)
    assert math.isnan(report.min)
    assert math.isnan(report.max)
