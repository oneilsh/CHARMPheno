"""NPMI topic coherence over a held-out BOW corpus.

Implements normalized pointwise mutual information per Roder et al. 2015,
with whole-document co-occurrence (no sliding window). The metric is the
unweighted mean over all unordered pairs of a topic's top-N terms.

See docs/decisions/0017-topic-coherence-evaluation.md and the spec at
docs/superpowers/specs/2026-05-11-topic-coherence-evaluation-design.md.
"""
from __future__ import annotations

import math
from itertools import combinations
from typing import TYPE_CHECKING

import numpy as np

from spark_vi.eval.topic.types import CoherenceReport

if TYPE_CHECKING:
    from pyspark import RDD

    from spark_vi.models.topic.types import BOWDocument


def _npmi_pair(p_i: float, p_j: float, p_ij: float) -> float:
    """NPMI for a single (w_i, w_j) pair.

    NPMI = log[p_ij / (p_i * p_j)] / -log p_ij.

    Edge cases:
      p_ij == 0:  return -1.0   (Roder et al. 2015 convention; pair never co-occurs)
      p_ij == 1:  return  1.0   (denominator -log(1) = 0; pair always co-occurs)

    These sentinel returns keep the per-pair value bounded in [-1, 1] with
    no NaN/Inf contamination, so the topic-level mean over pairs remains
    numerically well-defined.
    """
    if p_ij <= 0.0:
        return -1.0
    if p_ij >= 1.0:
        return 1.0
    pmi = math.log(p_ij / (p_i * p_j))
    return pmi / -math.log(p_ij)


def _top_n_terms_per_topic(topic_term: np.ndarray, top_n: int) -> np.ndarray:
    """Return the indices of the top-N terms for each topic, sorted descending.

    Ties broken by ascending index (lexicographic argsort over (-value, index)).
    Output shape: (K, top_n) with dtype int64.
    """
    K, V = topic_term.shape
    if top_n > V:
        raise ValueError(f"top_n={top_n} exceeds vocabulary size V={V}")
    # Stable sort on negated values gives descending order with tie-broken by
    # ascending original index — exactly what argsort on (-value) does when
    # stable.
    sorted_idx = np.argsort(-topic_term, axis=1, kind="stable")
    return sorted_idx[:, :top_n].astype(np.int64)


def _aggregate_topic_coherence(
    *,
    top_n_indices: np.ndarray,
    doc_freqs: dict[int, int],
    pair_freqs: dict[tuple[int, int], int],
    n_docs: int,
    min_pair_count: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-topic mean NPMI over the *scored* unordered pairs of top-N terms.

    A pair is "scored" iff its joint co-occurrence count in the reference
    corpus is >= ``min_pair_count``. Pairs below threshold contribute
    nothing to the topic mean — neither a -1 floor (as the Roder
    convention would) nor a 0. This is the C_NPMI handling per Aletras &
    Stevenson 2013 / Roder et al. 2015: rare pairs are unreliable
    co-occurrence estimates, so we honestly skip them and report
    coverage alongside NPMI.

    Args:
        top_n_indices: shape (K_scored, N). Term indices per topic.
        doc_freqs: term_index -> # reference docs containing that term.
        pair_freqs: (min_idx, max_idx) -> # reference docs containing both.
            Pairs with zero co-occurrence may be absent from this dict.
            Keys must be normalized as ``(min(w_i, w_j), max(w_i, w_j))`` —
            the canonical form this function looks up.
        n_docs: total # reference docs (the normalizer).
        min_pair_count: skip pairs whose joint count is < this value.
            Default 1 preserves the historical "only skip true zeros"
            behavior. Production drivers use 3 (so pairs with counts
            {0, 1, 2} are skipped).

    Returns:
        (per_topic_npmi, per_topic_scored_pairs):
        - per_topic_npmi: shape (K_scored,) float64. Mean NPMI over the
          scored pairs of each topic. ``NaN`` for topics where every
          pair was below threshold (zero coverage).
        - per_topic_scored_pairs: shape (K_scored,) int64. How many
          pairs cleared the threshold per topic.
    """
    K_scored, N = top_n_indices.shape
    out = np.empty(K_scored, dtype=np.float64)
    scored_counts = np.empty(K_scored, dtype=np.int64)
    for k in range(K_scored):
        terms = top_n_indices[k]
        total = 0.0
        n_scored = 0
        for w_i, w_j in combinations(terms, 2):
            a, b = (int(w_i), int(w_j)) if w_i < w_j else (int(w_j), int(w_i))
            joint = pair_freqs.get((a, b), 0)
            if joint < min_pair_count:
                continue
            p_i = doc_freqs.get(a, 0) / n_docs
            p_j = doc_freqs.get(b, 0) / n_docs
            p_ij = joint / n_docs
            total += _npmi_pair(p_i=p_i, p_j=p_j, p_ij=p_ij)
            n_scored += 1
        scored_counts[k] = n_scored
        out[k] = (total / n_scored) if n_scored > 0 else float("nan")
    return out, scored_counts


def _compute_doc_freqs(
    bow_rdd: "RDD[BOWDocument]",
    interest_set: set[int],
) -> dict[int, int]:
    """Per-term doc-frequency over the held-out corpus, restricted to interest_set.

    Returns {term_index: # docs containing it}, only for terms in interest_set.
    """
    interest_b = bow_rdd.context.broadcast(interest_set)

    def _emit_terms(doc):
        s = interest_b.value
        seen = set()
        for w in doc.indices:
            iw = int(w)
            if iw in s and iw not in seen:
                seen.add(iw)
                yield (iw, 1)

    counts = bow_rdd.flatMap(_emit_terms).reduceByKey(lambda a, b: a + b).collectAsMap()
    interest_b.unpersist(blocking=False)
    return dict(counts)


def _compute_pair_freqs(
    bow_rdd: "RDD[BOWDocument]",
    interest_set: set[int],
) -> dict[tuple[int, int], int]:
    """Pairwise co-occurrence over the held-out corpus, restricted to interest_set.

    Returns {(min_idx, max_idx): # docs containing both}, only for pairs where
    both indices are in interest_set and at least one doc co-occurrence exists.
    Keys are normalized to ``(min, max)`` order — sorting the per-doc interest
    terms before :func:`itertools.combinations` produces this naturally.
    """
    interest_b = bow_rdd.context.broadcast(interest_set)

    def _emit_pairs(doc):
        s = interest_b.value
        terms = sorted({int(w) for w in doc.indices if int(w) in s})
        for a, b in combinations(terms, 2):
            yield ((a, b), 1)

    counts = bow_rdd.flatMap(_emit_pairs).reduceByKey(lambda a, b: a + b).collectAsMap()
    interest_b.unpersist(blocking=False)
    return dict(counts)


def compute_npmi_coherence(
    topic_term: np.ndarray,
    reference_bow: "RDD[BOWDocument]",
    *,
    top_n: int = 20,
    topic_mask: np.ndarray | None = None,
    min_pair_count: int = 3,
) -> CoherenceReport:
    """NPMI coherence, mean over scored top-N pairs per topic.

    Args:
        topic_term: shape (K, V) row-stochastic; topic-term distributions
            E[beta]. For OnlineLDA pass lambda_ / lambda_.sum(axis=1, keepdims=True).
            For OnlineHDP same, with shape (T, V).
        reference_bow: RDD of BOWDocument used as the co-occurrence
            reference. Counts are ignored; only the set of indices per
            doc is used (binary co-occurrence). The driver chooses whether
            this is the holdout split alone or the full corpus
            (train ∪ holdout); see ADR 0017 revisions.

            Callers should pass an already-cached RDD (or one derived from a
            cached DataFrame). The orchestrator runs three actions over this
            RDD (``count()``, the doc-freqs map-reduce, and the pair-freqs
            map-reduce) and re-derivation cost will multiply otherwise. At
            local-test scale this is invisible; at cloud scale it is a
            measurable cost.
        top_n: number of top terms per topic. Default 20. Must be <= V.
        topic_mask: optional boolean array of length K matching topic_term's
            first axis. When provided, only mask==True rows are scored.
            None (default) scores all rows. Use cases include selecting the
            top-K-used HDP topics via top_k_used_topics(u, v, k), filtering
            out LDA's gracefully-unused tail (insight 0019), or restricting
            evaluation to a user-specified subset of topics.
        min_pair_count: skip pairs with joint count below this threshold
            in the reference corpus. Default 3 — pairs with counts in
            {0, 1, 2} contribute nothing to their topic mean (vs the
            pre-2026-05-12 behavior of flooring missing-pair NPMI at -1,
            which biased rare-phenotype topics toward maximally
            incoherent scores; see docs/insights/0007).

    Returns:
        CoherenceReport with per-topic NPMI, per-topic coverage counts,
        and nan-aware summary stats over the rated topics.
    """
    K_full, V = topic_term.shape
    if topic_mask is None:
        scored_rows = np.arange(K_full, dtype=np.int64)
    else:
        if topic_mask.shape != (K_full,):
            raise ValueError(
                f"topic_mask shape {topic_mask.shape} does not match topic_term K={K_full}"
            )
        scored_rows = np.flatnonzero(topic_mask).astype(np.int64)

    if scored_rows.size == 0:
        raise ValueError("topic_mask selected zero topics; nothing to score")

    filtered_topic_term = topic_term[scored_rows]
    top_n_indices = _top_n_terms_per_topic(filtered_topic_term, top_n=top_n)

    interest_set: set[int] = {int(w) for w in np.unique(top_n_indices)}

    n_docs = reference_bow.count()
    if n_docs == 0:
        raise ValueError("reference_bow is empty; cannot compute coherence")

    doc_freqs = _compute_doc_freqs(reference_bow, interest_set)
    pair_freqs = _compute_pair_freqs(reference_bow, interest_set)

    per_topic, per_topic_scored = _aggregate_topic_coherence(
        top_n_indices=top_n_indices,
        doc_freqs=doc_freqs,
        pair_freqs=pair_freqs,
        n_docs=n_docs,
        min_pair_count=min_pair_count,
    )

    rated_mask = ~np.isnan(per_topic)
    if rated_mask.any():
        rated = per_topic[rated_mask]
        mean = float(rated.mean())
        median = float(np.median(rated))
        stdev = float(rated.std(ddof=0))
        vmin = float(rated.min())
        vmax = float(rated.max())
    else:
        # No topic met the threshold for any pair. Surface that via NaN
        # summary rather than computing over an empty slice; the driver
        # banner will flag the unrated count.
        mean = median = stdev = vmin = vmax = float("nan")

    return CoherenceReport(
        per_topic_npmi=per_topic,
        per_topic_scored_pairs=per_topic_scored,
        top_term_indices=top_n_indices,
        topic_indices=scored_rows,
        reference_size=n_docs,
        n_holdout_docs=n_docs,
        per_topic_total_pairs=top_n * (top_n - 1) // 2,
        top_n=top_n,
        min_pair_count=min_pair_count,
        n_topics_unrated=int((~rated_mask).sum()),
        mean=mean,
        median=median,
        stdev=stdev,
        min=vmin,
        max=vmax,
    )
