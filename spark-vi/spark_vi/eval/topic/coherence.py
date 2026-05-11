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

    from spark_vi.core.types import BOWDocument


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
) -> np.ndarray:
    """Per-topic mean NPMI over the unordered pairs of its top-N terms.

    Args:
        top_n_indices: shape (K_scored, N). Term indices per topic.
        doc_freqs: term_index -> # held-out docs containing that term.
        pair_freqs: (min_idx, max_idx) -> # held-out docs containing both.
            Pairs with zero co-occurrence may be absent from this dict.
            Keys must be normalized as ``(min(w_i, w_j), max(w_i, w_j))`` —
            the canonical form this function looks up.
        n_docs: total # held-out docs (the normalizer).

    Returns:
        shape (K_scored,) float64 array of mean NPMI per topic.
    """
    K_scored, N = top_n_indices.shape
    out = np.empty(K_scored, dtype=np.float64)
    n_pairs = N * (N - 1) // 2
    for k in range(K_scored):
        terms = top_n_indices[k]
        total = 0.0
        for w_i, w_j in combinations(terms, 2):
            a, b = (int(w_i), int(w_j)) if w_i < w_j else (int(w_j), int(w_i))
            p_i = doc_freqs.get(a, 0) / n_docs
            p_j = doc_freqs.get(b, 0) / n_docs
            p_ij = pair_freqs.get((a, b), 0) / n_docs
            total += _npmi_pair(p_i=p_i, p_j=p_j, p_ij=p_ij)
        out[k] = total / n_pairs
    return out


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
    holdout_bow: "RDD[BOWDocument]",
    *,
    top_n: int = 20,
    hdp_topic_mask: np.ndarray | None = None,
) -> CoherenceReport:
    """NPMI coherence on held-out data, mean over top-N pairs per topic.

    Args:
        topic_term: shape (K, V) row-stochastic; topic-term distributions
            E[beta]. For OnlineLDA pass lambda_ / lambda_.sum(axis=1, keepdims=True).
            For OnlineHDP same, with shape (T, V).
        holdout_bow: RDD of BOWDocument. Counts are ignored; only the set of
            indices per doc is used (binary co-occurrence).

            Callers should pass an already-cached RDD (or one derived from a
            cached DataFrame). The orchestrator runs three actions over this
            RDD (``count()``, the doc-freqs map-reduce, and the pair-freqs
            map-reduce) and re-derivation cost will multiply otherwise. At
            local-test scale this is invisible; at cloud scale it is a
            measurable cost.
        top_n: number of top terms per topic. Default 20. Must be <= V.
        hdp_topic_mask: optional boolean array of length K (or T for HDP). When
            provided, only mask==True rows of topic_term are scored. None means
            score all rows (the LDA path).

    Returns:
        CoherenceReport with per-topic NPMI and descriptive summary stats.
    """
    K_full, V = topic_term.shape
    if hdp_topic_mask is None:
        scored_rows = np.arange(K_full, dtype=np.int64)
    else:
        if hdp_topic_mask.shape != (K_full,):
            raise ValueError(
                f"hdp_topic_mask shape {hdp_topic_mask.shape} does not match topic_term K={K_full}"
            )
        scored_rows = np.flatnonzero(hdp_topic_mask).astype(np.int64)

    if scored_rows.size == 0:
        raise ValueError("hdp_topic_mask selected zero topics; nothing to score")

    filtered_topic_term = topic_term[scored_rows]
    top_n_indices = _top_n_terms_per_topic(filtered_topic_term, top_n=top_n)

    interest_set: set[int] = {int(w) for w in np.unique(top_n_indices)}

    n_docs = holdout_bow.count()
    if n_docs == 0:
        raise ValueError("holdout_bow is empty; cannot compute coherence")

    doc_freqs = _compute_doc_freqs(holdout_bow, interest_set)
    pair_freqs = _compute_pair_freqs(holdout_bow, interest_set)

    per_topic = _aggregate_topic_coherence(
        top_n_indices=top_n_indices,
        doc_freqs=doc_freqs,
        pair_freqs=pair_freqs,
        n_docs=n_docs,
    )

    return CoherenceReport(
        per_topic_npmi=per_topic,
        top_term_indices=top_n_indices,
        topic_indices=scored_rows,
        n_holdout_docs=n_docs,
        top_n=top_n,
        mean=float(per_topic.mean()),
        median=float(np.median(per_topic)),
        stdev=float(per_topic.std(ddof=0)),
        min=float(per_topic.min()),
        max=float(per_topic.max()),
    )
