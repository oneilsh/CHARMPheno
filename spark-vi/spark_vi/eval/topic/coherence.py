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
    return {tuple(k): v for k, v in counts.items()}
