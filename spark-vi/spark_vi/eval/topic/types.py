"""Result types for spark_vi.eval.topic.

CoherenceReport is the frozen dataclass returned by compute_npmi_coherence;
the same shape supports LDA (all K topics scored) and HDP (mask-filtered
subset of T topics).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class CoherenceReport:
    """Per-topic NPMI with coverage and descriptive summary statistics.

    Attributes:
        per_topic_npmi: shape (K_scored,) per-topic mean NPMI over the
            *scored* pairs (joint count >= min_pair_count) of the top-N
            terms. Topics with zero scored pairs are NaN; they are
            excluded from the summary statistics below.
        per_topic_scored_pairs: shape (K_scored,) int64; how many of the
            top-N pairs cleared the joint-count threshold and contributed
            to that topic's mean.
        top_term_indices: shape (K_scored, N) — vocab indices of the top-N
            terms per scored topic, sorted by descending E[beta_t].
        topic_indices: shape (K_scored,) — original topic indices in the
            input topic-term matrix. Identity for LDA; mask-filtered subset
            for HDP.
        reference_size: number of documents in the reference corpus used
            for the co-occurrence statistics. The eval drivers currently
            pass the full BOW reproduced from the checkpoint's frozen
            vocab; in principle any RDD[BOWDocument] the caller hands to
            compute_npmi_coherence is fair game.
        n_holdout_docs: legacy alias for `reference_size` (older callers
            may read this; new code should prefer `reference_size`).
        per_topic_total_pairs: total number of unordered top-N pairs per
            topic; equals ``top_n * (top_n - 1) // 2``. Constant across
            topics, stored as a scalar.
        top_n: top-N parameter the report was computed with.
        min_pair_count: joint-count threshold below which a pair is
            skipped (not contributing to its topic's mean NPMI). 1 = only
            skip true zeros; higher values require statistical evidence
            per pair before scoring.
        n_topics_unrated: count of topics with zero scored pairs
            (per_topic_npmi NaN).
        mean, median, stdev, min, max: descriptive summary over the
            NON-NaN entries of per_topic_npmi (i.e. excluding unrated
            topics). Computed with nan-aware aggregates.

    Array fields are conceptually immutable — ``frozen=True`` prevents
    attribute reassignment but does not deep-freeze; callers should not
    mutate the contained arrays.
    """
    per_topic_npmi: np.ndarray
    per_topic_scored_pairs: np.ndarray
    top_term_indices: np.ndarray
    topic_indices: np.ndarray
    reference_size: int
    n_holdout_docs: int  # legacy alias for reference_size
    per_topic_total_pairs: int
    top_n: int
    min_pair_count: int
    n_topics_unrated: int
    mean: float
    median: float
    stdev: float
    min: float
    max: float
