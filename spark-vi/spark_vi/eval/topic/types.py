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
    """Per-topic NPMI and descriptive summary statistics.

    Attributes:
        per_topic_npmi: shape (K_scored,) per-topic mean NPMI over the
            unordered pairs of top-N terms.
        top_term_indices: shape (K_scored, N) — vocab indices of the top-N
            terms per scored topic, sorted by descending E[beta_t].
        topic_indices: shape (K_scored,) — original topic indices in the
            input topic-term matrix. Identity for LDA; mask-filtered subset
            for HDP.
        n_holdout_docs: number of documents in the held-out corpus used
            for the co-occurrence statistics.
        top_n: top-N parameter the report was computed with.
        mean, median, stdev, min, max: descriptive summary over
            per_topic_npmi. Not used to normalize.

    Array fields are conceptually immutable — ``frozen=True`` prevents
    attribute reassignment but does not deep-freeze; callers should not
    mutate the contained arrays.
    """
    per_topic_npmi: np.ndarray
    top_term_indices: np.ndarray
    topic_indices: np.ndarray
    n_holdout_docs: int
    top_n: int
    mean: float
    median: float
    stdev: float
    min: float
    max: float
