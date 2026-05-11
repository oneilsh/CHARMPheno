"""NPMI topic coherence over a held-out BOW corpus.

Implements normalized pointwise mutual information per Roder et al. 2015,
with whole-document co-occurrence (no sliding window). The metric is the
unweighted mean over all unordered pairs of a topic's top-N terms.

See docs/decisions/0017-topic-coherence-evaluation.md and the spec at
docs/superpowers/specs/2026-05-11-topic-coherence-evaluation-design.md.
"""
from __future__ import annotations

import math


def _npmi_pair(p_i: float, p_j: float, p_ij: float) -> float:
    """NPMI for a single (w_i, w_j) pair.

    NPMI = log[p_ij / (p_i * p_j)] / -log p_ij.

    Returns -1.0 when p_ij == 0 (Roder et al. 2015 convention) so the
    pair-aggregate stays in [-1, 1] without NaN/Inf contamination.
    """
    if p_ij <= 0.0:
        return -1.0
    pmi = math.log(p_ij / (p_i * p_j))
    return pmi / -math.log(p_ij)
