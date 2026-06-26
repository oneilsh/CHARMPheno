"""Shared per-iteration topic-word numerics for the topic models.

Pure, model-agnostic derivations from the (K, V) topic-word variational
parameter lambda — the math the LDA/STM/HDP cloud drivers' top-terms loggers
all share. Callers supply their own ordering key, per-topic annotation, and
vocab labeling; this helper never sees concept ids/names (engine layer is
domain-agnostic).
"""
from __future__ import annotations

import numpy as np


def topic_word_summary(lam: np.ndarray, top_n: int) -> dict[str, np.ndarray]:
    """Per-topic top-N term numerics from lambda (K, V).

    Returns a dict with row_sums (K,), peak (K,), mass_fraction (K,),
    top_indices (K, min(top_n, V)) and top_probs (same shape). top_indices are
    column indices into the vocabulary, descending by row-stochastic
    probability; top_probs are those probabilities. mass_fraction is the
    row-sum-normalized E[beta] used by LDA/STM (HDP supplies its own from the
    corpus sticks and ignores this field).
    """
    lam = np.asarray(lam, dtype=np.float64)
    row_sums = lam.sum(axis=1)
    denom = np.maximum(row_sums, 1e-12)
    peak = lam.max(axis=1) / denom
    topics = lam / denom[:, None]                       # row-stochastic
    n = min(int(top_n), lam.shape[1])
    top_indices = np.argsort(topics, axis=1)[:, ::-1][:, :n]
    top_probs = np.take_along_axis(topics, top_indices, axis=1)
    return {
        "row_sums": row_sums,
        "peak": peak,
        "mass_fraction": row_sums / max(float(row_sums.sum()), 1e-12),
        "top_indices": top_indices,
        "top_probs": top_probs,
    }
