"""Held-out within-document predictive log-likelihood: split one document's tokens into
a visible and a held-out part, and score the held-out part under a predictive token
distribution theta_hat @ beta.

Generic (any topic model with a per-document theta and a topic-term beta). The STM
shim's held-out scale sweeps and the per-document predictive-gain oracle use it; the
concentration-recovery simulator studies that first defined it now live with the
consuming application.
"""
from __future__ import annotations

import numpy as np

from spark_vi.models.topic.types import STMDocument


def heldout_split(
    doc: STMDocument, *, holdout_frac: float = 0.3, seed: int,
) -> tuple[STMDocument, np.ndarray, np.ndarray] | None:
    """Split ONE document's tokens into a visible half and a held-out half.

    Expands (indices, counts) to a token multiset (each index repeated by its
    count), shuffles with a seeded rng, holds out round(holdout_frac *
    n_tokens) tokens, and re-aggregates both halves back to (indices, counts)
    form. Returns (visible_doc, held_indices, held_counts) where visible_doc
    is a new STMDocument (same x/groups; indices/counts/length reflect only
    the visible half) and held_indices/held_counts are numpy arrays for the
    held-out half.

    Guards: a document with < 2 tokens, or one whose visible half would be
    empty, returns None so the caller can skip it (there is nothing to infer
    theta_hat from, or nothing held out to score).
    """
    n_tokens = int(round(float(doc.counts.sum())))
    if n_tokens < 2:
        return None

    rng = np.random.default_rng(seed)
    tokens = np.repeat(doc.indices, doc.counts.astype(np.int64))
    rng.shuffle(tokens)

    n_held = round(holdout_frac * n_tokens)
    held_tokens = tokens[:n_held]
    visible_tokens = tokens[n_held:]
    if visible_tokens.size == 0:
        return None

    v_idx, v_counts = np.unique(visible_tokens, return_counts=True)
    if held_tokens.size > 0:
        h_idx, h_counts = np.unique(held_tokens, return_counts=True)
    else:
        h_idx = np.array([], dtype=np.int32)
        h_counts = np.array([], dtype=np.float64)

    visible_doc = STMDocument(
        indices=v_idx.astype(np.int32),
        counts=v_counts.astype(np.float64),
        length=int(v_counts.sum()),
        x=doc.x,
        groups=doc.groups,
    )
    return visible_doc, h_idx.astype(np.int32), h_counts.astype(np.float64)


def _predictive_loglik(
    theta_hat: np.ndarray, beta: np.ndarray, held_indices: np.ndarray, held_counts: np.ndarray,
) -> float:
    """Held-out log-likelihood of held_indices/held_counts under the
    predictive token distribution `theta_hat @ beta` (length V). The 1e-12
    floor guards log(0) for a held-out term with zero predicted mass. Returns
    the SUM over held-out tokens in this one document (the caller averages
    per token across the whole corpus).
    """
    pred = theta_hat @ beta
    return float(np.sum(held_counts * np.log(pred[held_indices] + 1e-12)))
