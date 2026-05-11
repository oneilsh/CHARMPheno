"""HDP-specific helpers for eval.

OnlineHDP carries a length-(T-1) (u, v) parameter pair for the corpus-level
GEM stick, plus a (T, V) lambda. Most coherence evaluation wants to score
only the topics with non-trivial usage; this module computes the per-topic
expected stick weights E[beta_t] from (u, v) and exposes a top-K mask.
"""
from __future__ import annotations

import numpy as np


def _expected_corpus_betas(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Per-topic E[beta_t] from GEM stick-breaking parameters.

    For t = 0..T-2:  E[V_t] = u_t / (u_t + v_t)
                    E[beta_t] = E[V_t] * prod_{s<t} (1 - E[V_s])
    Topic T-1 receives the residual: 1 - sum(E[beta_0..T-2]).

    The closed form approximates the actual E[beta_t] (which involves a
    product of independent Beta random variables) by treating E[product]
    as product of expectations. This is the standard approximation used in
    the HDP literature (Wang/Paisley/Blei 2011) for ranking purposes.
    """
    e_vt = u / (u + v)
    log_one_minus = np.log1p(-e_vt)
    cum = np.concatenate([[0.0], np.cumsum(log_one_minus[:-1])])
    e_beta = np.empty(len(u) + 1, dtype=np.float64)
    e_beta[:-1] = e_vt * np.exp(cum)
    e_beta[-1] = max(0.0, 1.0 - e_beta[:-1].sum())
    return e_beta


def top_k_used_topics(*, u: np.ndarray, v: np.ndarray, k: int) -> np.ndarray:
    """Boolean mask of length T selecting the top-K topics by E[beta_t].

    Use as the `hdp_topic_mask` argument to compute_npmi_coherence.
    """
    if u.shape != v.shape:
        raise ValueError(f"u and v must have the same length; got {u.shape} and {v.shape}")
    T = len(u) + 1
    if k > T:
        raise ValueError(f"k={k} exceeds T={T}")
    e_beta = _expected_corpus_betas(u, v)
    # Top-K indices by descending E[beta]; ties broken by ascending index (stable sort).
    sorted_idx = np.argsort(-e_beta, kind="stable")[:k]
    mask = np.zeros(T, dtype=bool)
    mask[sorted_idx] = True
    return mask
