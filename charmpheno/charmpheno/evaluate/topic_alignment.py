"""Topic-recovery evaluation: JS divergence, prevalence ordering, biplot data.

Pure numpy + a single Spark aggregation in `ground_truth_from_oracle`. No
plotting — that lives in analysis/local/compare_lda_local.py.
"""
from __future__ import annotations

import numpy as np
from pyspark.sql import DataFrame
from pyspark.sql import functions as F


def js_divergence_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Pairwise Jensen-Shannon divergence between rows of A (K_a, V) and B (K_b, V).

    Returns (K_a, K_b) matrix in nats. Symmetric in (A, B).
    """
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    K_a, K_b = A.shape[0], B.shape[0]
    out = np.zeros((K_a, K_b))
    for i in range(K_a):
        for j in range(K_b):
            p, q = A[i], B[j]
            m = 0.5 * (p + q)
            out[i, j] = 0.5 * (_kl_safe(p, m) + _kl_safe(q, m))
    return out


def _kl_safe(p: np.ndarray, q: np.ndarray) -> float:
    """KL(p || q) computed only over the support of p (avoids 0 * log 0).

    The 1e-300 floor on q is defensive; in JS usage q is always the mixture
    0.5*(p+q_orig), so q >= 0.5*p > 0 wherever p > 0 and the floor never fires.
    """
    mask = p > 0
    return float(np.sum(p[mask] * (np.log(p[mask]) - np.log(q[mask] + 1e-300))))


def order_by_prevalence(
    topics: np.ndarray, prevalence: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Sort topic rows of `topics` (K, V) by `prevalence` descending.

    Returns (sorted_topics, perm). topics[perm] == sorted_topics.
    """
    perm = np.argsort(-np.asarray(prevalence))
    return topics[perm], perm


def alignment_biplot_data(
    topics_a: np.ndarray, prevalence_a: np.ndarray,
    topics_b: np.ndarray, prevalence_b: np.ndarray,
) -> dict:
    """Order both sets by prevalence and compute JS in the ordered frame.

    Returns dict with:
      js_matrix: (K_a, K_b)
      perm_a, perm_b: permutations applied (so caller can re-label)
      prevalence_a_sorted, prevalence_b_sorted: descending prevalence vectors
    """
    sorted_a, perm_a = order_by_prevalence(topics_a, prevalence_a)
    sorted_b, perm_b = order_by_prevalence(topics_b, prevalence_b)
    return {
        "js_matrix": js_divergence_matrix(sorted_a, sorted_b),
        "perm_a": perm_a,
        "perm_b": perm_b,
        "prevalence_a_sorted": np.asarray(prevalence_a)[perm_a],
        "prevalence_b_sorted": np.asarray(prevalence_b)[perm_b],
    }


def ground_truth_from_oracle(
    df: DataFrame,
    vocab_map: dict[int, int],
    K_true: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct (true_beta, true_prevalence) by aggregating the true_topic_id column.

    Robust to finite-sample noise: uses the empirical realization in this
    particular dataset (which is what we should hold recovery accountable
    to), not the simulator's parametric beta.

    Parameters:
        df: DataFrame with at least `true_topic_id` and `concept_id` columns.
        vocab_map: {concept_id: idx} as produced by to_bow_dataframe.
        K_true: number of true topics.

    Returns:
        true_beta: (K_true, V) row-normalized topic-word distributions.
        true_prevalence: (K_true,) total tokens per topic.
    """
    V = len(vocab_map)
    counts = (
        df.groupBy("true_topic_id", "concept_id")
          .agg(F.count("*").alias("n"))
          .collect()
    )
    beta = np.zeros((K_true, V), dtype=np.float64)
    for row in counts:
        cid = int(row["concept_id"])
        if cid not in vocab_map:
            continue
        k = int(row["true_topic_id"])
        if k < 0 or k >= K_true:
            continue
        beta[k, vocab_map[cid]] += float(row["n"])
    prevalence = beta.sum(axis=1)
    row_sums = beta.sum(axis=1, keepdims=True)
    beta = beta / np.where(row_sums > 0, row_sums, 1.0)
    return beta, prevalence
