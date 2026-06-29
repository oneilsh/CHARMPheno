"""Tests for the scalable (random-projection) spectral-init foundation.

Pins the per-doc rank-1 projected co-occurrence against the dense oracle
(spectral_init.word_cooccurrence) and the distributed accumulation pass
against a single-process numpy reference. See ADR 0032.
"""
from __future__ import annotations

import numpy as np
import pytest

from spark_vi.models.topic.types import STMDocument
from spark_vi.models.topic.partition import TopicBlockPartition
from spark_vi.models.topic.spectral_init import word_cooccurrence
from spark_vi.models.topic.spectral_init_scalable import (
    ProjectedCoocResult,
    _project_doc,
    _r_rows,
    default_projection_dim,
    projected_cooccurrence_rdd,
)


def _doc(indices, counts, x=None, groups=frozenset()):
    n = np.asarray(counts, dtype=np.float64)
    return STMDocument(
        indices=np.asarray(indices, dtype=np.int32),
        counts=n,
        length=int(n.sum()),
        x=np.array([1.0]) if x is None else np.asarray(x, dtype=np.float64),
        groups=groups,
    )


def _dense_block(n, L):
    return (np.outer(n, n) - np.diag(n)) / (L * (L - 1.0))


def test_default_projection_dim():
    import math

    # max(K, ceil(eps^-2 * ln(V)))
    assert default_projection_dim(K=4, V=1000, eps=0.1) == max(
        4, math.ceil(0.1 ** -2 * math.log(1000))
    )
    # K floor wins when JL dimension is tiny
    assert default_projection_dim(K=500, V=10, eps=0.5) == 500


def test_rank1_per_doc_identity():
    """qr_contrib equals the projected dense block (incl. diag correction)."""
    seed, d = 1234, 16
    idx = np.array([2, 5, 9, 11], dtype=np.int64)
    n = np.array([3.0, 1.0, 2.0, 4.0])
    doc = _doc(idx, n)
    L = float(n.sum())

    R_support = _r_rows(idx, seed, d)  # (len(idx), d)
    qr_contrib, _pw = _project_doc(idx, n, R_support)

    # Dense oracle: full V x d projection restricted to support rows.
    V = int(idx.max()) + 1
    R_full = _r_rows(np.arange(V, dtype=np.int64), seed, d)  # (V, d)
    block = _dense_block(n, L)  # (len, len) over support
    dense_proj = block @ R_full[idx]  # (len, d): block lives only on support cols

    np.testing.assert_allclose(qr_contrib, dense_proj, atol=1e-10, rtol=0)

    # Companion: dropping the -n_i*R[i] diag self-pair correction must disagree.
    s = (n[:, None] * R_support).sum(axis=0)
    wrong = (n[:, None] * s[None, :]) / (L * (L - 1.0))  # no diag term
    assert not np.allclose(wrong, dense_proj, atol=1e-10)


def test_pw_per_doc_identity():
    """pw_contrib equals the dense block's row sums on the support."""
    seed, d = 7, 8
    idx = np.array([0, 1, 4], dtype=np.int64)
    n = np.array([2.0, 5.0, 1.0])
    L = float(n.sum())

    R_support = _r_rows(idx, seed, d)
    _qr, pw_contrib = _project_doc(idx, n, R_support)

    block = _dense_block(n, L)
    np.testing.assert_allclose(pw_contrib, block.sum(axis=1), atol=1e-10, rtol=0)


def test_corpus_projected_sum_identity():
    """Sum over docs of scattered qr_contrib equals sum over docs of (block_d @ R).

    Normalization: word_cooccurrence divides by n_docs(L>=2) and by the grand
    total; we accumulate on the UNnormalized per-doc scale (Σ_d block_d), so the
    clean identity is against the un-averaged per-doc blocks, NOT against Q.
    """
    seed, d, V = 99, 12, 14
    docs = [
        _doc([1, 3, 5], [2.0, 1.0, 3.0]),
        _doc([0, 5, 9, 11], [1.0, 1.0, 1.0, 2.0]),
        _doc([3, 5], [4.0, 4.0]),
        _doc([7], [3.0]),  # L>=2 but single support -> block is all-zero off-diag
    ]
    R_full = _r_rows(np.arange(V, dtype=np.int64), seed, d)

    # left side: scattered qr_contrib accumulation
    acc = np.zeros((V, d), dtype=np.float64)
    for doc in docs:
        n = doc.counts
        L = float(n.sum())
        if L < 2:
            continue
        idx = np.asarray(doc.indices, dtype=np.int64)
        qr, _pw = _project_doc(idx, n, R_full[idx])
        acc[idx] += qr

    # right side: Σ_d (block_d @ R) scattered into V x d
    ref = np.zeros((V, d), dtype=np.float64)
    for doc in docs:
        n = doc.counts
        L = float(n.sum())
        if L < 2:
            continue
        idx = np.asarray(doc.indices, dtype=np.int64)
        block = _dense_block(n, L)
        ref[idx] += block @ R_full[idx]

    np.testing.assert_allclose(acc, ref, atol=1e-10, rtol=0)


def test_df_w_counts_multitoken_docs():
    """df_w[i] = number of docs (L>=2) whose support contains word i."""
    V = 8
    docs = [
        _doc([1, 3], [1.0, 1.0]),
        _doc([1, 5, 3], [1.0, 1.0, 1.0]),
        _doc([3], [5.0]),  # single support, still L>=2: contributes df for word 3
        _doc([2], [1.0]),  # L<2: contributes nothing
    ]
    df = np.zeros(V, dtype=np.int64)
    for doc in docs:
        n = doc.counts
        if float(n.sum()) < 2:
            continue
        df[np.asarray(doc.indices, dtype=np.int64)] += 1

    expected = np.zeros(V, dtype=np.int64)
    expected[1] = 2
    expected[3] = 3
    expected[5] = 1
    np.testing.assert_array_equal(df, expected)


def _numpy_reference(docs, partition, V, d, seed, dtype):
    """Single-process reference for the distributed pass."""
    pooled = np.zeros((V, d), dtype=dtype)
    group_QR = {g: np.zeros((V, d), dtype=dtype) for g in partition.groups}
    p_w = np.zeros(V, dtype=np.float64)
    df_w = np.zeros(V, dtype=np.int64)
    R_full = _r_rows(np.arange(V, dtype=np.int64), seed, d)
    for doc in docs:
        n = np.asarray(doc.counts, dtype=np.float64)
        L = float(n.sum())
        if L < 2:
            continue
        idx = np.asarray(doc.indices, dtype=np.int64)
        qr, pw = _project_doc(idx, n, R_full[idx])
        pooled[idx] += qr.astype(dtype)
        p_w[idx] += pw
        df_w[idx] += 1
        for g in doc.groups:
            if g in group_QR:
                group_QR[g][idx] += qr.astype(dtype)
    return pooled, group_QR, p_w, df_w


class TestProjectedCooccurrenceRDD:
    def test_distributed_matches_numpy_reference(self, spark):
        from tests._stm_synth import synthetic_gated_corpus

        docs, _planted, partition = synthetic_gated_corpus(
            groups=("cancer", "dementia"), fg_per_group=1, bg_k=2,
            V=24, D=18, doc_len=12, bg_frac=0.5, seed=3,
        )
        V, d, seed = 24, 10, 555

        rdd = spark.sparkContext.parallelize(docs, numSlices=4)
        result = projected_cooccurrence_rdd(
            rdd, partition, V=V, d=d, seed=seed, dtype=np.float64
        )

        assert isinstance(result, ProjectedCoocResult)
        ref_pooled, ref_group, ref_pw, ref_df = _numpy_reference(
            docs, partition, V, d, seed, np.float64
        )
        np.testing.assert_allclose(result.pooled_QR, ref_pooled, atol=1e-9, rtol=0)
        for g in partition.groups:
            np.testing.assert_allclose(
                result.group_QR[g], ref_group[g], atol=1e-9, rtol=0
            )
        np.testing.assert_allclose(result.p_w, ref_pw, atol=1e-12, rtol=0)
        np.testing.assert_array_equal(result.df_w, ref_df)

    def test_float32_default_close_to_float64(self, spark):
        from tests._stm_synth import synthetic_gated_corpus

        docs, _planted, partition = synthetic_gated_corpus(
            groups=("cancer", "dementia"), fg_per_group=1, bg_k=2,
            V=24, D=18, doc_len=12, bg_frac=0.5, seed=3,
        )
        V, d, seed = 24, 10, 555
        rdd = spark.sparkContext.parallelize(docs, numSlices=4)

        r32 = projected_cooccurrence_rdd(rdd, partition, V=V, d=d, seed=seed)
        r64 = projected_cooccurrence_rdd(
            rdd, partition, V=V, d=d, seed=seed, dtype=np.float64
        )
        assert r32.pooled_QR.dtype == np.float32
        denom = np.abs(r64.pooled_QR)
        denom[denom == 0] = 1.0
        rel = np.abs(r32.pooled_QR - r64.pooled_QR) / denom
        assert rel.max() < 1e-3
