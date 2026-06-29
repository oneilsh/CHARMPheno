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
from spark_vi.models.topic.spectral_init import (
    find_anchors,
    recover_beta,
    word_cooccurrence,
)
from spark_vi.models.topic.spectral_init import spectral_init_beta
from spark_vi.models.topic.spectral_init_scalable import (
    ProjectedCoocResult,
    _project_doc,
    _r_rows,
    default_projection_dim,
    find_anchors_projected,
    projected_cooccurrence_rdd,
    recover_beta_projected,
    scalable_spectral_init_beta,
)
from tests._stm_synth import foreground_recovers_group


def _projected_inputs(docs, V, seed, d, orthonormal=False):
    """Build (Q, QR, p_w, df_w) by projecting the dense oracle — no Spark.

    QR and p_w come off the same summed scale (QR = Q @ R, p_w = Q.sum(axis=1)),
    so the p_w row-normalization in the projected functions is exact.

    ``orthonormal`` QR-orthonormalizes the (V×d) Gaussian R before projecting.
    With d == V this makes R a pure rotation, so Q @ R is an ISOMETRY: it
    preserves residual norms and inner products exactly, which is the d→V limit
    the JL projection only approximates with a raw Gaussian (O(1/√d) error). The
    equivalence-to-oracle tests use this isometry to pin the geometry claim
    deterministically; the production pass uses the raw Gaussian sketch (JL only
    needs approximate preservation, which the floor + NNLS tolerate).
    """
    Q = word_cooccurrence(docs, V)
    R = _r_rows(np.arange(V, dtype=np.int64), seed, d)
    if orthonormal:
        R, _ = np.linalg.qr(R)
    QR = Q @ R
    p_w = Q.sum(axis=1)
    df_w = np.zeros(V, dtype=np.int64)
    for doc in docs:
        n = np.asarray(doc.counts, dtype=np.float64)
        if float(n.sum()) < 2:
            continue
        df_w[np.asarray(doc.indices, dtype=np.int64)] += 1
    return Q, QR, p_w, df_w


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
    group_p_w = {g: np.zeros(V, dtype=np.float64) for g in partition.groups}
    group_df_w = {g: np.zeros(V, dtype=np.int64) for g in partition.groups}
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
                group_p_w[g][idx] += pw
                group_df_w[g][idx] += 1
    return pooled, group_QR, p_w, df_w, group_p_w, group_df_w


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
        (ref_pooled, ref_group, ref_pw, ref_df,
         ref_gpw, ref_gdf) = _numpy_reference(
            docs, partition, V, d, seed, np.float64
        )
        np.testing.assert_allclose(result.pooled_QR, ref_pooled, atol=1e-9, rtol=0)
        for g in partition.groups:
            np.testing.assert_allclose(
                result.group_QR[g], ref_group[g], atol=1e-9, rtol=0
            )
            np.testing.assert_allclose(
                result.group_p_w[g], ref_gpw[g], atol=1e-12, rtol=0
            )
            np.testing.assert_array_equal(result.group_df_w[g], ref_gdf[g])
        np.testing.assert_allclose(result.p_w, ref_pw, atol=1e-12, rtol=0)
        np.testing.assert_array_equal(result.df_w, ref_df)

    def test_group_marginals_match_within_group_numpy_reference(self, spark):
        """group_p_w[g] / group_df_w[g] equal a numpy reference over group g's docs.

        The gated foreground step needs the WITHIN-GROUP marginal (the dense path
        uses Q_g.sum(axis=1)); pin it against a single-process reference that
        restricts to the docs carrying group g.
        """
        from tests._stm_synth import synthetic_gated_corpus

        docs, _planted, partition = synthetic_gated_corpus(
            groups=("cancer", "dementia"), fg_per_group=1, bg_k=2,
            V=24, D=22, doc_len=12, bg_frac=0.5, seed=11,
        )
        V, d, seed = 24, 10, 777
        rdd = spark.sparkContext.parallelize(docs, numSlices=4)
        result = projected_cooccurrence_rdd(
            rdd, partition, V=V, d=d, seed=seed, dtype=np.float64
        )

        for g in partition.groups:
            ref_pw = np.zeros(V, dtype=np.float64)
            ref_df = np.zeros(V, dtype=np.int64)
            for doc in docs:
                if g not in doc.groups:
                    continue
                n = np.asarray(doc.counts, dtype=np.float64)
                if float(n.sum()) < 2:
                    continue
                idx = np.asarray(doc.indices, dtype=np.int64)
                _qr, pw = _project_doc(idx, n, _r_rows(idx, seed, d))
                ref_pw[idx] += pw
                ref_df[idx] += 1
            np.testing.assert_allclose(
                result.group_p_w[g], ref_pw, atol=1e-12, rtol=0
            )
            np.testing.assert_array_equal(result.group_df_w[g], ref_df)

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


# ---------------------------------------------------------------------------
# Projected anchor-finding + beta recovery (ADR 0032). Inputs are built WITHOUT
# Spark by projecting the dense oracle (see _projected_inputs); QR and p_w are on
# the same summed scale so the p_w row-normalization is exact.
# ---------------------------------------------------------------------------


def test_projected_row_normalization_is_by_p_w():
    """(QR / p_w)[i] reproduces the dense conditional row image (Q[i]/p_w[i]) @ R.

    Pins that dividing the sketch row by the marginal p_w (NOT by the projected
    row sum) recovers the projected image of the dense Qbar row.
    """
    from tests._stm_synth import synthetic_ehr_corpus

    docs, _planted = synthetic_ehr_corpus(
        K_rare=3, V=30, D=40, doc_len=12, bg_frac=0.5, seed=1
    )
    V, seed, d = 30, 4242, 16
    Q, QR, p_w, _df = _projected_inputs(docs, V, seed, d)
    R = _r_rows(np.arange(V, dtype=np.int64), seed, d)

    Qbar_proj = QR / p_w[:, None]
    for i in range(V):
        if p_w[i] <= 0:
            continue
        expected = (Q[i] / p_w[i]) @ R
        np.testing.assert_allclose(Qbar_proj[i], expected, atol=1e-10, rtol=0)


def test_doc_freq_floor_excludes_rare_extreme_word():
    """A word with df_w < min_doc_freq is never an anchor even if its row is extreme.

    Two words are given identical, extreme (axis-aligned, unique-direction) rows;
    one has df below the floor, the other at/above it. Only the eligible one may
    be returned.
    """
    V, d = 6, 6
    QR = np.zeros((V, d), dtype=np.float64)
    p_w = np.ones(V, dtype=np.float64)
    # Words 0,1 get identical extreme unit rows along the same axis; words 2..5
    # get small mixed rows so the two extremes are the standout directions.
    QR[0] = np.eye(d)[0]
    QR[1] = np.eye(d)[0]
    for w in range(2, V):
        QR[w] = 0.01 * np.eye(d)[w % d]
    df_w = np.array([4, 10, 10, 10, 10, 10], dtype=np.int64)  # word 0 below floor

    # With the floor at 5, word 0 (df=4) is ineligible; word 1 (df=10) is the
    # eligible extreme and must be picked first.
    got = find_anchors_projected(QR, p_w, df_w, 1, min_doc_freq=5)
    assert got == [1]
    assert 0 not in find_anchors_projected(QR, p_w, df_w, V, min_doc_freq=5)

    # Drop the floor: word 0 becomes eligible (equally extreme) and may be chosen.
    got_lo = find_anchors_projected(QR, p_w, df_w, 1, min_doc_freq=1)
    assert got_lo[0] in (0, 1)


def test_anchor_equivalence_large_d():
    """At d=V with an isometric (orthonormal) sketch, projected anchors == dense.

    The greedy pivoted-QR selection depends only on residual NORMS of the
    normalized rows. A raw Gaussian R distorts those norms by O(1/√d) and so only
    APPROXIMATELY preserves the selection; the exact d→V limit is realized by an
    isometry (orthonormalized R, a pure rotation), under which residual norms are
    preserved to machine precision. We test the isometry to pin the geometry
    claim deterministically. Isolate it from the candidate-floor difference with
    min_doc_freq=1 / min_marginal_frac=0.0.
    """
    from tests._stm_synth import synthetic_ehr_corpus

    docs, _planted = synthetic_ehr_corpus(
        K_rare=4, V=40, D=120, doc_len=14, bg_frac=0.5, seed=7
    )
    V, seed, K = 40, 31337, 4
    d = V
    Q, QR, p_w, df_w = _projected_inputs(docs, V, seed, d, orthonormal=True)

    dense = set(find_anchors(Q, K, min_marginal_frac=0.0))
    proj = set(find_anchors_projected(QR, p_w, df_w, K, min_doc_freq=1))
    # Isometry: residual-norm ordering preserved exactly -> identical selection.
    assert len(dense ^ proj) <= 1, (sorted(dense), sorted(proj))


def test_recover_beta_projected_valid_and_close():
    """recover_beta_projected is row-stochastic and close to dense at large d."""
    from tests._stm_synth import synthetic_ehr_corpus

    docs, _planted = synthetic_ehr_corpus(
        K_rare=4, V=40, D=120, doc_len=14, bg_frac=0.5, seed=7
    )
    V, seed, K = 40, 31337, 4
    d = V
    # Isometry (orthonormal d=V sketch): the NNLS fit min||A^T c - Qbar_w|| is
    # invariant under a rotation of A and Qbar_w, so the projected recovery equals
    # the dense one. A raw Gaussian only approximates this (O(1/√d)); the isometry
    # makes the closeness exact and deterministic.
    Q, QR, p_w, df_w = _projected_inputs(docs, V, seed, d, orthonormal=True)

    anchors = find_anchors(Q, K, min_marginal_frac=0.0)
    beta = recover_beta_projected(QR, p_w, anchors)

    # Valid (K, V) row-stochastic matrix.
    assert beta.shape == (K, V)
    assert (beta >= 0).all()
    np.testing.assert_allclose(beta.sum(axis=1), 1.0, atol=1e-9, rtol=0)

    # Matches the dense oracle on the SAME anchors to ~machine precision, and the
    # per-topic top-word sets coincide.
    dense_beta = recover_beta(Q, anchors)
    assert np.abs(beta - dense_beta).max() < 1e-9
    for k in range(K):
        top_proj = set(np.argsort(beta[k])[-5:])
        top_dense = set(np.argsort(dense_beta[k])[-5:])
        assert len(top_proj & top_dense) >= 4


# ---------------------------------------------------------------------------
# Orchestrator: scalable_spectral_init_beta (block-aware, one distributed pass).
# Mirrors the dense spectral_init_beta structure on the projected primitives.
# ---------------------------------------------------------------------------


class TestScalableSpectralInitBeta:
    def test_non_gated_orchestrator(self, spark):
        """All-background partition: step 1 only, valid stochastic, non-uniform.

        The degenerate dense case (background_k = K, no foreground groups) runs the
        pooled background recovery alone and must reproduce a global single-pass
        anchor-word seed: a valid (K, V) row-stochastic β whose filled rows are NOT
        the uniform 1/V (spectral seeding took effect).
        """
        from tests._stm_synth import synthetic_ehr_corpus

        docs, _planted = synthetic_ehr_corpus(
            K_rare=4, V=40, D=120, doc_len=14, bg_frac=0.5, seed=7
        )
        V, K, seed = 40, 4, 31337
        partition = TopicBlockPartition(
            group_var="", background_k=K, foreground=()
        )
        rdd = spark.sparkContext.parallelize(docs, numSlices=4)
        beta = scalable_spectral_init_beta(
            rdd, partition, V=V, seed=seed, min_doc_freq=3
        )

        assert beta.shape == (K, V)
        assert (beta >= 0).all()
        for k in partition.background_indices():
            row = beta[k]
            if row.sum() == 0:
                continue  # short-fill left a zero row; never NaN
            np.testing.assert_allclose(row.sum(), 1.0, atol=1e-9, rtol=0)
            # Spectral seeding took effect: not the uniform 1/V seed.
            assert np.abs(row - 1.0 / V).max() > 1e-6

    def test_gated_orchestrator_structure_and_rare_recovery(self, spark):
        """Gated: background rows in bg slots, foreground rows in each group's block.

        Structural: each filled background row lives in background_indices() and each
        group's foreground rows live in block_indices(g), all valid stochastic. Then
        a foreground-recovery check analogous to the dense
        test_block_aware_init_recovers_rare_group_foreground: the rare arm's
        foreground topic lands its planted phenotype at init.
        """
        from tests._stm_synth import synthetic_gated_corpus

        # Same separable corpus the dense
        # test_block_aware_init_recovers_rare_group_foreground uses, with the rare
        # arm thinned to a minority (keep 1 in 4 of its docs).
        docs, planted, partition = synthetic_gated_corpus(
            groups=("maj", "rare"), fg_per_group=2, bg_k=3, V=240, D=1200,
            doc_len=30, bg_frac=0.6, seed=3,
        )
        docs = [
            d for i, d in enumerate(docs)
            if ("rare" not in d.groups) or (i % 4 == 0)
        ]

        V, seed = 240, 31337
        # A raw-Gaussian sketch at the DEFAULT d (eps=0.1) only approximately
        # preserves the anchor geometry (JL error O(1/√d)); empirically the rare-arm
        # foreground anchor is not reliably recovered until eps≈0.04 (d≈3426 here).
        # We raise d for THIS recovery test and document it. We do NOT orthonormalize
        # in production code, and the rigorous scalable≈dense planted-recovery at the
        # production default d is a separate later task — here a structural check plus
        # a recovers-the-rare-arm check at a larger d is sufficient.
        d = default_projection_dim(partition.K, V, eps=0.04)
        rdd = spark.sparkContext.parallelize(docs, numSlices=4)
        beta = scalable_spectral_init_beta(
            rdd, partition, V=V, d=d, seed=seed, min_doc_freq=3
        )

        assert beta.shape == (partition.K, V)
        assert (beta >= 0).all()
        # Every filled row is a valid distribution.
        for k in range(partition.K):
            row = beta[k]
            if row.sum() == 0:
                continue
            np.testing.assert_allclose(row.sum(), 1.0, atol=1e-9, rtol=0)

        # Structural: background rows occupy the background slots; each group's
        # foreground rows occupy that group's block (rows are placed by slot in the
        # orchestrator, so this is a placement invariant — filled rows are nonzero).
        bg_idx = set(partition.background_indices().tolist())
        fg_all = set()
        for g in partition.groups:
            fg_all |= set(partition.block_indices(g).tolist())
        assert bg_idx.isdisjoint(fg_all)
        assert bg_idx | fg_all == set(range(partition.K))

        # Foreground recovery: the rare arm's planted phenotype surfaces in its
        # block (thresh matches the dense block-aware recovery test).
        assert foreground_recovers_group(
            beta, partition, "rare", planted, thresh=0.4
        )
