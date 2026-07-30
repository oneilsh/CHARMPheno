import numpy as np


def _tiny():
    """A 2-node DAG layout + two per-domain lambdas + two BOWs sharing the K topics."""
    from spark_vi.models.topic.dag_placement import DagLayout
    lay = DagLayout({1: 0, 2: 0}, n_bg=2, tpn=1)          # K = 2 bg + 2 nodes = 4
    rng = np.random.default_rng(0)
    K = lay.K
    lam0 = rng.random((K, 6)) + 0.1                        # domain 0: V=6
    lam1 = rng.random((K, 4)) + 0.1                        # domain 1: V=4
    bow0 = rng.integers(0, 3, size=(5, 6)).astype(float)   # 5 docs
    bow1 = rng.integers(0, 3, size=(5, 4)).astype(float)
    return lay, {0: lam0, 1: lam1}, {0: bow0, 1: bow1}


def test_lr_background_sparse_dense_and_fixed_scoring_equivalence():
    """Fixed reference frequencies keep linear LR scoring independent of batch scale."""
    from scipy import sparse as sp
    from spark_vi.models.topic.dag_placement import (
        lr_background, lr_placement_scores)

    lay, lam, bows = _tiny()
    train = np.asarray(bows[0][:3], dtype=float)
    test = np.asarray(bows[0][3:], dtype=float)
    bg = lr_background(train)

    assert np.isclose(bg.sum(), 1.0)
    assert np.allclose(bg, lr_background(sp.csr_matrix(train)))
    raw = lr_placement_scores(test, lam[0], lay, alpha=float("inf"), background=bg)
    scaled = lr_placement_scores(
        test * 7.0, lam[0], lay, alpha=float("inf"), background=bg)
    assert np.allclose(scaled, raw * 7.0)


def test_multidomain_score_is_the_per_domain_sum():
    from spark_vi.models.topic.dag_placement import (
        lr_placement_scores, lr_placement_scores_multidomain)
    lay, lam, bows = _tiny()
    s_multi = lr_placement_scores_multidomain(bows, lam, lay, alpha=1.0)
    s0 = lr_placement_scores(bows[0], lam[0], lay, alpha=1.0)
    s1 = lr_placement_scores(bows[1], lam[1], lay, alpha=1.0)
    assert np.allclose(s_multi, s0 + s1)                   # additivity


def test_single_domain_ties_out_to_lr_placement_scores():
    from spark_vi.models.topic.dag_placement import (
        lr_placement_scores, lr_placement_scores_multidomain)
    lay, lam, bows = _tiny()
    s_multi_one = lr_placement_scores_multidomain({0: bows[0]}, {0: lam[0]}, lay, alpha=1.0)
    assert np.allclose(s_multi_one, lr_placement_scores(bows[0], lam[0], lay, alpha=1.0))


def test_domain_subset_selects_that_domain():
    from spark_vi.models.topic.dag_placement import (
        lr_placement_scores, lr_placement_scores_multidomain)
    lay, lam, bows = _tiny()
    s_sub = lr_placement_scores_multidomain(bows, lam, lay, alpha=1.0, domains=[1])
    assert np.allclose(s_sub, lr_placement_scores(bows[1], lam[1], lay, alpha=1.0))
    # leave-one-out: all minus dropped == the remaining domain
    s_drop0 = lr_placement_scores_multidomain(bows, lam, lay, alpha=1.0, domains=[1])
    s_all = lr_placement_scores_multidomain(bows, lam, lay, alpha=1.0)
    s0 = lr_placement_scores(bows[0], lam[0], lay, alpha=1.0)
    assert np.allclose(s_all - s0, s_drop0)


def test_auc_sweep_multidomain_matches_manual_auc():
    from spark_vi.models.topic.dag_placement import (
        _auc, lr_placement_scores_multidomain, lr_auc_sweep_multidomain)
    lay, lam, bows = _tiny()
    is_fg = np.array([1, 0, 1, 0, 1])
    sweep = lr_auc_sweep_multidomain(bows, lam, lay, is_fg, alpha_grid=[1.0, 10.0])
    for a in (1.0, 10.0):
        s = lr_placement_scores_multidomain(bows, lam, lay, alpha=a)
        assert np.isclose(sweep[a], _auc(s.max(axis=1), is_fg))


def test_empty_domains_raises():
    import pytest
    from spark_vi.models.topic.dag_placement import lr_placement_scores_multidomain
    lay, lam, bows = _tiny()
    with pytest.raises(ValueError):
        lr_placement_scores_multidomain(bows, lam, lay, alpha=1.0, domains=[])


def _tiny_scaled():
    """_tiny() but domain 1's BOW is a 10x copy of domain 0's (same V, same lam),
    so domain 1's raw score matrix is exactly 10x domain 0's: the LR score is
    linear in the counts and the background base rate is a normalized frequency,
    hence scale-invariant. Gives an EXACT target for scale equalization."""
    from spark_vi.models.topic.dag_placement import DagLayout
    lay = DagLayout({1: 0, 2: 0}, n_bg=2, tpn=1)
    rng = np.random.default_rng(0)
    lam0 = rng.random((lay.K, 6)) + 0.1
    bow0 = rng.integers(0, 3, size=(5, 6)).astype(float)
    return lay, {0: lam0, 1: lam0}, {0: bow0, 1: bow0 * 10.0}


def test_normalize_none_is_the_unchanged_per_domain_sum():
    # Regression: the default path must be bit-identical to the plain sum.
    from spark_vi.models.topic.dag_placement import (
        lr_placement_scores, lr_placement_scores_multidomain)
    lay, lam, bows = _tiny()
    s0 = lr_placement_scores(bows[0], lam[0], lay, alpha=1.0)
    s1 = lr_placement_scores(bows[1], lam[1], lay, alpha=1.0)
    s = lr_placement_scores_multidomain(bows, lam, lay, alpha=1.0, normalize=None)
    assert np.array_equal(s, s0 + s1)


def test_domain_score_matrices_sum_to_the_multidomain_score():
    from spark_vi.models.topic.dag_placement import (
        lr_domain_score_matrices, lr_placement_scores_multidomain)
    lay, lam, bows = _tiny()
    for rule in (None, "std", "length", "length+std"):
        mats = lr_domain_score_matrices(bows, lam, lay, alpha=1.0, normalize=rule)
        assert set(mats) == {0, 1}
        total = lr_placement_scores_multidomain(bows, lam, lay, alpha=1.0,
                                                normalize=rule)
        assert np.allclose(mats[0] + mats[1], total), rule


def test_std_normalization_equalizes_domain_scale_exactly():
    # Mechanism check. Domain 1 is a 10x copy of domain 0, so un-normalized its
    # scale is ~10x; after 'std' both matrices have unit std.
    from spark_vi.models.topic.dag_placement import (
        _domain_scale, domain_score_scale, lr_domain_score_matrices)
    lay, lam, bows = _tiny_scaled()
    raw = lr_domain_score_matrices(bows, lam, lay, alpha=1.0, normalize=None)
    assert domain_score_scale(raw[0]) == _domain_scale(raw[0])
    assert np.isclose(_domain_scale(raw[1]) / _domain_scale(raw[0]), 10.0)
    norm = lr_domain_score_matrices(bows, lam, lay, alpha=1.0, normalize="std")
    assert np.isclose(np.std(norm[0]), 1.0)
    assert np.isclose(np.std(norm[1]), 1.0)


def test_std_normalization_preserves_single_domain_ordering():
    # The invariance contract: one scalar per domain => affine => every
    # within-domain ordering survives (doc ranking AND max-over-nodes), so the
    # readout's only:<m> columns cannot move under 'std'.
    from spark_vi.models.topic.dag_placement import (
        _auc, lr_placement_scores_multidomain)
    lay, lam, bows = _tiny()
    one_bow, one_lam = {0: bows[0]}, {0: lam[0]}
    raw = lr_placement_scores_multidomain(one_bow, one_lam, lay, alpha=1.0)
    std = lr_placement_scores_multidomain(one_bow, one_lam, lay, alpha=1.0,
                                          normalize="std")
    y = np.array([1, 0, 1, 0, 1])
    assert np.isclose(_auc(raw.max(axis=1), y), _auc(std.max(axis=1), y))
    assert np.array_equal(raw.argmax(axis=1), std.argmax(axis=1))


def test_length_normalization_matches_per_domain_length_normalize():
    from spark_vi.models.topic.dag_placement import (
        lr_placement_scores, lr_placement_scores_multidomain)
    lay, lam, bows = _tiny()
    s = lr_placement_scores_multidomain(bows, lam, lay, alpha=1.0,
                                        normalize="length")
    expect = sum(lr_placement_scores(bows[m], lam[m], lay, alpha=1.0,
                                     length_normalize=True) for m in (0, 1))
    assert np.allclose(s, expect)


def test_domain_scale_falls_back_to_one_on_a_constant_domain():
    # An all-zero (no-token) domain has zero std; it must pass through as zeros
    # and contribute nothing, not produce inf/nan.
    from spark_vi.models.topic.dag_placement import _domain_scale, domain_score_scale
    for x in (np.zeros((4, 3)), np.full((4, 3), 2.5)):
        assert domain_score_scale(x) == _domain_scale(x) == 1.0


def test_combine_domain_score_matrices_applies_fixed_scales_and_weights():
    """A wrong per-domain divisor or multiplier changes the combined score."""
    from spark_vi.models.topic.dag_placement import combine_domain_score_matrices

    mats = {
        0: np.array([[2.0, 4.0], [6.0, 8.0]]),
        1: np.array([[10.0, 20.0], [30.0, 40.0]]),
    }
    got = combine_domain_score_matrices(
        mats, weights={0: 0.75, 1: 0.25}, scales={0: 2.0, 1: 10.0})
    expect = 0.75 * mats[0] / 2.0 + 0.25 * mats[1] / 10.0
    assert np.allclose(got, expect)


def test_combine_domain_score_matrices_identity_and_validation():
    """Invalid component shapes and multipliers cannot silently bias a ranking."""
    import pytest
    from spark_vi.models.topic.dag_placement import combine_domain_score_matrices

    mats = {0: np.ones((3, 2)), 1: np.full((3, 2), 2.0)}
    assert np.array_equal(combine_domain_score_matrices(mats), mats[0] + mats[1])
    with pytest.raises(ValueError, match="same shape"):
        combine_domain_score_matrices({0: mats[0], 1: np.ones((2, 2))})
    with pytest.raises(ValueError, match="scale"):
        combine_domain_score_matrices(mats, scales={1: 0.0})
    with pytest.raises(ValueError, match="weight"):
        combine_domain_score_matrices(mats, weights={1: -0.1})
    with pytest.raises(ValueError, match="at least one"):
        combine_domain_score_matrices(mats, weights={0: 0.0, 1: 0.0})


def test_unknown_normalize_raises():
    import pytest
    from spark_vi.models.topic.dag_placement import (
        lr_domain_score_matrices, lr_placement_scores_multidomain)
    lay, lam, bows = _tiny()
    with pytest.raises(ValueError):
        lr_placement_scores_multidomain(bows, lam, lay, alpha=1.0, normalize="zscore")
    with pytest.raises(ValueError):
        lr_domain_score_matrices(bows, lam, lay, alpha=1.0, normalize="zscore")


def test_auc_sweep_multidomain_forwards_normalize():
    from spark_vi.models.topic.dag_placement import (
        _auc, lr_placement_scores_multidomain, lr_auc_sweep_multidomain)
    lay, lam, bows = _tiny()
    is_fg = np.array([1, 0, 1, 0, 1])
    sweep = lr_auc_sweep_multidomain(bows, lam, lay, is_fg, alpha_grid=[1.0],
                                     normalize="std")
    s = lr_placement_scores_multidomain(bows, lam, lay, alpha=1.0, normalize="std")
    assert np.isclose(sweep[1.0], _auc(s.max(axis=1), is_fg))
