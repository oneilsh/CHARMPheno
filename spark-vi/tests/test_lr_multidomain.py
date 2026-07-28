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
