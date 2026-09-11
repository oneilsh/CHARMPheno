"""Alpha policy for the gated engine (exp 0121): `equalized_alpha` (children-
first initial alpha — equal TOTAL prior pseudo-count per block across the
corpus) and `GatedOnlineLDA.set_alpha_policy` (the post-construction seam the
Gated-PC estimator uses, since the frontier histogram exists only once the
document RDD does). Spark-free.
"""
import json

import numpy as np

from spark_vi.models.topic.dag_placement import DagLayout
from spark_vi.models.topic.gated_lda import GatedOnlineLDA, equalized_alpha


def _lay():
    # root 0; nodes 1, 2 under root; 3 under 1; 4 under 2 but NO training docs
    return DagLayout({1: 0, 2: 0, 3: 1, 4: 2}, n_bg=2, tpn=2)


_HIST = {frozenset({3}): 10, frozenset({2}): 40, frozenset(): 50}   # 100 docs


def test_equalized_alpha_gives_equal_total_prior_mass_per_block():
    lay = _lay()
    a = equalized_alpha(lay, _HIST, mean_alpha=0.5)
    assert a.shape == (lay.K,)
    assert abs(a.mean() - 0.5) < 1e-12                       # rescaled to the mean
    a_bg, a1, a2, a3, a4 = a[0], a[lay.block[1][0]], a[lay.block[2][0]], \
        a[lay.block[3][0]], a[lay.block[4][0]]
    # N: bg 100, node1 10 (seen via 3's closure), node3 10, node2 40, node4 0
    assert np.isclose(a1 * 10, a3 * 10) and np.isclose(a1 * 10, a2 * 40)
    assert np.isclose(a2 * 40, a_bg * 100)                   # equal N_b * alpha_b
    assert a3 > a2 > a_bg                                     # children first
    assert np.isclose(a4, a_bg)                               # unseen block = neutral
    assert np.array_equal(a[lay.block[1]], [a1, a1])         # tied within a block


def test_equalized_alpha_empty_histogram_is_uniform():
    lay = _lay()
    assert np.allclose(equalized_alpha(lay, {}, 0.3), 0.3)


def test_set_alpha_policy_reaches_initialize_global_and_turns_on_optimization():
    lay = _lay()
    m = GatedOnlineLDA(lay, vocab_size=6, alpha=0.5, random_seed=0)
    assert m.optimize_alpha is False
    m.set_alpha_policy(_HIST, optimize=True, init="equalized")
    assert m.optimize_alpha is True and m._frontier_histogram == _HIST
    gp = m.initialize_global(None)
    assert np.allclose(gp["alpha"], equalized_alpha(lay, _HIST, 0.5))
    # uniform keeps the constructor alpha
    m2 = GatedOnlineLDA(lay, vocab_size=6, alpha=0.5, random_seed=0)
    m2.set_alpha_policy(_HIST, optimize=False, init="uniform")
    assert np.allclose(m2.initialize_global(None)["alpha"], 0.5)


def test_set_alpha_policy_rejects_unknown_init():
    import pytest
    m = GatedOnlineLDA(_lay(), vocab_size=6, alpha=0.5, random_seed=0)
    with pytest.raises(ValueError, match="alpha init"):
        m.set_alpha_policy(_HIST, optimize=True, init="tilted")


def test_optimized_alpha_runs_one_local_global_cycle_after_policy():
    """The gated Newton step consumes the node-theta stat the E-step emits only
    when optimize_alpha is on; setting it post hoc must keep the two in sync."""
    from spark_vi.models.topic.types import GatedBOWDocument
    lay = _lay()
    m = GatedOnlineLDA(lay, vocab_size=6, alpha=0.5, random_seed=0)
    m.set_alpha_policy(_HIST, optimize=True, init="equalized")
    gp = m.initialize_global(None)
    docs = [GatedBOWDocument(indices=np.array([0, 3], np.int32),
                             counts=np.array([2.0, 1.0]), length=3,
                             frontier=frozenset({3})),
            GatedBOWDocument(indices=np.array([1, 4], np.int32),
                             counts=np.array([1.0, 1.0]), length=2,
                             frontier=frozenset())]
    stats = m.local_update(docs, gp)
    assert "e_log_theta_node_sum" in stats
    new_gp = m.update_global(gp, stats, 0.5)
    assert new_gp["alpha"].shape == (lay.K,) and np.all(new_gp["alpha"] > 0)


def test_pc_estimator_alpha_init_param_defaults_uniform():
    from spark_vi.mllib.topic.pc import OnlinePCLDAEstimator
    est = OnlinePCLDAEstimator(numLabels=3, weightY=0.0,
                               gateParent=json.dumps({"1": 0, "2": 0}),
                               gateNBg=2, gateTpn=1)
    assert est.getOrDefault("alphaInit") == "uniform"
    est2 = OnlinePCLDAEstimator(numLabels=3, weightY=0.0, alphaInit="equalized",
                                gateParent=json.dumps({"1": 0, "2": 0}),
                                gateNBg=2, gateTpn=1)
    assert est2.getOrDefault("alphaInit") == "equalized"
