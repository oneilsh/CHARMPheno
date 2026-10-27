"""Alpha policy for the gated engine: `GatedOnlineLDA.set_alpha_policy`, the
post-construction seam the MLlib estimator uses to turn on the per-node tied
empirical-Bayes alpha (the frontier histogram exists only once the document RDD
does). The equalized initial alpha (exp 0121) was removed by ADR 0050. Spark-free.
"""
import numpy as np

from spark_vi.models.topic.dag_placement import DagLayout
from spark_vi.models.topic.gated_lda import GatedOnlineLDA


def _lay():
    # root 0; nodes 1, 2 under root; 3 under 1; 4 under 2 but NO training docs
    return DagLayout({1: 0, 2: 0, 3: 1, 4: 2}, n_bg=2, tpn=2)


_HIST = {frozenset({3}): 10, frozenset({2}): 40, frozenset(): 50}   # 100 docs


def test_set_alpha_policy_turns_on_optimization_and_keeps_the_constructor_alpha():
    lay = _lay()
    m = GatedOnlineLDA(lay, vocab_size=6, alpha=0.5, random_seed=0)
    assert m.optimize_alpha is False
    m.set_alpha_policy(_HIST, optimize=True)
    assert m.optimize_alpha is True and m._frontier_histogram == _HIST
    assert np.allclose(m.initialize_global(None)["alpha"], 0.5)
    m2 = GatedOnlineLDA(lay, vocab_size=6, alpha=0.5, random_seed=0)
    m2.set_alpha_policy(_HIST, optimize=False)
    assert m2.optimize_alpha is False
    assert np.allclose(m2.initialize_global(None)["alpha"], 0.5)


def test_optimized_alpha_runs_one_local_global_cycle_after_policy():
    """The gated Newton step consumes the node-theta stat the E-step emits only
    when optimize_alpha is on; setting it post hoc must keep the two in sync."""
    from spark_vi.models.topic.types import GatedBOWDocument
    lay = _lay()
    m = GatedOnlineLDA(lay, vocab_size=6, alpha=0.5, random_seed=0)
    m.set_alpha_policy(_HIST, optimize=True)
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
