"""analysis/cloud/starvation.py: the flat-topic measure and the offline report."""
import numpy as np


def test_support_fraction_flags_a_flat_row_and_keeps_a_sharp_one():
    from starvation import flat_topics, support_fraction
    V = 100
    sharp = np.full(V, 0.01); sharp[:3] = 50.0
    flat = np.full(V, 1.0)
    lam = {0: np.stack([sharp, sharp, flat]), 1: np.full((3, 10), 0.01)}
    frac, mass = support_fraction(lam)
    assert frac[0] < 0.1 and frac[2] > 0.99
    np.testing.assert_allclose(mass[2], 100 + 0.1)
    n_flat, n_node, share = flat_topics(lam, n_bg=1)
    assert (n_flat, n_node) == (1, 2)
    assert 0 < share < 1


def test_support_fraction_uses_the_dominant_domain():
    from starvation import support_fraction
    a = np.full((1, 50), 1.0)                   # flat but light
    b = np.zeros((1, 10)); b[0, 0] = 500.0      # sharp and heavy: dominant
    frac, _ = support_fraction({0: a, 1: b})
    assert frac[0] < 0.2


def _fitted_run(tmp_path, name, iters):
    from spark_vi.core import LocalRunner, VIConfig
    from spark_vi.dag.layout import DagLayout
    from spark_vi.dag.nodeset import fit_node_set
    from spark_vi.io.export import save_result
    from spark_vi.models.topic.dag_gated_lda import DagGatedOnlineLDA
    from spark_vi.models.topic.types import GatedBOWDocument
    dag = {"a": None, "b": "a", "c": "a", "d": "b"}
    rng = np.random.default_rng(0)
    labels = [[["b"], ["c"], ["d"], []][i % 4] for i in range(80)]
    ns = fit_node_set(dag, labels)
    lay = DagLayout(ns.engine_parents, n_bg=2, tpn=1)
    m = DagGatedOnlineLDA(lay, 30, random_seed=0)
    m.extra_metadata = {"dag_node_set": {**ns.to_dict(), "n_bg": 2, "tpn": 1}}
    docs = []
    for lab in labels:
        idx, cnt = np.unique(rng.integers(0, 30, size=10), return_counts=True)
        docs.append(GatedBOWDocument(idx.astype(np.int32), cnt.astype(float),
                                     int(cnt.sum()), frontier=ns.frontier(lab)))
    res = LocalRunner(m, VIConfig(max_iterations=iters, convergence_tol=1e-12)).fit(docs)
    run = tmp_path / name
    save_result(res, str(run / "model" / "result"))
    return run


def test_report_reads_saved_models_and_prints_counts_only(tmp_path):
    from starvation import load_run, node_table, report
    r1, r2 = _fitted_run(tmp_path, "r1", 2), _fitted_run(tmp_path, "r2", 4)
    lam, ns, n_bg, tpn = load_run(r1)
    tab = node_table(lam, ns, n_bg, tpn)
    assert tab["starved"].size == len(ns.nodes)
    assert set(tab) >= {"support", "parent_share", "n_kids", "child_share", "depth"}
    out = report([r1, r2])
    assert "## r1:" in out and "## r2:" in out and "in the earlier runs" in out
    assert "by documents reaching the node" in out
