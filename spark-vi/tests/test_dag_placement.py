import numpy as np
from spark_vi.models.topic.dag_placement import DagLayout, label_from_coded, strip_dag_node_codes, fit_gated, profile, evaluate

PARENT = {1: 0, 2: 0, 3: 1, 4: 1, 5: 2, 6: 2}   # root 0 -> families 1,2 -> subtypes

def test_daglayout_structure_and_masks():
    lay = DagLayout(PARENT, n_bg=2, tpn=1)
    assert lay.nodes == [1, 2, 3, 4, 5, 6]
    assert lay.K == 2 + 6                      # bg + one topic per node
    assert lay.closure(3) == [0, 1, 3]         # root..v
    assert lay.subtree(1) == {1, 3, 4}
    assert lay.depth(3) == 2 and lay.depth(1) == 1
    # allowed(v) = bg ∪ blocks along closure(v), excluding root
    assert list(lay.allowed(3)) == [0, 1] + lay.block[1] + lay.block[3]
    assert list(lay.allowed(1)) == [0, 1] + lay.block[1]

def test_daglayout_tpn_two():
    lay = DagLayout(PARENT, n_bg=1, tpn=2)
    assert lay.K == 1 + 6 * 2
    assert len(lay.block[3]) == 2

def test_label_same_path_is_deepest():
    lay = DagLayout(PARENT)
    # {1,3} lie on one path root->1->3 : most-specific = deepest = 3
    assert label_from_coded([1, 3], lay) == 3
    assert label_from_coded([3], lay) == 3

def test_label_siblings_is_lca():
    lay = DagLayout(PARENT)
    # {3,4} are siblings under 1 : LCA = 1
    assert label_from_coded([3, 4], lay) == 1
    # {3,5} cross-branch under root : LCA = 0
    assert label_from_coded([3, 5], lay) == 0

def test_strip_dag_node_codes():
    doc = np.array([10, 3, 11, 1, 12])          # 3 and 1 are DAG-node codes
    out = strip_dag_node_codes(doc, {1, 3})
    assert list(out) == [10, 11, 12]

def test_dag_placement_corpus_shapes():
    from tests._stm_synth import dag_placement_corpus

    docs, labels, node_codes = dag_placement_corpus(
        parent=PARENT, node_prev={1: .18, 2: .18, 3: .16, 4: .16, 5: .16, 6: .16},
        V=120, doc_len=40, seed=0)
    assert len(docs) == len(labels)
    assert set(labels.tolist()) <= set(PARENT.keys())
    assert set(node_codes.keys()) == set(PARENT.keys())
    # a node's exact code appears in items labeled at/below that node
    below3 = [d for d, y in zip(docs, labels) if y in {3}]
    assert any(node_codes[3] in d for d in below3)

def test_fit_gated_learns_node_signatures():
    from tests._stm_synth import dag_placement_corpus
    docs, labels, _ = dag_placement_corpus(
        parent=PARENT, node_prev={1:.18,2:.18,3:.16,4:.16,5:.16,6:.16},
        V=120, doc_len=40, seed=1)
    lay = DagLayout(PARENT, n_bg=2, tpn=1)
    rng = np.random.default_rng(3)
    beta = fit_gated(docs[:1400], labels[:1400], lay, 120, n_iter=60, burn=30, rng=rng)
    assert beta.shape == (lay.K, 120)
    assert np.allclose(beta.sum(1), 1.0, atol=1e-6)

def test_profile_returns_node_affinity():
    lay = DagLayout(PARENT, n_bg=2, tpn=1)
    beta = np.full((lay.K, 30), 1e-3); beta /= beta.sum(1, keepdims=True)
    rng = np.random.default_rng(0)
    pr = profile(np.array([1, 2, 3, 4, 5]), beta, lay, n_iter=20, burn=10, rng=rng)
    assert set(pr.keys()) == set(lay.nodes)
    assert all(0.0 <= v <= 1.0 for v in pr.values())

def test_evaluate_perfect_profiles_score_high():
    lay = DagLayout(PARENT, n_bg=2, tpn=1)
    labels = np.array([3, 4, 5, 6, 1, 2] * 5)
    profiles = []
    for y in labels:                              # planted "perfect" affinity: closure-loaded
        cl = [u for u in lay.closure(y) if u != 0]   # true node + all its ancestors (not root)
        profiles.append({u: (1.0 if u in cl else 0.0) for u in lay.nodes})
    m = evaluate(profiles, labels, lay)
    assert m["mrr"] == 1.0 and m["top2"] == 1.0
    assert all(v >= 0.99 for v in m["node_auc"].values())
