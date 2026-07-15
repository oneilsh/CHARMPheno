import numpy as np
from spark_vi.models.topic.dag_placement import DagLayout, label_from_coded, frontier_from_coded, strip_dag_node_codes, fit_gated, profile, evaluate

PARENT = {1: 0, 2: 0, 3: 1, 4: 1, 5: 2, 6: 2}   # root 0 -> families 1,2 -> subtypes
DIAMOND = {1: 0, 2: 0, 3: 0, 4: [1, 2], 5: [1, 3]}  # multi-parent DAG: node 4,5 have two parents

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

def test_label_from_coded_empty_is_root():
    assert label_from_coded([], DagLayout(PARENT)) == 0

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

def test_evaluate_tolerates_root_label():
    lay = DagLayout(PARENT, n_bg=2, tpn=1)
    labels = np.array([3, 0, 4, 5])
    profiles = []
    for y in labels:
        if y == 0:
            profiles.append({u: 0.0 for u in lay.nodes})   # root: no informative affinity
        else:
            cl = [u for u in lay.closure(y) if u != 0]
            profiles.append({u: (1.0 if u in cl else 0.0) for u in lay.nodes})
    m = evaluate(profiles, labels, lay)                    # must not raise on the root label
    assert np.isfinite(m["mrr"])                            # 3 of 4 items are rankable

def test_identifiability_flags_near_identical_siblings():
    from spark_vi.models.topic.dag_placement import identifiability_annotation
    lay = DagLayout(PARENT, n_bg=2, tpn=1)
    beta = np.random.default_rng(0).random((lay.K, 20)) + 0.01
    # make siblings 3 and 4 near-identical topics
    beta[lay.block[4][0]] = beta[lay.block[3][0]].copy()
    beta /= beta.sum(1, keepdims=True)
    flagged = identifiability_annotation(beta, lay, tol=0.99)
    pairs = {(min(u, v), max(u, v)) for u, v, _ in flagged}
    assert (3, 4) in pairs
    assert (3, 5) not in pairs                    # cross-branch never reported

def test_render_profile_marks_true_and_shows_all_nodes():
    from spark_vi.models.topic.dag_placement import render_profile
    lay = DagLayout(PARENT, n_bg=2, tpn=1)
    aff = {1: 0.6, 2: 0.0, 3: 0.05, 4: 0.0, 5: 0.2, 6: 0.15}
    s = render_profile(aff, lay, true_node=1)
    assert "true" in s
    for u in lay.nodes:                                   # every node rendered
        assert str(u) in s
    assert s.count("\n") >= len(lay.nodes)

def test_end_to_end_recovers_family_and_subtype():
    from tests._stm_synth import dag_placement_corpus
    docs, labels, node_codes = dag_placement_corpus(
        parent=PARENT, node_prev={1:.18,2:.18,3:.16,4:.16,5:.16,6:.16},
        V=120, doc_len=40, seed=2)
    lay = DagLayout(PARENT, n_bg=2, tpn=1)
    ntr = int(0.7 * len(docs))
    rng = np.random.default_rng(5)
    beta = fit_gated(docs[:ntr], labels[:ntr], lay, 120, n_iter=80, burn=40, rng=rng)
    codes = set(node_codes.values())
    profs = [profile(strip_dag_node_codes(d, codes), beta, lay, n_iter=40, burn=20, rng=rng)
             for d in docs[ntr:]]
    m = evaluate(profs, labels[ntr:], lay)
    # gated-train places well; sim validated family ~0.99 / subtype ~0.97 (spec). Loose floors:
    assert m["auc_by_depth"][1] >= 0.85           # family level
    assert m["auc_by_depth"][2] >= 0.75           # subtype level
    assert m["mrr"] >= 0.6

def test_daglayout_multiparent_diamond():
    lay = DagLayout(DIAMOND, n_bg=2, tpn=1)
    assert lay.parents[4] == [1, 2] and lay.parents[5] == [1, 3]
    assert lay.closure(4) == [0, 1, 2, 4]          # all ancestors, depth-sorted, root first
    assert lay.closure(5) == [0, 1, 3, 5]
    assert lay.depth(4) == 2 and lay.depth(1) == 1 and lay.depth(0) == 0   # longest path
    assert lay.subtree(1) == {1, 4, 5} and lay.subtree(2) == {2, 4}
    want = {0, 1} | set(lay.block[1]) | set(lay.block[2]) | set(lay.block[3]) \
        | set(lay.block[4]) | set(lay.block[5])
    assert set(lay.allowed_set({4, 5}).tolist()) == want   # union of closures over the frontier

def test_daglayout_singleparent_backward_compat():
    lay = DagLayout(PARENT, n_bg=2, tpn=1)               # scalar-parent map still works
    assert lay.closure(3) == [0, 1, 3]                   # exact old list ordering
    assert list(lay.allowed(3)) == [0, 1] + lay.block[1] + lay.block[3]
    assert list(lay.allowed(1)) == [0, 1] + lay.block[1]
    assert lay.depth(3) == 2 and lay.depth(1) == 1
    assert lay.subtree(1) == {1, 3, 4}

def test_frontier_from_coded_cases():
    lay = DagLayout(DIAMOND)
    assert frontier_from_coded([1, 4], lay) == frozenset({4})       # same-path -> most-specific
    assert frontier_from_coded([4, 5], lay) == frozenset({4, 5})    # comorbid incomparable -> set
    assert frontier_from_coded([2, 3], lay) == frozenset({2, 3})    # contradictory siblings -> set
    assert frontier_from_coded([1, 2, 4], lay) == frozenset({4})    # both parents + child -> child
    # single-parent tree: ancestor+descendant collapses to the descendant
    assert frontier_from_coded([1, 3], DagLayout(PARENT)) == frozenset({3})

def test_fit_gated_accepts_frontier_sets():
    from tests._stm_synth import dag_placement_corpus
    # comorbid training labels (sets) must be accepted and produce a valid beta_hat
    docs, labels, _ = dag_placement_corpus(
        parent=PARENT, node_prev={1:.18,2:.18,3:.16,4:.16,5:.16,6:.16},
        V=120, doc_len=40, seed=1)
    lay = DagLayout(PARENT, n_bg=2, tpn=1)
    set_labels = [frozenset({int(y)}) for y in labels[:800]]   # scalars as singleton sets
    rng = np.random.default_rng(3)
    beta = fit_gated(docs[:800], set_labels, lay, 120, n_iter=40, burn=20, rng=rng)
    assert beta.shape == (lay.K, 120)
    assert np.allclose(beta.sum(1), 1.0, atol=1e-6)
