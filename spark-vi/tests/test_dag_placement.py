import numpy as np
from spark_vi.models.topic.dag_placement import DagLayout, frontier_from_coded, fit_gated, profile

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

def test_daglayout_depth_cycle_guarded():
    # DagLayout is the domain-agnostic public entry: a malformed cyclic parent map must
    # not recurse forever. The guard yields a finite depth rather than hanging.
    lay = DagLayout({1: 0, 2: [1, 3], 3: [2]})   # 2<->3 cycle
    assert lay.depth(1) == 1
    assert lay.depth(2) < float("inf") and lay.depth(3) < float("inf")

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

def test_daglayout_path_cousins_widens_contrast_set():
    # path_cousins = closure+siblings PLUS the siblings of every ancestor on the root-path.
    lay = DagLayout(PARENT, n_bg=2, tpn=1)
    sib = set(int(k) for k in lay.allowed_with_siblings(3))
    pc = set(int(k) for k in lay.allowed_with_path_cousins(3))
    assert sib <= pc                                    # superset of closure+siblings
    # node 3's ancestor 1 has sibling 2 (the "aunt"); path_cousins adds block[2], siblings does not
    assert set(lay.block[2]) <= pc
    assert not set(lay.block[2]).issubset(sib)
    assert 0 in pc and 1 in pc and len(pc) <= lay.K     # bg included, still bounded < dense K
    # multi-parent DAG: does not crash, stays a superset of the siblings support
    d = DagLayout(DIAMOND, n_bg=2, tpn=1)
    ds = set(int(k) for k in d.allowed_with_siblings(4))
    dp = set(int(k) for k in d.allowed_with_path_cousins(4))
    assert ds <= dp and len(dp) <= d.K
    # +kids adds v's OWN children's blocks (node 1 has children 3,4); superset of path_cousins
    pck = set(int(k) for k in lay.allowed_with_path_cousins_kids(1))
    assert set(int(k) for k in lay.allowed_with_path_cousins(1)) <= pck
    assert set(lay.block[3]) <= pck and set(lay.block[4]) <= pck   # node 1's children
    assert len(pck) <= lay.K


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

def test_dag_placement_corpus_multi_shapes():
    from tests._stm_synth import dag_placement_corpus_multi

    docs, labels, node_codes = dag_placement_corpus_multi(
        parent=DIAMOND, leaf_prev={4: .5, 5: .5}, comorbid_rate=0.3,
        V=120, doc_len=48, seed=0)
    assert len(docs) == len(labels)
    assert all(isinstance(f, frozenset) and len(f) >= 1 for f in labels)
    assert set(node_codes.keys()) == set(DIAMOND.keys())
    assert any(len(f) > 1 for f in labels)                  # some comorbid patients exist


def test_daglayout_descendants_is_proper_and_mirrors_closure():
    from spark_vi.models.topic.dag_placement import DagLayout
    # DAG: 1 -> 2 -> 4, 1 -> 3, and 4 also a child of 3 (multi-parent diamond)
    lay = DagLayout({2: 1, 3: 1, 4: 2}, n_bg=1, tpn=1)
    lay.parents.setdefault(4, [])
    if 3 not in lay.parents[4]:
        lay.parents[4].append(3)          # 4 has parents {2,3}
    lay.children.setdefault(3, [])
    if 4 not in lay.children[3]:
        lay.children[3].append(4)
    # node 1 (anchor) has every other node as a descendant
    assert set(lay.descendants(1)) == {2, 3, 4}
    # node 4 (leaf) has none
    assert lay.descendants(4) == []
    # descendants excludes u itself and is disjoint from proper ancestors
    assert 2 not in lay.descendants(2)
    assert set(lay.descendants(2)) == {4}
    # sorted by (depth, id)
    d = lay.descendants(1)
    assert d == sorted(d, key=lambda x: (lay.depth(x), x))


def test_allowed_with_siblings_adds_sibling_blocks():
    """Localized-head support: allowed(v) + siblings' blocks (the closure contrast
    set). Layout {1:0,2:0,3:1}, n_bg=2, tpn=1 -> block[1]=[2],block[2]=[3],block[3]=[4]."""
    from spark_vi.models.topic.dag_placement import DagLayout
    lay = DagLayout({1: 0, 2: 0, 3: 1}, n_bg=2, tpn=1)
    def s(v):
        return set(int(k) for k in lay.allowed_with_siblings(v))
    # node 1: allowed {0,1,2} + sibling 2's block {3}  (both are root's children)
    assert s(1) == {0, 1, 2, 3}
    assert s(2) == {0, 1, 2, 3}
    # node 3: parent is 1, which has only child 3 -> no siblings -> == allowed(3)
    assert s(3) == {0, 1, 2, 4} == set(int(k) for k in lay.allowed(3))
    # root: no parents, no siblings -> background only
    assert s(0) == {0, 1}
    # siblings only ADD (superset of allowed), and stay within [0, K)
    for v in (0, 1, 2, 3):
        assert set(int(k) for k in lay.allowed(v)) <= s(v)
        assert max(s(v)) < lay.K


def test_cost_report_flags_support_and_dense_vs_localized():
    """Pre-flight cost profile: support sizes, fan-out, and dense-vs-localized head
    matrix costs. A deep bounded-fan-out tree localizes well (support << K); a flat
    high-fan-out DAG does not (a node's siblings ~ K, so localized ~ dense)."""
    from spark_vi.models.topic.dag_placement import DagLayout
    # deep, bounded fan-out: root -> A,B ; A -> A1,A2 ; B -> B1,B2  (each node few sibs)
    deep = {1: 0, 2: 0, 3: 1, 4: 1, 5: 2, 6: 2}
    lay = DagLayout(deep, n_bg=2, tpn=1)                    # K = 2 + 6 = 8
    d, s = lay.cost_report(C=7, vocab_size=100, localized=True)
    assert d["C"] == 7 and d["K"] == 8
    assert d["support_max"] <= d["K"]                       # never exceeds K
    assert d["localized_head_mem_bytes"] <= d["dense_head_mem_bytes"]
    # collected = the padded (C, S, S) actually emitted: packed floor <= collected
    # <= dense wall (this is the size that hits driver.maxResultSize).
    assert d["collected_head_mem_bytes"] == 7 * d["support_max"] ** 2 * 8
    assert (d["localized_head_mem_bytes"] <= d["collected_head_mem_bytes"]
            <= d["dense_head_mem_bytes"])
    assert d["lambda_bytes"] == 8 * 100 * 8                 # K*V*8
    assert "SIZE/COST PROFILE" in s and "support/node" in s and "collected" in s

    # flat high-fan-out: 6 nodes all under root -> each sees ~5 siblings ~ K.
    flat = {c: 0 for c in range(1, 7)}
    lay2 = DagLayout(flat, n_bg=2, tpn=1)                   # K = 8
    d2, _ = lay2.cost_report(C=7, localized=True)
    assert d2["fanout_max"] == 6
    # a root child's support = bg + own + 5 siblings = 2+1+5 = 8 = K (localization
    # gives nothing on a flat DAG — the report makes that visible).
    assert d2["support_max"] == d2["K"]
    assert d2["lambda_bytes"] is None                       # vocab_size omitted
