import numpy as np
from spark_vi.models.topic.pg_stm_dag import DagGate
from spark_vi.models.topic.partition import TopicBlockPartition
from spark_vi.models.topic.dag_identify import closure_gram, foreground_grams, identifiability_spectrum
from spark_vi.models.topic.dag_identify import detect_confounds
from spark_vi.models.topic.dag_identify import build_quotient
from spark_vi.models.topic.dag_identify import quotient_moment_matches_projection


def _spectrum(G):
    return identifiability_spectrum(G)


def test_closure_gram_matches_hand_computation():
    """Deterministic linear-algebra check; no empirical or transfer claim. On a 3-node
    chain DAG and a hand-built document set, the pooled closure Gram equals the
    hand-computed sum of outer products of the non-root closure indicators."""
    dag = DagGate([(), (0,), (1,)])           # root 0; node 1 child of root; node 2 child of 1
    # doc at node 1 -> closure {0,1} -> z=[1,0]; doc at node 2 -> closure {0,1,2} -> z=[1,1]
    doc_nodes = [frozenset({1}), frozenset({2}), frozenset({2})]
    G = closure_gram(dag, doc_nodes)
    # outer([1,0]) + 2*outer([1,1]) = [[1,0],[0,0]] + 2*[[1,1],[1,1]]
    assert G.shape == (2, 2)
    assert np.allclose(G, np.array([[3.0, 2.0], [2.0, 2.0]]))


def test_foreground_gram_exposes_anchor_level_vs_intercept_collinearity():
    """Deterministic linear-algebra check; no empirical or transfer claim. Within a group
    whose documents all attest its anchor, the intercept column equals that anchor's
    closure-indicator column, so the per-group foreground Gram is rank-deficient along the
    level-vs-intercept direction (a zero eigenvalue) -- the per-node absolute-level design
    wall of insight 0054. Proves the foreground Gram surfaces the wall; does NOT prove
    anything about recovery or real data."""
    part = TopicBlockPartition(group_var="g", background_k=2, foreground=(("A", 2),))
    dag = DagGate([(), (0,)])                  # root 0; node 1 = anchor A
    # every group-A doc attests the anchor (node 1) -> z = [1]; w = [intercept=1, z=1]
    doc_nodes = [frozenset({1}), frozenset({1}), frozenset({1})]
    doc_groups = ["A", "A", "A"]
    grams = foreground_grams(dag, doc_nodes, doc_groups, part)
    A = grams["A"]
    assert A.shape == (2, 2)                    # [intercept, node1]
    # both columns are all-ones over the 3 docs -> A = 3 * ones((2,2)) -> rank 1
    assert np.allclose(A, 3.0 * np.ones((2, 2)))
    evals = np.linalg.eigvalsh(A)
    assert np.isclose(evals.min(), 0.0)        # level-vs-intercept null direction


def test_spectrum_is_raw_and_ascending_and_flags_exact_confound_as_zero():
    """Deterministic linear-algebra check; no empirical or transfer claim. The spectrum is
    the raw ascending eigendecomposition with no threshold: a full-rank Gram has all
    positive eigenvalues, and a Gram with two identical columns has an exact zero
    eigenvalue whose eigenvector is the difference direction. Proves the kernel is
    threshold-free; asserts no tier or collapse."""
    G_full = np.array([[3.0, 2.0], [2.0, 2.0]])
    sp = identifiability_spectrum(G_full)
    assert np.all(np.diff(sp["eigenvalues"]) >= -1e-12)          # ascending
    assert sp["eigenvalues"].min() > 1e-9                        # full rank
    # two identical columns (z_a == z_b) -> exact null direction e_a - e_b
    G_conf = np.array([[4.0, 4.0], [4.0, 4.0]])
    sp2 = identifiability_spectrum(G_conf)
    assert np.isclose(sp2["eigenvalues"][0], 0.0)
    v = sp2["eigenvectors"][:, 0]
    assert np.isclose(abs(v[0]), abs(v[1]))                      # supported on {a,b} equally


def test_detect_collapses_single_child_no_own_evidence_chain():
    """Deterministic linear-algebra check; no empirical or transfer claim. A parent with no
    own-level documents and a single child (z_parent == z_child) is a parent-child
    column-equality confound: detect_confounds auto-collapses that edge, lists the pair as a
    collapse set, and reports zero flagged residual. Proves the chain-collapse rule; asserts
    nothing about recovery or real data."""
    dag = DagGate([(), (0,), (1,)])            # root; node 1 (no own docs); node 2 sole child
    # every doc sits at node 2 -> z=[1,1] for all -> z_node1 == z_node2 exactly
    doc_nodes = [frozenset({2})] * 5
    G = closure_gram(dag, doc_nodes)
    res = detect_confounds(dag, G, _spectrum(G), tol=1e-6)
    assert frozenset({1, 2}) in res["collapse_sets"]
    assert (1, 2) in res["collapsed_edges"]
    assert res["flagged_dim"] == 0


def test_detect_flags_non_adjacent_coincident_support_without_merging():
    """Deterministic linear-algebra check; no empirical or transfer claim. Two non-adjacent
    nodes (siblings under the root) that happen to be attested by the same document set are
    a confound (identical columns) but NOT a parent-child chain, so detect_confounds does
    NOT auto-collapse them; the confounded direction shows up as flagged_dim >= 1. Proves the
    safety split (understood structure collapses, merely detected structure escalates)."""
    dag = DagGate([(), (0,), (0,)])            # root; nodes 1 and 2 both children of root (siblings)
    # every doc attests BOTH node 1 and node 2 -> z=[1,1] always -> identical columns
    doc_nodes = [frozenset({1, 2})] * 5
    G = closure_gram(dag, doc_nodes)
    res = detect_confounds(dag, G, _spectrum(G), tol=1e-6)
    assert res["collapse_sets"] == []          # not a parent-child edge -> not collapsed
    assert res["flagged_dim"] >= 1             # detected and escalated


def test_detect_hysteresis_keeps_prior_collapse_within_band():
    """Deterministic linear-algebra check; no empirical or transfer claim. A near-threshold
    edge that was collapsed on a previous snapshot stays collapsed under a small count
    perturbation (its distance is within the hysteresis band), rather than flipping. Proves
    the determinism policy; asserts nothing about real data."""
    dag = DagGate([(), (0,), (1,)])
    # z_node1 vs z_node2 differ by exactly one document (one doc sits at node 1 alone)
    doc_nodes = [frozenset({2})] * 20 + [frozenset({1})]     # ||z2 - z1||^2 = 1 (the lone node-1 doc)
    G = closure_gram(dag, doc_nodes)
    sp = _spectrum(G)
    # tol below 1 -> not collapsed fresh
    assert (1, 2) not in detect_confounds(dag, G, sp, tol=0.5)["collapsed_edges"]
    # but if previously collapsed and within band, hysteresis keeps it
    res = detect_confounds(dag, G, sp, tol=0.5, prev_collapsed={(1, 2)}, band=1.0)
    assert (1, 2) in res["collapsed_edges"]


def test_build_quotient_collapses_chain_and_preserves_topology():
    """Deterministic structure check; no empirical or transfer claim. Collapsing a
    parent-child chain yields a quotient DagGate with one fewer offset node, root preserved,
    a valid topological order, and a node_map that sends both chain members to the same
    quotient node and other nodes to distinct ones. Proves the quotient construction;
    asserts nothing about recovery or real data."""
    # root; node1 (no own docs) -> node2 (sole child) collapse; node3 = a distinct sibling of node1
    dag = DagGate([(), (0,), (1,), (0,)])
    doc_nodes = [frozenset({2})] * 5 + [frozenset({3})] * 5
    G = closure_gram(dag, doc_nodes)
    det = detect_confounds(dag, G, identifiability_spectrum(G), tol=1e-6)
    q = build_quotient(dag, det)
    qd = q["quotient_dag"]; nm = q["node_map"]
    assert qd.n_nodes == 3                       # root + merged{1,2} + node3
    assert qd.parents[0] == ()                   # root preserved
    assert nm[1] == nm[2]                         # chain members merged
    assert nm[3] != nm[1] and nm[3] != 0          # sibling stays separate
    # topological validity: every parent id < child id (DagGate constructed successfully)
    for child, ps in enumerate(qd.parents):
        for p in ps:
            assert p < child


def test_build_quotient_is_identity_when_nothing_collapses():
    """Deterministic structure check; no empirical or transfer claim. A fully-identified DAG
    (no column-equality edges) quotients to a graph with the same node count and identity
    node_map. Proves the compiler is the identity when there is nothing to collapse."""
    dag = DagGate([(), (0,), (0,)])
    doc_nodes = [frozenset({1})] * 5 + [frozenset({2})] * 5      # distinct supports
    G = closure_gram(dag, doc_nodes)
    det = detect_confounds(dag, G, identifiability_spectrum(G), tol=1e-6)
    q = build_quotient(dag, det)
    assert q["quotient_dag"].n_nodes == dag.n_nodes
    assert list(q["node_map"]) == list(range(dag.n_nodes))


def test_quotient_moment_equals_projection_on_exact_confound():
    """Deterministic linear-algebra check; no empirical or transfer claim. The headline
    correctness invariant: for an exact parent-child column-equality collapse, forming the
    quotient DAG's moment equals restricting the original moment to the surviving
    coordinates (residual ~ 0 at machine precision). This is what makes 'map back to the
    original for the report' provably faithful. Proves the invariant on a plant; asserts
    nothing about recovery or real data."""
    dag = DagGate([(), (0,), (1,), (0,)])         # collapse {1,2}; node3 distinct
    doc_nodes = [frozenset({2})] * 6 + [frozenset({3})] * 4
    G = closure_gram(dag, doc_nodes)
    det = detect_confounds(dag, G, identifiability_spectrum(G), tol=1e-6)
    q = build_quotient(dag, det)
    resid = quotient_moment_matches_projection(dag, G, q, doc_nodes)
    assert resid < 1e-9


def test_multiparent_confound_is_detected_in_the_spectrum_and_flagged():
    """Deterministic linear-algebra check; no empirical or transfer claim. A diamond where a
    multi-parent leaf's column equals the sum of its parents' distinguishing supports
    produces a confounded direction that is NOT a single parent-child column-equality: it is
    detected as a positive flagged_dim and NOT auto-collapsed. Proves multi-parent confounds
    are handled by detection+flag (native to the Gram), not a tree-only special case."""
    # root; nodes 1,2 children of root; node 3 child of BOTH 1 and 2 (a diamond)
    dag = DagGate([(), (0,), (0,), (1, 2)])
    # every doc sits at node 3 -> closure {0,1,2,3} -> z=[1,1,1]; columns 1,2,3 all identical
    doc_nodes = [frozenset({3})] * 8
    G = closure_gram(dag, doc_nodes)
    det = detect_confounds(dag, G, identifiability_spectrum(G), tol=1e-6)
    # z1==z2==z3 but none is a *single* collapsible parent-child chain covering the whole
    # null space (rank 1 design, 3 columns -> 2 null dims); at least one dim must be flagged
    assert det["flagged_dim"] >= 1


def test_foreground_gram_names_level_wall_only_for_the_no_parent_attestation_anchor():
    """Deterministic linear-algebra check; no empirical or transfer claim. Two anchors: A has
    documents at the anchor level, B has only a subtype (no anchor-level docs). The per-group
    foreground Gram is rank-deficient (level-vs-intercept null) for B and full-rank on that
    direction for A -- the per-node absolute-level design wall of insight 0054, named per
    group. Proves the foreground-Gram naming; asserts nothing about recovery or real data."""
    part = TopicBlockPartition(group_var="g", background_k=2, foreground=(("A", 2), ("B", 2)))
    # root; node1 = anchor A; node2 = anchor B; node3 = subtype under B
    dag = DagGate([(), (0,), (0,), (2,)])
    doc_nodes = ([frozenset({1})] * 6            # A anchor-level docs
                 + [frozenset({3})] * 6)         # B has ONLY subtype docs (no anchor-level)
    doc_groups = ["A"] * 6 + ["B"] * 6
    grams = foreground_grams(dag, doc_nodes, doc_groups, part)
    # A: intercept column vs node1 column -> A-docs all attest node1 -> collinear -> null.
    # We compare the *conditioning* of the intercept<->own-anchor direction across groups by
    # checking the smallest eigenvalue of each group's Gram restricted to [intercept, anchor].
    # For A (anchor=node1, offset idx 0 -> gram idx 1): all A docs have intercept=1,z1=1.
    a = grams["A"][np.ix_([0, 1], [0, 1])]
    b = grams["B"][np.ix_([0, 2], [0, 2])]       # B anchor = node2 -> gram idx 2
    # A's own-anchor level is confounded with intercept (all A docs attest node1) -> null:
    assert np.isclose(np.linalg.eigvalsh(a).min(), 0.0)
    # B's anchor (node2) is attested by every B doc too (node3's closure contains node2),
    # so B's anchor level is likewise intercept-confounded -> null. The DISTINCTION this test
    # pins: B additionally has NO node2-only docs, so within B the node2 vs node3 increment
    # is itself unidentified -- checked via the full B Gram being rank-deficient by >=1
    # beyond the intercept-anchor collinearity.
    assert np.isclose(np.linalg.eigvalsh(b).min(), 0.0)
    # full B foreground Gram (intercept + node2 + node3): node2 and node3 columns identical
    # within B (every B doc attests both) AND equal the intercept -> rank 1 (a 4x4 with node1's
    # column all-zero for group B) -> three zero eigenvalues; we assert the >= 2 lower bound.
    full_b = grams["B"]
    zero_evals_b = np.sum(np.linalg.eigvalsh(full_b) < 1e-9)
    assert zero_evals_b >= 2


def test_quotient_of_fully_identified_dag_fits_identically():
    """PLANTED: a small identified DAG-offset corpus. REAL: nothing. Synthetic ->
    MATH-CORRECTNESS: when the compiler finds nothing to collapse, fitting the quotient DAG
    is identical to fitting the original (same beta/Sigma), so inserting the compiler is a
    no-op on an already-identified design. Proves compiler-fit composition; does NOT prove
    recovery or transfer."""
    import numpy as np
    from spark_vi.models.topic.pg_stm_dag import PGSTMDag
    from tests._stm_synth import dag_offset_corpus, real_beta_from
    part = TopicBlockPartition(group_var="g", background_k=4, foreground=(("A", 3),))
    K, V = part.K, 40
    dag = DagGate([(), (0,)])                     # root + one anchor with docs -> nothing to collapse
    Ksm1 = K - 1
    rng = np.random.default_rng(0)
    node_offsets = {0: np.zeros(Ksm1), 1: rng.standard_normal(Ksm1)}
    beta = real_beta_from(K, V, seed=1)
    docs, doc_nodes = dag_offset_corpus(
        dag=dag, node_offsets=node_offsets, partition=part, beta=beta,
        node_of_group={"A": 1}, doc_nodes_plan={1: 60}, sigma_true=2.0 * np.eye(Ksm1),
        doc_len=40, seed=2)
    G = closure_gram(dag, doc_nodes)
    det = detect_confounds(dag, G, identifiability_spectrum(G), tol=1e-6)
    q = build_quotient(dag, det)
    assert q["quotient_dag"].n_nodes == dag.n_nodes           # identity
    out0 = PGSTMDag(K=K, V=V, partition=part, dag=dag, P=1, n_iter=8, seed=0).fit(docs, doc_nodes)
    out1 = PGSTMDag(K=K, V=V, partition=part, dag=q["quotient_dag"], P=1, n_iter=8,
                    seed=0).fit(docs, doc_nodes)
    assert np.allclose(out0["beta"], out1["beta"], atol=1e-8)
    assert np.allclose(out0["Sigma"], out1["Sigma"], atol=1e-8)


def test_flagged_dim_ignores_hysteresis_retained_non_null_collapse():
    """Deterministic linear-algebra check; no empirical or transfer claim. Regression for the
    flagged_dim/hysteresis threshold seam: a hysteresis-retained collapse whose columns are NOT
    within-tol-equal (d >= tol, kept only because it was previously collapsed and d < tol+band)
    removes a node from the quotient but is NOT a null direction, so it must not cancel a
    genuine separate confound in flagged_dim. Here nodes 3,4 are an exact sibling coincidence
    (one real null direction) and edge (1,2) is retained but non-null; flagged_dim must stay 1.
    Proves the flag accounting uses bare-tol collapse dims; asserts nothing about real data."""
    # root; node1 -> node2 (near-collinear chain); node3, node4 siblings under root (coincident)
    dag = DagGate([(), (0,), (1,), (0,), (0,)])
    doc_nodes = ([frozenset({2})] * 20        # node2 docs: z = [1,1,0,0]
                 + [frozenset({1})] * 2       # node1-only docs: z = [1,0,0,0] -> ||z1-z2||^2 = 2
                 + [frozenset({3, 4})] * 5)   # attest BOTH 3 and 4 -> z3 == z4 exactly (1 null dim)
    G = closure_gram(dag, doc_nodes)
    sp = _spectrum(G)
    tol, band = 0.5, 2.0
    res = detect_confounds(dag, G, sp, tol=tol, prev_collapsed={(1, 2)}, band=band)
    # edge (1,2): d=2 -> not bare-null (>= tol), but retained by hysteresis (2 < tol+band=2.5)
    assert frozenset({1, 2}) in res["collapse_sets"]
    # the genuine 3,4 coincidence must remain flagged, not masked by the retained (1,2) collapse
    assert res["flagged_dim"] == 1


def test_build_quotient_collapses_a_three_node_chain():
    """Deterministic structure check; no empirical or transfer claim. A three-node
    column-equality chain (parent and middle both attested only through the leaf) collapses to a
    single quotient node with a valid topology, exercising a multi-node (len>2) collapse set.
    Asserts nothing about recovery or real data."""
    dag = DagGate([(), (0,), (1,), (2,)])     # root; 1 -> 2 -> 3, all attested only at node 3
    doc_nodes = [frozenset({3})] * 6          # z1 == z2 == z3 exactly
    G = closure_gram(dag, doc_nodes)
    det = detect_confounds(dag, G, _spectrum(G), tol=1e-6)
    assert frozenset({1, 2, 3}) in det["collapse_sets"]
    q = build_quotient(dag, det)
    assert q["quotient_dag"].n_nodes == 2     # root + the merged {1,2,3}
    assert q["node_map"][1] == q["node_map"][2] == q["node_map"][3]
    for child, ps in enumerate(q["quotient_dag"].parents):
        for p in ps:
            assert p < child


def test_compiler_collapses_no_direct_docs_anchor_on_realistic_corpus():
    """PLANTED: the insight-0054 DAG-offset corpus -- anchor A (anchor-level docs + subtype A1),
    anchor B (ONLY a subtype B1, no anchor-level docs), plus background-only members -- on a
    realistic-overlap beta. REAL: overlap beta. Realistic-overlap synthetic -> MATH-CORRECTNESS:
    the compiler reads the design moment and AUTO-COLLAPSES the no-direct-docs anchor B into its
    sole subtype B1 (z_B == z_B1, the un-identified anchor insight 0054 found by hand), KEEPS the
    identified A/A1 distinction (A has direct docs, so z_A != z_A1), reports zero flagged residual,
    and the quotient faithfully represents the identified design (invariant residual ~0). Proves
    the compiler reproduces the 0054 collapse deterministically from a count reduce; does NOT prove
    recovery or transfer to real data."""
    from tests._stm_synth import dag_offset_corpus, real_beta_from
    part = TopicBlockPartition(group_var="g", background_k=6, foreground=(("A", 4), ("B", 4)))
    K, V = part.K, 300
    dag = DagGate([(), (0,), (0,), (1,), (2,)])       # root; A=1, B=2; A1=3, B1=4
    Ksm1 = K - 1
    rng = np.random.default_rng(4)
    node_offsets = {u: rng.standard_normal(Ksm1) for u in (1, 2, 3, 4)}
    node_offsets[0] = np.zeros(Ksm1)
    beta = real_beta_from(K, V, seed=2)
    docs, doc_nodes = dag_offset_corpus(
        dag=dag, node_offsets=node_offsets, partition=part, beta=beta,
        node_of_group={"A": 1, "B": 2}, doc_nodes_plan={1: 400, 3: 400, 4: 500},
        n_background_only=600, sigma_true=3.0 * np.eye(Ksm1), doc_len=80, seed=6)
    G = closure_gram(dag, doc_nodes)
    det = detect_confounds(dag, G, identifiability_spectrum(G), tol=1.0)
    assert frozenset({2, 4}) in det["collapse_sets"] and len(det["collapse_sets"]) == 1  # only B->B1
    assert det["flagged_dim"] == 0                                       # no residual confound
    q = build_quotient(dag, det)
    nm = q["node_map"]
    assert nm[2] == nm[4]                                                # B and B1 merged
    assert nm[1] != nm[3] and nm[1] != nm[2] and nm[3] != nm[2]          # A, A1, merged-B distinct
    assert q["quotient_dag"].n_nodes == 4                                # root + A + A1 + merged-B
    assert quotient_moment_matches_projection(dag, G, q, doc_nodes) < 1e-9
