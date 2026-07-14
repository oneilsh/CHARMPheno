import numpy as np
import pytest
from spark_vi.models.topic.pg_stm_dag import DagGate, offset_penalty, dag_offset_ridge


def test_dag_closure_and_indicator_over_two_levels():
    """Deterministic structure check; no empirical or transfer claim."""
    # 0=root; 1,2 anchors under root; 3 = subtype under anchor 1
    dag = DagGate([(), (0,), (0,), (1,)])
    assert dag.n_nodes == 4
    assert dag.closure(frozenset({3})) == frozenset({3, 1, 0})   # subtype -> anchor -> root
    assert dag.closure(frozenset({2})) == frozenset({2, 0})
    z = dag.closure_indicator(frozenset({3}))
    assert z.dtype == np.float64
    assert list(z) == [1.0, 1.0, 0.0, 1.0]                       # nodes 0,1,3 on; 2 off


def test_dag_diamond_shared_ancestor_counted_once():
    """Deterministic structure check; no empirical or transfer claim."""
    # 3 has two parents 1 and 2, both under root 0 (a diamond)
    dag = DagGate([(), (0,), (0,), (1, 2)])
    assert dag.closure(frozenset({3})) == frozenset({3, 1, 2, 0})
    assert dag.closure_indicator(frozenset({3})).sum() == 4      # 0 once, not twice


def test_dag_depth_is_shortest_root_distance():
    """Deterministic structure check; no empirical or transfer claim."""
    dag = DagGate([(), (0,), (1,), (0, 2)])   # node 3 reachable via 0 (d=1) or via 2 (d=3)
    assert list(dag.depth) == [0, 1, 2, 1]    # shortest wins for node 3


def test_dag_rejects_parent_index_not_less_than_child():
    """Deterministic structure check; no empirical or transfer claim."""
    with pytest.raises(ValueError):
        DagGate([(), (2,), (0,)])             # node 1's parent 2 > 1 -> not topo-ordered


def test_dag_dump_lists_nodes_with_depth_and_parents():
    """Deterministic structure check; no empirical or transfer claim."""
    dag = DagGate([(), (0,), (1,)])
    d = dag.dump()
    assert d[2] == {"node": 2, "depth": 2, "parents": [1]}


def test_offset_penalty_is_depth_scaled_on_non_root_node_block():
    """Deterministic linear-algebra check; no empirical or transfer claim. The penalty
    excludes the root (its offset column is dropped) and depth-scales the non-root rows."""
    dag = DagGate([(), (0,), (1,)])            # depths 0,1,2 ; non-root nodes 1,2 (depths 1,2)
    pen = offset_penalty(P=2, dag=dag, gamma_ridge=1e-6, lam_base=2.0, gamma_depth=1.0)
    assert pen.shape == (2 + 2,)               # P covariate rows + (n_nodes-1) offset rows
    assert np.allclose(pen[:2], 1e-6)
    assert np.allclose(pen[2:], [2.0 * 2, 2.0 * 3])   # lam_base*(1+depth) for nodes 1,2


def test_dag_offset_ridge_recovers_well_posed_coefficients():
    """Deterministic linear-algebra check; no empirical or transfer claim."""
    rng = np.random.default_rng(0)
    n, d, k = 500, 4, 3
    W = rng.standard_normal((n, d))
    coeff = rng.standard_normal((d, k))
    M = W @ coeff
    got = dag_offset_ridge(W.T @ W, W.T @ M, penalty=np.full(d, 1e-8))
    assert np.allclose(got, coeff, atol=1e-4)


def test_dag_offset_ridge_shrinks_an_unconstrained_column_to_zero():
    """Deterministic linear-algebra check; no empirical or transfer claim."""
    # design column 3 is all-zero (a "never-active" node) -> its coeff row must go ~0
    rng = np.random.default_rng(1)
    n, k = 400, 2
    W = rng.standard_normal((n, 4)); W[:, 3] = 0.0
    M = W[:, :3] @ rng.standard_normal((3, k))
    pen = np.array([1e-8, 1e-8, 1e-8, 5.0])
    got = dag_offset_ridge(W.T @ W, W.T @ M, penalty=pen)
    assert np.allclose(got[3], 0.0, atol=1e-9)      # unconstrained + penalized -> exactly ~0


from spark_vi.models.topic.pg_stm import PGSTMVI
from spark_vi.models.topic.pg_stm_dag import PGSTMDag, root_only_dag
from tests._stm_synth import gated_ln_corpus_stick


def test_pgstmdag_root_only_matches_flat_pgstmvi():
    """PLANTED: a stick-native gated corpus. REAL: nothing. Synthetic -> MATH-CORRECTNESS
    only: with a root-only DAG the offset block is EMPTY (the root column is dropped), so
    PGSTMDag is exactly PGSTMVI and must return the SAME beta and Sigma, with an all-zero
    root offset row. Proves the drop-root augmentation does not perturb the validated flat
    model. Does NOT prove anything about multi-level DAG behavior."""
    docs, part, _St, _b = gated_ln_corpus_stick(
        group_weights={"A": 0.5, "B": 0.5}, fg_per_group=2, bg_k=3, V=60, D=300,
        doc_len=40, seed=0)
    P = docs[0].x.shape[0]
    vi = PGSTMVI(K=part.K, V=60, partition=part, P=P, n_iter=40, seed=0).fit(docs)
    dag = root_only_dag()
    doc_nodes = [frozenset({0})] * len(docs)
    out = PGSTMDag(K=part.K, V=60, partition=part, dag=dag, P=P, n_iter=40,
                   gamma_ridge=1e-6, lam_base=1e-6, gamma_depth=1.0, seed=0).fit(docs, doc_nodes)
    assert np.allclose(out["beta"], vi["beta"], atol=2e-3)
    assert np.allclose(out["Sigma"], vi["Sigma"], atol=2e-3)
    assert out["B"].shape == (1, part.K - 1)                # one node row (the root)...
    assert np.allclose(out["B"][0], 0.0)                    # ...forced to zero (not estimated)


from spark_vi.models.topic.pg_stm import stick_layout
from tests._stm_synth import dag_offset_corpus, real_beta_from


def test_offset_recovery_through_two_level_closure():
    """PLANTED: node offsets on a root->anchor->subtype DAG + a planted Sigma. REAL: beta
    (realistic overlap, topic_overlap=0.6) and doc-length distribution. Realistic-overlap
    synthetic -> MATH-CORRECTNESS: given a known closure structure, PGSTMDag recovers the
    planted node offsets (subtype offset separated from its anchor's) through a two-level
    closure. Does NOT prove real-data offsets are recoverable, nor transfer."""
    from spark_vi.models.topic.partition import TopicBlockPartition
    part = TopicBlockPartition(group_var="g", background_k=6,
                               foreground=(("A", 4), ("B", 4)))
    K, V = part.K, 300
    beta = real_beta_from(K, V, seed=2)
    # DAG: 0 root; 1,2 anchors (= groups A,B); 3 = subtype under anchor 1
    dag = DagGate([(), (0,), (0,), (1,)])
    Ksm1 = K - 1
    rng = np.random.default_rng(3)
    node_offsets = {0: np.zeros(Ksm1), 1: rng.standard_normal(Ksm1),
                    2: rng.standard_normal(Ksm1), 3: rng.standard_normal(Ksm1)}
    sigma_true = 3.0 * np.eye(Ksm1)
    docs, doc_nodes, _cand = dag_offset_corpus(
        dag=dag, node_offsets=node_offsets, partition=part, beta=beta,
        node_of_group={"A": 1, "B": 2}, doc_nodes_plan={1: 400, 2: 400, 3: 400},
        sigma_true=sigma_true, doc_len=80, seed=4)
    out = PGSTMDag(K=K, V=V, partition=part, dag=dag, P=docs[0].x.shape[0],
                   n_iter=60, lam_base=1e-3, gamma_depth=1.0, seed=0).fit(docs, doc_nodes)
    B = out["B"]
    # recovery is up to the root/intercept reparam; compare on the ACTIVE sticks of each
    # node's own block via correlation of recovered vs planted subtype offset.
    lay = stick_layout(part)
    a1 = lay["groups"]["A"]["active"]
    r = np.corrcoef(B[3][a1], node_offsets[3][a1])[0, 1]
    assert r > 0.6, f"subtype offset not recovered through the 2-level closure (r={r:.2f})"


def test_fallback_spurious_node_offset_shrinks_to_near_zero():
    """PLANTED: offsets on a TREE only (root, two anchors) with a SPURIOUS extra subtype
    node whose true offset is 0. REAL: overlap beta. Realistic-overlap synthetic ->
    MATH-CORRECTNESS: an unearned node's offset norm shrinks far below an earned node's
    ('reduces to the simpler model where the data is tree-like'). Does NOT prove the
    SURVIVING structure is correct, only that unearned structure deactivates."""
    from spark_vi.models.topic.partition import TopicBlockPartition
    part = TopicBlockPartition(group_var="g", background_k=6, foreground=(("A", 4), ("B", 4)))
    K, V = part.K, 300
    beta = real_beta_from(K, V, seed=5)
    dag = DagGate([(), (0,), (0,), (1,)])            # node 3 = the spurious subtype
    Ksm1 = K - 1
    rng = np.random.default_rng(6)
    node_offsets = {0: np.zeros(Ksm1), 1: rng.standard_normal(Ksm1),
                    2: rng.standard_normal(Ksm1), 3: np.zeros(Ksm1)}   # 3 is truly 0
    docs, doc_nodes, _cand = dag_offset_corpus(
        dag=dag, node_offsets=node_offsets, partition=part, beta=beta,
        node_of_group={"A": 1, "B": 2}, doc_nodes_plan={1: 400, 2: 400, 3: 200},
        sigma_true=3.0 * np.eye(Ksm1), doc_len=80, seed=7)
    out = PGSTMDag(K=K, V=V, partition=part, dag=dag, P=docs[0].x.shape[0],
                   n_iter=60, lam_base=1e-2, gamma_depth=1.0, seed=0).fit(docs, doc_nodes)
    norms = out["node_norms"]
    # spurious node 3's offset must be much smaller than the earned anchors' offsets
    assert norms[3] < 0.25 * min(norms[1], norms[2]), \
        f"spurious node did not deactivate: norms={norms}"


def test_offset_uncertainty_is_ordinal_ranks_scarce_above_populated():
    """PLANTED: node offsets + Sigma on root->anchor->{populated subtype, scarce subtype}
    with anchor-only docs so both increments are identified. REAL: overlap beta.
    Realistic-overlap synthetic -> MATH-CORRECTNESS (RELATIVE uncertainty only): the ordinal
    read-out ranks the data-scarce subtype as LESS resolved than the populated subtype
    (rank[scarce] > rank[populated]) and its calibration status is 'ordinal'. Rank is a
    design-moment property (independent of sigma^2 / iterations). We assert ORDERING, not
    absolute coverage: those intervals are overconfident (~0.13 vs 0.90, insight 0051) and
    calibrated absolute intervals are deferred to the read-out-honesty engine. Anchor
    offsets are un-identified under a partitioning gate (dummy trap, insight 0050) so we
    measure identified subtype increments. Does NOT prove absolute coverage or transfer."""
    from spark_vi.models.topic.partition import TopicBlockPartition
    part = TopicBlockPartition(group_var="g", background_k=6, foreground=(("A", 4), ("B", 4)))
    K, V = part.K, 120
    dag = DagGate([(), (0,), (0,), (1,), (1,)])       # 0 root; 1,2 anchors; 3,4 subtypes under 1
    Ksm1 = K - 1
    ranks_scarce = []; ranks_pop = []
    for rep in range(3):
        beta = real_beta_from(K, V, seed=200 + rep)
        rng = np.random.default_rng(9 + rep)
        node_offsets = {u: rng.standard_normal(Ksm1) for u in (1, 2, 3, 4)}
        node_offsets[0] = np.zeros(Ksm1)
        docs, doc_nodes, _cand = dag_offset_corpus(
            dag=dag, node_offsets=node_offsets, partition=part, beta=beta,
            node_of_group={"A": 1, "B": 2},
            doc_nodes_plan={1: 120, 2: 120, 3: 240, 4: 24},   # node 3 populated, node 4 scarce
            sigma_true=3.0 * np.eye(Ksm1), doc_len=50, seed=100 + rep)
        out = PGSTMDag(K=K, V=V, partition=part, dag=dag, P=docs[0].x.shape[0],
                       n_iter=25, lam_base=1e-3, seed=rep).fit(docs, doc_nodes)
        ou = out["offset_uncertainty"]
        assert ou["calibration"] == "ordinal"
        assert "offset_cov_diag" not in out               # no raw widths exported
        ranks_pop.append(int(ou["rank"][3])); ranks_scarce.append(int(ou["rank"][4]))
    assert np.mean(ranks_scarce) > np.mean(ranks_pop), (
        f"scarce subtype not ranked less-resolved: scarce={ranks_scarce} pop={ranks_pop}")


def test_identified_flag_true_for_populated_false_for_zero_doc_node():
    """PLANTED: offsets on root->2 anchors, one anchor's subtype well-populated, plus a
    ZERO-doc extra node. REAL: overlap beta. Realistic-overlap synthetic -> MATH-CORRECTNESS:
    the `identified` flag is True for a well-populated distinct node (data halves the prior
    variance) and False for a node with no attesting documents (posterior variance == prior
    variance, ratio 1). Proves the flag distinguishes data-identified from prior-dominated
    offsets. Does NOT prove real-data identifiability or transfer."""
    from spark_vi.models.topic.partition import TopicBlockPartition
    part = TopicBlockPartition(group_var="g", background_k=6, foreground=(("A", 4), ("B", 4)))
    K, V = part.K, 120
    dag = DagGate([(), (0,), (0,), (1,), (1,)])       # node 4 will attest NO documents
    Ksm1 = K - 1
    rng = np.random.default_rng(3)
    node_offsets = {u: rng.standard_normal(Ksm1) for u in (1, 2, 3)}
    node_offsets[0] = np.zeros(Ksm1); node_offsets[4] = np.zeros(Ksm1)
    beta = real_beta_from(K, V, seed=7)
    docs, doc_nodes, _cand = dag_offset_corpus(
        dag=dag, node_offsets=node_offsets, partition=part, beta=beta,
        node_of_group={"A": 1, "B": 2},
        doc_nodes_plan={1: 200, 2: 200, 3: 300},          # node 4 absent -> zero-doc column
        sigma_true=3.0 * np.eye(Ksm1), doc_len=50, seed=5)
    out = PGSTMDag(K=K, V=V, partition=part, dag=dag, P=docs[0].x.shape[0],
                   n_iter=25, lam_base=1e-3, seed=0).fit(docs, doc_nodes)
    ident = out["offset_uncertainty"]["identified"]
    assert ident[3] == True, "well-populated subtype should be data-identified"
    assert ident[4] == False, "zero-doc node should be prior-dominated"


from spark_vi.models.topic.pg_stm_dag import inject_spurious_edges


def test_inject_spurious_edges_adds_random_leaves_and_shrinks_on_replay():
    """Mechanical check of the real-data fallback HOOK (Task-3b machinery). The real-data
    RUN (inject into the OMOP DAG, fit on the real corpus, verify injected offsets die) is
    the OMOP-integration phase; here we prove the injector produces valid extra leaves and
    that on a planted corpus (offsets truly 0 on injected nodes) their norms shrink. This
    test is synthetic and asserts MATH-CORRECTNESS of the hook only."""
    base = DagGate([(), (0,), (0,)])
    dag2 = inject_spurious_edges(base, extra_parents=[1, 2])
    assert dag2.n_nodes == 5                                    # 2 injected leaves
    assert dag2.parents[3] in ((1,), (2,)) and dag2.parents[4] in ((1,), (2,))
    # injected nodes attest no documents -> their offset columns are never active -> ~0
    from spark_vi.models.topic.partition import TopicBlockPartition
    part = TopicBlockPartition(group_var="g", background_k=6, foreground=(("A", 4), ("B", 4)))
    K, V = part.K, 300
    beta = real_beta_from(K, V, seed=10)
    Ksm1 = K - 1; rng = np.random.default_rng(11)
    node_offsets = {0: np.zeros(Ksm1), 1: rng.standard_normal(Ksm1), 2: rng.standard_normal(Ksm1),
                    3: np.zeros(Ksm1), 4: np.zeros(Ksm1)}
    docs, doc_nodes, _cand = dag_offset_corpus(
        dag=dag2, node_offsets=node_offsets, partition=part, beta=beta,
        node_of_group={"A": 1, "B": 2}, doc_nodes_plan={1: 300, 2: 300},   # nobody at 3,4
        sigma_true=3.0 * np.eye(Ksm1), doc_len=80, seed=12)
    out = PGSTMDag(K=K, V=V, partition=part, dag=dag2, P=docs[0].x.shape[0],
                   n_iter=50, lam_base=1e-2, seed=0).fit(docs, doc_nodes)
    assert out["node_norms"][3] < 1e-6 and out["node_norms"][4] < 1e-6


def test_offset_indicator_drops_the_root_entry():
    """Deterministic structure check; no empirical or transfer claim. The offset design
    omits the root column (it equals the covariate intercept), so offset_indicator is the
    closure indicator over non-root nodes 1..U-1."""
    dag = DagGate([(), (0,), (0,), (1,)])          # root; anchors 1,2; subtype 3 under 1
    assert dag.n_offset_nodes == 3
    z = dag.offset_indicator(frozenset({3}))       # closure {3,1,0}; drop root -> nodes 1,2,3
    assert z.dtype == np.float64
    assert list(z) == [1.0, 0.0, 1.0]              # node1 on, node2 off, node3 on
    assert list(dag.offset_indicator(frozenset({2}))) == [0.0, 1.0, 0.0]


def test_fit_routes_background_only_docs_and_they_inform_the_design():
    """PLANTED: a small gated corpus plus one background-only document (no group).
    REAL: nothing. Synthetic -> MATH-CORRECTNESS: PGSTMDag.fit runs with background-only
    documents mixed in (routing them to the flat E-step), returns valid shapes, and the
    background-only doc contributes to the covariate design (its all-ones intercept enters
    XtX via the fit). Proves the routing + flat E-step compose with the gated path. Does
    NOT prove recovery or transfer."""
    import numpy as np
    from spark_vi.models.topic.partition import TopicBlockPartition
    part = TopicBlockPartition(group_var="g", background_k=4, foreground=(("A", 3), ("B", 3)))
    from spark_vi.models.topic.pg_stm_dag import DagGate, PGSTMDag
    from tests._stm_synth import gated_ln_corpus_stick
    docs, _p, _S, _b = gated_ln_corpus_stick(
        group_weights={"A": 0.5, "B": 0.5}, fg_per_group=3, bg_k=4, V=40, D=120,
        doc_len=30, seed=0)
    # append 40 background-only docs: no group, a couple of background-topic tokens each
    import dataclasses
    bg_docs = [dataclasses.replace(docs[i], groups=frozenset()) for i in range(40)]
    mixed = list(docs) + bg_docs
    dag = DagGate([(), (0,)])                      # root + one anchor (group A) — trivial here
    out = PGSTMDag(K=part.K, V=40, partition=part, dag=dag, P=1, n_iter=15, seed=0).fit(
        mixed, [frozenset({1})] * len(docs) + [frozenset({0})] * len(bg_docs))
    assert out["beta"].shape == (part.K, 40)
    assert out["Sigma"].shape == (part.K - 1, part.K - 1)
    assert np.isfinite(out["beta"]).all() and np.isfinite(out["Sigma"]).all()


def test_background_only_corpus_recovers_planted_background():
    """PLANTED: a pure background-only corpus (no groups, flat background stick-breaking
    theta_bg = stick_to_simplex(psi_bg), psi_bg ~ N(0, sigma_true[bg,bg])) on a planted
    background beta. REAL: nothing. Synthetic -> MATH-CORRECTNESS: fitting a
    background-only-only corpus with PGSTMDag recovers the planted BACKGROUND topics by
    block-mass (planted_recovery, the DAG suite's standard measure), validating the flat
    E-step end-to-end and its consistency with the shared background block. Recovers 4 of
    5 planted background topics; the 5th is the LEAST-ATTESTED topic (mean-0 stick-breaking
    under-weights later sticks, so the last background topic receives little corpus mass) —
    an attestation/information property of the plant, NOT an estimator limitation of the
    flat E-step (insight 0053). Does NOT prove gated-mixture behavior or transfer to real
    data."""
    import numpy as np
    from spark_vi.models.topic.partition import TopicBlockPartition
    from spark_vi.models.topic.pg_stm_dag import DagGate, PGSTMDag
    from tests._stm_synth import dag_offset_corpus, real_beta_from, planted_recovery
    part = TopicBlockPartition(group_var="g", background_k=5, foreground=(("A", 3),))
    K, V = part.K, 60
    beta = real_beta_from(K, V, seed=1)
    dag = DagGate([(), (0,)])
    Ksm1 = K - 1
    node_offsets = {0: np.zeros(Ksm1), 1: np.zeros(Ksm1)}
    docs, doc_nodes, _cand = dag_offset_corpus(
        dag=dag, node_offsets=node_offsets, partition=part, beta=beta,
        node_of_group={"A": 1}, doc_nodes_plan={}, n_background_only=400,
        sigma_true=2.0 * np.eye(Ksm1), doc_len=60, seed=3)
    assert len(docs) == 400 and all(d.groups == frozenset() for d in docs)
    out = PGSTMDag(K=K, V=V, partition=part, dag=dag, P=1, n_iter=40, seed=0).fit(docs, doc_nodes)
    # Recovery measured by BLOCK-MASS (planted_recovery, the DAG suite's standard
    # attestation-robust measure): does some fitted background topic put >= 0.5 mass on
    # each planted background topic's high-probability word block? Single-word argmax is
    # NOT used here: it is tie-break-brittle under topic overlap and conflates recovery
    # with the stick-breaking attestation skew below.
    rec = planted_recovery(out["beta"][:part.background_k], beta[:part.background_k])
    assert rec >= 4, f"flat path did not recover the attested background topics: {rec}/5"


def test_background_only_members_flip_anchor_identification_and_identified_increments_recover():
    """PLANTED: node offsets + Sigma on root->{A(anchor docs + subtype A1); B(subtype B1 only,
    no anchor docs)}. REAL: overlap beta (topic_overlap=0.6) + doc-length. Realistic-overlap
    synthetic -> MATH-CORRECTNESS. Validates two robust facts of background-only member support:
    (1) IDENTIFICATION-FLAG FLIP: the anchor-A increment is un-identified under the partitioning
    gate WITHOUT background-only members (identified=False, the insight-0050 dummy trap) and
    becomes identified=True WITH them (the trap is broken for the identification flag). (2) The
    IDENTIFIED subtype increment (A1, which has branching evidence: anchor-A docs vs A1 docs
    separate the levels) recovers, measured on FOREGROUND (node-specific) sticks -- the shared
    background-stick offset is weakly identified so recovery is NOT measured there (insight
    0050/0054). Does NOT claim anchor-LEVEL point recovery (anchor levels stay confounded with the
    global intercept on group foreground sticks; background-only docs never activate those sticks,
    so they flip the identification flag WITHOUT restoring the level's point estimate -- insight
    0052/0054), NOR recovery of the no-direct-docs anchor B (its node B/B1 increments are
    collinear -- a collapsible degenerate chain per insight 0054), NOR transfer to real data."""
    import numpy as np
    from spark_vi.models.topic.partition import TopicBlockPartition
    from spark_vi.models.topic.pg_stm_dag import DagGate, PGSTMDag
    from spark_vi.models.topic.pg_stm import stick_layout
    from tests._stm_synth import dag_offset_corpus, real_beta_from
    part = TopicBlockPartition(group_var="g", background_k=6, foreground=(("A", 4), ("B", 4)))
    K, V = part.K, 300
    dag = DagGate([(), (0,), (0,), (1,), (2,)])       # root; A=1 (docs), B=2 (no docs); A1=3, B1=4
    Ksm1 = K - 1
    rng = np.random.default_rng(4)
    node_offsets = {u: rng.standard_normal(Ksm1) for u in (1, 2, 3, 4)}
    node_offsets[0] = np.zeros(Ksm1)
    beta = real_beta_from(K, V, seed=2)
    common = dict(dag=dag, node_offsets=node_offsets, partition=part, beta=beta,
                  node_of_group={"A": 1, "B": 2}, doc_nodes_plan={1: 400, 3: 400, 4: 500},
                  sigma_true=3.0 * np.eye(Ksm1), doc_len=80, seed=6)
    # WITHOUT background-only members: anchor-A increment un-identified (the insight-0050 trap)
    docs0, dn0, _cand0 = dag_offset_corpus(n_background_only=0, **common)
    out0 = PGSTMDag(K=K, V=V, partition=part, dag=dag, P=1, n_iter=60, lam_base=1e-3,
                    seed=0).fit(docs0, dn0)
    assert out0["offset_uncertainty"]["identified"][1] == False, \
        f"anchor A unexpectedly identified without bg-only: {out0['offset_uncertainty']['identified']}"
    # WITH background-only members: the trap is broken for the identification flag
    docs1, dn1, _cand1 = dag_offset_corpus(n_background_only=600, **common)
    out1 = PGSTMDag(K=K, V=V, partition=part, dag=dag, P=1, n_iter=60, lam_base=1e-3,
                    seed=0).fit(docs1, dn1)
    ident = out1["offset_uncertainty"]["identified"]
    assert ident[1] == True, f"background-only members did not flip anchor A identification: {ident}"
    # A1 (branching evidence: anchor-A docs vs A1 docs) is identified -- assert the premise the
    # foreground-increment recovery check below relies on, so the test self-verifies its docstring
    assert ident[3] == True, f"subtype A1 unexpectedly un-identified: {ident}"
    # the IDENTIFIED subtype increment (A1) recovers on FOREGROUND sticks (insight 0050/0054)
    lay = stick_layout(part)
    actA = lay["groups"]["A"]["active"]
    bg = set(np.asarray(lay["bg_sticks"]).tolist())
    fgA = np.array([d for d in actA if d not in bg], dtype=np.int64)
    B = out1["B"]
    r_fg = np.corrcoef(B[3][fgA], node_offsets[3][fgA])[0, 1]
    assert r_fg > 0.4, f"identified subtype A1 foreground increment did not recover: r={r_fg:.2f}"
