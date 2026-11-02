"""Pure-logic tests for the native-Mondo label space (exp 0110; spec 2026-10-30).

Repo convention: Spark wiring is cluster-covered, pure python kernels are
unit-tested. So `build_mondo_native_fit_inputs` (BigQuery + Spark) has no test here,
and everything it composes does: the id encoding, the code ladder, the subgraph
(coded terms + their Mondo ancestors) and the attestation provider (local Spark).

Powering, landing, the reduction and the splice moved from this build into the
DAG-gated estimator (`spark_vi.dag.nodeset`); the properties the native build used
to promise for them (multi-parenthood kept, no subsumed sibling, thin chains
spliced, an unpowered term landing on every nearest kept ancestor) are checked here
on the subgraph this module hands over, through the estimator's node-set learning.
"""
import pytest

from charmpheno.mondo import native_dag as mn
from spark_vi.dag.nodeset import ROOT, fit_node_set


# --------------------------------------------------------------------------- #
# id space                                                                     #
# --------------------------------------------------------------------------- #
def test_mondo_cid_roundtrips_and_rejects_non_curies():
    assert mn.mondo_cid("MONDO:0004995") == 4995
    assert mn.mondo_curie(4995) == "MONDO:0004995"
    assert mn.mondo_curie(mn.mondo_cid("MONDO:0000001")) == "MONDO:0000001"
    for bad in ("HP:0000118", "0004995", "MONDO:abc", "MONDO:"):
        with pytest.raises(ValueError):
            mn.mondo_cid(bad)


def test_root_cannot_collide_with_a_term_and_has_no_curie():
    # Mondo numeric ids are positive; the integer tables' root is -1, so the two id
    # families cannot overlap.
    assert mn.MONDO_NATIVE_ROOT_CID == -1
    assert mn.mondo_cid("MONDO:0000001") > mn.MONDO_NATIVE_ROOT_CID
    with pytest.raises(ValueError):
        mn.mondo_curie(mn.MONDO_NATIVE_ROOT_CID)


#   A            A is the top; B and C are mid-level; D is under BOTH (a diamond);
#  / \           E is under C only.
# B   C
#  \ / \
#   D   E
_DIAMOND_CHILDREN = {"MONDO:0000001": ["MONDO:0000002", "MONDO:0000003"],
                     "MONDO:0000002": ["MONDO:0000004"],
                     "MONDO:0000003": ["MONDO:0000004", "MONDO:0000005"]}
_A, _B, _C, _D, _E = (f"MONDO:000000{i}" for i in range(1, 6))


def _diamond_parents():
    return mn.parent_adjacency(_DIAMOND_CHILDREN)


def test_parent_adjacency_inverts_and_dedupes():
    pa = _diamond_parents()
    assert pa[_D] == [_B, _C]          # multi-parent preserved, sorted
    assert pa[_E] == [_C]
    assert _A not in pa                # the top has no parent


def test_ancestor_closure_is_reflexive_and_visits_a_diamond_once():
    pa = _diamond_parents()
    assert mn.ancestor_closure(_D, pa) == {_A, _B, _C, _D}
    assert mn.ancestor_closure(_A, pa) == {_A}


# --------------------------------------------------------------------------- #
# the subgraph handed to the fit                                               #
# --------------------------------------------------------------------------- #
def test_subgraph_is_the_coded_terms_and_all_their_ancestors():
    """A mid-level term with no code of its own (B, C) is in the subgraph: it can
    earn a block on its descendants' documents, which is what closure-support
    powering used to keep. A term nothing resolves to below (E) is not."""
    pa = _diamond_parents()
    sg = mn.subgraph_parents([_D], pa)
    assert sg == {_A: [], _B: [_A], _C: [_A], _D: [_B, _C]}


def test_subgraph_keeps_mondos_own_edges_including_the_diamond():
    pa = _diamond_parents()
    sg = mn.subgraph_parents([_D, _E], pa)
    assert sg[_D] == [_B, _C] and sg[_E] == [_C]
    assert sorted(sg) == [_A, _B, _C, _D, _E]


def test_subgraph_within_a_branch_drops_outside_terms_and_ancestors():
    """`branch_root` = C: E and D are under C (D also under B); the subgraph keeps
    C's subtree only, C becomes top-level, and D's out-of-branch parent B goes."""
    pa = _diamond_parents()
    within = {_C, _D, _E}
    sg = mn.subgraph_parents([_D, _E, _B], pa, within=within)
    assert sg == {_C: [], _D: [_C], _E: [_C]}


def test_subgraph_parses_as_the_estimators_dag_in_curies():
    from spark_vi.dag.nodeset import parse_dag
    parsed = parse_dag(mn.subgraph_parents([_D, _E], _diamond_parents()))
    assert parsed.parents[_A] == (ROOT,)
    assert all(isinstance(n, str) for n in parsed.nodes)


# --------------------------------------------------------------------------- #
# the node set the estimator learns on that subgraph (what the build used to   #
# do: powering, reduction, splice, landing)                                    #
# --------------------------------------------------------------------------- #
def _docs(**n):
    """Documents labelled at the most-specific term: {term: count}."""
    return [[t] for t, k in n.items() for _ in range(k)]


def test_an_unpowered_term_lands_on_every_nearest_kept_ancestor():
    """D below the threshold lands on BOTH arms (a diamond is not silently a tree);
    its documents keep their evidence for B and C."""
    pa = _diamond_parents()
    sg = mn.subgraph_parents([_D, _E], pa)
    docs = ([[_D]] * 2) + ([[_E]] * 5) + ([[_B]] * 5)
    ns = fit_node_set(sg, docs, min_docs_per_node=5)
    assert ns.pruned[_D] == (_B, _C)
    assert ns.support[_C] == 7 and ns.support[_B] == 7


def test_no_kept_node_is_a_sibling_of_its_own_descendant():
    """The structural acceptance property (exp 0104's 619 category anchors): the
    kept DAG is a transitive reduction, so a subsumed sibling cannot exist — over
    every threshold, including ones that drop an intermediate."""
    pa = _diamond_parents()
    sg = mn.subgraph_parents([_D, _E], pa)
    docs = ([[_D]] * 6) + ([[_E]] * 3) + ([[_C]] * 2)
    for k in (1, 3, 5, 7, 9, 12):
        ns = fit_node_set(sg, docs, min_docs_per_node=k)
        kept = ns.parents
        children = {}
        for c, ps in kept.items():
            for p in ps:
                children.setdefault(p, []).append(c)
        for kids in children.values():
            for a in kids:
                for b in kids:
                    if a is ROOT or b is ROOT or a == b:
                        continue
                    assert a not in mn.ancestor_closure(b, pa), (k, a, b)


def test_an_uncoded_thin_chain_is_spliced_and_a_labelled_one_is_kept():
    """`collapse`'s splice, now the estimator's: B (no document of its own, one kept
    child D) is spliced; C has documents of its own and stays although it too has a
    single child."""
    children = {_A: [_B, _C], _B: [_D], _C: [_E]}
    pa = mn.parent_adjacency(children)
    sg = mn.subgraph_parents([_D, _E, _C], pa)
    docs = ([[_D]] * 4) + ([[_E]] * 4) + ([[_C]] * 2)
    ns = fit_node_set(sg, docs, min_docs_per_node=1)
    assert _B in ns.spliced
    assert _C in ns.engine_id
    assert ns.parents[_D] == (_A,)          # reattached past the spliced B


# --------------------------------------------------------------------------- #
# the ladder as pure logic                                                     #
# --------------------------------------------------------------------------- #
def test_resolve_code_terms_prefers_exact_and_never_climbs_past_it():
    pa = _diamond_parents()
    out = mn.resolve_code_terms(standard_pairs=[(11, _D)],
                                climb_pairs=[(11, _A), (22, _C)], parent_adj=pa)
    assert out[11] == [_D]             # the climb candidate for 11 is discarded
    assert out[22] == [_C]


def test_resolve_code_terms_reduces_a_nested_tie_to_the_most_specific():
    pa = _diamond_parents()
    out = mn.resolve_code_terms([], [(7, _D), (7, _C), (7, _A)], pa)
    assert out[7] == [_D]


def test_resolve_code_terms_keeps_a_genuine_orthogonal_tie():
    pa = _diamond_parents()
    out = mn.resolve_code_terms([], [(7, _B), (7, _E)], pa)
    assert out[7] == [_B, _E]          # neither is an ancestor of the other


def test_resolve_code_terms_maps_one_code_to_several_exact_terms():
    pa = _diamond_parents()
    out = mn.resolve_code_terms([(11, _D), (11, _E)], [], pa)
    assert out[11] == [_D, _E]


# --------------------------------------------------------------------------- #
# the readout DAG's reduction (readout_dag.py uses induced_hasse_parents)       #
# --------------------------------------------------------------------------- #
def test_nearest_mapped_parents_alone_would_reintroduce_a_subsumed_sibling():
    from spark_vi.dag.ops import nearest_kept_parents as nearest_mapped_parents
    pa = _diamond_parents()
    assert nearest_mapped_parents({_A, _C, _D}, pa)[_D] == [_A, _C]
    assert _C in mn.ancestor_closure(_D, pa)          # ...and C subsumes D


def test_induced_hasse_parents_drops_the_redundant_ancestor_edge():
    pa = _diamond_parents()
    assert mn.induced_hasse_parents({_A, _C, _D}, pa)[_D] == [_C]
    assert mn.induced_hasse_parents({_A, _C, _D}, pa)[_C] == [_A]
    assert mn.induced_hasse_parents({_A, _C, _D}, pa)[_A] == []


def test_induced_hasse_parents_keeps_incomparable_parents():
    pa = _diamond_parents()
    assert mn.induced_hasse_parents({_A, _B, _C, _D}, pa)[_D] == [_B, _C]


# --------------------------------------------------------------------------- #
# reports                                                                      #
# --------------------------------------------------------------------------- #
def test_build_reports_carry_counts_and_no_threshold():
    stats = {"version": mn.MONDO_NATIVE_VERSION, "n_coded_terms": 2, "n_nodes": 5,
             "n_multi_parent": 1, "n_top": 1, "branch": "",
             "n_codes_resolved": 3, "n_codes_outside_branch": 0}
    line = mn.format_native_build_report(stats)
    assert mn.MONDO_NATIVE_VERSION in line and "5 node(s)" in line
    assert "minDocsPerNode" in line
    assert "3 standard code(s)" in mn.format_native_mapping_report(stats)


# --------------------------------------------------------------------------- #
# multi-parent checks against spark-vi's engine layout and readout targets     #
# --------------------------------------------------------------------------- #
def _diamond_layout():
    """Engine-id diamond: 0=root, 1=B, 2=C, 3=D(under both), 4=E(under C)."""
    from spark_vi.dag.layout import DagLayout
    return DagLayout({1: [0], 2: [0], 3: [1, 2], 4: [2]}, n_bg=2, tpn=1)


def test_daglayout_closure_visits_a_diamond_once():
    lay = _diamond_layout()
    clo = lay.closure(3)
    assert sorted(clo) == [0, 1, 2, 3]
    assert len(clo) == len(set(clo))          # no double-visit
    assert clo[0] == 0                        # root first (depth, id) ordering


def test_daglayout_allowed_set_on_a_diamond_has_no_repeated_topic():
    lay = _diamond_layout()
    allowed = list(lay.allowed_set({3}))
    assert allowed == sorted(set(allowed))
    assert len(allowed) == lay.n_bg + 3 * lay.tpn


def test_daglayout_subtree_and_depth_are_diamond_safe():
    lay = _diamond_layout()
    assert lay.subtree(0) == {0, 1, 2, 3, 4}
    assert lay.depth(3) == 2                  # longest path root->B->D
    assert sorted(lay.descendants(2)) == [3, 4]


def _diamond_labeler(mode):
    from spark_vi.readout.labels import NodeLabeler
    ns = fit_node_set(mn.subgraph_parents([_D, _E], _diamond_parents()),
                      [[_D], [_E], [_B], [_A]], min_docs_per_node=1)
    return ns, NodeLabeler.from_node_set(ns, mask_mode=mode)


def test_closure_mask_sibling_expansion_unions_over_all_parents():
    """The readout's closure mask observes the children of EVERY parent of every
    active node: over a multi-parent node that union is what stops a diamond from
    hiding one arm's siblings."""
    ns, lab = _diamond_labeler("closure")
    pos, obs = lab([_D])
    e = ns.engine_id
    assert set(pos) == {0, e[_A], e[_B], e[_C], e[_D]}
    assert e[_E] in obs                # a sibling of D through its SECOND parent C


def test_closure_mask_single_parent_arm():
    ns, lab = _diamond_labeler("closure")
    _, obs = lab([_E])
    e = ns.engine_id
    assert e[_D] in obs                # D is a sibling of E under C
    assert e[_B] in obs                # B is a sibling of C under A


def test_dag_children_and_min_depth_is_multi_parent_safe():
    from spark_vi.dag.ops import children_and_min_depth as _dag_children_and_min_depth
    children, depth = _dag_children_and_min_depth(
        {0: [], 1: [0], 2: [0], 3: [1, 2], 4: [2]}, 5)
    assert 3 in children[1] and 3 in children[2]
    assert depth[3] == 2


# --------------------------------------------------------------------------- #
# the attestation seam (local Spark; the BQ build above it is cluster-covered)  #
# --------------------------------------------------------------------------- #
def _code_map(spark, rows):
    import pandas as pd
    return spark.createDataFrame(pd.DataFrame(
        {"std_cid": [c for c, _ in rows], "node_id": [t for _, t in rows],
         "node_cid": [mn.mondo_cid(t) for _, t in rows]}))


def test_native_attested_provider_labels_docs_with_mondo_curies(spark):
    """A doc's labels are the Mondo terms its in-window codes resolve to (curies,
    `array<string>`); a doc whose codes resolve to nothing survives with an empty
    array (background)."""
    import pandas as pd
    from charmpheno.omop.doc_spec import PatientCohortDocSpec

    code_map = _code_map(spark, [(5001, _D), (5002, _D), (5003, _E)])
    events = spark.createDataFrame(pd.DataFrame({
        "person_id": [1, 2, 2, 3],
        "concept_id": [5001, 5003, 5002, 9999],   # p3's 9999 resolves to nothing
        "source_cohort": ["population"] * 4}))
    out = mn.make_mondo_native_attested_provider(
        code_map, doc_spec=PatientCohortDocSpec())(events)
    assert dict(out.dtypes)["labels"] == "array<string>"
    got = {r["doc_id"]: list(r["labels"]) for r in out.collect()}
    assert got["population:1"] == [_D]
    assert got["population:2"] == [_D, _E]
    assert got["population:3"] == []           # background doc survives, empty


def test_native_provider_keeps_an_ancestor_attested_by_another_code(spark):
    """No cross-code reduction: the estimator's label contract allows ancestors, and
    its frontier reduction is internal."""
    import pandas as pd
    from charmpheno.omop.doc_spec import PatientCohortDocSpec

    code_map = _code_map(spark, [(5001, _D), (5002, _B)])
    events = spark.createDataFrame(pd.DataFrame(
        {"person_id": [1, 1], "concept_id": [5001, 5002],
         "source_cohort": ["population"] * 2}))
    out = mn.make_mondo_native_attested_provider(
        code_map, doc_spec=PatientCohortDocSpec())(events)
    assert list(out.collect()[0]["labels"]) == [_B, _D]


# --------------------------------------------------------------------------- #
# The code ladder's climb (moved from analysis/cloud/tests/test_mondo_usage_core.py). #
# --------------------------------------------------------------------------- #
def test_nearest_mapped_standard_ancestors_picks_min_distance():
    # concept_ancestor climb (source_climb space): descendant 100 has mapped
    # standard ancestors at levels 1 and 2; only the NEAREST is kept.
    edges = [(100, 10, 1), (100, 20, 2)]
    assert mn.nearest_mapped_standard_ancestors(edges) == {100: [10]}


def test_nearest_mapped_standard_ancestors_keeps_ties():
    # two mapped ancestors equally near (both level 2) -> keep BOTH so the term
    # rows can be flagged as a shared-attribution collision. Level-3 dropped.
    edges = [(100, 10, 2), (100, 20, 2), (100, 30, 3)]
    assert mn.nearest_mapped_standard_ancestors(edges) == {100: [10, 20]}


def test_nearest_mapped_standard_ancestors_excludes_distance_zero_and_empty():
    # a self/exact edge (level 0) is NOT a climb; no mapped ancestor -> absent.
    assert mn.nearest_mapped_standard_ancestors([(100, 100, 0)]) == {}
    assert mn.nearest_mapped_standard_ancestors([]) == {}
    # independent descendants are resolved separately
    assert mn.nearest_mapped_standard_ancestors(
        [(1, 5, 1), (2, 6, 4), (2, 7, 4)]) == {1: [5], 2: [6, 7]}


def test_reduce_tie_map_keeps_most_specific():
    # Mondo child->parents: carbuncle -> pyoderma -> skin; hereditary & endocrine are roots.
    parents = {"carbuncle": ["pyoderma"], "pyoderma": ["skin"], "skin": [],
               "hereditary": [], "endocrine": []}
    pairs = [
        (1, "carbuncle"), (1, "pyoderma"), (1, "skin"),   # nested chain -> just carbuncle
        (2, "hereditary"), (2, "endocrine"),              # orthogonal tie -> both kept
        (3, "skin"),                                       # single -> unchanged
    ]
    out = mn.reduce_tie_map(pairs, parents)
    assert out[1] == ["carbuncle"]                         # ancestors pyoderma/skin dropped
    assert out[2] == ["endocrine", "hereditary"]           # neither is the other's ancestor
    assert out[3] == ["skin"]
