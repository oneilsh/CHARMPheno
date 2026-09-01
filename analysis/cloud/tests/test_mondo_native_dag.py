"""Pure-logic tests for the native-Mondo label DAG (exp 0110).

Repo convention (`mondo_dag`, `mondo_collapse`, `case_finding_assembly`): Spark
wiring is cluster-covered, pure numpy/pandas/python kernels are unit-tested. So
`build_mondo_native_fit_inputs` (BigQuery + Spark) has no test here, and
everything it composes does: the id encoding, the closure-support powering, the
kept-set Hasse, the splice interaction, the terminal-as-property behavior, and
the roll-to-kept landing rule.

The last group is the plan's "multi-parenthood, verified not assumed" checklist
(§3): the layout and the closure mask take parent LISTS, but exp 0104's
accidental tree meant real diamonds were never exercised. Those tests read
`DagLayout` and `frontier_to_label` as they are and assert AGAINST them — both
modules are source-hashed into every bundle cache key and are not to be touched.
"""
import pytest

import mondo_native_dag as mn
from mondo_collapse import collapse_only_child_classes


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
    # Mondo numeric ids are positive; the forest root is -1 (the shared
    # _FOREST_ROOT_CID convention), so the two id families cannot overlap.
    assert mn.MONDO_NATIVE_ROOT_CID == -1
    assert mn.mondo_cid("MONDO:0000001") > mn.MONDO_NATIVE_ROOT_CID
    with pytest.raises(ValueError):
        mn.mondo_curie(mn.MONDO_NATIVE_ROOT_CID)


# --------------------------------------------------------------------------- #
# closure-support powering                                                     #
# --------------------------------------------------------------------------- #
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


def _curies(dag):
    """`{curie: [parent curies]}` for a built DAG, rendering the forest root (-1,
    which has no curie) as 'ROOT'."""
    def _c(cid):
        return "ROOT" if cid == mn.MONDO_NATIVE_ROOT_CID else mn.mondo_curie(cid)
    return {_c(c): sorted(_c(p) for p in ps) for c, ps in dag.parents.items()}


def test_parent_adjacency_inverts_and_dedupes():
    pa = _diamond_parents()
    assert pa[_D] == [_B, _C]          # multi-parent preserved, sorted
    assert pa[_E] == [_C]
    assert _A not in pa                # the top has no parent


def test_ancestor_closure_is_reflexive_and_visits_a_diamond_once():
    pa = _diamond_parents()
    assert mn.ancestor_closure(_D, pa) == {_A, _B, _C, _D}
    assert mn.ancestor_closure(_A, pa) == {_A}


def test_closure_support_rolls_persons_up_and_counts_each_person_once():
    pa = _diamond_parents()
    # p1 attests D (under both B and C); p2 attests E; p3 attests D AND E.
    support = mn.closure_support(
        [("p1", _D), ("p2", _E), ("p3", _D), ("p3", _E)], pa)
    assert support[_D] == 2            # direct: p1, p3
    assert support[_E] == 2            # direct: p2, p3
    assert support[_B] == 2            # rolled from D
    assert support[_C] == 3            # D (p1,p3) union E (p2,p3) -> p1,p2,p3
    assert support[_A] == 3            # everybody
    # closure support >= direct support, everywhere (the plan's scale claim)
    direct = {_D: 2, _E: 2}
    assert all(support[t] >= n for t, n in direct.items())


def test_closure_support_does_not_double_count_a_person_across_two_paths():
    # p1 reaches C twice (via D and via E). C must still count them once.
    pa = _diamond_parents()
    assert mn.closure_support([("p1", _D), ("p1", _E)], pa)[_C] == 1


def test_closure_rows_cover_the_reflexive_closure_of_every_term():
    pa = _diamond_parents()
    rows = mn.closure_rows([_D, _E], pa)
    assert (_D, _D) in rows and (_D, _A) in rows and (_D, _B) in rows
    assert (_E, _C) in rows and (_E, _B) not in rows
    assert rows == sorted(set(rows))   # deterministic, deduped


def test_powering_keeps_a_mid_level_term_with_no_direct_code():
    """The plan's headline powering claim: 'directly coded' is a PROPERTY, not a
    node type — an uncoded mid-level term clears min_positives on the support of
    its descendants."""
    pa = _diamond_parents()
    support = mn.closure_support([(f"p{i}", _D) for i in range(150)], pa)
    kept = {t for t, n in support.items() if n >= 100}
    assert kept == {_A, _B, _C, _D}    # B and C carry no code of their own
    assert _E not in support           # nothing under E


# --------------------------------------------------------------------------- #
# kept-set Hasse over a diamond                                                #
# --------------------------------------------------------------------------- #
def test_kept_set_hasse_preserves_multi_parenthood_through_a_dropped_middle():
    """A diamond whose LEFT arm is unpowered must not silently become a tree: D
    keeps C as a parent and picks up A (B's nearest kept ancestor) — not one
    arbitrary parent."""
    pa = _diamond_parents()
    # A and C are coded so the splice leaves the shape alone (the thin-chain
    # post-pass is exercised separately below).
    dag, _ = mn.build_native_label_dag({_A, _C, _D}, pa, coded_ids={_A, _C, _D},
                                       names={_A: "a", _C: "c", _D: "d"})
    parents = _curies(dag)
    assert parents[_C] == [_A]
    # A is REDUNDANT: it is already an ancestor of D through C. Keeping it would
    # make C and D siblings under A with C subsuming D — the 619 shape.
    assert parents[_D] == [_C]


def test_nearest_mapped_parents_alone_would_reintroduce_a_subsumed_sibling():
    """The bug the reduction closes, pinned on the PORTED function so a future
    unification cannot quietly drop the extra step: with B unpowered, D's nearest
    mapped parents are {A, C} — and C is an ancestor of D, so under A they are
    siblings. This is exp 0104's 619-category-anchor shape, rebuilt."""
    from mondo_usage_core import nearest_mapped_parents

    pa = _diamond_parents()
    assert nearest_mapped_parents({_A, _C, _D}, pa)[_D] == [_A, _C]
    assert _C in mn.ancestor_closure(_D, pa)          # ...and C subsumes D


def test_induced_hasse_parents_drops_the_redundant_ancestor_edge():
    pa = _diamond_parents()
    assert mn.induced_hasse_parents({_A, _C, _D}, pa)[_D] == [_C]
    assert mn.induced_hasse_parents({_A, _C, _D}, pa)[_C] == [_A]
    assert mn.induced_hasse_parents({_A, _C, _D}, pa)[_A] == []


def test_induced_hasse_parents_keeps_incomparable_parents():
    """Only COMPARABLE parents are reduced: a genuine orthogonal diamond (the
    ~50%-of-Mondo multi-parent case) survives intact."""
    pa = _diamond_parents()
    assert mn.induced_hasse_parents({_A, _B, _C, _D}, pa)[_D] == [_B, _C]


def test_kept_set_hasse_keeps_both_arms_when_both_survive():
    pa = _diamond_parents()
    dag, stats = mn.build_native_label_dag(
        {_A, _B, _C, _D}, pa, coded_ids={_A, _B, _C, _D}, names={})
    parents = _curies(dag)
    assert parents[_D] == [_B, _C]     # the diamond survives as a diamond
    assert stats["n_hasse_multi_parent"] >= 1
    assert stats["n_final_multi_parent"] >= 1


def test_no_kept_node_is_a_sibling_of_its_own_descendant():
    """The structural acceptance property (plan §3): in a transitive reduction a
    subsumed sibling cannot exist, which is what makes exp 0104's 619
    category-anchor trap impossible rather than merely unlikely."""
    pa = _diamond_parents()
    # Every kept subset, including the ones with a dropped intermediate — those
    # are exactly the cases nearest-per-branch alone gets wrong.
    subsets = [{_A, _B, _C, _D, _E}, {_A, _C, _D}, {_A, _C, _D, _E},
               {_A, _B, _D, _E}, {_A, _D, _E}, {_B, _C, _D}]
    for kept in subsets:
        dag, _ = mn.build_native_label_dag(kept, pa, coded_ids=kept, names={})
        children = dag.children()
        anc = {c: mn.ancestor_closure(mn.mondo_curie(c), pa) for c in dag.nodes()
               if c != mn.MONDO_NATIVE_ROOT_CID}
        for _p, kids in children.items():
            for a in kids:
                for b in kids:
                    if a == b or mn.MONDO_NATIVE_ROOT_CID in (a, b):
                        continue
                    assert mn.mondo_curie(a) not in anc[b], (
                        f"kept={sorted(kept)}: {a} is a sibling AND an "
                        f"ancestor of {b}")


# --------------------------------------------------------------------------- #
# splice interaction + terminal-as-property                                    #
# --------------------------------------------------------------------------- #
def test_splice_removes_an_uncoded_thin_chain_and_keeps_the_coded_one():
    """`is_terminal` is a PROPERTY lookup here, not exp 0109's positive-concept-id
    sign test: an uncoded only-child rung is spliced out, a coded one — a real,
    attestable disease — is not, even though both are structurally identical."""
    #  A -> B -> D   (B uncoded, only child D)   ; A -> C -> E (C CODED, only child E)
    children = {_A: [_B, _C], _B: [_D], _C: [_E]}
    pa = mn.parent_adjacency(children)
    dag, stats = mn.build_native_label_dag(
        {_A, _B, _C, _D, _E}, pa, coded_ids={_C, _D, _E}, names={})
    nodes = {mn.mondo_curie(c) for c in dag.nodes()
             if c != mn.MONDO_NATIVE_ROOT_CID}
    assert _B not in nodes             # uncoded rung: spliced
    assert _C in nodes                 # coded rung: kept
    assert nodes == {_A, _C, _D, _E}
    parents = _curies(dag)
    assert parents[_D] == [_A]         # D reattached past the spliced B
    assert stats["collapse"]["spliced"] == 1


def test_splice_preserves_multi_parenthood_when_it_rewires():
    """`collapse_only_child_classes` takes `{child: [parents]}` and rewires to the
    nearest surviving ANCESTORS (plural). Read here as a contract test on the
    module exp 0110 reuses rather than reimplements."""
    #  A -> X -> D ; B -> X ; so X has two parents and one child -> spliced, and
    #  D must inherit BOTH A and B. A and B each carry a second (terminal) child
    #  so they are not themselves only-child classes.
    parent_of = {"X": ["A", "B"], "D": ["X"], "T1": ["A"], "T2": ["B"],
                 "A": [], "B": []}
    out, stats = collapse_only_child_classes(
        parent_of, root="ROOT", is_terminal=lambda n: n in {"D", "T1", "T2"})
    assert "X" not in out
    assert out["D"] == ["A", "B"]
    assert stats["spliced"] == 1


def test_native_build_report_names_the_predicted_degenerate_count():
    pa = _diamond_parents()
    _, stats = mn.build_native_label_dag({_A, _B, _C, _D}, pa,
                                         coded_ids={_D}, names={})
    line = mn.format_native_build_report(stats)
    assert "native-mondo-v1" in line
    assert "predicted residual degenerate" in line


# --------------------------------------------------------------------------- #
# landing rule: attestations never fall through to the root                    #
# --------------------------------------------------------------------------- #
def test_roll_terms_to_kept_is_identity_on_kept_terms():
    pa = _diamond_parents()
    assert mn.roll_terms_to_kept([_D], {_D, _C}, pa) == {_D: [_D]}


def test_roll_terms_to_kept_lands_on_every_nearest_kept_ancestor():
    pa = _diamond_parents()
    # D unpowered: it lands on BOTH arms' nearest kept ancestors, not one.
    assert mn.roll_terms_to_kept([_D], {_B, _C, _A}, pa) == {_D: [_B, _C]}
    # only A kept: the walk climbs past both B and C.
    assert mn.roll_terms_to_kept([_D], {_A}, pa) == {_D: [_A]}
    # nothing kept above it: empty, NOT the root (the caller drops the code).
    assert mn.roll_terms_to_kept([_D], set(), pa) == {_D: []}


def test_every_attested_id_is_a_node_of_the_final_dag():
    """The invariant that keeps `attach_frontiers`'s root fallback from firing: an
    id the DAG has never heard of rolls up to `{root}` there, silently. Resolving
    the code map against the POST-splice node set is what prevents it."""
    children = {_A: [_B, _C], _B: [_D], _C: [_E]}
    pa = mn.parent_adjacency(children)
    coded = {_A, _D, _E}
    # D is CODED but unpowered (not in the kept set); B is powered but has no kept
    # child, C is a powered uncoded rung — both go, so D's landing is A.
    dag, _ = mn.build_native_label_dag({_A, _B, _C, _E}, pa, coded_ids=coded,
                                       names={})
    final = {mn.mondo_curie(c) for c in dag.nodes()
             if c != mn.MONDO_NATIVE_ROOT_CID}
    landing = mn.roll_terms_to_kept(coded, final, pa)
    for term, lands in landing.items():
        for land in lands:
            assert mn.mondo_cid(land) in dag.nodes(), (term, land)
    assert landing[_D] == [_A]         # unpowered, but it still attests upward


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
    # a code that ties to D and its own ancestors A/C keeps only D.
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
# plan §3 multi-parent pre-flight checklist, tested against the real modules   #
# --------------------------------------------------------------------------- #
def _diamond_layout():
    """Engine-id diamond: 0=root, 1=B, 2=C, 3=D(under both), 4=E(under C)."""
    from spark_vi.models.topic.dag_placement import DagLayout
    return DagLayout({1: [0], 2: [0], 3: [1, 2], 4: [2]}, n_bg=2, tpn=1)


def test_daglayout_closure_visits_a_diamond_once():
    lay = _diamond_layout()
    clo = lay.closure(3)
    assert sorted(clo) == [0, 1, 2, 3]
    assert len(clo) == len(set(clo))          # no double-visit
    assert clo[0] == 0                        # root first (depth, id) ordering


def test_daglayout_allowed_set_on_a_diamond_has_no_repeated_topic():
    lay = _diamond_layout()
    allowed = list(lay.allowed(3))
    assert allowed == sorted(set(allowed))
    # background block + one block per closure node except the root
    assert len(allowed) == lay.n_bg + 3 * lay.tpn


def test_daglayout_subtree_and_depth_are_diamond_safe():
    lay = _diamond_layout()
    assert lay.subtree(0) == {0, 1, 2, 3, 4}
    assert lay.depth(3) == 2                  # longest path root->B->D
    assert lay.descendants(2) == [4, 3] or lay.descendants(2) == [3, 4]


def test_frontier_to_label_sibling_expansion_unions_over_all_parents():
    """`frontier_to_label`'s closure mask observes the children of EVERY parent of
    every active node. Over a multi-parent node that union is what stops a diamond
    from hiding one arm's siblings. Read from `case_finding_assembly` as-is (it is
    source-hashed into every bundle key); asserted against, never modified."""
    from charmpheno.omop.case_finding_assembly import frontier_to_label

    lay = _diamond_layout()
    label, mask = frontier_to_label([3], lay, 5, label_mask_mode="closure")
    # positives: D's closure = {root, B, C, D}
    assert list(label) == [1.0, 1.0, 1.0, 1.0, 0.0]
    # observed adds E — a sibling of D reached through D's SECOND parent C.
    assert mask[4] == 1.0
    assert list(mask) == [1.0] * 5


def test_frontier_to_label_single_parent_arm_does_not_see_the_other_arm():
    """The control for the test above: from E (one parent) the mask reaches D,
    which is a sibling under C — but nothing that is only reachable via B."""
    from charmpheno.omop.case_finding_assembly import frontier_to_label

    lay = _diamond_layout()
    _, mask = frontier_to_label([4], lay, 5, label_mask_mode="closure")
    assert mask[3] == 1.0              # D is a sibling of E under C
    assert mask[1] == 1.0              # B is a sibling of C under the root


def test_dag_children_and_depth_is_multi_parent_safe():
    """`_dag_children_and_depth` feeds the conditional readout's cohorts; a child
    with two kept parents must appear under BOTH."""
    from gated_pc_cloud import _dag_children_and_depth

    children, depth = _dag_children_and_depth(
        {0: [], 1: [0], 2: [0], 3: [1, 2], 4: [2]}, 5)
    assert 3 in children[1] and 3 in children[2]
    assert depth[3] == 2


def test_engine_dag_roundtrips_mondo_ids_through_to_engine():
    """The id-agnostic spine, exercised on Mondo ids: `to_engine` must produce a
    contiguous 0..N map whose int2cid values are the Mondo node ids (ints), which
    is what `bundle_drift_report`'s `{int(i): int(c)}` re-read requires."""
    pa = _diamond_parents()
    dag, _ = mn.build_native_label_dag({_A, _B, _C, _D}, pa,
                                       coded_ids={_A, _B, _C, _D}, names={_A: "a"})
    parent_int, int2cid, cid2int = dag.to_engine()
    assert int2cid[0] == mn.MONDO_NATIVE_ROOT_CID
    assert sorted(int2cid) == list(range(len(int2cid)))
    assert all(isinstance(c, int) for c in int2cid.values())
    assert cid2int[mn.mondo_cid(_D)] in parent_int
    # every parent list is engine ids, and the diamond survived the remap
    assert len(parent_int[cid2int[mn.mondo_cid(_D)]]) == 2


# --------------------------------------------------------------------------- #
# the attestation seam (local Spark; the BQ build above it is cluster-covered)  #
# --------------------------------------------------------------------------- #
def test_native_attested_provider_maps_codes_to_label_nodes(spark):
    """A doc attests the Mondo label nodes its in-window codes resolve to; a doc
    whose codes resolve to nothing survives with an empty `attested_cids` (a `[]`
    frontier downstream), exactly like the SNOMED and legacy-Mondo providers."""
    import pandas as pd
    from charmpheno.omop.doc_spec import PatientCohortDocSpec

    code_map = spark.createDataFrame(pd.DataFrame({
        "std_cid": [5001, 5002, 5003],
        "node_cid": [mn.mondo_cid(_D), mn.mondo_cid(_D), mn.mondo_cid(_E)]}))
    events = spark.createDataFrame(pd.DataFrame({
        "person_id": [1, 2, 2, 3],
        "concept_id": [5001, 5003, 5002, 9999],   # p3's 9999 resolves to nothing
        "source_cohort": ["population"] * 4}))
    provider = mn.make_mondo_native_attested_provider(
        code_map, doc_spec=PatientCohortDocSpec())
    got = {r["doc_id"]: sorted(r["attested_cids"])
           for r in provider(events).collect()}

    assert got["population:1"] == [mn.mondo_cid(_D)]
    assert got["population:2"] == sorted({mn.mondo_cid(_D), mn.mondo_cid(_E)})
    assert got["population:3"] == []           # background doc survives, empty


def test_native_attested_ids_are_bigints_the_assembler_can_cast(spark):
    """`case_finding_assembly.attach_frontiers` hard-casts every attested id with
    `int(c)`; the column must therefore be `array<bigint>`, not strings. This is
    the test that pins the engine-id decision recorded in the module docstring."""
    import pandas as pd
    from charmpheno.omop.doc_spec import PatientCohortDocSpec

    code_map = spark.createDataFrame(pd.DataFrame(
        {"std_cid": [5001], "node_cid": [mn.mondo_cid(_D)]}))
    events = spark.createDataFrame(pd.DataFrame(
        {"person_id": [1], "concept_id": [5001], "source_cohort": ["population"]}))
    out = mn.make_mondo_native_attested_provider(
        code_map, doc_spec=PatientCohortDocSpec())(events)
    assert dict(out.dtypes)["attested_cids"] == "array<bigint>"
    assert [int(c) for c in out.collect()[0]["attested_cids"]] == [mn.mondo_cid(_D)]
