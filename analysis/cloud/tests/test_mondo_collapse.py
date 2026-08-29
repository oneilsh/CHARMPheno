"""Unit tests for the splice-to-fixpoint label-DAG reduction (exp 0109).

The claim under test is the one the whole experiment rests on: exp 0104's 763
degenerate readout cells are exactly ``{root} u {class nodes with one kept
child}``, so removing the only-children (and the childless classes that removal
leaves behind) drives the predicted degenerate count to 1 — the root alone.

Everything here is pure integer-graph work in `mondo_dag`'s id space: terminals
(the powered OMOP anchors) are POSITIVE, class nodes are SYNTHETIC NEGATIVES, the
forest root is -1. The tests use small hand-built maps rather than Mondo frames,
because the property being asserted is graph-structural.
"""
import pytest

from mondo_collapse import (DAG_COLLAPSE_VERSION, collapse_engine_dag,
                            collapse_only_child_classes, format_collapse_report)

ROOT = -1


def _children(parent_of, root=ROOT):
    """{parent: sorted children} over a {child: [parents]} map, for assertions."""
    out = {}
    for child, parents in parent_of.items():
        for p in parents:
            out.setdefault(p, set()).add(child)
    return {p: sorted(cs) for p, cs in out.items()}


def _no_only_child_classes(parent_of, root=ROOT):
    """The prediction the diagnostic publishes: after the fixpoint, no non-root
    class node has exactly one kept child."""
    ch = _children(parent_of, root)
    return [n for n in ({root} | set(parent_of))
            if n != root and n < 0 and len(ch.get(n, ())) == 1]


# --------------------------------------------------------------------------- #
# 1. the splice itself                                                         #
# --------------------------------------------------------------------------- #
def test_chain_of_class_nodes_collapses_to_the_terminal():
    """A -> B -> C -> terminal: B and C are only-children rungs, so the terminal
    ends up hanging directly off A. This is the exact shape Mondo's multi-axis
    cover-stealing produces at whole-Mondo scale."""
    # class ids -2 (A), -3 (B), -4 (C); terminal 1001.
    parent_of = {-2: [ROOT], -3: [-2], -4: [-3], 1001: [-4], 2002: [-2]}
    out, stats = collapse_only_child_classes(parent_of, ROOT)

    assert out == {-2: [ROOT], 1001: [-2], 2002: [-2]}
    assert stats["spliced"] == 2 and stats["dropped_childless"] == 0
    # ONE pass: the batch rewire walks the terminal past C AND B in a single
    # nearest-surviving-ancestor walk.
    assert stats["passes"] == 1
    assert stats["n_nodes_before"] == 6 and stats["n_nodes_after"] == 4
    assert _no_only_child_classes(out) == []


def test_terminals_are_never_spliced():
    """A terminal with exactly one child is a REAL disease node (patients attest
    it); only abstract class nodes are scaffolding."""
    parent_of = {1001: [ROOT], 1002: [1001]}
    out, stats = collapse_only_child_classes(parent_of, ROOT)
    assert out == {1001: [ROOT], 1002: [1001]}
    assert stats["spliced"] == 0 and stats["n_nodes_after"] == stats["n_nodes_before"]


def test_multi_child_class_nodes_are_untouched():
    """A genuine branch point discriminates between its children, so it has a
    non-degenerate observed cell and must survive verbatim."""
    parent_of = {-2: [ROOT], 1001: [-2], 1002: [-2], 1003: [-2]}
    out, stats = collapse_only_child_classes(parent_of, ROOT)
    assert out == parent_of
    assert stats["spliced"] == 0 and stats["dropped_childless"] == 0
    assert stats["passes"] == 0


def test_childless_class_nodes_are_dropped():
    """A class node serving no kept child serves nothing at all — it is a topic
    block and a readout row with no label content behind it."""
    parent_of = {-2: [ROOT], -3: [ROOT], 1001: [-2], 1002: [-2]}   # -3 has no kids
    out, stats = collapse_only_child_classes(parent_of, ROOT)
    assert -3 not in out
    assert out == {-2: [ROOT], 1001: [-2], 1002: [-2]}
    assert stats["dropped_childless"] == 1 and stats["spliced"] == 0


def test_root_survives_even_though_it_stays_degenerate():
    """Structural: the root is what makes the forest connected, and it is ONE node.
    exp 0109 predicts a residual degenerate count of exactly 1 for that reason."""
    parent_of = {-2: [ROOT], 1001: [-2]}
    out, stats = collapse_only_child_classes(parent_of, ROOT)
    assert out == {1001: [ROOT]}                 # -2 spliced, terminal onto root
    assert stats["predicted_degenerate"] == 1
    assert stats["residual_only_children"] == 0


# --------------------------------------------------------------------------- #
# 2. the FIXPOINT (one pass is not enough)                                     #
# --------------------------------------------------------------------------- #
def test_fixpoint_when_a_pass_creates_a_new_only_child():
    """-2's two children are separate chains onto the SAME terminal. Pass 1 splices
    both chains; -2 is then an only-child parent itself and pass 2 removes it. A
    single-pass implementation would leave a degenerate node behind."""
    parent_of = {-2: [ROOT], -3: [-2], -4: [-2], 1001: [-3, -4], 2002: [ROOT]}
    out, stats = collapse_only_child_classes(parent_of, ROOT)
    assert stats["passes"] >= 2
    assert out == {1001: [ROOT], 2002: [ROOT]}
    assert _no_only_child_classes(out) == []


def test_fixpoint_when_dropping_a_childless_class_orphans_its_parent():
    """-3 is childless. Dropping it leaves -2 with one child, which the next pass
    splices — the drop and the splice are the same fixpoint."""
    parent_of = {-2: [ROOT], -3: [-2], 1001: [-2], 2002: [ROOT]}
    out, stats = collapse_only_child_classes(parent_of, ROOT)
    assert stats["dropped_childless"] == 1 and stats["spliced"] == 1
    assert stats["passes"] >= 2
    assert out == {1001: [ROOT], 2002: [ROOT]}


def test_nested_chains_reach_the_prediction_no_only_child_remains():
    """A deliberately gnarly mix — deep chains, a real branch point, a childless
    class, a multi-parent class — must land on the prediction that drives the whole
    experiment: NO non-root class node with a single kept child."""
    parent_of = {
        -2: [ROOT],                      # real branch point (-3, -6 below)
        -3: [-2], -4: [-3], -5: [-4],    # chain down to two terminals
        1001: [-5], 1002: [-5],
        -6: [-2], -7: [-6],              # chain down to one terminal
        1003: [-7],
        -8: [-2],                        # childless
        -9: [-3, -6], 1004: [-9],        # multi-parent class, one child
    }
    out, stats = collapse_only_child_classes(parent_of, ROOT)
    assert _no_only_child_classes(out) == []
    assert stats["residual_only_children"] == 0
    assert stats["predicted_degenerate"] == 1
    # every terminal survives, and nothing else positive appeared
    assert {n for n in out if n > 0} == {1001, 1002, 1003, 1004}
    assert stats["n_terminals"] == 4


def test_multi_parent_class_rewires_its_child_to_all_kept_ancestors():
    """Splicing keeps the DAG's multi-parent structure: the surviving child gets
    the spliced node's parents, not just one of them."""
    parent_of = {-2: [ROOT], -3: [ROOT], -4: [-2, -3], 1001: [-4],
                 1002: [-2], 1003: [-3]}
    out, stats = collapse_only_child_classes(parent_of, ROOT)
    assert stats["spliced"] == 1                      # only -4 qualifies
    assert out[1001] == sorted([-2, -3])              # BOTH parents, deduped+sorted
    assert -4 not in out


def test_everything_collapses_to_a_bare_root_when_there_is_one_terminal():
    parent_of = {-2: [ROOT], -3: [-2], 1001: [-3]}
    out, stats = collapse_only_child_classes(parent_of, ROOT)
    assert out == {1001: [ROOT]}
    assert stats["n_classes_after"] == 0 and stats["predicted_degenerate"] == 1


def test_is_terminal_override_lets_a_caller_name_its_own_id_space():
    """The positive/negative convention is `mondo_dag`'s, not the algorithm's."""
    parent_of = {"a": [], "b": ["a"], "t": ["b"]}
    out, stats = collapse_only_child_classes(
        parent_of, "root", is_terminal=lambda n: n == "t")
    # "a"/"b" are only-child classes; both splice, "t" lands on the root.
    assert out == {"t": ["root"]}
    assert stats["spliced"] == 2


# --------------------------------------------------------------------------- #
# 3. the ConditionDag wrapper + the diagnostic line                            #
# --------------------------------------------------------------------------- #
def test_collapse_engine_dag_preserves_names_and_terminals():
    from mondo_dag import build_mondo_engine_dag

    # anchor:1001 under a chain MONDO:A -> MONDO:B; anchor:1002 beside it.
    parent_of = {"anchor:1001": ["MONDO:B"], "MONDO:B": ["MONDO:A"],
                 "MONDO:A": [], "anchor:1002": []}
    dag = build_mondo_engine_dag(
        parent_of, anchor_names={1001: "a1", 1002: "a2"},
        class_names={"MONDO:A": "class a", "MONDO:B": "class b"})
    assert len(dag.nodes()) == 5                       # root + 2 classes + 2 anchors

    collapsed, stats = collapse_engine_dag(dag)
    assert collapsed.anchor == dag.anchor
    assert collapsed.nodes() == {ROOT, 1001, 1002}     # both classes spliced away
    assert collapsed.parents[1001] == [ROOT]
    assert stats["spliced"] == 2
    # names survive for the survivors and are dropped for the spliced nodes.
    assert collapsed.names[1001] == "a1" and collapsed.names[1002] == "a2"
    assert set(collapsed.names) == {ROOT, 1001, 1002}
    # and it still remaps into engine-id space (root -> 0), which is all the
    # multi-domain assembler needs of it.
    parent_int, int2cid, cid2int = collapsed.to_engine()
    assert cid2int[ROOT] == 0
    assert parent_int[cid2int[1001]] == [0]


def test_format_collapse_report_names_the_numbers_that_matter():
    parent_of = {-2: [ROOT], -3: [-2], 1001: [-3], 1002: [ROOT], -4: [ROOT]}
    _, stats = collapse_only_child_classes(parent_of, ROOT)
    line = format_collapse_report(stats)
    assert line.startswith("[mondo]   dag-collapse (")
    assert DAG_COLLAPSE_VERSION in line
    assert "spliced 2" in line and "dropped 1 childless" in line
    assert "predicted residual degenerate = 1" in line


def test_collapse_is_idempotent():
    """Running the reduction on an already-reduced DAG must be a no-op — that is
    what 'fixpoint' means, and it is what lets a cached collapsed bundle be
    re-derived without drifting."""
    parent_of = {-2: [ROOT], -3: [-2], -4: [-2], 1001: [-3], 1002: [-3],
                 1003: [-4], 1004: [-4]}
    once, _ = collapse_only_child_classes(parent_of, ROOT)
    twice, stats2 = collapse_only_child_classes(once, ROOT)
    assert twice == once
    assert stats2["passes"] == 0 and stats2["spliced"] == 0
