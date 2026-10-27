"""`spark_vi.dag.ops`: the generic graph primitives (ADR 0051).

The transitive reduction and the closure product have their own tests in the charmpheno
native-DAG suite and the readout's stacked tests; these pin the primitives below them.
"""
import numpy as np

from spark_vi.dag.ops import (ancestor_closure, children_and_depth, closure_matrix,
                              descendants, nearest_kept_ancestors, nearest_kept_parents,
                              parent_adjacency, transitive_reduction_parents)


def test_nearest_mapped_parents_collapses_unmapped_intermediates():
    # child -> parents; only ROOTMAP and LEAF are mapped, MID is an unmapped
    # intermediate that must be skipped so LEAF attaches to ROOTMAP.
    parent_adj = {"LEAF": ["MID"], "MID": ["ROOTMAP"], "ROOTMAP": ["TOP"]}
    mapped = {"LEAF", "ROOTMAP"}
    out = nearest_kept_parents(mapped, parent_adj)
    assert out["LEAF"] == ["ROOTMAP"]         # MID collapsed
    assert out["ROOTMAP"] == []               # no mapped ancestor -> root


def test_nearest_mapped_parents_stops_at_first_mapped():
    # two mapped ancestors on one chain: keep only the nearest.
    parent_adj = {"LEAF": ["NEAR"], "NEAR": ["FAR"]}
    out = nearest_kept_parents({"LEAF", "NEAR", "FAR"}, parent_adj)
    assert out["LEAF"] == ["NEAR"]            # FAR not included (past NEAR)


def test_closure_descendants_and_adjacency_inversion():
    child_adj = {"a": ["b", "c"], "b": ["d"], "c": ["d"]}
    parents = parent_adjacency(child_adj)
    assert parents == {"b": ["a"], "c": ["a"], "d": ["b", "c"]}
    assert ancestor_closure("d", parents) == {"a", "b", "c", "d"}
    assert descendants(child_adj, "a") == {"b", "c", "d"}


def test_nearest_kept_ancestors_lands_on_the_first_kept_node_per_branch():
    parents = {"d": ["b", "c"], "b": ["a"], "c": ["a"]}
    assert nearest_kept_ancestors("d", {"a", "c"}, parents) == {"a", "c"}
    assert nearest_kept_ancestors("c", {"a", "c"}, parents) == {"c"}
    assert nearest_kept_ancestors("z", {"a"}, parents) == set()


def test_transitive_reduction_drops_the_redundant_sibling_edge():
    """A - B - D and A - C - D with B dropped: nearest-per-branch gives D the
    parents {A, C}, but C is below A, so the reduction keeps only C."""
    parents = {"b": ["a"], "c": ["a"], "d": ["b", "c"]}
    assert nearest_kept_parents({"a", "c", "d"}, parents)["d"] == ["a", "c"]
    red = transitive_reduction_parents({"a", "c", "d"}, parents)
    assert red == {"a": [], "c": ["a"], "d": ["c"]}


def test_depth_and_closure_matrix_on_a_diamond():
    parent_int = {1: [0], 2: [0], 3: [1, 2]}
    _children, depth = children_and_depth(parent_int, 4)
    assert depth == {0: 0, 1: 1, 2: 1, 3: 2}
    M = closure_matrix(parent_int, 4)
    np.testing.assert_array_equal(M[3], [1, 1, 1, 1])     # root counted once
    np.testing.assert_array_equal(M[1], [1, 1, 0, 0])
