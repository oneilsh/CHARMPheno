"""Tests for the nesting rule (maximal_anchors)."""


def test_nested_chain_keeps_deepest_clearing():
    from anchor_neighborhoods import maximal_anchors

    # 1 -> 2 -> 3 (1 is broadest, 3 most specific), all clear the floor.
    clearing = {1, 2, 3}
    ancestry = [(1, 2), (1, 3), (2, 3)]  # includes the transitive (1,3)
    assert maximal_anchors(clearing, ancestry) == {3}


def test_incomparable_anchors_all_kept():
    from anchor_neighborhoods import maximal_anchors

    clearing = {10, 20, 30}
    assert maximal_anchors(clearing, []) == {10, 20, 30}


def test_ancestor_of_non_clearing_descendant_is_kept():
    from anchor_neighborhoods import maximal_anchors

    # 1 -> 2, but 2 does not clear the floor -> 1 is the most specific clearing.
    clearing = {1}
    ancestry = [(1, 2)]
    assert maximal_anchors(clearing, ancestry) == {1}


def test_self_pairs_ignored():
    from anchor_neighborhoods import maximal_anchors

    clearing = {5}
    assert maximal_anchors(clearing, [(5, 5)]) == {5}


def test_forest_keeps_deepest_per_chain():
    from anchor_neighborhoods import maximal_anchors

    # chain 1->2->3 and separate chain 4->5, plus isolate 9
    clearing = {1, 2, 3, 4, 5, 9}
    ancestry = [(1, 2), (2, 3), (1, 3), (4, 5)]
    assert maximal_anchors(clearing, ancestry) == {3, 5, 9}
