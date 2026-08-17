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


# --- freeze stage (select_frozen_anchors) ---


def _cand(cid, name, count, maximal=True):
    return {"standard_concept_id": cid, "standard_concept_name": name,
            "positive_count": count, "is_maximal": str(maximal)}


def test_freeze_pins_rare6_applies_band_and_nesting():
    from anchor_neighborhoods import select_frozen_anchors

    rows = [
        _cand(100, "keep me", 800),                      # in band, maximal -> kept
        _cand(101, "too broad", 27000),                  # over ceiling -> dropped
        _cand(102, "too thin", 10),                      # under floor -> dropped
        _cand(103, "nested ancestor", 500, maximal=False),  # not maximal -> dropped
        _cand(104, "excluded umbrella", 3000),           # in exclude list -> dropped
    ]
    frozen = select_frozen_anchors(
        rows, floor=50, ceiling=10000,
        rare6={999: "Sarcoidosis"}, exclude_ids=[104],
    )
    by_id = {r["concept_id"]: r for r in frozen}
    assert set(by_id) == {999, 100}          # rare6 pinned + the one banded/maximal anchor
    assert by_id[999]["source"] == "rare6"
    assert by_id[100]["source"] == "prioritised"
    # rare6 first in output order
    assert frozen[0]["concept_id"] == 999


def test_freeze_rare6_in_candidates_is_pinned_not_duplicated():
    from anchor_neighborhoods import select_frozen_anchors

    # A rare6 anchor also appears among the prioritised candidates (e.g. MG).
    rows = [_cand(76685, "Myasthenia gravis", 787)]
    frozen = select_frozen_anchors(rows, floor=50, ceiling=10000)  # default rare6 map
    mg = [r for r in frozen if r["concept_id"] == 76685]
    assert len(mg) == 1 and mg[0]["source"] == "rare6"
    assert mg[0]["positive_count"] == 787  # count carried from candidates
