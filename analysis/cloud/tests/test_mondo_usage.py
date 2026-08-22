"""Pure-logic tests for the whole-Mondo EHR-usage export (exact map, no roll-up):
the three-state small-cell rule, collision detection, nearest-mapped-ancestor
edges, and payload assembly. The BQ/Spark path is cluster-covered, not unit-tested.
"""
import mondo_usage_cloud as m


def test_three_state_suppression():
    # unused vs used-small vs reported are DISTINCT states — a used-small term is
    # never conflated with 0 and never given an exact number.
    assert m.usage_state(0) == ("unused", "0", 0)
    assert m.usage_state(1) == ("used_small", "<20", None)
    assert m.usage_state(19) == ("used_small", "<20", None)
    assert m.usage_state(20) == ("reported", "20", 20)
    assert m.usage_state(6543) == ("reported", "6543", 6543)


def test_collision_map_and_siblings():
    # std 100 is shared by two Mondo terms (OMOP Maps-to coarsening); std 200 is not.
    pairs = [(100, "MONDO:A"), (100, "MONDO:B"), (200, "MONDO:A")]
    s2m = m.collision_map(pairs)
    assert s2m[100] == ["MONDO:A", "MONDO:B"]
    assert s2m[200] == ["MONDO:A"]
    term_std = {"MONDO:A": [100, 200], "MONDO:B": [100]}
    sib = m.term_collision_siblings(term_std, s2m)
    assert sib["MONDO:A"] == ["MONDO:B"]      # shares std 100 with B
    assert sib["MONDO:B"] == ["MONDO:A"]
    # a term whose only std concept is unshared has no siblings
    assert m.term_collision_siblings({"MONDO:C": [300]}, s2m)["MONDO:C"] == []


def test_nearest_mapped_parents_collapses_unmapped_intermediates():
    # child -> parents; only ROOTMAP and LEAF are mapped, MID is an unmapped
    # intermediate that must be skipped so LEAF attaches to ROOTMAP.
    parent_adj = {"LEAF": ["MID"], "MID": ["ROOTMAP"], "ROOTMAP": ["TOP"]}
    mapped = {"LEAF", "ROOTMAP"}
    out = m.nearest_mapped_parents(mapped, parent_adj)
    assert out["LEAF"] == ["ROOTMAP"]         # MID collapsed
    assert out["ROOTMAP"] == []               # no mapped ancestor -> root


def test_nearest_mapped_parents_stops_at_first_mapped():
    # two mapped ancestors on one chain: keep only the nearest.
    parent_adj = {"LEAF": ["NEAR"], "NEAR": ["FAR"]}
    out = m.nearest_mapped_parents({"LEAF", "NEAR", "FAR"}, parent_adj)
    assert out["LEAF"] == ["NEAR"]            # FAR not included (past NEAR)


def _rows():
    return [
        {"mondo_id": "MONDO:HTN", "label": "hypertension", "is_internal": True,
         "parents": [], "std_concepts": [316866], "n_persons": 40000,
         "collision_siblings": []},
        {"mondo_id": "MONDO:RARE", "label": "rare thing", "is_internal": False,
         "parents": ["MONDO:HTN"], "std_concepts": [999], "n_persons": 5,
         "collision_siblings": ["MONDO:DUP"]},
        {"mondo_id": "MONDO:DUP", "label": "dup thing", "is_internal": False,
         "parents": ["MONDO:HTN"], "std_concepts": [999], "n_persons": 0,
         "collision_siblings": ["MONDO:RARE"]},
    ]


def test_assemble_payload_states_stats_and_root():
    p = m.assemble_payload(meta={"rollup": False}, term_rows=_rows())
    by_id = {n["id"]: n for n in p["nodes"]}
    assert by_id["root"]["kind"] == "root"
    assert by_id["MONDO:HTN"]["state"] == "reported" and by_id["MONDO:HTN"]["count"] == 40000
    assert by_id["MONDO:RARE"]["state"] == "used_small" and by_id["MONDO:RARE"]["count"] is None
    assert by_id["MONDO:DUP"]["state"] == "unused"
    # HTN had no mapped parent -> attaches to root; RARE keeps its mapped parent.
    assert by_id["MONDO:HTN"]["parents"] == ["root"]
    assert by_id["MONDO:RARE"]["parents"] == ["MONDO:HTN"]
    # depth: root 0, HTN 1, RARE 2
    assert by_id["MONDO:HTN"]["depth"] == 1 and by_id["MONDO:RARE"]["depth"] == 2
    # four display categories: HTN reported, RARE used_small, DUP is 0-count but
    # sits ABOVE nothing used itself yet IS a child of HTN... it's a used_small's
    # sibling — DUP has no used descendant, so it is "other".
    assert by_id["MONDO:HTN"]["category"] == "reported"
    assert by_id["MONDO:RARE"]["category"] == "used_small"
    assert by_id["MONDO:DUP"]["category"] == "other"
    s = p["stats"]
    assert s["mapped_terms"] == 3
    assert s["used_terms"] == 2 and s["used_small_terms"] == 1 and s["reported_terms"] == 1
    assert s["unused_terms"] == 1
    assert s["used_branch_terms"] == 0 and s["other_terms"] == 1
    assert s["collision_terms"] == 2          # RARE + DUP both flagged
    assert s["internal_terms"] == 1 and s["internal_used_terms"] == 1


def test_rare_flag_passthrough_and_stats():
    rows = [
        {"mondo_id": "MONDO:R", "label": "rare dx", "is_internal": False, "parents": [],
         "std_concepts": [1], "n_persons": 30, "collision_siblings": [],
         "rare": True, "rare_src": ["Orphanet", "GARD"]},
        {"mondo_id": "MONDO:C", "label": "common dx", "is_internal": False, "parents": [],
         "std_concepts": [2], "n_persons": 500, "collision_siblings": [],
         "rare": False, "rare_src": []},
        {"mondo_id": "MONDO:RU", "label": "rare unused", "is_internal": False, "parents": [],
         "std_concepts": [3], "n_persons": 0, "collision_siblings": [],
         "rare": True, "rare_src": ["NORD"]},
    ]
    p = m.assemble_payload(meta={}, term_rows=rows)
    by = {n["id"]: n for n in p["nodes"]}
    assert by["MONDO:R"]["rare"] and by["MONDO:R"]["rare_src"] == ["Orphanet", "GARD"]
    assert by["MONDO:C"]["rare"] is False
    assert p["stats"]["rare_terms"] == 2          # R + RU
    assert p["stats"]["rare_used_terms"] == 1     # only R is used (RU has 0 patients)


def test_used_branch_category_for_zero_count_ancestor():
    # A 0-count internal ancestor that sits ABOVE a used leaf is "used_branch",
    # not "other".
    rows = [
        {"mondo_id": "MONDO:SYS", "label": "system disorder", "is_internal": True,
         "parents": [], "std_concepts": [1], "n_persons": 0, "collision_siblings": []},
        {"mondo_id": "MONDO:USED", "label": "used leaf", "is_internal": False,
         "parents": ["MONDO:SYS"], "std_concepts": [2], "n_persons": 50,
         "collision_siblings": []},
    ]
    by_id = {n["id"]: n for n in m.assemble_payload(meta={}, term_rows=rows)["nodes"]}
    assert by_id["MONDO:SYS"]["state"] == "unused"
    assert by_id["MONDO:SYS"]["category"] == "used_branch"   # 0 count, above a used node
    assert by_id["MONDO:USED"]["category"] == "reported"


def test_summary_renders_and_suppresses_nothing_exact():
    p = m.assemble_payload(meta={}, term_rows=_rows())
    out = m.format_summary(p["stats"])
    assert "WHOLE-MONDO EHR USAGE" in out and "no roll-up" in out
    assert "40000" not in out                 # exact patient counts never in the summary
    assert "3" in out                          # term counts (exact) do appear
