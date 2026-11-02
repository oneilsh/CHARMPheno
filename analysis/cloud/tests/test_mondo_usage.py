"""Pure-logic tests for the whole-Mondo EHR-usage export (exact map, no roll-up):
the three-state small-cell rule, collision detection, nearest-mapped-ancestor
edges, and payload assembly. The BQ/Spark path is cluster-covered, not unit-tested.
"""
import mondo_usage_cloud as m


def test_three_state_suppression():
    # unused vs used-small vs reported are DISTINCT states — a used-small term is
    # never conflated with 0 and never given an exact number.
    assert m.usage_state(0) == ("unused", "0", 0)
    assert m.usage_state(1) == ("used_small", "≤20", None)
    assert m.usage_state(19) == ("used_small", "≤20", None)
    # floor is INCLUSIVE — a count of exactly 20 is masked (matches AoU's "≤ 20"),
    # only counts strictly above the floor are reported exactly.
    assert m.usage_state(20) == ("used_small", "≤20", None)
    assert m.usage_state(21) == ("reported", "21", 21)
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


def test_meaningful_skeleton_collapses_linear_keeps_branch_points():
    # linear chain LEAF -> MID -> TOP: MID/TOP are single-child pass-throughs -> dropped
    assert m.meaningful_skeleton({"LEAF"}, {"LEAF": ["MID"], "MID": ["TOP"], "TOP": []}) == {"LEAF"}
    # BP has two seed-bearing children (A, B) -> kept; TOP above it is linear -> dropped
    padj = {"A": ["BP"], "B": ["BP"], "BP": ["TOP"], "TOP": []}
    assert m.meaningful_skeleton({"A", "B"}, padj) == {"A", "B", "BP"}
    # a seed node that is itself also an ancestor stays; an un-attributed leaf-side node
    # with only ONE seed-bearing child collapses even if it has other non-seed children
    padj2 = {"X": ["MID"], "MID": ["ROOT"], "Y": ["ROOT"], "ROOT": []}
    # seed {X, Y}: ROOT has two seed-bearing children (MID->X, and Y) -> ROOT kept; MID linear -> dropped
    assert m.meaningful_skeleton({"X", "Y"}, padj2) == {"X", "Y", "ROOT"}


def test_meaningful_skeleton_empty_and_seed_only():
    assert m.meaningful_skeleton(set(), {}) == set()
    # a lone seed with un-attributed single-parent chain keeps only the seed
    assert m.meaningful_skeleton({"S"}, {"S": ["P"], "P": []}) == {"S"}


def test_nearest_mapped_standard_ancestors_picks_min_distance():
    # concept_ancestor climb (source_climb space): descendant 100 has mapped
    # standard ancestors at levels 1 and 2; only the NEAREST is kept.
    edges = [(100, 10, 1), (100, 20, 2)]
    assert m.nearest_mapped_standard_ancestors(edges) == {100: [10]}


def test_nearest_mapped_standard_ancestors_keeps_ties():
    # two mapped ancestors equally near (both level 2) -> keep BOTH so the term
    # rows can be flagged as a shared-attribution collision. Level-3 dropped.
    edges = [(100, 10, 2), (100, 20, 2), (100, 30, 3)]
    assert m.nearest_mapped_standard_ancestors(edges) == {100: [10, 20]}


def test_nearest_mapped_standard_ancestors_excludes_distance_zero_and_empty():
    # a self/exact edge (level 0) is NOT a climb; no mapped ancestor -> absent.
    assert m.nearest_mapped_standard_ancestors([(100, 100, 0)]) == {}
    assert m.nearest_mapped_standard_ancestors([]) == {}
    # independent descendants are resolved separately
    assert m.nearest_mapped_standard_ancestors(
        [(1, 5, 1), (2, 6, 4), (2, 7, 4)]) == {1: [5], 2: [6, 7]}


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


def test_code_multiplicity_passthrough_and_stats():
    rows = [
        {"mondo_id": "MONDO:A", "label": "a", "is_internal": False, "parents": [],
         "std_concepts": [1, 2], "n_persons": 40, "collision_siblings": [],
         "codes": [{"id": 11, "vocab": "SNOMED", "code": "x"},
                   {"id": 22, "vocab": "ICD10CM", "code": "y"}],
         "n_codes": 2, "codes_by_vocab": {"SNOMED": 1, "ICD10CM": 1}},
        {"mondo_id": "MONDO:B", "label": "b", "is_internal": False, "parents": [],
         "std_concepts": [3], "n_persons": 25, "collision_siblings": [],
         "codes": [{"id": 33, "vocab": "SNOMED", "code": "z"}],
         "n_codes": 1, "codes_by_vocab": {"SNOMED": 1}},
    ]
    p = m.assemble_payload(meta={}, term_rows=rows)
    by = {n["id"]: n for n in p["nodes"]}
    assert by["MONDO:A"]["n_codes"] == 2 and by["MONDO:A"]["codes_by_vocab"]["ICD10CM"] == 1
    assert len(by["MONDO:A"]["codes"]) == 2
    assert p["stats"]["total_codes"] == 3            # 2 + 1
    assert p["stats"]["multi_code_terms"] == 1       # only A maps >1 code


def test_reduce_tie_map_keeps_most_specific():
    # Mondo child->parents: carbuncle -> pyoderma -> skin; hereditary & endocrine are roots.
    parents = {"carbuncle": ["pyoderma"], "pyoderma": ["skin"], "skin": [],
               "hereditary": [], "endocrine": []}
    pairs = [
        (1, "carbuncle"), (1, "pyoderma"), (1, "skin"),   # nested chain -> just carbuncle
        (2, "hereditary"), (2, "endocrine"),              # orthogonal tie -> both kept
        (3, "skin"),                                       # single -> unchanged
    ]
    out = m.reduce_tie_map(pairs, parents)
    assert out[1] == ["carbuncle"]                         # ancestors pyoderma/skin dropped
    assert out[2] == ["endocrine", "hereditary"]           # neither is the other's ancestor
    assert out[3] == ["skin"]


def test_classify_collision_kinds_splits_mechanism():
    # code 100 hits A,B via EXACT (standard coarsening) -> shared_concept;
    # code 200 hits B,C via CLIMB -> climb_tie; code 300 hits D exact + E climbed -> mixed.
    pairs = [(100, "A", "standard_exact"), (100, "B", "standard_exact"),
             (200, "B", "climbed"), (200, "C", "climbed"),
             (300, "D", "source_exact"), (300, "E", "climbed"),
             (9, "Z", "source_exact")]                 # code 9 -> one term, no collision
    siblings, term_kind, code_kind = m.classify_collision_kinds(pairs)
    assert code_kind == {100: "shared_concept", 200: "climb_tie", 300: "mixed"}
    assert siblings["A"] == ["B"] and term_kind["A"] == "shared_concept"
    assert siblings["C"] == ["B"] and term_kind["C"] == "climb_tie"
    # B collides via a shared_concept code (100) AND a climb_tie code (200) -> mixed
    assert siblings["B"] == ["A", "C"] and term_kind["B"] == "mixed"
    assert term_kind["D"] == "mixed" and term_kind["E"] == "mixed"
    assert "Z" not in siblings           # single-term code is not a collision


def test_suppress_count_floor_inclusive():
    assert m.suppress_count(0) == "0"
    assert m.suppress_count(1) == "≤20"
    assert m.suppress_count(20) == "≤20"        # floor inclusive
    assert m.suppress_count(21) == "21"
    assert m.suppress_count(None) == "n/a"


def test_parse_hpo_dag():
    obo = """format-version: 1.2

[Term]
id: HP:0000118
name: Phenotypic abnormality

[Term]
id: HP:0002917
name: Hypomagnesemia
is_a: HP:0004363 ! Abnormal circulating metabolite concentration
is_a: HP:0012418 ! Hypomagnesemia parent two

[Term]
id: HP:0000001
name: All

[Term]
id: HP:9999999
name: obsolete thing
is_a: HP:0000118 ! Phenotypic abnormality
is_obsolete: true

[Typedef]
id: part_of
"""
    labels, parents = m.parse_hpo_dag(obo)
    assert labels["HP:0002917"] == "Hypomagnesemia"
    assert labels["HP:0000118"] == "Phenotypic abnormality"
    assert sorted(parents["HP:0002917"]) == ["HP:0004363", "HP:0012418"]
    assert parents.get("HP:0000118", []) == []          # a root: no is_a
    assert "HP:9999999" not in labels                    # obsolete excluded
    assert "HP:9999999" not in parents


def test_normalize_xref_vocab():
    assert m.normalize_xref_vocab("SNOMEDCT_US") == "SNOMED"
    assert m.normalize_xref_vocab("snomedct_us") == "SNOMED"     # case-insensitive
    assert m.normalize_xref_vocab("ICD-10") == "ICD10CM"
    assert m.normalize_xref_vocab("ICD9CM") == "ICD9CM"
    assert m.normalize_xref_vocab("UMLS") == "UMLS"
    assert m.normalize_xref_vocab("ORPHA") is None              # not one we match


def test_parse_hpo_xrefs():
    obo = """format-version: 1.2

[Term]
id: HP:0002917
name: Hypomagnesemia
def: "A decreased magnesium level." []
xref: SNOMEDCT_US:190855004
xref: UMLS:C0151723 {source="MONDO"}
xref: ICD-10:E83.42 ! Disorders of magnesium metabolism

[Term]
id: HP:0000001
name: All
xref: UMLS:C0444868

[Typedef]
id: part_of
xref: SNOMEDCT_US:9999999
"""
    rows = m.parse_hpo_xrefs(obo)
    # SNOMED + normalized ICD-10 + UMLS captured for the real term; comments/qualifiers stripped
    assert ("HP:0002917", "Hypomagnesemia", "SNOMED", "190855004") in rows
    assert ("HP:0002917", "Hypomagnesemia", "ICD10CM", "E83.42") in rows
    assert ("HP:0002917", "Hypomagnesemia", "UMLS", "C0151723") in rows
    # Typedef xref is NOT a term xref (ignored)
    assert all(r[0] == "HP:0002917" or r[0] == "HP:0000001" for r in rows)
    assert not any(r[3] == "9999999" for r in rows)
    # a term with only an unmatched-vocab xref contributes nothing matchable... UMLS is kept
    assert ("HP:0000001", "All", "UMLS", "C0444868") in rows


def test_build_safe_summary_renders_hpo_probe_suppressed():
    results = [{
        "space": "source_climb", "min_cell": 20, "mondo_version": "2026-06-02",
        "generated_utc": "2026-08-27T00:00:00Z",
        "stats": {"mapped_terms": 8895, "used_terms": 4836, "used_fraction": 0.54,
                  "reported_terms": 3074, "used_small_terms": 1762,
                  "collision_terms": 2232, "rare_used_terms": 2176},
        "n_total": 626396, "n_coded": 349815, "n_on_mondo": 342394,
        "survey": {"persons_source_exact": 247122, "persons_standard_exact": 325284,
                   "persons_climbed": 329296, "unmatched_codes_by_vocab": {"ICD10CM": 9540},
                   "hpo": {"hpo_snomed_terms": 9800,
                           "climb": {"concepts": 1200, "with_hpo": 780, "mass": 500000,
                                     "mass_hpo": 320000},
                           # a DROP bucket whose HPO-recoverable mass is a small cell -> suppress
                           "drop": {"concepts": 40, "with_hpo": 6, "mass": 300, "mass_hpo": 12},
                           "examples": [{"snomed": "190855004", "hp_id": "HP:0002917",
                                         "hp_label": "Hypomagnesemia",
                                         "climbs_to": ["metabolic disease"]}]}},
    }]
    out = m.build_safe_summary(results)
    assert "HPO phenotype-gap probe" in out
    assert "780 of 1200 concepts (65%)" in out            # climb coverage % rendered
    assert "person-mass 320000 of 500000" in out          # large masses shown raw
    assert "6 of 40 concepts (15%)" in out                # drop coverage
    assert "≤20 of 300" in out                            # drop mass_hpo (12) suppressed, mass (300) raw
    assert "SNOMED 190855004` → HP:0002917 Hypomagnesemia" in out
    assert "climbs to: metabolic disease" in out
    # example carries NO per-concept patient number (only identities + labels)
    import re
    assert not re.search(r"Hypomagnesemia.*\b\d{2,}\b", out)


def test_build_safe_summary_hpo_axis_row():
    results = [{
        "space": "source_climb", "min_cell": 20, "mondo_version": "2026-06-02",
        "generated_utc": "2026-08-27T00:00:00Z",
        "stats": {"mapped_terms": 8895, "used_terms": 4800, "used_fraction": 0.54,
                  "reported_terms": 3000, "used_small_terms": 1800,
                  "collision_terms": 2000, "rare_used_terms": 2100},
        "n_total": 626396, "n_coded": 349815, "n_on_mondo": 342394, "survey": {},
        "hpo_axis": {"used_terms": 512, "reported_terms": 300, "persons": 7},  # persons -> ≤20
    }]
    out = m.build_safe_summary(results)
    assert "HPO axis" in out and "512" in out
    assert "persons ≤20" in out and " 7 " not in out


def test_build_safe_summary_suppresses_and_leaks_nothing():
    results = [
        {"space": "source", "min_cell": 20, "mondo_version": "2026-06-02",
         "generated_utc": "2026-08-26T00:00:00Z",
         "stats": {"mapped_terms": 8894, "used_terms": 4000, "used_fraction": 0.45,
                   "reported_terms": 2700, "used_small_terms": 1300,
                   "collision_terms": 0, "rare_used_terms": 2000},
         "survey": {}, "n_total": 400000, "n_coded": 390000, "n_on_mondo": 300000},
        {"space": "source_climb", "min_cell": 20, "mondo_version": "2026-06-02",
         "generated_utc": "2026-08-26T00:00:00Z",
         "stats": {"mapped_terms": 8894, "used_terms": 4600, "used_fraction": 0.52,
                   "reported_terms": 2740, "used_small_terms": 1860,
                   "collision_terms": 211, "rare_used_terms": 2275},
         # small PERSON tier counts MUST be suppressed; unmatched CODE counts are code
        # identities (safe, unsuppressed) so they show raw.
         "survey": {"persons_source_exact": 250000, "persons_standard_exact": 40000,
                    "persons_climbed": 7, "unmatched_codes_by_vocab": {"ICD9CM": 812}},
         "n_total": 400000, "n_coded": 390000, "n_on_mondo": 330000},
    ]
    out = m.build_safe_summary(results)
    # both spaces summarized; term counts (safe) present
    assert "`source`" in out and "`source_climb`" in out and "8894" in out
    # small PERSON figures are ≤-suppressed, never shown raw
    assert "climbed ≤20" in out          # persons_climbed 7 -> ≤20
    assert " 7" not in out
    # clean coverage: persons with NO mapped term = coded - on_mondo (390000-330000)
    assert "no mapped term 60000" in out
    # unmatched CODE counts are code identities -> shown raw (not a patient number)
    assert "ICD9CM=812" in out
    # no workbench/project identifier leaks (we never pass args.cdr in); a
    # `wb-<name>-<n>.rNNNN` workbench id or a `.rNNNN` release suffix must be absent.
    import re
    assert "wb-" not in out
    assert not re.search(r"\br\d{4}\b", out.replace("2026", ""))


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


def test_dag_structures():
    parents = {"A": [], "B": ["A"], "C": ["A", "B"], "D": ["Z"]}   # Z unknown
    parent_adj, has_child = m.dag_structures(parents)
    assert parent_adj["C"] == ["A", "B"]
    assert parent_adj["D"] == []            # unknown parent Z dropped
    assert has_child == {"A", "B"}          # A parents B,C; B parents C; D has no children
    assert "C" not in has_child


# --- complementary suppression of the source-code -> term roll-up marginal ----------
def _rep(N):
    return {"state": "reported", "count": N, "display": str(N)}


def test_chain_linear():
    assert m.shortest_climb_chain(1, 3, 2, {1: [2], 2: [3]}, {2: 1, 3: 0}) == ([1, 2, 3], False)


def test_chain_diamond_is_ambiguous_and_deterministic():
    parents = {1: [3, 2], 2: [4], 3: [4]}
    assert m.shortest_climb_chain(1, 4, 2, parents, {2: 1, 3: 1}) == ([1, 2, 4], True)


def test_chain_missing_returns_none():
    assert m.shortest_climb_chain(1, 9, 2, {1: [2]}, {2: 1}) == (None, False)


def test_chain_ignores_off_shortest_parent():
    # 1 -> 5 is a parent but sits 2 levels from the target (a longer route); skip it
    parents = {1: [5, 2], 2: [3], 5: [6], 6: [3]}
    assert m.shortest_climb_chain(1, 3, 2, parents, {2: 1, 5: 2, 6: 1}) == ([1, 2, 3], False)


def test_build_climb_paths_groups_by_target():
    # std 10 climbs to Mondo X via anc 30 (10 -> 20 -> 30) and to Mondo Y via anc 40 (10 -> 40)
    pairs = [(10, "MONDO:X", 30, 2), (10, "MONDO:Y", 40, 1)]
    ca_rows = [(20, 30, 1)]
    isa = [(10, 20), (20, 30), (10, 40)]
    out = m.build_climb_paths(pairs, ca_rows, isa)
    assert out[(10, "MONDO:X")] == ([10, 20, 30], False)
    assert out[(10, "MONDO:Y")] == ([10, 40], False)


import pytest  # noqa: E402


@pytest.mark.slow
@pytest.mark.slow
def test_collect_climb_paths_end_to_end(spark):
    import pandas as pd
    # std 11 climbs 11 -> 20 -> 30 (30 carries MONDO:X); std 10 climbs 10 -> 40 (MONDO:Y)
    ca = spark.createDataFrame([(20, 11, 1), (30, 11, 2), (30, 20, 1), (40, 10, 1)],
                               "ancestor_concept_id long, descendant_concept_id long, "
                               "min_levels_of_separation long")
    ca_f = spark.createDataFrame([(11, 30, 2), (10, 40, 1)], "std_cid long, anc_cid long, lev long")
    nearest = ca_f.select("std_cid", "anc_cid")
    anc_map = spark.createDataFrame([(30, "MONDO:X"), (40, "MONDO:Y")],
                                    "anc_cid long, anc_mondo string")
    reduced = pd.DataFrame({"std_cid": [11, 10], "mondo_id": ["MONDO:X", "MONDO:Y"]})
    isa = spark.createDataFrame([(11, 20), (20, 30), (10, 40), (99, 98)],
                                "concept_id_1 long, concept_id_2 long")
    paths, n = m.collect_climb_paths(spark, ca, ca_f, nearest, anc_map, reduced, isa)
    assert paths == {(11, "MONDO:X"): ([11, 20, 30], False), (10, "MONDO:Y"): ([10, 40], False)}
    assert n == 5                                   # {10, 11, 20, 30, 40}


def test_hpo_cache_name_is_keyed_by_url():
    a = m.hpo_cache_name("http://purl.obolibrary.org/obo/hp.obo")
    b = m.hpo_cache_name("http://purl.obolibrary.org/obo/hp/releases/2026-06-23/hp.obo")
    assert a != b and a.endswith(".obo") and b.startswith("hp-")


def test_hpo_data_version():
    obo = "format-version: 1.2\ndata-version: hp/releases/2026-06-23\n\n[Term]\nid: HP:1\n"
    assert m.hpo_data_version(obo) == "hp/releases/2026-06-23"
    assert m.hpo_data_version("format-version: 1.2\n") is None


# --- additive diagnosis cells (spec Revision 2) ---------------------------------------
def _row(mid, parents=(), cells=(), **kw):
    return dict({"mondo_id": mid, "label": mid.lower(), "is_internal": False,
                 "parents": list(parents), "std_concepts": [], "n_persons": 0,
                 "collision_siblings": [], "source_codes": [dict(c) for c in cells]}, **kw)


def _cell(cid, n, step, vocab="ICD10CM"):
    return {"id": cid, "vocab": vocab, "code": f"C{cid}", "step": step,
            "n": m.publish_cell(n)}


def test_publish_cell():
    assert [m.publish_cell(n) for n in (0, 1, 20, 21, 456)] == [0, 10, 10, 21, 456]


def test_term_count_is_sum_of_published_cells():
    # the user's example: ICDXX with ≤20 diagnoses -> 10, ICDXX2 with 456 -> term 466
    p = m.assemble_payload(meta={}, term_rows=[
        _row("MONDO:T", cells=[_cell(1, 7, 2), _cell(2, 456, 2)])])
    t = {n["id"]: n for n in p["nodes"]}["MONDO:T"]
    assert t["count"] == 466 and t["display"] == "466" and t["state"] == "reported"
    assert t["imputed"] == 1 and sorted(c["n"] for c in t["source_codes"]) == [10, 456]
    assert t["split"] == {"named": 466, "climbed": 0}


def test_split_parts_are_sums_and_partition_the_count():
    p = m.assemble_payload(meta={}, term_rows=[
        _row("MONDO:T", cells=[_cell(1, 300, 1), _cell(2, 5, 2), _cell(3, 90, 4)])])
    t = {n["id"]: n for n in p["nodes"]}["MONDO:T"]
    assert t["split"] == {"named": 310, "climbed": 90} and t["count"] == 400


def test_all_imputed_term_is_used_small():
    p = m.assemble_payload(meta={}, term_rows=[
        _row("MONDO:S", cells=[_cell(1, 3, 2), _cell(2, 19, 4)]), _row("MONDO:Z")])
    by = {n["id"]: n for n in p["nodes"]}
    assert by["MONDO:S"]["count"] == 20 and by["MONDO:S"]["state"] == "used_small"
    assert by["MONDO:Z"]["count"] == 0 and by["MONDO:Z"]["state"] == "unused"
    assert p["stats"]["used_small_terms"] == 1 and p["stats"]["imputed_cells"] == 2


def test_rollup_counts_each_code_once_per_subtree():
    # code 7 reaches both children of P (50 patients via SNOMED in A, climbed into B);
    # in P's roll-up it counts once, at its largest cell, in its lowest-step part.
    rows = [_row("MONDO:P"),
            _row("MONDO:A", parents=["MONDO:P"], cells=[_cell(7, 50, 2)]),
            _row("MONDO:B", parents=["MONDO:P"], cells=[_cell(7, 50, 4), _cell(8, 4, 4)])]
    by = {n["id"]: n for n in m.assemble_payload(meta={}, term_rows=rows)["nodes"]}
    assert by["MONDO:P"]["roll"] == {"all": 60, "named": 50, "climbed": 10, "imputed": 1}
    assert by["MONDO:B"]["roll"]["all"] == 60 and by["MONDO:A"]["roll"]["all"] == 50
    assert by["root"]["roll"]["all"] == 60
    assert by["MONDO:B"]["count"] == 60     # within a term each code is its own cell


def test_rollup_cells_multi_parent_term_counts_once():
    cells = {"C": [{"id": 1, "n": 30, "step": 2}]}
    parents = {"C": ["A", "B"], "A": ["R"], "B": ["R"], "R": []}
    r = m.rollup_cells(cells, parents)
    assert r["R"]["all"] == 30 and r["A"]["all"] == 30 and r["B"]["all"] == 30


def test_catalog_cap_keeps_heaviest_but_sums_everything():
    p = m.assemble_payload(meta={}, catalog_cap=1, term_rows=[
        _row("MONDO:T", cells=[_cell(1, 3, 2), _cell(2, 400, 4)])])
    t = {n["id"]: n for n in p["nodes"]}["MONDO:T"]
    assert [c["n"] for c in t["source_codes"]] == [400] and t["n_source_codes"] == 2
    assert t["count"] == 410


def test_source_code_catalog_passthrough():
    c = {"id": 9, "vocab": "ICD10CM", "code": "E27.3", "step": 4, "n": 120,
         "path": [11, 12], "path_ambiguous": True}
    t = {n["id"]: n for n in m.assemble_payload(
        meta={}, term_rows=[_row("MONDO:T", cells=[c])])["nodes"]}["MONDO:T"]
    assert t["source_codes"] == [c]


def test_frac_is_never_published():
    rows = [_row("MONDO:A", cells=[_cell(1, 30, 2)], n_frac=12.4)]
    for nd in m.assemble_payload(meta={}, term_rows=rows)["nodes"]:
        assert "frac" not in nd and "frac_display" not in nd and "frac_range" not in nd


def test_assert_cells_safe_rejects_unsafe_payloads():
    import pytest
    good = {"id": "X", "state": "reported", "cells": True, "count": 466,
            "source_codes": [{"n": 10}, {"n": 456}], "split": {"named": 466, "climbed": 0},
            "roll": {"all": 466, "named": 466, "climbed": 0, "imputed": 1}}
    cells = {"X": [{"id": 1, "n": 10, "step": 2}, {"id": 2, "n": 456, "step": 2}]}
    m.assert_cells_safe([good], cells)
    for bad in (dict(good, source_codes=[{"n": 7}, {"n": 456}]),          # raw small cell
                dict(good, split={"named": 460, "climbed": 0}),           # split != count
                dict(good, count=470),                                     # count != Σ cells
                dict(good, roll={"all": 400, "named": 400, "climbed": 0, "imputed": 1}),
                dict(good, frac=466)):
        with pytest.raises(AssertionError):
            m.assert_cells_safe([bad], cells)
    with pytest.raises(AssertionError):                                    # legacy small total
        m.assert_cells_safe([{"id": "L", "state": "reported", "count": 5}])


def test_legacy_rows_without_cells_keep_masked_totals():
    rows = [{"mondo_id": "MONDO:L", "label": "l", "is_internal": False, "parents": [],
             "std_concepts": [1], "n_persons": 5, "collision_siblings": []},
            {"mondo_id": "MONDO:M", "label": "m", "is_internal": False, "parents": [],
             "std_concepts": [2], "n_persons": 500, "collision_siblings": []}]
    by = {n["id"]: n for n in m.assemble_payload(meta={}, term_rows=rows)["nodes"]}
    assert by["MONDO:L"]["count"] is None and by["MONDO:L"]["display"] == "≤20"
    assert by["MONDO:M"]["count"] == 500 and "roll" not in by["MONDO:M"]


@pytest.mark.slow
def test_catalog_steps_lowest_wins(spark):
    rows = [  # person, mondo, origin, via, k_src, k_std
        (1, "T", 100, "standard_exact", 100, 7), (2, "T", 100, "climbed", 100, 8),
        (3, "T", 200, "climbed", 200, 9), (4, "T", 100, "standard_exact", 100, 7),
        (5, "T", 100, "standard_exact", 100, 6)]
    att = spark.createDataFrame(rows, "person_id long, mondo_id string, origin_cid long, "
                                      "via string, k_src long, k_std long")
    steps = m.catalog_steps_df(att)
    assert {(r.mondo_id, r.origin_cid): r.step for r in steps.collect()} == \
        {("T", 100): 2, ("T", 200): 4}
    std = {(r.mondo_id, r.origin_cid): (r.std_cid, r.multi_std)
           for r in m.code_std_df(att, steps).collect()}
    assert std[("T", 100)] == (7, True) and std[("T", 200)] == (9, False)
    assert m.rolled_up_only_persons(att, steps) == 1
