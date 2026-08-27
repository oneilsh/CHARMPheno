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


def test_fractional_count_suppressed_independently():
    # frac is patient-derived: it can dip below the floor even when the EXACT count
    # clears it (a term whose patients arrive via shared codes), so it gets its own gate.
    rows = [
        # exact 30 (reported) but fractional 12.4 -> must suppress the frac to ≤20
        {"mondo_id": "MONDO:A", "label": "a", "is_internal": False, "parents": [],
         "std_concepts": [1], "n_persons": 30, "n_frac": 12.4, "collision_siblings": []},
        # exact and frac both clear the floor; frac rounds to int
        {"mondo_id": "MONDO:B", "label": "b", "is_internal": False, "parents": [],
         "std_concepts": [2], "n_persons": 5000, "n_frac": 2500.6, "collision_siblings": []},
        # no n_frac supplied -> falls back to the exact count
        {"mondo_id": "MONDO:C", "label": "c", "is_internal": False, "parents": [],
         "std_concepts": [3], "n_persons": 90, "collision_siblings": []},
    ]
    by = {n["id"]: n for n in m.assemble_payload(meta={}, term_rows=rows)["nodes"]}
    assert by["MONDO:A"]["count"] == 30 and by["MONDO:A"]["frac"] is None      # frac withheld
    assert by["MONDO:A"]["frac_display"] == "≤20"
    assert by["MONDO:B"]["frac"] == 2501 and by["MONDO:B"]["frac_display"] == "2501"
    assert by["MONDO:C"]["frac"] == 90                                          # fell back to exact
    assert by["root"]["frac"] is None and by["root"]["frac_display"] == ""


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


def test_volume_band_ranges_and_floor():
    # bottom band IS the small-cell floor (identical to a suppressed cell) — no exact
    # count in 1..floor ever escapes as a band.
    assert m.volume_band(0) == "0"
    assert m.volume_band(1) == "≤20"
    assert m.volume_band(20) == "≤20"          # floor inclusive
    assert m.volume_band(21) == "21–100"
    assert m.volume_band(100) == "21–100"
    assert m.volume_band(101) == "101–1k"
    assert m.volume_band(1000) == "101–1k"
    assert m.volume_band(1001) == "1k–10k"
    assert m.volume_band(10_000) == "1k–10k"
    assert m.volume_band(10_001) == "10k–100k"
    assert m.volume_band(100_000) == "10k–100k"
    assert m.volume_band(100_001) == ">100k"
    assert m.volume_band(5_000_000) == ">100k"
    assert m.volume_band(None) == "n/a"
    # no band label is ever a bare small-cell integer (1..floor)
    for n in (1, 5, 20, 21, 999, 12345, 200000):
        lab = m.volume_band(n)
        assert not (lab.isdigit() and 1 <= int(lab) <= 20)


def test_volume_band_respects_custom_floor():
    assert m.volume_band(50, min_cell=50) == "≤50"
    assert m.volume_band(51, min_cell=50) == "51–100"


def test_band_histogram_counts_codes_heavy_first():
    bands = (["≤20"] * 184 + ["1k–10k"] * 3 + [">100k"] * 2 + ["101–1k"] * 8)
    hist = m.band_histogram(bands)
    # heavy -> light order, empty bands omitted, counts are CODE counts
    assert hist == [
        {"band": ">100k", "codes": 2},
        {"band": "1k–10k", "codes": 3},
        {"band": "101–1k", "codes": 8},
        {"band": "≤20", "codes": 184},
    ]
    # the "codes" figure is a count of codes (public identities), so a small value like
    # 2 or 3 is NOT a small-cell disclosure and is reported raw.
    assert hist[0]["codes"] == 2
    assert m.band_histogram([]) == []


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


def test_source_code_catalog_passthrough():
    # source_climb space: each term catalogs the originating source codes (ICD etc.)
    # that reached it, as identity + a total; assemble_payload passes them through
    # and defaults them to empty for the exact-map spaces.
    rows = [
        {"mondo_id": "MONDO:A", "label": "a", "is_internal": False, "parents": [],
         "std_concepts": [1], "n_persons": 40, "collision_siblings": [],
         "source_codes": [{"id": 11, "vocab": "ICD10CM", "code": "E11.9", "via": "exact",
                           "band": "1k–10k"},
                          {"id": 12, "vocab": "ICD10CM", "code": "E11.21", "via": "climbed",
                           "band": "≤20"}],
         "n_source_codes": 2,
         "source_bands": [{"band": "1k–10k", "codes": 1}, {"band": "≤20", "codes": 1}]},
        {"mondo_id": "MONDO:B", "label": "b", "is_internal": False, "parents": [],
         "std_concepts": [2], "n_persons": 25, "collision_siblings": []},
    ]
    p = m.assemble_payload(meta={}, term_rows=rows)
    by = {n["id"]: n for n in p["nodes"]}
    assert by["MONDO:A"]["n_source_codes"] == 2
    assert [c["code"] for c in by["MONDO:A"]["source_codes"]] == ["E11.9", "E11.21"]
    assert by["MONDO:A"]["source_codes"][1]["via"] == "climbed"
    # codes carry identity + band only, no embedded name (names resolved client-side)
    assert "name" not in by["MONDO:A"]["source_codes"][0]
    # per-code band + per-term band histogram pass through (the "where's the weight" shape)
    assert by["MONDO:A"]["source_codes"][0]["band"] == "1k–10k"
    assert by["MONDO:A"]["source_bands"] == [{"band": "1k–10k", "codes": 1},
                                             {"band": "≤20", "codes": 1}]
    # a term with no catalog defaults cleanly; the root too
    assert by["MONDO:B"]["source_codes"] == [] and by["MONDO:B"]["n_source_codes"] == 0
    assert by["MONDO:B"]["source_bands"] == []
    assert by["root"]["source_codes"] == [] and by["root"]["n_source_codes"] == 0
    assert by["root"]["source_bands"] == []


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
