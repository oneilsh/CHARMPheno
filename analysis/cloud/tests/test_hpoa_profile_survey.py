"""Pure-logic tests for the stage-1 HPOA profile survey (offline, ontology-only).

Covers the format hazards the HPOA spec calls out (NOT qualifier, aspect codes,
the three frequency shapes), the Mondo-xref -> HPOA-key prefix rewrite
(Orphanet: -> ORPHA:), and the true-path direction of SNOMED realizability
(ancestor-or-self of a direct-xref term; a sibling subtree must NOT leak).
The download/CLI path is exercised by running the survey, not unit-tested.
"""
import pandas as pd

import hpoa_profile_survey as s

# --- fixtures ---------------------------------------------------------------

_HP_OBO = """format-version: 1.2
data-version: hp/releases/2099-01-01

[Term]
id: HP:0000001
name: All

[Term]
id: HP:0001626
name: Abnormality of the cardiovascular system
is_a: HP:0000001

[Term]
id: HP:0001635
name: Congestive heart failure
is_a: HP:0001626
xref: SNOMEDCT_US:42343007

[Term]
id: HP:0031547
name: Very specific unxreffed finding
is_a: HP:0001626

[Term]
id: HP:0099999
name: Unrelated finding
is_a: HP:0000001

[Typedef]
id: part_of
"""

_HPOA = """#description: test
#version: 2099-01-02
database_id\tdisease_name\tqualifier\thpo_id\treference\tevidence\tonset\tfrequency\tsex\tmodifier\taspect\tbiocuration
OMIM:100000\tdisease A\t\tHP:0001635\tPMID:1\tPCS\t\tHP:0040281\t\t\tP\tX
OMIM:100000\tdisease A\t\tHP:0031547\tPMID:1\tPCS\t\t7/13\t\t\tP\tX
OMIM:100000\tdisease A\tNOT\tHP:0099999\tPMID:1\tPCS\t\t\t\t\tP\tX
OMIM:100000\tdisease A\t\tHP:0000001\tPMID:1\tPCS\t\t\t\t\tI\tX
ORPHA:2000\tdisease A orpha\t\tHP:0001635\tPMID:2\tPCS\t\t17%\t\t\tP\tX
OMIM:200000\tdisease B\t\tHP:0001635\tPMID:3\tPCS\t\t0%\t\t\tP\tX
"""

_NODES = pd.DataFrame({
    "id": ["MONDO:1", "MONDO:2", "MONDO:3"],
    "category": ["biolink:Disease"] * 3,
    "name": ["cv root", "disease A", "disease B"],
    "xref": ["ICD10CM:I00-I99", "OMIM:100000|Orphanet:2000", "OMIM:200000"],
})
_EDGES = pd.DataFrame({
    "subject": ["MONDO:2", "MONDO:3"],
    "predicate": ["biolink:subclass_of"] * 2,
    "object": ["MONDO:1", "MONDO:1"],
})


def _profiles():
    keys = s.mondo_hpoa_keys(_NODES, {"MONDO:1", "MONDO:2", "MONDO:3"})
    return s.build_profiles(s.parse_hpoa(_HPOA), keys)


# --- frequency normalization ------------------------------------------------

def test_normalize_frequency_three_shapes_and_edge_cases():
    assert s.normalize_frequency("HP:0040281") == 0.895     # HP frequency term
    assert s.normalize_frequency("7/13") == 7 / 13          # patient ratio
    assert s.normalize_frequency("17%") == 0.17             # percent
    assert s.normalize_frequency("30%-79%") == 0.545        # range -> midpoint
    assert s.normalize_frequency("") is None
    assert s.normalize_frequency(None) is None
    assert s.normalize_frequency(float("nan")) is None
    assert s.normalize_frequency("0/0") is None             # degenerate ratio
    assert s.normalize_frequency("garbage") is None


# --- HPOA parse + Mondo key rewrite -----------------------------------------

def test_parse_hpoa_skips_comments_and_keeps_12_columns():
    df = s.parse_hpoa(_HPOA)
    assert len(df) == 6
    assert {"database_id", "qualifier", "hpo_id", "frequency", "aspect"} <= set(df.columns)


def test_mondo_keys_rewrite_orphanet_prefix_and_drop_unmatched():
    keys = s.mondo_hpoa_keys(_NODES, {"MONDO:1", "MONDO:2", "MONDO:3"})
    got = set(zip(keys["mondo_id"], keys["db_id"]))
    # ICD10CM xref on the root is NOT an HPOA key; Orphanet: -> ORPHA:.
    assert got == {("MONDO:2", "OMIM:100000"), ("MONDO:2", "ORPHA:2000"),
                   ("MONDO:3", "OMIM:200000")}


# --- profile semantics ------------------------------------------------------

def test_profiles_aspect_filter_not_split_and_source_pooling():
    g = _profiles()
    a = g[g["mondo_id"] == "MONDO:2"].set_index("hpo_id")
    # aspect I row dropped entirely
    assert "HP:0000001" not in a.index
    # NOT row lands as negative polarity
    assert bool(a.loc["HP:0099999", "neg"]) is True
    # same term from two sources (OMIM 0.895, ORPHA 0.17): freq pools by max
    assert a.loc["HP:0001635", "freq"] == 0.895
    assert a.loc["HP:0001635", "n_sources"] == 2
    assert bool(a.loc["HP:0001635", "neg"]) is False


def test_zero_frequency_means_excluded_and_goes_negative():
    g = _profiles()
    b = g[g["mondo_id"] == "MONDO:3"].set_index("hpo_id")
    # "0%" = tested-and-absent; boosting it would invert the annotation
    assert bool(b.loc["HP:0001635", "neg"]) is True


# --- realizability closure ---------------------------------------------------

def test_snomed_closure_is_ancestor_or_self_no_sibling_leak():
    direct, closure, labels = s.hpo_realizability(_HP_OBO)
    assert direct == {"HP:0001635"}
    # ancestors of the direct term are closure-realizable (true-path)...
    assert {"HP:0001635", "HP:0001626", "HP:0000001"} <= closure
    # ...but a SIBLING with no xreffed descendant must not leak in
    assert "HP:0031547" not in closure
    assert "HP:0099999" not in closure
    assert labels["HP:0031547"] == "Very specific unxreffed finding"


# --- emit-codes closure collection -------------------------------------------

def test_profile_code_rows_descendant_pickup_no_sibling_leak_negatives_carried():
    """A profile term collects SNOMED codes from its descendant-OR-SELF closure
    (true-path, emit direction); a sibling's code must NOT leak sideways; NOT
    terms are carried with neg=True; a term with no code yields no rows."""
    labels, parents = s.parse_hpo_dag(_HP_OBO)
    xrefs = s.parse_hpo_xrefs(_HP_OBO)
    profiles = pd.DataFrame({
        # the ANCESTOR of the xreffed term: picks the descendant code up
        "mondo_id": ["MONDO:2", "MONDO:2", "MONDO:2", "MONDO:3"],
        "hpo_id": ["HP:0001626", "HP:0031547", "HP:0099999", "HP:0001635"],
        "neg": [False, False, True, False],
        "freq": [0.9, None, None, 0.5],
    })
    rows = s.profile_code_rows(profiles, parents, xrefs)
    got = set(zip(rows["mondo_id"], rows["hp_id"], rows["neg"], rows["code"]))
    # ancestor picks up the descendant's code; self-xref term keeps its own
    assert ("MONDO:2", "HP:0001626", False, "42343007") in got
    assert ("MONDO:3", "HP:0001635", False, "42343007") in got
    # SIBLING of the xreffed term (HP:0031547) must not leak the code in
    assert not any(hp == "HP:0031547" for _m, hp, _n, _c in got)
    # negative term carried — but HP:0099999's closure holds no code, so it
    # yields no rows either; a negative WITH a code keeps neg=True
    neg_rows = s.profile_code_rows(
        pd.DataFrame({"mondo_id": ["MONDO:2"], "hpo_id": ["HP:0001635"],
                      "neg": [True], "freq": [0.0]}), parents, xrefs)
    assert list(neg_rows["neg"]) == [True]
    # column contract for stage 2
    assert list(rows.columns) == ["mondo_id", "hp_id", "neg", "freq",
                                  "vocab", "code"]
    assert set(rows["vocab"]) == {"SNOMED"}


def test_profile_code_rows_freq_blank_when_unknown(tmp_path):
    """NaN freq round-trips to a BLANK TSV cell (stage 2 reads 'empty when
    unknown', not the string 'nan')."""
    labels, parents = s.parse_hpo_dag(_HP_OBO)
    xrefs = s.parse_hpo_xrefs(_HP_OBO)
    profiles = pd.DataFrame({"mondo_id": ["MONDO:2"], "hpo_id": ["HP:0001635"],
                             "neg": [False], "freq": [float("nan")]})
    rows = s.profile_code_rows(profiles, parents, xrefs)
    path = tmp_path / "codes.tsv"
    rows.to_csv(path, sep="\t", index=False)
    line = path.read_text().splitlines()[1]
    assert "nan" not in line
    assert line.split("\t")[3] == ""        # the freq cell, blank


# --- survey rows + branch closure --------------------------------------------

def test_branch_closure_depths_and_survey_rows():
    closure, depth, names = s.mondo_branch_closure(_NODES, _EDGES, "MONDO:1")
    assert closure == {"MONDO:1", "MONDO:2", "MONDO:3"}
    assert depth == {"MONDO:1": 0, "MONDO:2": 1, "MONDO:3": 1}

    g = _profiles()
    direct, cl, _ = s.hpo_realizability(_HP_OBO)
    rows = s.survey_rows(g, closure, depth, names, direct, cl).set_index("mondo_id")
    # the root has NO profile and still gets a row — absence is the finding
    assert rows.loc["MONDO:1", "profile_n"] == 0
    a = rows.loc["MONDO:2"]
    assert (a["profile_n"], a["not_n"]) == (2, 1)
    assert a["freq_known_n"] == 2
    assert (a["snomed_direct_n"], a["snomed_closure_n"]) == (1, 1)
    # disease B's only term went negative (0%), so its positive profile is empty
    assert rows.loc["MONDO:3", "profile_n"] == 0


# --- descendant profile roll-up (--rollup) ------------------------------------

# A deeper toy branch than _NODES/_EDGES: root -> {A, sib}, A -> leaf.
_RU_ADJ = {"MONDO:1": ["MONDO:2", "MONDO:3"], "MONDO:2": ["MONDO:4"]}
_RU_CLOSURE = {"MONDO:1", "MONDO:2", "MONDO:3", "MONDO:4"}


def _ru(profiles):
    return s.rollup_profiles(pd.DataFrame(profiles), _RU_ADJ, _RU_CLOSURE)


def test_rollup_credits_descendants_to_ancestors_inherited_true():
    """A leaf's profile is credited to EVERY branch ancestor (multi-level) with
    inherited=True; the leaf's own row keeps inherited=False."""
    g = _ru({"mondo_id": ["MONDO:4"], "hpo_id": ["HP:0001635"],
             "neg": [False], "freq": [0.5]}).set_index("mondo_id")
    assert set(g.index) == {"MONDO:1", "MONDO:2", "MONDO:4"}
    assert bool(g.loc["MONDO:1", "inherited"]) is True
    assert bool(g.loc["MONDO:2", "inherited"]) is True
    assert bool(g.loc["MONDO:4", "inherited"]) is False
    # the annotation itself rides along unchanged
    assert g.loc["MONDO:1", "freq"] == 0.5
    assert bool(g.loc["MONDO:1", "neg"]) is False


def test_rollup_self_plus_inherited_merge_takes_freq_max_inherited_false():
    """When the credited node has its OWN annotation for the term, the merged
    row is inherited=False and freq pools by max over contributors (NaN only
    if ALL contributors lack one)."""
    g = _ru({"mondo_id": ["MONDO:2", "MONDO:4"],
             "hpo_id": ["HP:0001635"] * 2,
             "neg": [False, False], "freq": [0.2, 0.9]}).set_index("mondo_id")
    assert g.loc["MONDO:2", "freq"] == 0.9          # descendant's max wins
    assert bool(g.loc["MONDO:2", "inherited"]) is False  # own annotation
    # a self-NaN freq still pools to the descendant's value; all-NaN stays NaN
    g2 = _ru({"mondo_id": ["MONDO:2", "MONDO:4"],
              "hpo_id": ["HP:0001635"] * 2,
              "neg": [False, False],
              "freq": [float("nan"), 0.9]}).set_index("mondo_id")
    assert g2.loc["MONDO:2", "freq"] == 0.9
    g3 = _ru({"mondo_id": ["MONDO:4"], "hpo_id": ["HP:0001635"],
              "neg": [False], "freq": [float("nan")]}).set_index("mondo_id")
    assert pd.isna(g3.loc["MONDO:1", "freq"])


def test_rollup_neg_unanimity_positive_anywhere_wins():
    """neg=True only if ALL contributors are negative; one positive contributor
    anywhere in the subtree makes the union row positive."""
    unanimous = _ru({"mondo_id": ["MONDO:2", "MONDO:4"],
                     "hpo_id": ["HP:0099999"] * 2,
                     "neg": [True, True],
                     "freq": [None, None]}).set_index("mondo_id")
    assert bool(unanimous.loc["MONDO:1", "neg"]) is True
    split = _ru({"mondo_id": ["MONDO:2", "MONDO:4"],
                 "hpo_id": ["HP:0099999"] * 2,
                 "neg": [True, False],
                 "freq": [None, None]}).set_index("mondo_id")
    assert bool(split.loc["MONDO:1", "neg"]) is False
    assert bool(split.loc["MONDO:2", "neg"]) is False  # own NOT overridden too


def test_rollup_sibling_profile_never_credited():
    """Credit flows strictly descendant -> ancestor: a SIBLING's profile must
    not appear on MONDO:2 (only on the shared ancestor and the sibling)."""
    g = _ru({"mondo_id": ["MONDO:3"], "hpo_id": ["HP:0001635"],
             "neg": [False], "freq": [0.7]})
    assert set(g["mondo_id"]) == {"MONDO:1", "MONDO:3"}
    assert "MONDO:2" not in set(g["mondo_id"])
    assert "MONDO:4" not in set(g["mondo_id"])


def test_rollup_off_emission_columns_and_rows_unchanged():
    """Without --rollup, profile_code_rows sees no `inherited` column and its
    emission keeps the OLD six-column contract, row for row; a rolled-up frame
    appends `inherited` after `code`."""
    labels, parents = s.parse_hpo_dag(_HP_OBO)
    xrefs = s.parse_hpo_xrefs(_HP_OBO)
    profiles = _profiles()
    old = s.profile_code_rows(profiles, parents, xrefs)
    assert list(old.columns) == ["mondo_id", "hp_id", "neg", "freq",
                                 "vocab", "code"]
    rolled = s.rollup_profiles(
        profiles, {"MONDO:1": ["MONDO:2", "MONDO:3"]},
        {"MONDO:1", "MONDO:2", "MONDO:3"})
    new = s.profile_code_rows(rolled, parents, xrefs)
    assert list(new.columns) == ["mondo_id", "hp_id", "neg", "freq",
                                 "vocab", "code", "inherited"]
    # here no descendant shares a self-annotated term, so the self-credited
    # rows survive the roll-up byte-for-byte (in general max-pool/unanimity
    # may change a self row's freq/neg when descendants share the term)
    assert old.equals(new[new["inherited"] == False]  # noqa: E712
                      [old.columns].reset_index(drop=True))
