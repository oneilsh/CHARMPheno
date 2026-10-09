"""The DisMech ribbon as an explicit native-Mondo label set (spec 2026-10-07 §D1).

Three claims:

  1. **The reader is faithful and the artifact is pinned.** `kb/disorders/*.yaml`
     -> rows -> TSV -> rows round-trips; a disorder without a MONDO disease_term is
     skipped and counted; the TSV refuses to load without its commit header; the
     identity moves with the member set and NOT with the descriptive columns.

  2. **The filter is a receipt, not a silent intersection.** `apply_label_set_filter`
     keeps `powered & ribbon` and accounts for every member that did not make it
     (unknown to this Mondo release / unpowered), and reports how flat the ribbon
     really is (nested pairs within the kept set) without acting on it.

  3. **No existing key moves.** `label_set` folds into the bundle key ONLY on the
     native path and ONLY when non-empty; the pinned SNOMED and Mondo hashes in
     test_case_finding_cache_mondo.py stay the authoritative tripwire, and the
     spec -> key path the fit and the re-readout share threads the field.

Pure; no Spark.
"""
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CLOUD = REPO_ROOT / "analysis" / "cloud"
for _p in (str(REPO_ROOT), str(CLOUD), str(REPO_ROOT / "charmpheno"),
           str(REPO_ROOT / "spark-vi"), str(REPO_ROOT / "tests" / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import _case_finding_cache as ccache  # noqa: E402
import dismech_ribbon as dr  # noqa: E402
import gated_pc_cloud as gpc  # noqa: E402
import gated_pc_readout as gpr  # noqa: E402
import mondo_native_dag as mnd  # noqa: E402
from test_case_finding_cache_mondo import (  # noqa: E402
    _MD_BASE, _MONDO_KEY_NO_COLLAPSE, _MONDO_SPEC, _SNOMED_BASE, _SNOMED_KEY,
    _mondo_manifest)

_YAML_EDS = """
name: Ehlers-Danlos Syndrome
category: Genetic
disease_term:
  preferred_term: Ehlers-Danlos syndrome
  term:
    id: MONDO:0020066
    label: Ehlers-Danlos syndrome
classifications:
  harrisons_chapter:
  - classification_value: IMMUNE_RHEUMATOLOGIC
  - classification_value: GENETICS_ENVIRONMENT_DISEASE
  isds_skeletal_category:
    classification_value: overgrowth_syndromes
"""
_YAML_DCM = """
name: Dilated Cardiomyopathy
category: Complex
disease_term:
  term:
    id: MONDO:0005021
    label: dilated cardiomyopathy
classifications:
  harrisons_chapter:
  - classification_value: CARDIOVASCULAR
"""
_YAML_NO_TERM = """
name: Aconitine Poisoning
category: Environmental
"""
_YAML_ROOT = """
name: Dorsalgia
category: Complex
disease_term:
  term:
    id: MONDO:0000001
    label: disease
"""
_YAML_DUP = """
name: EDS (duplicate curation)
category: Mendelian
disease_term:
  term:
    id: MONDO:0020066
"""


@pytest.fixture
def kb(tmp_path):
    d = tmp_path / "disorders"
    d.mkdir()
    (d / "Ehlers-Danlos_Syndrome.yaml").write_text(_YAML_EDS)
    (d / "Dilated_Cardiomyopathy.yaml").write_text(_YAML_DCM)
    (d / "Aconitine_Poisoning.yaml").write_text(_YAML_NO_TERM)
    (d / "EDS_dup.yaml").write_text(_YAML_DUP)
    (d / "Dorsalgia.yaml").write_text(_YAML_ROOT)
    return d


# --------------------------------------------------------------------------- #
# 1. reader + artifact                                                          #
# --------------------------------------------------------------------------- #
def test_reader_keeps_mondo_terms_and_counts_the_rest(kb):
    rows, skipped = dr.read_disorders(kb)
    # no MONDO term, and the Mondo ROOT (DisMech's Dorsalgia curation error): both out
    assert skipped == ["Aconitine_Poisoning.yaml", "Dorsalgia.yaml"]
    assert "MONDO:0000001" in dr.EXCLUDED_MONDO_IDS
    assert [r.mondo_id for r in rows] == [
        "MONDO:0005021", "MONDO:0020066", "MONDO:0020066"]
    eds = next(r for r in rows if r.dismech_file == "Ehlers-Danlos_Syndrome.yaml")
    assert eds.category == "Genetic"
    # a list-valued classification keeps every value, pipe-joined, in order
    assert eds.classifications["harrisons_chapter"] == (
        "IMMUNE_RHEUMATOLOGIC|GENETICS_ENVIRONMENT_DISEASE")
    assert eds.classifications["isds_skeletal_category"] == "overgrowth_syndromes"


def test_tsv_round_trips_and_dedups_on_the_label_set(kb, tmp_path):
    rows, _ = dr.read_disorders(kb)
    out = tmp_path / "ribbon.tsv"
    dr.write_ribbon_tsv(rows, out, commit="71cd0452358b01ef7d0b3cc213297f9837be7e54")
    rb = dr.load_ribbon(out)
    assert rb.commit.startswith("71cd0452")
    assert len(rb.rows) == 3                       # both EDS curations survive as rows
    assert rb.mondo_ids == {"MONDO:0005021", "MONDO:0020066"}   # one label node
    assert rb.rows[1].classifications == dr.load_ribbon(out).rows[1].classifications
    sets = rb.classification_sets("harrisons_chapter")
    assert sets["CARDIOVASCULAR"] == {"MONDO:0005021"}
    assert sets["IMMUNE_RHEUMATOLOGIC"] == {"MONDO:0020066"}


def test_tsv_without_a_commit_header_is_refused(tmp_path):
    p = tmp_path / "bare.tsv"
    p.write_text("mondo_id\tdisorder_name\nMONDO:1\tx\n")
    with pytest.raises(ValueError, match="dismech_commit"):
        dr.load_ribbon(p)


def test_identity_moves_with_members_not_with_descriptions(kb, tmp_path):
    rows, _ = dr.read_disorders(kb)
    a = tmp_path / "a.tsv"
    dr.write_ribbon_tsv(rows, a, commit="aaaa")
    ident = dr.label_set_identity(dr.load_ribbon(a))
    assert ident.startswith("dismech:aaaa:2:")
    # renaming / retagging: same identity
    renamed = [dr.RibbonRow(r.mondo_id, "X", "Y", {}, r.dismech_file) for r in rows]
    b = tmp_path / "b.tsv"
    dr.write_ribbon_tsv(renamed, b, commit="aaaa")
    assert dr.label_set_identity(dr.load_ribbon(b)) == ident
    # a different member set, or a different DisMech commit: different identity
    c = tmp_path / "c.tsv"
    dr.write_ribbon_tsv(rows[:1], c, commit="aaaa")
    assert dr.label_set_identity(dr.load_ribbon(c)) != ident
    d = tmp_path / "d.tsv"
    dr.write_ribbon_tsv(rows, d, commit="bbbb")
    assert dr.label_set_identity(dr.load_ribbon(d)) != ident


# --------------------------------------------------------------------------- #
# 2. the filter is a receipt                                                    #
# --------------------------------------------------------------------------- #
#   root -> A -> B -> C ; root -> D ; E is a member unknown to this release
_PARENT = {"A": ["root"], "B": ["A"], "C": ["B"], "D": ["root"], "root": []}
_KNOWN = set(_PARENT)


def test_filter_keeps_the_intersection_and_accounts_for_every_member():
    powered = {"A", "B", "C", "D", "root"}
    ribbon = {"A", "C", "D", "E", "Z"}           # B powered but not a member
    kept, st = mnd.apply_label_set_filter(powered, ribbon, _PARENT,
                                          known_terms=_KNOWN, name="t")
    assert kept == {"A", "C", "D"}
    assert st["n_members"] == 5 and st["n_kept"] == 3
    assert st["n_unknown"] == 2                 # E, Z not in this Mondo release
    assert st["n_unpowered"] == 0
    # A is an ancestor of C within the kept set: reported, not acted on
    assert st["n_nested_pairs"] == 1 and st["n_nested_members"] == 1
    assert st["name"] == "t"


def test_filter_reports_unpowered_members_and_a_flat_ribbon_has_no_pairs():
    powered = {"A", "D"}
    kept, st = mnd.apply_label_set_filter(powered, {"A", "C", "D"}, _PARENT,
                                          known_terms=_KNOWN)
    assert kept == {"A", "D"}
    assert st["n_unpowered"] == 1 and st["n_unknown"] == 0
    assert st["n_nested_pairs"] == 0
    line = mnd.format_label_set_report({"label_set": st})
    assert "3 member(s) -> 2 kept" in line and "1 unpowered" in line
    assert mnd.format_label_set_report({"label_set": None}).endswith(
        "(whole powered set)")


# --------------------------------------------------------------------------- #
# 3. the key                                                                    #
# --------------------------------------------------------------------------- #
def test_label_set_folds_only_on_the_native_path_and_only_when_set():
    k = ccache.compute_bundle_cache_key
    native = dict(_MD_BASE, mondo_native=True, mondo_native_version="native-mondo-v1")
    base = k(**native)
    assert k(**native, label_set="") == base
    assert k(**native, label_set="dismech:71cd04523580:3218:abc") != base
    assert (k(**native, label_set="dismech:71cd04523580:3218:abc")
            != k(**native, label_set="dismech:71cd04523580:3217:def"))
    # anchor-hierarchy and SNOMED keys are frozen even if a caller passes it
    assert k(**_MD_BASE, label_set="dismech:x:1:y") == _MONDO_KEY_NO_COLLAPSE
    assert k(**_SNOMED_BASE, label_set="dismech:x:1:y") == _SNOMED_KEY


def test_spec_threads_label_set_to_the_key_and_the_readout_recovers_it():
    native = dict(_MONDO_SPEC, dag_source="mondo_native")
    off = gpc.multidomain_cache_key(native)
    assert gpc.multidomain_cache_key(dict(native, label_set="")) == off
    on_spec = dict(native, label_set="dismech:71cd04523580:3218:abc",
                   label_set_path="analysis/cloud/anchor_selection_data/x.tsv")
    on = gpc.multidomain_cache_key(on_spec)
    assert on != off
    # the path is a rebuild input, never a key input
    assert gpc.multidomain_cache_key(dict(on_spec, label_set_path="/moved.tsv")) == on
    # manifest -> spec -> key, the re-readout's path
    m = _mondo_manifest()
    m["dag_source"] = m["corpus_manifest"]["dag_source"] = "mondo_native"
    m["corpus_manifest"]["label_set"] = on_spec["label_set"]
    m["corpus_manifest"]["label_set_path"] = on_spec["label_set_path"]
    spec = gpr.corpus_spec_from_manifest(m)
    assert spec["label_set"] == on_spec["label_set"]
    assert spec["label_set_path"] == on_spec["label_set_path"]
    assert gpc.multidomain_cache_key(spec) == on
    # the CLI may move the path but never the identity
    spec2 = gpr.corpus_spec_from_manifest(m, label_set_path="/elsewhere.tsv")
    assert spec2["label_set_path"] == "/elsewhere.tsv"
    assert gpc.multidomain_cache_key(spec2) == on
    # a manifest from before the field existed means ''
    legacy = _mondo_manifest()
    legacy["dag_source"] = legacy["corpus_manifest"]["dag_source"] = "mondo_native"
    assert gpr.corpus_spec_from_manifest(legacy)["label_set"] == ""


def test_label_set_is_refused_off_the_native_path(tmp_path):
    class A:
        dag_source = "mondo"
        label_set = str(tmp_path / "r.tsv")
    with pytest.raises(ValueError, match="mondo_native"):
        gpc._label_set_spec_fields(A())

    class B:
        dag_source = "mondo_native"
        label_set = ""
    assert gpc._label_set_spec_fields(B()) == {"label_set": "", "label_set_path": ""}


def test_run_experiment_passes_label_set_through_repo_relative(monkeypatch):
    """Front matter `label_set:` -> `--label-set <abs path>` (repo-root-relative,
    so the doc reads the same on a laptop and the cluster checkout); absent, the
    argv is byte-identical to before the key existed."""
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    import run_experiment as rex
    monkeypatch.setenv("WORKSPACE_CDR", "cdr")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "proj")
    base = {"source_table": "t", "person_mod": 1, "vocab_size": 5000, "min_df": 20,
            "min_patient_count": 20, "doc_min_length": 10, "max_iter": 100,
            "min_n": 0, "n_bg": 8, "tpn": 1, "seed": 0, "dag_source": "mondo_native",
            "readout_mode": "distributed"}
    rel = "analysis/cloud/anchor_selection_data/dismech_ribbon.tsv"
    argv = rex.build_gated_pc_args(dict(base, label_set=rel), "/tmp/out")
    got = argv[argv.index("--label-set") + 1]
    assert got == str(REPO_ROOT / rel) and Path(got).is_absolute()
    assert "--label-set" not in rex.build_gated_pc_args(dict(base), "/tmp/out")


def test_the_committed_ribbon_loads_and_matches_the_spec_pin():
    rb = dr.load_ribbon(REPO_ROOT / "analysis/cloud/anchor_selection_data/dismech_ribbon.tsv")
    assert rb.commit == "71cd0452358b01ef7d0b3cc213297f9837be7e54"
    assert len(rb.mondo_ids) == 3226 and len(rb.rows) == 3245
    assert dr.label_set_identity(rb) == "dismech:71cd0452358b:3226:f27a6b36ca62"
    assert all(i.startswith("MONDO:") for i in rb.mondo_ids)
    assert "MONDO:0000001" not in rb.mondo_ids


def test_flat_label_dag_puts_every_member_under_the_root():
    """Exp 0135's first launch: the induced Hasse nested peripartum CM under DCM
    (depth 4 overall). With flat=True every kept member is a root child, whatever
    Mondo says, and the attestation roll-up is unchanged (most specific member)."""
    A, B, C, D, R = ("MONDO:0000001", "MONDO:0000002", "MONDO:0000003",
                     "MONDO:0000004", "MONDO:0000000")
    #  R -> A -> B -> C ;  A, C kept (B unpowered); D kept, separate branch
    parent = {A: [R], B: [A], C: [B], D: [R], R: []}
    kept = {A, C, D}
    nested, _ = mnd.build_native_label_dag(kept, parent, coded_ids=kept, flat=False)
    flat, st = mnd.build_native_label_dag(kept, parent, coded_ids=kept, flat=True)
    root = mnd.MONDO_NATIVE_ROOT_CID
    assert all(ps == [root] for ps in flat.parents.values()), flat.parents
    assert nested.parents[mnd.mondo_cid(C)] == [mnd.mondo_cid(A)]
    assert st["n_hasse_multi_parent"] == 0
    assert mnd.flat_label_parents({"X", "Y"}, {}) == {"X": [], "Y": []}
    # roll-up is the same under both: a term under C lands on C only, never on A
    assert mnd.roll_terms_to_kept([C], kept, parent) == {C: [C]}


def test_member_ancestor_pairs_among_final_nodes():
    E, H, C, R = "MONDO:0020066", "MONDO:0007523", "MONDO:0005021", "MONDO:0000000"
    parent = {E: [R], H: [E], C: [R], R: []}
    pairs = mnd.member_ancestor_pairs({E, H, C}, parent)
    assert pairs == [(mnd.mondo_cid(E), mnd.mondo_cid(H))]
    assert mnd.member_ancestor_pairs({H, C}, parent) == []


@pytest.mark.slow
def test_provider_keeps_the_descendant_only_when_a_doc_attests_both(spark):
    """A patient coded with both 'EDS' and 'hypermobile EDS' attests hEDS only;
    one coded with EDS alone attests EDS; a doc with no mapped code attests
    nothing; without `ancestor_pairs` the provider is unchanged (both)."""
    from charmpheno.omop.doc_spec import PatientCohortDocSpec
    E, H = mnd.mondo_cid("MONDO:0020066"), mnd.mondo_cid("MONDO:0007523")
    code_map = spark.createDataFrame([(101, E), (102, H), (103, E)],
                                     ["std_cid", "node_cid"])
    ev = spark.createDataFrame(
        [(1, "fg", 101), (1, "fg", 102),        # both codes -> hEDS only
         (2, "fg", 103),                        # EDS alone -> EDS
         (3, "fg", 999)],                       # unmapped -> []
        ["person_id", "source_cohort", "concept_id"])
    spec = PatientCohortDocSpec()
    reduced = mnd.make_mondo_native_attested_provider(
        code_map, doc_spec=spec, ancestor_pairs=[(E, H)])(ev)
    got = {r["person_id"]: sorted(r["attested_cids"]) for r in reduced.collect()}
    assert got == {1: [H], 2: [E], 3: []}
    plain = mnd.make_mondo_native_attested_provider(code_map, doc_spec=spec)(ev)
    got0 = {r["person_id"]: sorted(r["attested_cids"]) for r in plain.collect()}
    assert got0[1] == sorted([E, H]) and got0[2] == [E] and got0[3] == []


# --------------------------------------------------------------------------- #
# exp 0136: the multi-map guard (peripartum CM <- "Finding related to pregnancy") #
# --------------------------------------------------------------------------- #
def test_multimap_guard_drops_the_broad_target_and_keeps_the_disorder():
    from mondo_native_dag import (drop_ancestor_multimap_targets,
                                  format_native_powering_report)
    O903, PPCM, PREG = 45576878, 4037495, 444094
    J45, ASTHMA = 1, 317009
    X, A, B = 2, 10, 20                       # incomparable targets: both kept
    src = {O903: {PPCM, PREG}, J45: {ASTHMA}, X: {A, B}}
    # PREG is an ancestor of PPCM; A and B unrelated
    kept, dropped = drop_ancestor_multimap_targets(src, [(PREG, PPCM), (99, A)])
    assert kept == {O903: {PPCM}, J45: {ASTHMA}, X: {A, B}}
    assert dropped == [(O903, PREG)]
    rep = format_native_powering_report(dict(
        n_codes_resolved=1, n_coded_terms=1, n_terms_with_any_support=1, n_powered=1,
        min_positives=100, min_support_kept=100, n_codes_attesting=1,
        n_multimap_sources=2, n_multimap_dropped=1, multimap_dropped_targets=[PREG]))
    assert "multi-map guard: 2 source code(s)" in rep and "1 ancestor target(s)" in rep


def test_multimap_guard_fan_in_drops_a_context_concept_shared_by_unrelated_terms():
    """Main's insight 0076: 444094 is the exact concept of peripartum CM AND
    preeclampsia (unrelated in Mondo) — dropped even when SNOMED does not place it
    above the disorder. A concept shared by NESTED terms is kept, and a 1:1 map is
    never touched."""
    from mondo_native_dag import drop_ancestor_multimap_targets
    O903, PPCM, PREG, O14, PREE = 1, 4037495, 444094, 2, 439393
    PCM, PRE, HYP = "MONDO:0018920", "MONDO:0005081", "MONDO:0001134"
    padj = {PRE: [HYP], PCM: []}
    src = {O903: {PPCM, PREG}, O14: {PREE, PREG}, 3: {PREG}}
    terms = {PPCM: {PCM}, PREG: {PCM, PRE}, PREE: {PRE}}
    kept, dropped = drop_ancestor_multimap_targets(
        src, [], terms_of_target=terms, parent_adj=padj)       # no SNOMED ancestry
    assert kept[O903] == {PPCM} and kept[O14] == {PREE}
    assert kept[3] == {PREG}                                     # 1:1 map untouched
    # shared only by nested terms (preeclampsia under hypertensive disorder): kept
    kept2, _ = drop_ancestor_multimap_targets(
        {O14: {PREE, PREG}}, [], terms_of_target={PREG: {PRE, HYP}, PREE: {PRE}},
        parent_adj=padj)
    assert kept2[O14] == {PREE, PREG}


def test_code_map_audit_counts_shared_codes_by_mechanism():
    from mondo_native_dag import code_map_audit, format_code_map_audit
    rows = [(1, 10), (2, 10), (3, 10), (3, 20), (4, 20), (4, 30), (5, 30)]
    a = code_map_audit(rows, exact_codes={1, 3}, names={10: "PPCM", 20: "DCM"}, top=2)
    assert a["n_codes"] == 5 and a["n_nodes"] == 3
    assert a["n_shared_codes"] == 2 and a["n_shared_exact"] == 1 and a["n_shared_climb"] == 1
    assert a["n_nodes_sharing"] == 3
    assert a["top_nodes"] == [("PPCM", 3), ("DCM", 2)]
    assert "2 code(s) attest >= 2 nodes (1 exact-shared, 1 climb-tie)" in format_code_map_audit(a)


def test_hierarchy_test_drops_overbroad_targets_and_spares_consistent_ones():
    """v4 (0137 launch 2 withdrew the anchor test): an ICD target that SNOMED puts
    above >= 3 Mondo terms Mondo does not put under the disease is over-broad
    (tularemia <- 'Disorder of gastrointestinal tract'); a target whose subsumed
    terms are all Mondo descendants is kept (DCM's concept over peripartum and
    familial DCM); SNOMED same_as rows are never dropped."""
    from mondo_native_dag import overbroad_exact_rows
    TUL, CROHN, DIVERT, GERD = "MONDO:1", "MONDO:2", "MONDO:3", "MONDO:4"
    DCM, PPCM, FDCM, ALC = "MONDO:10", "MONDO:11", "MONDO:12", "MONDO:13"
    GI, DCM_C = 500, 600
    padj = {PPCM: [DCM], FDCM: [DCM]}
    rows = [(TUL, "ICD10CM", 1, GI),          # over-broad -> drop
            (TUL, "SNOMED", 2, GI),           # same target, SNOMED-declared -> keep
            (DCM, "ICD10CM", 3, DCM_C)]       # subsumes only DCM's own children (+1)
    subsumed = {GI: {CROHN, DIVERT, GERD}, DCM_C: {PPCM, FDCM, ALC}}
    kept, dropped, counts = overbroad_exact_rows(rows, subsumed, padj, min_unrelated=3)
    assert dropped == [(TUL, "ICD10CM", 1, GI)]
    assert (DCM, "ICD10CM", 3, DCM_C) in kept          # 1 unrelated (ALC) < 3
    assert counts == {(TUL, GI): 3}
    # the v5 default (10) keeps a concept SNOMED merely subdivides more finely
    k10, d10, c10 = overbroad_exact_rows(rows, subsumed, padj)
    assert d10 == [] and c10 == {(TUL, GI): 3}


def test_curated_xref_exclusion_and_snomed_suspects():
    """v6: Mondo's 'thrombophilia due to thrombin defect' -> SCTID 111293003 'Venous
    thrombosis' is excluded by name; a SNOMED row that fails the hierarchy test is
    reported as a suspect and never dropped."""
    import pandas as pd
    from mondo_native_dag import (EXCLUDED_MONDO_XREFS, drop_excluded_xrefs,
                                  overbroad_exact_rows)
    assert ("MONDO:0008559", "SNOMED", "111293003") in EXCLUDED_MONDO_XREFS
    df = pd.DataFrame({"mondo_id": ["MONDO:0008559", "MONDO:0008559", "MONDO:0002305"],
                       "vocabulary_id": ["SNOMED", "ICD10CM", "SNOMED"],
                       "concept_code": ["111293003", "D68.59", "234467004"]})
    kept, n = drop_excluded_xrefs(df)
    assert n == 1 and list(kept["concept_code"]) == ["D68.59", "234467004"]
    rows = [("MONDO:9", "SNOMED", 1, 777)]
    k, d, _c = overbroad_exact_rows(rows, {777: {f"MONDO:{i}" for i in range(20, 32)}}, {})
    assert d == [] and k == rows
    assert overbroad_exact_rows.snomed_suspects == {("MONDO:9", 777): 12}
