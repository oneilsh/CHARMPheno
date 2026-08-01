"""Tests for charmpheno.omop.case_finding_assembly (piece 2 of the case-finding
cluster driver) + the diabetes disease-registry entry it depends on."""


def test_disease_registry_has_diabetes_anchor_201820():
    from charmpheno.omop.cohorts import _DISEASE_REGISTRY
    assert _DISEASE_REGISTRY["diabetes"] == {
        "inclusion_ancestors": (201820,),
        "exclusion_ancestors": (),
    }


from charmpheno.omop.condition_dag import build_condition_dag
from spark_vi.models.topic.dag_placement import DagLayout


def _diamond_dag():
    # concept-id DAG rooted at anchor 100:
    #   100 -> 200, 300 ; 200 -> 400 ; 300 -> 400 (diamond) ; 200 -> 500
    edges = [(100, 200), (100, 300), (200, 400), (300, 400), (200, 500)]
    node_ids = [200, 300, 400, 500]
    return build_condition_dag(edges, anchor=100, node_ids=node_ids)


def test_descendants_walks_transitively():
    from charmpheno.omop.case_finding_assembly import _descendants
    dag = _diamond_dag()
    ch = dag.children()
    assert _descendants(ch, 200) == {400, 500}
    assert _descendants(ch, 400) == set()


def test_most_specific_cids_drops_attested_ancestors_keeps_incomparable():
    from charmpheno.omop.case_finding_assembly import most_specific_cids
    dag = _diamond_dag()
    # attest 200 and its descendant 400 -> only 400 is most-specific.
    assert most_specific_cids({200, 400}, dag) == {400}
    # attest incomparable 400 and 500 -> both kept.
    assert most_specific_cids({400, 500}, dag) == {400, 500}
    # single node -> itself.
    assert most_specific_cids({300}, dag) == {300}


def test_roll_up_to_survivors_reattaches_dropped_to_nearest_ancestor():
    from charmpheno.omop.case_finding_assembly import roll_up_to_survivors
    dag = _diamond_dag()
    keep = {100, 200, 300}          # 400 and 500 pruned
    # 400 (pruned) rolls up to BOTH surviving parents 200 and 300.
    assert roll_up_to_survivors({400}, dag, keep) == {200, 300}
    # a kept node stays itself; a dropped one rolls up, in one call.
    assert roll_up_to_survivors({200, 500}, dag, keep) == {200}


def test_doc_frontier_engine_ids_maps_and_reduces_to_most_specific():
    from charmpheno.omop.case_finding_assembly import doc_frontier_engine_ids
    dag = _diamond_dag()
    keep = dag.nodes()                        # nothing pruned
    parent_int, int2cid, cid2int = dag.to_engine()
    lay = DagLayout(parent_int, n_bg=2, tpn=1)
    # attest 200 + 400: 400 is a descendant of 200 -> frontier = {engine(400)}.
    fr = doc_frontier_engine_ids({200, 400}, dag, keep, cid2int, lay)
    assert fr == [cid2int[400]]
    # empty attestation (background doc) -> [].
    assert doc_frontier_engine_ids(set(), dag, keep, cid2int, lay) == []
    # incomparable 400 + 500 -> both, in engine space, sorted.
    fr2 = doc_frontier_engine_ids({400, 500}, dag, keep, cid2int, lay)
    assert fr2 == sorted([cid2int[400], cid2int[500]])


def test_doc_frontier_engine_ids_rolls_pruned_attestation_up():
    from charmpheno.omop.case_finding_assembly import doc_frontier_engine_ids
    dag = _diamond_dag()
    # prune 400: a patient attesting only 400 rolls up to 200 and 300, which are
    # incomparable survivors -> frontier = {engine(200), engine(300)}.
    keep = {100, 200, 300, 500}
    from charmpheno.omop.condition_dag import prune_by_attestation
    counts = {200: 99, 300: 99, 500: 99, 400: 0}
    after = prune_by_attestation(dag, counts, min_n=1)
    parent_int, int2cid, cid2int = after.to_engine()
    lay = DagLayout(parent_int, n_bg=2, tpn=1)
    fr = doc_frontier_engine_ids({400}, dag, after.nodes(), cid2int, lay)
    assert fr == sorted([cid2int[200], cid2int[300]])


def test_strip_features_drops_only_named_dims_preserving_size():
    from pyspark.ml.linalg import SparseVector
    from charmpheno.omop.case_finding_assembly import strip_features
    v = SparseVector(10, [1, 3, 5, 7], [2.0, 1.0, 4.0, 1.0])
    out = strip_features(v, {3, 7})
    assert out.size == 10
    assert dict(zip(out.indices.tolist(), out.values.tolist())) == {1: 2.0, 5: 4.0}


def test_strip_features_empty_drop_is_identity():
    from pyspark.ml.linalg import SparseVector
    from charmpheno.omop.case_finding_assembly import strip_features
    v = SparseVector(5, [0, 2], [1.0, 3.0])
    out = strip_features(v, set())
    assert out.size == 5
    assert dict(zip(out.indices.tolist(), out.values.tolist())) == {0: 1.0, 2: 3.0}


def test_strip_features_all_dropped_yields_empty_vector():
    from pyspark.ml.linalg import SparseVector
    from charmpheno.omop.case_finding_assembly import strip_features
    v = SparseVector(4, [1, 2], [5.0, 6.0])
    out = strip_features(v, {1, 2})
    assert out.size == 4
    assert out.indices.tolist() == [] and out.values.tolist() == []


import datetime as dt
from charmpheno.omop.doc_spec import PatientCohortDocSpec


def _events(spark, rows):
    # rows: (person_id, concept_id, source_cohort, start_date)
    return spark.createDataFrame(
        rows,
        ["person_id", "concept_id", "source_cohort", "condition_era_start_date"],
    )


def test_doc_attested_nodes_keeps_only_dag_nodes_and_background_empty(spark):
    from charmpheno.omop.case_finding_assembly import doc_attested_nodes
    node_cids = {200, 300, 400}
    ev = _events(spark, [
        (1, 200, "diabetes", dt.date(2015, 1, 1)),   # node
        (1, 999, "diabetes", dt.date(2015, 2, 1)),   # non-node (rides along)
        (1, 400, "diabetes", dt.date(2015, 3, 1)),   # node
        (2, 888, "general",  dt.date(2016, 1, 1)),   # background, no node code
    ])
    out = {
        r["doc_id"]: (r["person_id"], r["source_cohort"], sorted(r["attested_cids"]))
        for r in doc_attested_nodes(
            ev, node_cids, doc_spec=PatientCohortDocSpec()).collect()
    }
    assert out["diabetes:1"] == (1, "diabetes", [200, 400])
    assert out["general:2"] == (2, "general", [])       # background survives, empty


def test_doc_attested_nodes_rollup_maps_descendants_to_nearest_dag_node(spark):
    from charmpheno.omop.case_finding_assembly import doc_attested_nodes
    node_cids = {200, 400}  # 200 = a class node, 400 = an anchor
    # concept_ancestor (ancestor, descendant): 200 covers 999 (a non-node
    # descendant) and itself; 400 covers itself. min_sep column not needed here.
    ca = spark.createDataFrame(
        [(200, 200), (200, 999), (400, 400)],
        ["ancestor_concept_id", "descendant_concept_id"])
    ev = _events(spark, [
        (1, 999, "general", dt.date(2015, 1, 1)),   # non-node code UNDER class 200
        (2, 400, "rare",    dt.date(2016, 1, 1)),   # anchor code
        (3, 888, "general", dt.date(2016, 1, 1)),   # unrelated -> background
    ])
    out = {r["doc_id"]: sorted(r["attested_cids"])
           for r in doc_attested_nodes(
               ev, node_cids, doc_spec=PatientCohortDocSpec(), ancestor_df=ca).collect()}
    assert out["general:1"] == [200]   # migraine-like code rolled UP to its class
    assert out["rare:2"] == [400]      # anchor unchanged
    assert out["general:3"] == []      # no descendant of any node -> background

    # Without ancestor_df, the non-node code 999 attests nothing (exact-match).
    exact = {r["doc_id"]: sorted(r["attested_cids"])
             for r in doc_attested_nodes(
                 ev, node_cids, doc_spec=PatientCohortDocSpec()).collect()}
    assert exact["general:1"] == []    # 999 is not a DAG node -> not attested


def test_doc_attested_nodes_distinct_within_doc(spark):
    from charmpheno.omop.case_finding_assembly import doc_attested_nodes
    ev = _events(spark, [
        (1, 200, "diabetes", dt.date(2015, 1, 1)),
        (1, 200, "diabetes", dt.date(2015, 6, 1)),   # same node twice in the window
    ])
    row = doc_attested_nodes(ev, {200}, doc_spec=PatientCohortDocSpec()).collect()[0]
    assert sorted(row["attested_cids"]) == [200]


def test_node_patient_counts_counts_distinct_patients_not_docs(spark):
    from charmpheno.omop.case_finding_assembly import (
        doc_attested_nodes, node_patient_counts,
    )
    ev = _events(spark, [
        (1, 200, "diabetes", dt.date(2015, 1, 1)),
        (2, 200, "diabetes", dt.date(2016, 1, 1)),   # node 200: 2 distinct patients
        (2, 300, "diabetes", dt.date(2016, 2, 1)),   # node 300: 1 patient
        (3, 999, "general",  dt.date(2017, 1, 1)),   # no node -> contributes nothing
    ])
    att = doc_attested_nodes(ev, {200, 300}, doc_spec=PatientCohortDocSpec())
    assert node_patient_counts(att) == {200: 2, 300: 1}


from pyspark.ml.linalg import SparseVector


def test_attach_frontiers_emits_engine_ids_and_empty_for_background(spark):
    from charmpheno.omop.case_finding_assembly import (
        doc_attested_nodes, attach_frontiers,
    )
    from charmpheno.omop.condition_dag import build_condition_dag
    from spark_vi.models.topic.dag_placement import DagLayout
    edges = [(100, 200), (200, 400)]
    dag = build_condition_dag(edges, anchor=100, node_ids=[200, 400])
    parent_int, int2cid, cid2int = dag.to_engine()
    lay = DagLayout(parent_int, n_bg=2, tpn=1)

    ev = _events(spark, [
        (1, 200, "diabetes", dt.date(2015, 1, 1)),
        (1, 400, "diabetes", dt.date(2015, 2, 1)),   # descendant -> frontier {400}
        (2, 777, "general",  dt.date(2016, 1, 1)),   # background -> []
    ])
    att = doc_attested_nodes(ev, dag.nodes(), doc_spec=PatientCohortDocSpec())
    out = {
        r["doc_id"]: sorted(r["frontier"])
        for r in attach_frontiers(att, dag, dag.nodes(), cid2int, lay).collect()
    }
    assert out["diabetes:1"] == [cid2int[400]]
    assert out["general:2"] == []


def test_split_train_test_is_deterministic_and_person_disjoint(spark):
    from charmpheno.omop.case_finding_assembly import split_train_test
    df = spark.createDataFrame(
        [(pid, f"diabetes:{pid}") for pid in range(200)],
        ["person_id", "doc_id"],
    )
    tr1, te1 = split_train_test(df, holdout_frac=0.25, split_salt=20260716)
    tr2, te2 = split_train_test(df, holdout_frac=0.25, split_salt=20260716)
    test_ids_1 = {r["person_id"] for r in te1.collect()}
    test_ids_2 = {r["person_id"] for r in te2.collect()}
    train_ids_1 = {r["person_id"] for r in tr1.collect()}
    assert test_ids_1 == test_ids_2                       # deterministic
    assert test_ids_1 & train_ids_1 == set()              # disjoint
    assert test_ids_1 | train_ids_1 == set(range(200))    # a partition
    assert 0.15 < len(test_ids_1) / 200 < 0.35            # roughly holdout_frac


def test_strip_test_features_removes_named_vocab_dims(spark):
    from charmpheno.omop.case_finding_assembly import strip_test_features
    df = spark.createDataFrame(
        [(1, SparseVector(6, [0, 2, 4], [1.0, 2.0, 3.0]))],
        ["person_id", "features"],
    )
    out = strip_test_features(df, {2}).collect()[0]["features"]
    assert out.size == 6
    assert dict(zip(out.indices.tolist(), out.values.tolist())) == {0: 1.0, 4: 3.0}


def test_assemble_from_events_end_to_end(spark):
    """Full assembly on synthetic events + a tiny DAG: schema, frontier engine-ids,
    leakage strip on TEST only, and a DagLayout-loadable bundle."""
    from charmpheno.omop.case_finding_assembly import assemble_from_events
    from charmpheno.omop.condition_dag import build_condition_dag
    from spark_vi.models.topic.dag_placement import DagLayout

    # DAG: anchor 100 -> 200 (T2), 300 (T1); 200 -> 400 (T2-with-complication node).
    edges = [(100, 200), (100, 300), (200, 400)]
    before = build_condition_dag(
        edges, anchor=100, node_ids=[200, 300, 400],
        names={100: "diabetes", 200: "T2", 300: "T1", 400: "T2-renal"},
    )

    # 30 diabetes patients attest 200 (+ some 400) + a rides-along non-node 999;
    # 30 background patients attest only non-node codes. One 365-day window each,
    # collapsed to one doc by PatientCohortDocSpec.
    rows = []
    for pid in range(30):                       # diabetes / foreground
        rows.append((pid, 200, "diabetes", dt.date(2015, 1, 1)))
        rows.append((pid, 999, "diabetes", dt.date(2015, 2, 1)))
        if pid % 2 == 0:
            rows.append((pid, 400, "diabetes", dt.date(2015, 3, 1)))
    for pid in range(100, 130):                 # background / general
        rows.append((pid, 888, "general", dt.date(2016, 1, 1)))
        rows.append((pid, 777, "general", dt.date(2016, 2, 1)))
    ev = spark.createDataFrame(
        rows, ["person_id", "concept_id", "source_cohort",
               "condition_era_start_date"])

    from charmpheno.omop.doc_spec import PatientCohortDocSpec
    bundle = assemble_from_events(
        ev, before, doc_spec=PatientCohortDocSpec(min_doc_length=0),
        min_n=1, holdout_frac=0.3, split_salt=20260716,
        vocab_size=100, min_df=1, min_patient_count=1, n_bg=2, tpn=1)

    # bundle plumbing
    assert set(bundle.parent_int) and 0 not in bundle.parent_int      # anchor has no parent
    lay = DagLayout(bundle.parent_int, n_bg=2, tpn=1)
    # K_nodes (pruning_ledger's `kept`) always includes the anchor/root; DagLayout.nodes
    # always excludes it (root=0 has no entry in the parent map) -- structural invariant
    # K_nodes == len(lay.nodes) + 1, independent of what min_n prunes. (Plan transcription
    # note: the plan's literal assertion had `+ 0`, which is mathematically impossible given
    # the existing, read-only pruning_ledger/DagLayout semantics -- corrected to `+ 1` here;
    # see the Task 6 report for the derivation.)
    assert bundle.ledger["K_nodes"] == len(lay.nodes) + 1
    assert bundle.name_by_id[200] == "T2"

    # schema the shim consumes
    for df in (bundle.train_df, bundle.test_df):
        assert set(["person_id", "doc_id", "features", "frontier",
                    "source_cohort"]) <= set(df.columns)

    # a foreground TEST doc that attested 400 has frontier == {engine(400)};
    # a background doc has frontier == [].
    test_rows = {r["doc_id"]: r for r in bundle.test_df.collect()}
    cid2int = bundle.cid2int
    fg = [r for did, r in test_rows.items() if did.startswith("diabetes:")]
    bg = [r for did, r in test_rows.items() if did.startswith("general:")]
    assert fg and bg
    for r in fg:
        assert set(r["frontier"]) in ({cid2int[200]}, {cid2int[400]})
    for r in bg:
        assert list(r["frontier"]) == []

    # leakage strip: TEST foreground features must NOT contain the node-200 vocab
    # dim, but MUST retain the rides-along non-node 999.
    vm = bundle.vocab_map
    node200_idx = vm[200]
    non_node_idx = vm[999]
    for r in fg:
        assert node200_idx not in set(r["features"].indices.tolist())
        assert non_node_idx in set(r["features"].indices.tolist())

    # train features are NOT stripped: a train foreground doc keeps node 200.
    train_fg = [r for r in bundle.train_df.collect()
                if r["doc_id"].startswith("diabetes:")]
    assert train_fg
    assert any(node200_idx in set(r["features"].indices.tolist()) for r in train_fg)


def test_assemble_prunes_and_fits_vocab_on_train_only(spark):
    """Leakage check: a node attested by 2 patients (one train, one test) with
    min_n=2 must be PRUNED under train-only counting (train count 1 < 2) even
    though the combined train+test count (2) would satisfy min_n; a token that
    appears ONLY in a test patient's document must be absent from the
    train-fit (and frozen-for-test) vocab.

    Person ids 500 (lands TRAIN) / 507 (lands TEST) are pinned by directly
    inspecting split_train_test(..., holdout_frac=0.3, split_salt=20260716) --
    the assertion right after building `ev` verifies the pin still holds
    rather than assuming it, so this test stays honest if the hash/threshold
    ever changes."""
    from charmpheno.omop.case_finding_assembly import (
        assemble_from_events, split_train_test,
    )
    from charmpheno.omop.condition_dag import build_condition_dag
    from charmpheno.omop.doc_spec import PatientCohortDocSpec
    import datetime as dt

    edges = [(100, 200), (100, 300), (200, 400)]
    before = build_condition_dag(edges, anchor=100, node_ids=[200, 300, 400],
                                 names={100: "dm", 200: "T2", 300: "T1", 400: "T2r"})

    rows = []
    for pid in range(40):                        # baseline diabetes patients
        rows.append((pid, 200, "diabetes", dt.date(2015, 1, 1)))
        rows.append((pid, 999, "diabetes", dt.date(2015, 2, 1)))
    for pid in range(100, 140):                   # baseline background patients
        rows.append((pid, 888, "general", dt.date(2016, 1, 1)))
        rows.append((pid, 777, "general", dt.date(2016, 2, 1)))
    # Node 400 (T2r) attested by exactly two patients: 500 (train) and 507
    # (test) under split_salt=20260716 / holdout_frac=0.3. Combined count = 2
    # (>= min_n) but TRAIN-only count = 1 (< min_n) -- this is exactly the
    # transductive leakage the split-first fix closes.
    train_pid, test_pid = 500, 507
    rows.append((train_pid, 200, "diabetes", dt.date(2015, 1, 1)))
    rows.append((train_pid, 400, "diabetes", dt.date(2015, 1, 2)))
    rows.append((test_pid, 200, "diabetes", dt.date(2015, 1, 1)))
    rows.append((test_pid, 400, "diabetes", dt.date(2015, 1, 2)))
    # Token 555 appears ONLY in the test patient's document.
    rows.append((test_pid, 555, "diabetes", dt.date(2015, 1, 3)))

    ev = spark.createDataFrame(
        rows, ["person_id", "concept_id", "source_cohort",
               "condition_era_start_date"])

    # Verify the pinned split assignment before trusting the rest of the fixture.
    _, te_check = split_train_test(ev, holdout_frac=0.3, split_salt=20260716)
    test_ids = {r["person_id"] for r in te_check.select("person_id").distinct().collect()}
    assert train_pid not in test_ids and test_pid in test_ids

    bundle = assemble_from_events(
        ev, before, doc_spec=PatientCohortDocSpec(min_doc_length=0),
        min_n=2, holdout_frac=0.3, split_salt=20260716,
        vocab_size=100, min_df=1, min_patient_count=1, n_bg=2, tpn=1)

    tr = {r["person_id"] for r in bundle.train_df.collect()}
    te = {r["person_id"] for r in bundle.test_df.collect()}
    assert tr and te and (tr & te == set())
    assert train_pid in tr and test_pid in te

    # node 400: TRAIN count 1 < min_n=2 -> pruned; absent from the train-fit
    # (pruned) DAG engine map, regardless of the combined train+test count.
    assert 400 not in bundle.cid2int

    # token 555: zero occurrences in TRAIN docs -> absent from the (train-fit,
    # frozen-for-test) vocab, even though it occurs once in the raw corpus.
    assert 555 not in bundle.vocab_map

    # ledger reports test coarsening.
    assert "test_coarsening_rate" in bundle.ledger
    assert "test_fg_docs" in bundle.ledger
    assert bundle.ledger["test_fg_docs"] > 0
    assert 0.0 <= bundle.ledger["test_coarsening_rate"] <= 1.0

    # DagLayout loads; K emergent from TRAIN-surviving nodes.
    from spark_vi.models.topic.dag_placement import DagLayout
    DagLayout(bundle.parent_int, n_bg=2, tpn=1)


def test_condition_dag_from_frames_builds_taxonomy_from_omop_frames(spark):
    from charmpheno.omop.case_finding_assembly import _condition_dag_from_frames
    # concept: anchor 100 + standard conditions 200,300,400; 999 non-standard,
    # 555 wrong-domain -> excluded as nodes.
    concept = spark.createDataFrame(
        [
            (100, "diabetes", "S", "Condition"),
            (200, "T2",       "S", "Condition"),
            (300, "T1",       "S", "Condition"),
            (400, "T2-renal", "S", "Condition"),
            (999, "non-std",  None, "Condition"),   # not standard
            (555, "a drug",   "S", "Drug"),          # wrong domain
        ],
        ["concept_id", "concept_name", "standard_concept", "domain_id"],
    )
    # concept_ancestor: descendants of 100 + min-sep-1 edges. Include a sep-2 row
    # (100->400) that must NOT become a direct edge.
    ca = spark.createDataFrame(
        [
            (100, 200, 1), (100, 300, 1), (100, 400, 2),
            (200, 400, 1),
            (100, 999, 1), (100, 555, 1),   # candidates filtered out by concept join
        ],
        ["ancestor_concept_id", "descendant_concept_id", "min_levels_of_separation"],
    )
    dag = _condition_dag_from_frames(concept, ca, anchors=[100])
    assert dag.nodes() == {100, 200, 300, 400}
    assert dag.parents[400] == [200]            # sep-1 edge only, not 100->400
    assert set(dag.parents[200]) == {100}
    assert dag.names[200] == "T2"


def test_condition_dag_from_frames_builds_forest_over_multiple_anchors(spark):
    from charmpheno.omop.case_finding_assembly import _condition_dag_from_frames
    # Two disjoint disease subtrees: anchor 100 -> 200; anchor 500 -> 600.
    concept = spark.createDataFrame(
        [
            (100, "diseaseA", "S", "Condition"),
            (200, "A-subtype", "S", "Condition"),
            (500, "diseaseB", "S", "Condition"),
            (600, "B-subtype", "S", "Condition"),
        ],
        ["concept_id", "concept_name", "standard_concept", "domain_id"],
    )
    ca = spark.createDataFrame(
        [(100, 200, 1), (500, 600, 1)],
        ["ancestor_concept_id", "descendant_concept_id", "min_levels_of_separation"],
    )
    dag = _condition_dag_from_frames(concept, ca, anchors=[100, 500], root=-1)
    # synthetic root over both anchors; each anchor a depth-1 child of the root.
    assert dag.anchor == -1
    assert dag.nodes() == {-1, 100, 200, 500, 600}
    assert dag.parents[100] == [-1]
    assert dag.parents[500] == [-1]
    assert dag.parents[200] == [100]            # within-subtree structure preserved
    assert dag.parents[600] == [500]
    assert dag.depth(200) == 2 and dag.depth(100) == 1
    # to_engine() maps the synthetic root to engine-id 0.
    _, int2cid, cid2int = dag.to_engine()
    assert cid2int[-1] == 0


def test_condition_dag_from_frames_rejects_multi_anchor_without_root(spark):
    import pytest
    from charmpheno.omop.case_finding_assembly import _condition_dag_from_frames
    concept = spark.createDataFrame(
        [(100, "a", "S", "Condition")],
        ["concept_id", "concept_name", "standard_concept", "domain_id"])
    ca = spark.createDataFrame(
        [(100, 100, 0)],
        ["ancestor_concept_id", "descendant_concept_id", "min_levels_of_separation"])
    with pytest.raises(ValueError, match="forest"):
        _condition_dag_from_frames(concept, ca, anchors=[100, 500])


def test_assemble_case_finding_corpus_importable_signature():
    import inspect
    from charmpheno.omop.case_finding_assembly import assemble_case_finding_corpus
    p = inspect.signature(assemble_case_finding_corpus).parameters
    assert {"disease", "cdr", "billing", "person_mod", "min_n", "vocab_size",
            "holdout_frac", "n_bg", "tpn"} <= set(p)
    assert p["disease"].default == "diabetes"
    assert "anchor" not in p                     # anchor is derived from disease now


def test_strip_mode_both_strips_train_features(spark):
    from charmpheno.omop.case_finding_assembly import assemble_from_events
    from charmpheno.omop.condition_dag import build_condition_dag
    from charmpheno.omop.doc_spec import PatientCohortDocSpec
    import datetime as dt
    edges = [(100, 200), (100, 300)]
    before = build_condition_dag(edges, anchor=100, node_ids=[200, 300],
                                 names={100: "dm", 200: "T2", 300: "T1"})
    rows = []
    for pid in range(40):
        rows.append((pid, 200, "diabetes", dt.date(2015, 1, 1)))
        rows.append((pid, 999, "diabetes", dt.date(2015, 2, 1)))
    ev = spark.createDataFrame(rows, ["person_id", "concept_id", "source_cohort",
                                      "condition_era_start_date"])
    kw = dict(doc_spec=PatientCohortDocSpec(min_doc_length=0), min_n=2,
              holdout_frac=0.3, split_salt=20260716, vocab_size=100, min_df=1,
              min_patient_count=1, n_bg=2, tpn=1)
    b_test = assemble_from_events(ev, before, strip_mode="test_only", **kw)
    b_both = assemble_from_events(ev, before, strip_mode="both", **kw)
    node200 = b_both.vocab_map.get(200)
    if node200 is not None:
        train_has = any(node200 in set(r["features"].indices.tolist())
                        for r in b_test.train_df.collect())
        train_stripped = all(node200 not in set(r["features"].indices.tolist())
                             for r in b_both.train_df.collect())
        assert train_has and train_stripped


def test_assemble_from_events_label_events_decouples_features_from_frontier(spark):
    # Feature frame carries ONLY non-node tokens (pre-index phenotype); the label
    # frame carries the DAG node code. The frontier must come from the label frame
    # and the BOW features from the feature frame — never mixed.
    import datetime as dt
    from charmpheno.omop.case_finding_assembly import assemble_from_events, _condition_dag_from_frames
    from charmpheno.omop.doc_spec import PatientCohortDocSpec
    # Minimal DAG: root anchor 100, one child node 200 (single-disease).
    concept = spark.createDataFrame(
        [(100, "anchor", "S", "Condition"), (200, "sub", "S", "Condition")],
        ["concept_id", "concept_name", "standard_concept", "domain_id"])
    ca = spark.createDataFrame(
        [(100, 200, 1)],
        ["ancestor_concept_id", "descendant_concept_id", "min_levels_of_separation"])
    dag = _condition_dag_from_frames(concept, ca, anchors=100, root=None)

    def ev(rows):   # (person, concept, source_cohort, date)
        return spark.createDataFrame(
            [(p, c, s, d) for (p, c, s, d) in rows],
            ["person_id", "concept_id", "source_cohort", "condition_era_start_date"])
    feats, labels = [], []
    for p in range(1, 13):
        feats += [(p, 900, "dis", dt.date(2014, 1, 1)), (p, 901, "dis", dt.date(2014, 2, 1))]
        labels += [(p, 200, "dis", dt.date(2015, 1, 1))]        # node code, post-index
    for p in range(100, 130):
        feats += [(p, 900, "general", dt.date(2014, 1, 1))]     # background feature only
    feature_events, label_events = ev(feats), ev(labels)

    doc_spec = PatientCohortDocSpec(min_doc_length=0)
    bundle = assemble_from_events(
        feature_events, dag, doc_spec=doc_spec, min_n=2, holdout_frac=0.25,
        vocab_size=50, min_df=1, min_patient_count=1, n_bg=2, tpn=1,
        label_events=label_events)
    # Node code 200 defines the DAG; it must NOT be in the feature vocab (features
    # are the 900/901 tokens only) — proving features came from the feature frame.
    assert 200 not in bundle.vocab_map
    assert 900 in bundle.vocab_map
    # Foreground docs got a non-empty frontier (from the label frame's node code).
    fr = {r["doc_id"]: r["frontier"]
          for r in bundle.train_df.select("doc_id", "frontier").collect()}
    assert any(len(v) > 0 for k, v in fr.items() if k.startswith("dis:"))
    assert all(len(v) == 0 for k, v in fr.items() if k.startswith("general:"))


def test_assemble_case_finding_corpus_accepts_window_mode_params():
    import inspect
    from charmpheno.omop.case_finding_assembly import assemble_case_finding_corpus
    sig = inspect.signature(assemble_case_finding_corpus)
    assert sig.parameters["window_mode"].default == "forward"
    assert sig.parameters["lookback_days"].default == 365
    assert sig.parameters["label_window_days"].default == 365
