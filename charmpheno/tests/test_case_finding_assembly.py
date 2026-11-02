"""Tests for charmpheno.omop.case_finding_assembly (the case-finding bundle) + the
diabetes disease-registry entry it depends on.

The bundle contract (spec 2026-10-30): the whole label DAG (`dag`, user ids) and
per document the attested node ids (`labels`). No frontier, no dense label vectors,
no pruning: the node set is learned by the estimator from the training documents
(`spark_vi.dag.nodeset`), which the last tests exercise on the bundle's own output."""
import datetime as dt

import pytest
from pyspark.ml.linalg import SparseVector

from charmpheno.omop.condition_dag import build_condition_dag
from charmpheno.omop.doc_spec import PatientCohortDocSpec


def test_disease_registry_has_diabetes_anchor_201820():
    from charmpheno.omop.cohorts import _DISEASE_REGISTRY
    assert _DISEASE_REGISTRY["diabetes"] == {
        "inclusion_ancestors": (201820,),
        "exclusion_ancestors": (),
    }


# --------------------------------------------------------------------------- #
# The DAG as JSON-ready edges                                                  #
# --------------------------------------------------------------------------- #
def test_dag_edges_round_trip_keeps_top_nodes_and_id_types():
    from charmpheno.omop.case_finding_assembly import dag_edges, dag_from_edges
    for dag in ({100: [], 200: [100], 300: [100], 400: [200, 300]},
                {"MONDO:0000001": [], "MONDO:0000002": ["MONDO:0000001"]}):
        edges = dag_edges(dag)
        assert [100, None] in edges or ["MONDO:0000001", None] in edges
        assert dag_from_edges(edges) == dag
    # a node that appears only as a parent is still a node
    assert dag_from_edges([[2, 1]]) == {1: [], 2: [1]}


# --------------------------------------------------------------------------- #
# Leakage strip                                                                #
# --------------------------------------------------------------------------- #
def test_strip_features_drops_only_named_dims_preserving_size():
    from charmpheno.omop.case_finding_assembly import strip_features
    v = SparseVector(10, [1, 3, 5, 7], [2.0, 1.0, 4.0, 1.0])
    out = strip_features(v, {3, 7})
    assert out.size == 10
    assert dict(zip(out.indices.tolist(), out.values.tolist())) == {1: 2.0, 5: 4.0}


def test_strip_features_empty_drop_is_identity():
    from charmpheno.omop.case_finding_assembly import strip_features
    v = SparseVector(5, [0, 2], [1.0, 3.0])
    out = strip_features(v, set())
    assert out.size == 5
    assert dict(zip(out.indices.tolist(), out.values.tolist())) == {0: 1.0, 2: 3.0}


def test_strip_features_all_dropped_yields_empty_vector():
    from charmpheno.omop.case_finding_assembly import strip_features
    v = SparseVector(4, [1, 2], [5.0, 6.0])
    out = strip_features(v, {1, 2})
    assert out.size == 4
    assert out.indices.tolist() == [] and out.values.tolist() == []


# --------------------------------------------------------------------------- #
# Spark pieces                                                                 #
# --------------------------------------------------------------------------- #
def _events(spark, rows):
    # rows: (person_id, concept_id, source_cohort, start_date)
    return spark.createDataFrame(
        rows,
        ["person_id", "concept_id", "source_cohort", "condition_era_start_date"],
    )


def test_doc_attested_nodes_keeps_only_dag_nodes_and_background_empty(spark):
    from charmpheno.omop.case_finding_assembly import doc_attested_nodes
    node_ids = {200, 300, 400}
    ev = _events(spark, [
        (1, 200, "diabetes", dt.date(2015, 1, 1)),   # node
        (1, 999, "diabetes", dt.date(2015, 2, 1)),   # non-node (rides along)
        (1, 400, "diabetes", dt.date(2015, 3, 1)),   # node
        (2, 888, "general",  dt.date(2016, 1, 1)),   # background, no node code
    ])
    att = doc_attested_nodes(ev, node_ids, doc_spec=PatientCohortDocSpec())
    assert dict(att.dtypes)["labels"] == "array<bigint>"
    out = {r["doc_id"]: (r["person_id"], r["source_cohort"], list(r["labels"]))
           for r in att.collect()}
    # every attested node, ancestors included (the estimator reduces), sorted
    assert out["diabetes:1"] == (1, "diabetes", [200, 400])
    assert out["general:2"] == (2, "general", [])       # background survives, empty


def test_doc_attested_nodes_distinct_within_doc(spark):
    from charmpheno.omop.case_finding_assembly import doc_attested_nodes
    ev = _events(spark, [
        (1, 200, "diabetes", dt.date(2015, 1, 1)),
        (1, 200, "diabetes", dt.date(2015, 6, 1)),   # same node twice in the window
    ])
    row = doc_attested_nodes(ev, {200}, doc_spec=PatientCohortDocSpec()).collect()[0]
    assert list(row["labels"]) == [200]


def test_attach_label_column_fills_background_with_a_typed_empty_array(spark):
    from charmpheno.omop.case_finding_assembly import attach_label_column
    bow = spark.createDataFrame([("a:1",), ("a:2",)], ["doc_id"])
    att = spark.createDataFrame([("a:1", "a", ["MONDO:0000002"])],
                                "doc_id string, source_cohort string, "
                                "labels array<string>")
    out = attach_label_column(bow, att)
    assert dict(out.dtypes)["labels"] == "array<string>"
    got = {r["doc_id"]: list(r["labels"]) for r in out.collect()}
    assert got == {"a:1": ["MONDO:0000002"], "a:2": []}


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


def _diabetes_dag():
    # anchor 100 -> 200 (T2), 300 (T1); 200 -> 400 (T2-with-complication node).
    cdag = build_condition_dag(
        [(100, 200), (100, 300), (200, 400)], anchor=100, node_ids=[200, 300, 400],
        names={100: "diabetes", 200: "T2", 300: "T1", 400: "T2-renal"})
    return cdag.user_dag(), dict(cdag.names)


def test_assemble_from_events_end_to_end(spark):
    """Full assembly on synthetic events + a tiny DAG: schema, the whole DAG handed
    over, labels per document (attested nodes), leakage strip on TEST only."""
    from charmpheno.omop.case_finding_assembly import assemble_from_events

    dag, names = _diabetes_dag()
    rows = []
    for pid in range(30):                       # diabetes / foreground
        rows.append((pid, 200, "diabetes", dt.date(2015, 1, 1)))
        rows.append((pid, 999, "diabetes", dt.date(2015, 2, 1)))
        if pid % 2 == 0:
            rows.append((pid, 400, "diabetes", dt.date(2015, 3, 1)))
    for pid in range(100, 130):                 # background / general
        rows.append((pid, 888, "general", dt.date(2016, 1, 1)))
        rows.append((pid, 777, "general", dt.date(2016, 2, 1)))
    ev = _events(spark, rows)

    bundle = assemble_from_events(
        ev, dag, names=names, doc_spec=PatientCohortDocSpec(min_doc_length=0),
        holdout_frac=0.3, split_salt=20260716,
        vocab_size=100, min_df=1, min_patient_count=1)

    assert bundle.dag == {100: [], 200: [100], 300: [100], 400: [200]}
    assert bundle.name_by_id[200] == "T2"
    assert bundle.ledger["n_dag_nodes"] == 4
    assert bundle.ledger["train_fg_docs"] > 0 and bundle.ledger["test_fg_docs"] > 0

    for df in (bundle.train_df, bundle.test_df):
        assert {"person_id", "doc_id", "features", "labels",
                "source_cohort"} <= set(df.columns)
        assert "frontier" not in df.columns and "label" not in df.columns

    test_rows = {r["doc_id"]: r for r in bundle.test_df.collect()}
    fg = [r for did, r in test_rows.items() if did.startswith("diabetes:")]
    bg = [r for did, r in test_rows.items() if did.startswith("general:")]
    assert fg and bg
    for r in fg:
        assert list(r["labels"]) in ([200], [200, 400])
    for r in bg:
        assert list(r["labels"]) == []

    # leakage strip: TEST foreground features must NOT contain the node-200 vocab
    # dim, but MUST retain the rides-along non-node 999.
    vm = bundle.vocab_map
    for r in fg:
        assert vm[200] not in set(r["features"].indices.tolist())
        assert vm[999] in set(r["features"].indices.tolist())
    train_fg = [r for r in bundle.train_df.collect()
                if r["doc_id"].startswith("diabetes:")]
    assert train_fg
    assert any(vm[200] in set(r["features"].indices.tolist()) for r in train_fg)


def test_vocab_is_fit_on_train_and_the_node_set_is_learned_from_train_labels(spark):
    """Leakage, both halves. A token that appears ONLY in a test patient's document is
    absent from the train-fit (frozen-for-test) vocab. And a node attested by two
    patients (one train, one test) is PRUNED at minDocsPerNode=2 when the node set
    is learned from the bundle's TRAIN labels (the estimator's contract: support
    counted on what `fit` is given) although train+test would clear it. The bundle
    itself keeps the node: it no longer depends on the threshold.

    Person ids 500 (TRAIN) / 507 (TEST) are pinned by inspecting split_train_test;
    the assertion after building `ev` verifies the pin."""
    from charmpheno.omop.case_finding_assembly import (
        assemble_from_events, split_train_test)
    from spark_vi.dag.nodeset import fit_node_set

    dag, names = _diabetes_dag()
    rows = []
    for pid in range(40):
        rows.append((pid, 200, "diabetes", dt.date(2015, 1, 1)))
        rows.append((pid, 999, "diabetes", dt.date(2015, 2, 1)))
    for pid in range(100, 140):
        rows.append((pid, 888, "general", dt.date(2016, 1, 1)))
    train_pid, test_pid = 500, 507
    for p in (train_pid, test_pid):
        rows.append((p, 200, "diabetes", dt.date(2015, 1, 1)))
        rows.append((p, 400, "diabetes", dt.date(2015, 1, 2)))
    rows.append((test_pid, 555, "diabetes", dt.date(2015, 1, 3)))
    ev = _events(spark, rows)
    _, te_check = split_train_test(ev, holdout_frac=0.3, split_salt=20260716)
    test_ids = {r["person_id"] for r in te_check.select("person_id").distinct().collect()}
    assert train_pid not in test_ids and test_pid in test_ids

    bundle = assemble_from_events(
        ev, dag, names=names, doc_spec=PatientCohortDocSpec(min_doc_length=0),
        holdout_frac=0.3, split_salt=20260716,
        vocab_size=100, min_df=1, min_patient_count=1)
    assert 555 not in bundle.vocab_map
    assert 400 in bundle.dag                      # the bundle keeps every node

    train_labels = [list(r["labels"]) for r in bundle.train_df.collect()]
    ns = fit_node_set(bundle.dag, train_labels, min_docs_per_node=2)
    assert 400 in ns.pruned and ns.pruned[400] == (200,)   # lands on its parent
    assert 200 in ns.engine_id


def test_strip_mode_both_strips_train_features(spark):
    from charmpheno.omop.case_finding_assembly import assemble_from_events
    cdag = build_condition_dag([(100, 200), (100, 300)], anchor=100,
                               node_ids=[200, 300],
                               names={100: "dm", 200: "T2", 300: "T1"})
    rows = []
    for pid in range(40):
        rows.append((pid, 200, "diabetes", dt.date(2015, 1, 1)))
        rows.append((pid, 999, "diabetes", dt.date(2015, 2, 1)))
    ev = _events(spark, rows)
    kw = dict(doc_spec=PatientCohortDocSpec(min_doc_length=0),
              holdout_frac=0.3, split_salt=20260716, vocab_size=100, min_df=1,
              min_patient_count=1)
    b_test = assemble_from_events(ev, cdag.user_dag(), strip_mode="test_only", **kw)
    b_both = assemble_from_events(ev, cdag.user_dag(), strip_mode="both", **kw)
    node200 = b_both.vocab_map[200]
    assert any(node200 in set(r["features"].indices.tolist())
               for r in b_test.train_df.collect())
    assert all(node200 not in set(r["features"].indices.tolist())
               for r in b_both.train_df.collect())
    with pytest.raises(ValueError, match="strip_mode"):
        assemble_from_events(ev, cdag.user_dag(), strip_mode="never", **kw)


def test_assemble_from_events_label_events_decouples_features_from_labels(spark):
    # Feature frame carries ONLY non-node tokens (pre-index phenotype); the label
    # frame carries the DAG node code. Labels must come from the label frame and
    # the BOW features from the feature frame — never mixed.
    from charmpheno.omop.case_finding_assembly import (
        assemble_from_events, _condition_dag_from_frames)
    concept = spark.createDataFrame(
        [(100, "anchor", "S", "Condition"), (200, "sub", "S", "Condition")],
        ["concept_id", "concept_name", "standard_concept", "domain_id"])
    ca = spark.createDataFrame(
        [(100, 200, 1)],
        ["ancestor_concept_id", "descendant_concept_id", "min_levels_of_separation"])
    cdag = _condition_dag_from_frames(concept, ca, anchors=100, root=None)

    feats, labels = [], []
    for p in range(1, 13):
        feats += [(p, 900, "dis", dt.date(2014, 1, 1)), (p, 901, "dis", dt.date(2014, 2, 1))]
        labels += [(p, 200, "dis", dt.date(2015, 1, 1))]        # node code, post-index
    for p in range(100, 130):
        feats += [(p, 900, "general", dt.date(2014, 1, 1))]     # background feature only
    bundle = assemble_from_events(
        _events(spark, feats), cdag.user_dag(), names=cdag.names,
        doc_spec=PatientCohortDocSpec(min_doc_length=0), holdout_frac=0.25,
        vocab_size=50, min_df=1, min_patient_count=1,
        label_events=_events(spark, labels))
    assert 200 not in bundle.vocab_map
    assert 900 in bundle.vocab_map
    lab = {r["doc_id"]: list(r["labels"])
           for r in bundle.train_df.select("doc_id", "labels").collect()}
    assert any(v == [200] for k, v in lab.items() if k.startswith("dis:"))
    assert all(v == [] for k, v in lab.items() if k.startswith("general:"))


def test_condition_dag_from_frames_builds_taxonomy_from_omop_frames(spark):
    from charmpheno.omop.case_finding_assembly import _condition_dag_from_frames
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
    # the single anchor is a real concept: a top-level node of the handed-over DAG
    assert dag.user_dag() == {100: [], 200: [100], 300: [100], 400: [200]}


def test_condition_dag_from_frames_builds_forest_over_multiple_anchors(spark):
    from charmpheno.omop.case_finding_assembly import _condition_dag_from_frames
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
    assert dag.anchor == -1
    assert dag.nodes() == {-1, 100, 200, 500, 600}
    assert dag.parents[100] == [-1] and dag.parents[500] == [-1]
    assert dag.parents[200] == [100] and dag.parents[600] == [500]
    assert dag.depth(200) == 2 and dag.depth(100) == 1
    # the synthetic forest root is dropped: the anchors are top-level
    assert dag.user_dag(synthetic_root=-1) == {100: [], 200: [100], 500: [],
                                               600: [500]}


def test_condition_dag_from_frames_rejects_multi_anchor_without_root(spark):
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
    assert {"disease", "cdr", "billing", "person_mod", "vocab_size",
            "holdout_frac"} <= set(p)
    assert p["disease"].default == "diabetes"
    assert "anchor" not in p                     # anchor is derived from disease now
    # the threshold, the layout and the readout's mask are not bundle inputs
    for gone in ("min_n", "n_bg", "tpn", "emit_labels", "label_mask_mode"):
        assert gone not in p


def test_assemble_case_finding_corpus_accepts_window_mode_params():
    import inspect
    from charmpheno.omop.case_finding_assembly import assemble_case_finding_corpus
    sig = inspect.signature(assemble_case_finding_corpus)
    assert sig.parameters["window_mode"].default == "forward"
    assert sig.parameters["lookback_days"].default == 365
    assert sig.parameters["label_window_days"].default == 365
