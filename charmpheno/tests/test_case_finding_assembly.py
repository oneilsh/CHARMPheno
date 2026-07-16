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
