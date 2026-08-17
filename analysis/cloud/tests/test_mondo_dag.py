"""Unit tests for the whole-Mondo label DAG seams (analysis/cloud/mondo_dag.py).

Covers the two SNOMED-specific seams a Mondo fit swaps in:
  1. build_mondo_engine_dag / mondo_reduced_hierarchy — the integer-id engine DAG
     (terminals = OMOP anchor cids, class nodes = synthetic negatives, root = -1),
     plus the template-branch restriction;
  2. make_mondo_attested_provider — the SNOMED-climb per-patient attestation.
"""
import pandas as pd
import pytest

from mondo_dag import (
    build_mondo_engine_dag, mondo_reduced_hierarchy, branch_mondo_id_set,
    powered_anchor_climb, make_mondo_attested_provider, MONDO_ROOT_CID)


# --- synthetic Mondo disease frames (a tiny two-branch tree) ---------------
def _mondo_frames():
    ROOT = "MONDO:0700096"  # human disease (the reduction's stop ceiling)
    rows = [
        (ROOT, "human disease"),
        ("MONDO:CV", "cardiovascular disorder"),
        ("MONDO:NEU", "nervous system disorder"),
        ("MONDO:A1", "cv sub one"),
        ("MONDO:A2", "cv sub two"),
        ("MONDO:B1", "neuro sub one"),
    ]
    nodes_df = pd.DataFrame(
        {"id": [r[0] for r in rows], "name": [r[1] for r in rows],
         "category": ["biolink:Disease"] * len(rows)})
    edges = [("MONDO:CV", ROOT), ("MONDO:NEU", ROOT), ("MONDO:A1", "MONDO:CV"),
             ("MONDO:A2", "MONDO:CV"), ("MONDO:B1", "MONDO:NEU")]
    edges_df = pd.DataFrame(
        {"subject": [e[0] for e in edges], "object": [e[1] for e in edges],
         "predicate": ["biolink:subclass_of"] * len(edges)})
    mapping = pd.DataFrame(
        {"standard_concept_id": [1001, 1002, 2001],
         "standard_concept_name": ["a1", "a2", "b1"],
         "mondo_id": ["MONDO:A1", "MONDO:A2", "MONDO:B1"]})
    return edges_df, nodes_df, mapping


def test_build_mondo_engine_dag_id_space():
    """Terminals -> positive OMOP cids, class nodes -> synthetic negatives,
    root -> -1 (engine 0); to_engine() gives the expected parent structure."""
    parent_of = {
        "anchor:1001": ["MONDO:CV"],
        "anchor:1002": ["MONDO:CV"],
        "anchor:2001": [],            # no kept class -> attaches to root
        "MONDO:CV": [],               # a top class
    }
    dag = build_mondo_engine_dag(
        parent_of, anchor_names={1001: "a1", 1002: "a2", 2001: "b1"},
        class_names={"MONDO:CV": "cardio"})

    parent_int, int2cid, cid2int = dag.to_engine()
    # root is engine 0 and the negative sentinel concept-id.
    assert cid2int[MONDO_ROOT_CID] == 0
    # CONCEPT-id space: terminals keep their positive OMOP cids; the class node is
    # a synthetic negative; only the root shares the negative half with it.
    class_cid = dag_class_cid(dag)          # negative by construction
    assert {1001, 1002, 2001} <= dag.nodes()
    assert all(c > 0 for c in dag.nodes() if c not in (MONDO_ROOT_CID, class_cid))
    # structure (ENGINE-id space): CV and the orphan anchor hang off root; the two
    # cv anchors hang off CV.
    assert parent_int[cid2int[1001]] == [cid2int[class_cid]]
    assert parent_int[cid2int[1002]] == [cid2int[class_cid]]
    assert parent_int[cid2int[2001]] == [0]
    assert parent_int[cid2int[class_cid]] == [0]
    # names carried through both id kinds.
    assert dag.names[1001] == "a1"
    assert dag.names[class_cid] == "cardio"


def dag_class_cid(dag):
    """The single synthetic (negative, non-root) class id in the test DAG."""
    return next(c for c in dag.nodes() if c < 0 and c != MONDO_ROOT_CID)


def test_mondo_reduced_hierarchy_keeps_branch_point():
    """Over the tiny tree, the cardiovascular term is the one kept class (covers
    two anchors); the neuro anchor is unclustered (its parent covers only one)."""
    edges_df, nodes_df, mapping = _mondo_frames()
    reduced, anchor_names, class_names = mondo_reduced_hierarchy(
        mapping, {1001, 1002, 2001}, edges_df=edges_df, nodes_df=nodes_df)

    assert reduced["n_classes"] == 1
    (class_id,) = reduced["classes"]
    assert class_id == "MONDO:CV"
    assert reduced["classes"][class_id]["size"] == 2
    assert reduced["parent_of"]["anchor:1001"] == ["MONDO:CV"]
    assert reduced["parent_of"]["anchor:2001"] == []   # unclustered -> root
    assert anchor_names[1001] == "a1"
    assert class_names["MONDO:CV"] == "cardiovascular disorder"


def test_branch_restriction_drops_other_body_systems():
    """Restricting to the cardiovascular subtree keeps only its two anchors; the
    neuro anchor is dropped before the reduction (the Step-A template knob)."""
    edges_df, nodes_df, mapping = _mondo_frames()
    branch = branch_mondo_id_set("MONDO:CV", edges_df=edges_df, nodes_df=nodes_df)
    assert branch == {"MONDO:CV", "MONDO:A1", "MONDO:A2"}

    reduced, _, _ = mondo_reduced_hierarchy(
        mapping, {1001, 1002, 2001}, edges_df=edges_df, nodes_df=nodes_df,
        branch_mondo_ids=branch)
    terminals = [n for n in reduced["parent_of"] if n.startswith("anchor:")]
    assert set(terminals) == {"anchor:1001", "anchor:1002"}
    assert "anchor:2001" not in reduced["parent_of"]

    # and it assembles into an engine DAG of exactly root + CV + 2 anchors.
    dag = build_mondo_engine_dag(reduced["parent_of"], anchor_names=anchor_names_from(mapping),
                                 class_names={"MONDO:CV": "cardio"})
    assert len(dag.nodes()) == 4  # root, CV, 1001, 1002


def anchor_names_from(mapping):
    return dict(zip(mapping["standard_concept_id"].astype(int),
                    mapping["standard_concept_name"]))


def test_mondo_attested_provider_climb(spark):
    """A patient attests the powered anchor(s) their condition codes climb to; a
    patient with no anchor-mapped code survives with an empty attested_cids."""
    from charmpheno.omop.doc_spec import PatientCohortDocSpec

    # climb: anchor 1001 covers descendants {5001, 5002}; anchor 1002 covers {5003}.
    ca = spark.createDataFrame(pd.DataFrame({
        "ancestor_concept_id": [1001, 1001, 1002, 7],
        "descendant_concept_id": [5001, 5002, 5003, 8]}))
    climb = powered_anchor_climb(ca, {1001, 1002}, spark=spark)

    events = spark.createDataFrame(pd.DataFrame({
        "person_id": [1, 2, 2, 3],
        "concept_id": [5001, 5003, 5002, 9999],   # p3's 9999 maps to no anchor
        "source_cohort": ["population"] * 4}))
    provider = make_mondo_attested_provider(
        climb, doc_spec=PatientCohortDocSpec())
    got = {r["doc_id"]: sorted(r["attested_cids"])
           for r in provider(events).collect()}

    assert got["population:1"] == [1001]
    assert got["population:2"] == [1001, 1002]
    assert got["population:3"] == []          # background doc survives, empty
