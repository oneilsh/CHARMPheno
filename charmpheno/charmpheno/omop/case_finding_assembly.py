# charmpheno/charmpheno/omop/case_finding_assembly.py
"""Assemble the labeled hierarchical case-finding corpus from OMOP.

One document per patient tagged with its set-valued DAG frontier (the clinical
truth), the pruned label DagLayout, and a held-out split with the leakage strip
applied. Piece 2 of the cluster driver (piece 1 = condition_dag.py; piece 3 =
the cloud driver). Composes the piece-1 DAG builder, the population+disease
cohort machinery (cohorts.apply_population_disease_cohort), to_bow_dataframe,
and the engine's frontier helpers (spark_vi.models.topic.dag_placement).

This is the concept-id DOMAIN bridge: concept-ids are expected here. The engine
stays integer-id agnostic. Three id spaces are threaded with care (the tests pin
the translations):

  concept-id  raw OMOP concept_id; DAG build/prune/counts/roll-up.
  engine-id   contiguous 0..N from ConditionDag.to_engine() (anchor->0);
              DagLayout, frontier_from_coded, the emitted frontier column.
  vocab-index [0,V) from vocab_map {concept_id: idx}; features SparseVector,
              leakage strip.

See docs/superpowers/specs/2026-07-15-case-finding-assembly-design.md.
"""
from __future__ import annotations

from charmpheno.omop.condition_dag import _nearest_surviving_ancestors
from spark_vi.models.topic.dag_placement import frontier_from_coded


def _descendants(children_map: dict[int, list[int]], root: int) -> set[int]:
    """Proper descendants of `root` in a {parent: [children]} concept-id map."""
    out: set[int] = set()
    stack = list(children_map.get(root, []))
    while stack:
        x = stack.pop()
        if x in out:
            continue
        out.add(x)
        stack.extend(children_map.get(x, []))
    return out


def most_specific_cids(attested_cids, before_dag) -> set[int]:
    """The most-specific attested concept-ids: attested nodes with no attested
    proper descendant. Concept-id-space analogue of frontier_from_coded, used for
    the pruning ledger's coarsening accounting (which measures depths in the
    pre-prune ontology)."""
    C = set(attested_cids)
    ch = before_dag.children()
    return {c for c in C if not (_descendants(ch, c) & (C - {c}))}


def roll_up_to_survivors(attested_cids, before_dag, keep) -> set[int]:
    """Map each attested concept-id to a surviving node: itself if kept, else its
    nearest surviving ancestors (transitive walk up the PRE-PRUNE DAG). Mirrors
    prune_by_attestation's rewire (same _nearest_surviving_ancestors walk), so a
    rolled-up patient lands exactly where the pruned DAG reattaches its node."""
    surv: set[int] = set()
    for c in attested_cids:
        if c in keep:
            surv.add(c)
        else:
            surv |= _nearest_surviving_ancestors(before_dag, c, keep)
    return surv


def doc_frontier_engine_ids(attested_cids, before_dag, keep, cid2int, lay) -> list[int]:
    """The set-valued frontier in ENGINE-id space (sorted). Roll pruned
    attestations up to survivors (concept-id), map via cid2int, then
    frontier_from_coded over the pruned DagLayout (most-specific engine nodes;
    incomparable survivors kept as a set). Empty attestation (background doc) ->
    []."""
    if not attested_cids:
        return []
    survivors = roll_up_to_survivors(attested_cids, before_dag, keep)
    engine_ids = [cid2int[c] for c in survivors if c in cid2int]
    return sorted(frontier_from_coded(engine_ids, lay))


def strip_features(vec, drop_idxs):
    """Return a SparseVector equal to `vec` with the vocab dims in `drop_idxs`
    removed (leakage strip; held-out docs only). `vec.size` is preserved so the
    vector still matches the model vocabulary; the dropped indices simply become
    zero (absent from the sparse representation). This is the case-finding test:
    a held-out patient must not read its own DAG-node type code off its features."""
    from pyspark.ml.linalg import SparseVector
    if not drop_idxs:
        return vec
    drop = {int(i) for i in drop_idxs}
    kept = [(int(i), float(v)) for i, v in zip(vec.indices, vec.values)
            if int(i) not in drop]
    if not kept:
        return SparseVector(vec.size, [], [])
    idxs, vals = zip(*kept)
    return SparseVector(vec.size, list(idxs), list(vals))


def doc_attested_nodes(events_df, node_cids, *, doc_spec):
    """Per document, the distinct in-window condition concept-ids that are DAG
    nodes. Derives doc_id via `doc_spec`, then LEFT-joins a full doc roster
    against the node-filtered attestations so background docs (no DAG-node code)
    survive with an empty `attested_cids` (they get a `[]` frontier downstream).

    Returns [doc_id, person_id, source_cohort, attested_cids: array<bigint>].
    person_id and source_cohort are constant within a doc_id (the cohort arms are
    disjoint by person and doc_id encodes source_cohort), so F.first is
    well-defined."""
    from pyspark.sql import functions as F

    ev = doc_spec.derive_docs(events_df)
    roster = ev.groupBy("doc_id").agg(
        F.first("person_id").alias("person_id"),
        F.first("source_cohort").alias("source_cohort"),
    )
    attested = (
        ev.where(F.col("concept_id").isin(list(node_cids)))
          .groupBy("doc_id")
          .agg(F.collect_set(F.col("concept_id").cast("long")).alias("attested_cids"))
    )
    return (
        roster.join(attested, on="doc_id", how="left")
        .withColumn(
            "attested_cids",
            F.coalesce(F.col("attested_cids"),
                       F.array().cast("array<bigint>")),
        )
    )


def node_patient_counts(attested_df) -> dict[int, int]:
    """Distinct `person_id` per attested node concept-id (patient count, the
    learnability measure the prune uses — NOT patient-year count). Collected to a
    small driver dict (one entry per DAG node)."""
    from pyspark.sql import functions as F

    exploded = attested_df.select(
        "person_id", F.explode("attested_cids").alias("node_cid"),
    ).distinct()
    rows = (
        exploded.groupBy("node_cid")
        .agg(F.countDistinct("person_id").alias("n"))
        .collect()
    )
    return {int(r["node_cid"]): int(r["n"]) for r in rows}
