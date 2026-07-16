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

from dataclasses import dataclass

from charmpheno.omop.condition_dag import _nearest_surviving_ancestors
from spark_vi.models.topic.dag_placement import frontier_from_coded

# Fixed salt for the deterministic train/test split. Hashing person_id with a
# constant salt makes the split reproducible + resume-stable across runs (Spark's
# F.rand() is not), while spreading patients pseudo-uniformly. Mirrors
# cohorts._RANDOM_WINDOW_SALT.
_SPLIT_SALT = 20260716


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


def attach_frontiers(attested_df, before_dag, keep, cid2int, lay):
    """Add a `frontier: array<bigint>` column (ENGINE-id space) to `attested_df`
    by applying `doc_frontier_engine_ids` per row. The DAG/keep/cid2int/lay
    structures are small and picklable; they are captured in the UDF closure and
    broadcast with the task."""
    from pyspark.sql import functions as F
    from pyspark.sql.types import ArrayType, LongType

    def _fr(cids):
        return [int(x) for x in doc_frontier_engine_ids(
            [int(c) for c in (cids or [])], before_dag, keep, cid2int, lay)]

    udf = F.udf(_fr, ArrayType(LongType()))
    return attested_df.withColumn("frontier", udf(F.col("attested_cids")))


def split_train_test(df, *, holdout_frac, split_salt=_SPLIT_SALT):
    """Deterministic salted-hash split on person_id (resume-stable; F.hash, not
    F.rand). A person's docs never straddle the split — the bucket is a pure
    function of person_id + salt — so a patient-keyed holdout stays correct even
    if the doc unit ever becomes many-per-patient. Returns (train_df, test_df)."""
    from pyspark.sql import functions as F

    bucket = F.pmod(F.hash(F.col("person_id"), F.lit(split_salt)), F.lit(10000))
    thresh = int(round(holdout_frac * 10000))
    tagged = df.withColumn("_split_bucket", bucket)
    test = tagged.where(F.col("_split_bucket") < thresh).drop("_split_bucket")
    train = tagged.where(F.col("_split_bucket") >= thresh).drop("_split_bucket")
    return train, test


def strip_test_features(test_df, drop_idxs, *, features_col="features"):
    """Apply the SparseVector leakage strip to `features_col`, removing the vocab
    dims in `drop_idxs` (the DAG-node type codes). Held-out docs only — the caller
    passes only the test split here."""
    from pyspark.sql import functions as F
    from pyspark.ml.linalg import VectorUDT

    drop = {int(i) for i in drop_idxs}

    def _strip(v):
        return strip_features(v, drop)

    udf = F.udf(_strip, VectorUDT())
    return test_df.withColumn(features_col, udf(F.col(features_col)))


@dataclass
class CaseFindingBundle:
    """The assembled case-finding corpus. `train_df`/`test_df` carry
    [person_id, doc_id, features, frontier(engine-ids), source_cohort] — the exact
    shape GatedLDAEstimator(labelCol="frontier").fit consumes and dag_placement's
    evaluate scores. `parent_int`/`int2cid`/`cid2int` bridge engine <-> concept-id;
    `vocab_map` is {concept_id: vocab_idx}; `name_by_id` is {concept_id:
    concept_name} for interpretation (render_profile); `ledger` is the pruning
    receipt (kept/dropped/K_nodes + coarsening)."""
    train_df: object
    test_df: object
    parent_int: dict
    int2cid: dict
    cid2int: dict
    vocab_map: dict
    name_by_id: dict
    ledger: dict


def assemble_from_events(events_df, before_dag, *, doc_spec, min_n,
                         holdout_frac=0.2, split_salt=_SPLIT_SALT,
                         vocab_size, min_df, min_patient_count,
                         n_bg=2, tpn=1) -> CaseFindingBundle:
    """Assemble the case-finding bundle from already-windowed events (with a
    `source_cohort` column) + the pre-prune concept-id DAG. This is the testable
    core: no BigQuery, pure Spark + domain logic. See the module docstring for the
    id-space ordering.

    `cohort_frontiers` for the ledger's coarsening rate is computed from the
    FOREGROUND docs only (background attestations are empty) and collected to the
    driver — foreground scale, run once at prep time."""
    from pyspark.sql import functions as F
    from charmpheno.omop.condition_dag import prune_by_attestation, pruning_ledger
    from charmpheno.omop.topic_prep import to_bow_dataframe
    from spark_vi.models.topic.dag_placement import DagLayout

    attested = doc_attested_nodes(
        events_df, before_dag.nodes(), doc_spec=doc_spec).cache()
    try:
        counts = node_patient_counts(attested)
        after_dag = prune_by_attestation(before_dag, counts, min_n)
        parent_int, int2cid, cid2int = after_dag.to_engine()
        lay = DagLayout(parent_int, n_bg=n_bg, tpn=tpn)
        keep = after_dag.nodes()

        fg_sets = [
            {int(c) for c in r["attested_cids"]}
            for r in attested.where(F.size("attested_cids") > 0)
                             .select("attested_cids").collect()
        ]
        cohort_frontiers = [most_specific_cids(s, before_dag) for s in fg_sets]
        ledger = pruning_ledger(before_dag, after_dag, counts,
                                cohort_frontiers=cohort_frontiers)

        fr = attach_frontiers(attested, before_dag, keep, cid2int, lay)

        bow_df, vocab_map = to_bow_dataframe(
            events_df, doc_spec=doc_spec, vocab_size=vocab_size,
            min_df=min_df, min_patient_count=min_patient_count)

        labeled = (
            bow_df.join(fr.select("doc_id", "frontier", "source_cohort"),
                        on="doc_id", how="left")
            .withColumn("frontier",
                        F.coalesce(F.col("frontier"),
                                   F.array().cast("array<bigint>")))
        )
        train_df, test_df = split_train_test(
            labeled, holdout_frac=holdout_frac, split_salt=split_salt)

        drop_idxs = {vocab_map[c] for c in before_dag.nodes() if c in vocab_map}
        test_df = strip_test_features(test_df, drop_idxs)

        return CaseFindingBundle(
            train_df=train_df, test_df=test_df, parent_int=parent_int,
            int2cid=int2cid, cid2int=cid2int, vocab_map=vocab_map,
            name_by_id=dict(before_dag.names), ledger=ledger)
    finally:
        attested.unpersist()
