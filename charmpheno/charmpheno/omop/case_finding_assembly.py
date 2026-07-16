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

# Synthetic concept-id for the multi-disease forest root. Real OMOP concept_ids
# are positive, so -1 is a safe sentinel that cannot collide with an attested
# code; it becomes engine-id 0 (the never-pruned DAG root) after to_engine().
# Used only when a disease resolves to more than one DAG anchor (the rare6
# forest); single-anchor diseases root the DAG at the anchor itself.
_FOREST_ROOT_CID = -1


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
    concept_name} for interpretation; `ledger` is the pruning receipt
    (kept/dropped/K_nodes + coarsening).

    NOTE (id space): `name_by_id` is keyed on CONCEPT-ids, but
    dag_placement.render_profile keys its `names` on ENGINE-ids (lay.nodes). Remap
    before rendering:  {i: name_by_id[c] for i, c in int2cid.items() if c in name_by_id}.
    Passing name_by_id straight into render_profile silently mislabels every node."""
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
                         n_bg=2, tpn=1, strip_mode="test_only") -> CaseFindingBundle:
    """Assemble the case-finding bundle from already-windowed events (with a
    `source_cohort` column) + the pre-prune concept-id DAG. This is the testable
    core: no BigQuery, pure Spark + domain logic. See the module docstring for the
    id-space ordering.

    Split-first, leakage-free: patients are split into train/test BEFORE
    anything else runs. DAG pruning (`min_n`) and vocabulary fitting both see
    TRAIN patients only; TEST attestations are rolled onto the TRAIN-pruned DAG
    (`attach_frontiers` walks each dropped test node up to its nearest
    surviving ancestor) and TEST documents are bag-of-worded with the FROZEN
    train vocabulary. `ledger["test_coarsening_rate"]`/`["test_fg_docs"]`
    report how much the test foreground was coarsened by a DAG pruned on train
    counts alone. `cohort_frontiers` (train) for the ledger's (train)
    coarsening rate is computed from the FOREGROUND docs only (background
    attestations are empty) and collected to the driver — foreground scale,
    run once at prep time."""
    from pyspark.sql import functions as F
    from charmpheno.omop.condition_dag import prune_by_attestation, pruning_ledger
    from charmpheno.omop.topic_prep import to_bow_dataframe
    from spark_vi.models.topic.dag_placement import DagLayout

    node_cids = before_dag.nodes()
    # 1) split PATIENTS first (events carry person_id); nothing downstream sees
    #    the other side. Same deterministic hash as the doc-level split.
    train_events, test_events = split_train_test(
        events_df, holdout_frac=holdout_frac, split_salt=split_salt)

    # 2) prune the DAG on TRAIN patient counts only.
    train_att = doc_attested_nodes(train_events, node_cids, doc_spec=doc_spec).cache()
    test_att = doc_attested_nodes(test_events, node_cids, doc_spec=doc_spec).cache()
    try:
        counts = node_patient_counts(train_att)
        after_dag = prune_by_attestation(before_dag, counts, min_n)
        parent_int, int2cid, cid2int = after_dag.to_engine()
        lay = DagLayout(parent_int, n_bg=n_bg, tpn=tpn)
        keep = after_dag.nodes()

        # 3) ledger: TRAIN coarsening (as before) + TEST coarsening (new).
        def _fg_ms(att_df):
            return [most_specific_cids({int(c) for c in r["attested_cids"]}, before_dag)
                    for r in att_df.where(F.size("attested_cids") > 0)
                                   .select("attested_cids").collect()]
        train_fg = _fg_ms(train_att)
        ledger = pruning_ledger(before_dag, after_dag, counts,
                                cohort_frontiers=train_fg)
        test_fg = _fg_ms(test_att)
        test_coarsened = sum(1 for ms in test_fg if any(c not in keep for c in ms))
        ledger["test_fg_docs"] = len(test_fg)
        ledger["test_coarsening_rate"] = (
            test_coarsened / len(test_fg) if test_fg else 0.0)

        # 4) frontiers for both sides via the TRAIN DAG (test attestations to
        #    pruned nodes roll up to nearest surviving ancestor).
        train_fr = attach_frontiers(train_att, before_dag, keep, cid2int, lay)
        test_fr = attach_frontiers(test_att, before_dag, keep, cid2int, lay)

        # 5) vocab fit on TRAIN; frozen for TEST.
        train_bow, vocab_map = to_bow_dataframe(
            train_events, doc_spec=doc_spec, vocab_size=vocab_size,
            min_df=min_df, min_patient_count=min_patient_count)
        vocab_list = [None] * len(vocab_map)
        for cid, idx in vocab_map.items():
            vocab_list[idx] = cid
        test_bow, _ = to_bow_dataframe(test_events, doc_spec=doc_spec, vocab=vocab_list)

        def _label(bow, fr):
            return (bow.join(fr.select("doc_id", "frontier", "source_cohort"),
                             on="doc_id", how="left")
                    .withColumn("frontier",
                                F.coalesce(F.col("frontier"),
                                           F.array().cast("array<bigint>"))))
        train_df = _label(train_bow, train_fr)
        test_df = _label(test_bow, test_fr)

        # 6) leakage strip (test_only): drop DAG-node type codes from TEST features.
        drop_idxs = {vocab_map[c] for c in before_dag.nodes() if c in vocab_map}
        test_df = strip_test_features(test_df, drop_idxs)
        if strip_mode == "both":
            train_df = strip_test_features(train_df, drop_idxs)
        elif strip_mode != "test_only":
            raise ValueError(
                f"strip_mode must be 'test_only' or 'both', got {strip_mode!r}")

        return CaseFindingBundle(
            train_df=train_df, test_df=test_df, parent_int=parent_int,
            int2cid=int2cid, cid2int=cid2int, vocab_map=vocab_map,
            name_by_id=dict(before_dag.names), ledger=ledger)
    finally:
        train_att.unpersist(); test_att.unpersist()


def _condition_dag_from_frames(concept_df, ca_df, anchors, root=None):
    """Build the concept-id ConditionDag from `concept` + `concept_ancestor`
    frames.

    `anchors` is one anchor concept-id or a sequence of them. Nodes = standard-
    condition (standard_concept='S', domain_id='Condition') descendants of ANY
    anchor (+ the anchors themselves); edges = min-sep-1 concept_ancestor pairs
    among the nodes (node membership is pushed into the edge scan so only
    ~DAG-size rows collect, not the full concept_ancestor table); names from
    `concept`. Delegates assembly to piece-1 build_condition_dag.

    Single-disease case (`root=None`): exactly one anchor is required and the DAG
    is rooted at it directly (unchanged legacy behavior).

    Multi-disease forest (`root` given, a synthetic sentinel concept-id): the DAG
    is rooted at `root`, and each disease anchor is wired as a depth-1 child of
    the root via an explicit (root, anchor) edge, so the anchors' subtrees hang
    side by side under one connected forest. A descendant whose only min-sep-1
    parent was filtered out (non-standard/wrong-domain) still orphan-attaches to
    the root rather than its disease anchor — surfaced via `dag.orphans`, and
    typically pruned by min_n as a rare subtype; acceptable for v1."""
    from pyspark.sql import functions as F
    from charmpheno.omop.condition_dag import build_condition_dag

    anchor_list = [int(a) for a in ([anchors] if isinstance(anchors, int) else anchors)]
    if root is None:
        if len(anchor_list) != 1:
            raise ValueError(
                "single-anchor DAG requires exactly one anchor; pass a synthetic "
                f"`root` to build a forest over {len(anchor_list)} anchors")
        root = anchor_list[0]
    root = int(root)

    desc = (
        ca_df.where(F.col("ancestor_concept_id").isin(anchor_list))
             .select(F.col("descendant_concept_id").alias("concept_id"))
    )
    std_cond = (
        concept_df.where((F.col("standard_concept") == "S")
                         & (F.col("domain_id") == "Condition"))
                  .select("concept_id", "concept_name")
    )
    node_rows = desc.join(std_cond, on="concept_id", how="inner").collect()
    node_ids = [int(r["concept_id"]) for r in node_rows]
    names = {int(r["concept_id"]): r["concept_name"] for r in node_rows}
    # nodeset for the edge scan = descendants + anchors (+ root added by
    # build_condition_dag). The synthetic root is never in concept_ancestor, so
    # min-sep-1 edges only reconstruct within-subtree structure.
    nodeset = list(set(node_ids) | set(anchor_list))

    edges = [
        (int(r["ancestor_concept_id"]), int(r["descendant_concept_id"]))
        for r in ca_df.where(F.col("min_levels_of_separation") == 1)
                      .where(F.col("ancestor_concept_id").isin(nodeset))
                      .where(F.col("descendant_concept_id").isin(nodeset))
                      .select("ancestor_concept_id", "descendant_concept_id")
                      .collect()
    ]
    # Names for the anchors (they may or may not be standard conditions in
    # `concept`, and are needed regardless as DAG nodes).
    anchor_rows = (concept_df.where(F.col("concept_id").isin(anchor_list))
                   .select("concept_id", "concept_name").collect())
    for r in anchor_rows:
        names[int(r["concept_id"])] = r["concept_name"]

    if root not in anchor_list:
        # Forest: connect the synthetic root to each disease anchor explicitly so
        # the anchors are depth-1 children (not orphans), and give the root a name.
        edges += [(root, a) for a in anchor_list]
        names.setdefault(root, "rare-disease forest root")

    # node_ids passed to build_condition_dag must include the anchors so they are
    # in the nodeset (build_condition_dag adds only the root).
    return build_condition_dag(edges, root, node_ids + anchor_list, names)


def load_condition_dag(spark, *, anchors, cdr, billing, root=None):
    """Read `concept` + `concept_ancestor` from BigQuery and build the condition
    DAG (concept-id space) over one or more `anchors`. BQ wrapper around
    _condition_dag_from_frames; pass a synthetic `root` for a multi-anchor
    forest (see that function)."""
    def _read(table):
        return (spark.read.format("bigquery")
                .option("table", f"{cdr}.{table}")
                .option("parentProject", billing).load())

    concept = _read("concept").select(
        "concept_id", "concept_name", "standard_concept", "domain_id")
    ca = _read("concept_ancestor").select(
        "ancestor_concept_id", "descendant_concept_id", "min_levels_of_separation")
    return _condition_dag_from_frames(concept, ca, anchors, root=root)


def assemble_case_finding_corpus(spark, *, disease="diabetes", cdr, billing,
                                 source_table="condition_era", person_mod,
                                 min_n, holdout_frac=0.2, split_salt=_SPLIT_SALT,
                                 vocab_size, min_df, min_patient_count,
                                 n_bg=2, tpn=1, doc_min_length=0,
                                 prior_obs_days=365, window_days=365,
                                 strip_mode="test_only"):
    """End-to-end BQ assembly: load OMOP (person_mod sample), apply the
    `disease`+background cohort (one `window_days` window per patient), build the
    disease's label DAG, and assemble the bundle.

    `disease` is the single knob. Its DAG anchors come from the cohort registry
    (cohorts.disease_anchors) — the SAME concept ancestors that define the
    foreground arm — so a single-disease name (diabetes, eds) roots the DAG at
    one anchor while a multi-disease name (rare6) builds a forest of the six
    anchors' subtrees under a synthetic root (_FOREST_ROOT_CID). Thin wrapper
    over assemble_from_events; the per-doc unit is PatientCohortDocSpec (doc_id =
    source_cohort:person_id) so each patient's single window is exactly one
    document. Requires a live CDR; unit tests cover assemble_from_events and
    _condition_dag_from_frames directly."""
    from charmpheno.omop import load_omop_bigquery
    from charmpheno.omop.cohorts import apply_population_disease_cohort, disease_anchors
    from charmpheno.omop.doc_spec import PatientCohortDocSpec

    omop = load_omop_bigquery(
        spark=spark, cdr_dataset=cdr, billing_project=billing,
        person_sample_mod=person_mod, source_table=source_table)
    date_col = "condition_era_start_date"
    events = apply_population_disease_cohort(
        omop, disease=disease, window_days=window_days,
        spark=spark, cdr_dataset=cdr, billing_project=billing,
        date_col=date_col, prior_obs_days=prior_obs_days)
    anchors = disease_anchors(disease)
    root = _FOREST_ROOT_CID if len(anchors) > 1 else None
    before_dag = load_condition_dag(
        spark, anchors=anchors, root=root, cdr=cdr, billing=billing)
    doc_spec = PatientCohortDocSpec(min_doc_length=doc_min_length)
    return assemble_from_events(
        events, before_dag, doc_spec=doc_spec, min_n=min_n,
        holdout_frac=holdout_frac, split_salt=split_salt, vocab_size=vocab_size,
        min_df=min_df, min_patient_count=min_patient_count, n_bg=n_bg, tpn=tpn,
        strip_mode=strip_mode)
