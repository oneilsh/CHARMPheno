# charmpheno/charmpheno/omop/case_finding_assembly.py
"""Assemble the labeled case-finding corpus from OMOP.

One document per patient carrying its features and the label DAG nodes it ATTESTS
(the clinical truth), a held-out split, and the leakage strip applied. The bundle
holds the label DAG itself (`dag`, `{node: [parents]}` in the user's node ids) and
per document the attested node ids (`labels`), which is exactly what the DAG-gated
estimator and the readout heads take (spec 2026-10-30 R1/R2/R6): no frontier, no
dense label vectors, no engine ids, no threshold. Which nodes get a topic block
(support, prune, landing, reduction, splice) is learned by the estimator from the
TRAINING documents; the dense per-node targets are built from the fitted node set
by the readout (`spark_vi.readout.labels`).

Composes the SNOMED DAG builder (condition_dag.py), the population+disease cohort
machinery (cohorts.py) and `to_bow_dataframe`. Two id spaces:

  node id     the label DAG's ids: OMOP concept ids (int) on the SNOMED path, Mondo
              curies (str) on the native Mondo path (`charmpheno.mondo.native_dag`).
              `labels` is `array<bigint>` / `array<string>` accordingly.
  vocab-index [0,V) from vocab_map {concept_id: idx}; features SparseVector,
              leakage strip.

See docs/superpowers/specs/2026-07-15-case-finding-assembly-design.md and
docs/architecture/LABEL_RULES.md.
"""
from __future__ import annotations

from dataclasses import dataclass

# Fixed salt for the deterministic train/test split. Hashing person_id with a
# constant salt makes the split reproducible + resume-stable across runs (Spark's
# F.rand() is not), while spreading patients pseudo-uniformly. Mirrors
# cohorts._RANDOM_WINDOW_SALT.
_SPLIT_SALT = 20260716

# Synthetic concept-id for the multi-disease forest root. Real OMOP concept_ids
# are positive, so -1 is a safe sentinel that cannot collide with an attested
# code. Used only when a disease resolves to more than one DAG anchor (the rare6
# forest); it is dropped from the handed-over DAG (`ConditionDag.user_dag`), whose
# parentless anchors hang from the estimator's own synthetic root.
_FOREST_ROOT_CID = -1

# Lookback mode requires at least one year of pre-index observation so the
# feature window [index - lookback_days, index) is anchored on real recorded
# history (the "≥1yr of history before index" requirement; see the lookback
# design doc's "Cohort gate" section). This floor is a fixed design invariant
# of the pre-diagnosis framing, not the forward-mode `prior_obs_days` knob:
# the up-to-5yr feature window (lookback_days=1825) is deliberately clipped to
# whatever history exists beyond this 1-year floor, so the gate stays at 365
# regardless of lookback_days. Kept intrinsic to the lookback path (rather than
# threaded from prior_obs_days) so a lookback experiment cannot silently
# disable it by inheriting prior_obs_days=0 from a forward config.
_LOOKBACK_PRIOR_OBS_DAYS = 365

# The bundle's label columns (spec 2026-10-30 "What changes outside spark-vi"):
# LABELS_COL is the label-year attestation (every bundle); GATE_LABELS_COL the
# all-ever attestation a gate may be fit on instead (exp 0146, R8), added to the
# TRAIN split by the fit driver only when asked for.
LABELS_COL = "labels"
GATE_LABELS_COL = "gate_labels"


def dag_edges(dag) -> list:
    """`{node: [parents]}` -> a JSON-ready edge list `[[child, parent | None], ...]`
    (a top-level node is `[node, None]`), sorted. JSON keeps an int and a str apart,
    so the ids round-trip with their type (`dag_from_edges`)."""
    out = []
    for n in sorted(dag):
        ps = sorted(dag[n])
        if not ps:
            out.append([n, None])
        out.extend([n, p] for p in ps)
    return out


def dag_from_edges(edges) -> dict:
    """The inverse of `dag_edges`: `{node: [parents]}`, every node a key."""
    out: dict = {}
    for c, p in edges:
        out.setdefault(c, [])
        if p is not None:
            out[c].append(p)
            out.setdefault(p, [])
    return {n: sorted(ps) for n, ps in out.items()}


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


def doc_attested_nodes(events_df, node_ids, *, doc_spec):
    """Per document, the distinct in-window condition concept-ids that are DAG
    nodes (the SNOMED path's exact-node attestation). Derives doc_id via
    `doc_spec`, then LEFT-joins a full doc roster against the node-filtered
    attestations so background docs (no DAG-node code) survive with an empty
    `labels`.

    Returns [doc_id, person_id, source_cohort, labels: array<bigint>] (sorted).
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
        ev.where(F.col("concept_id").isin([int(n) for n in node_ids]))
          .groupBy("doc_id")
          .agg(F.array_sort(F.collect_set(F.col("concept_id").cast("long")))
               .alias(LABELS_COL))
    )
    return (
        roster.join(attested, on="doc_id", how="left")
        .withColumn(LABELS_COL, F.coalesce(F.col(LABELS_COL),
                                           F.array().cast("array<bigint>")))
    )


def attach_label_column(bow, attested, *, col=LABELS_COL, extra=("source_cohort",)):
    """`bow` LEFT-joined with `attested`'s `col` (and `extra` columns) on doc_id; a
    document with no attestation row gets an EMPTY array of the column's own
    element type (background), never a null."""
    from pyspark.sql import functions as F
    dt = attested.schema[col].dataType.simpleString()
    return (bow.join(attested.select("doc_id", col, *extra), on="doc_id", how="left")
            .withColumn(col, F.coalesce(F.col(col), F.array().cast(dt))))


def label_ledger(train_att, test_att, n_nodes, *, col=LABELS_COL) -> dict:
    """The bundle's receipt: the DAG size and the foreground (non-empty `labels`)
    document counts per split. Counts of documents and nodes only. The node-set
    receipt (kept / pruned / spliced) is the fitted model's."""
    from pyspark.sql import functions as F
    return {"n_dag_nodes": int(n_nodes),
            "train_fg_docs": int(train_att.where(F.size(col) > 0).count()),
            "test_fg_docs": int(test_att.where(F.size(col) > 0).count())}


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
    [person_id, doc_id, features, labels, source_cohort]: `labels` the attested node
    ids (the estimator's `labelsCol`). `dag` is the label DAG `{node: [parents]}` in
    those ids (the estimator's `setDag`); `vocab_map` is {concept_id: vocab_idx};
    `name_by_id` is {node id: name} for interpretation; `ledger` the receipt
    (`label_ledger`)."""
    train_df: object
    test_df: object
    dag: dict
    vocab_map: dict
    name_by_id: dict
    ledger: dict


def assemble_from_events(events_df, label_dag, *, names=None, doc_spec,
                         holdout_frac=0.2, split_salt=_SPLIT_SALT,
                         vocab_size, min_df, min_patient_count,
                         strip_mode="test_only",
                         label_events=None) -> CaseFindingBundle:
    """Assemble the case-finding bundle from already-windowed events (with a
    `source_cohort` column) + the label DAG (`{concept_id: [parents]}`, e.g.
    `ConditionDag.user_dag()`). This is the testable core: no BigQuery, pure Spark +
    domain logic.

    Split-first, leakage-free: patients are split into train/test BEFORE anything
    else runs. The vocabulary is fit on TRAIN patients only and frozen for TEST.
    Each document's `labels` are the DAG nodes its label-frame codes attest
    (exact-node attestation, `doc_attested_nodes`); no node is dropped here: the
    estimator learns the node set from the training documents.

    `label_events` (optional): when given, `labels` come from `label_events` instead
    of `events_df`, while features still come from `events_df`. This is lookback
    mode: a patient's features are their pre-index history and their labels are
    read from a separate forward window. The patient split is applied to BOTH frames
    with the same `split_train_test(holdout_frac, split_salt)` so a person's feature
    and label rows land on the same side.

    Leakage strip (`strip_mode`): the vocabulary dims of the DAG's node ids are
    removed from TEST features (`"both"`: train too), so a held-out patient is not
    scored on the code that defines its label."""
    from charmpheno.omop.topic_prep import to_bow_dataframe

    if strip_mode not in ("test_only", "both"):
        raise ValueError(
            f"strip_mode must be 'test_only' or 'both', got {strip_mode!r}")
    node_ids = set(label_dag)
    # 1) split PATIENTS first (events carry person_id); nothing downstream sees
    #    the other side.
    train_events, test_events = split_train_test(
        events_df, holdout_frac=holdout_frac, split_salt=split_salt)
    if label_events is None:
        train_lab, test_lab = train_events, test_events
    else:
        train_lab, test_lab = split_train_test(
            label_events, holdout_frac=holdout_frac, split_salt=split_salt)

    # 2) attestation per document (exact node), both splits.
    train_att = doc_attested_nodes(train_lab, node_ids, doc_spec=doc_spec).cache()
    test_att = doc_attested_nodes(test_lab, node_ids, doc_spec=doc_spec).cache()
    try:
        ledger = label_ledger(train_att, test_att, len(node_ids))

        # 3) vocab fit on TRAIN; frozen for TEST.
        train_bow, vocab_map = to_bow_dataframe(
            train_events, doc_spec=doc_spec, vocab_size=vocab_size,
            min_df=min_df, min_patient_count=min_patient_count)
        vocab_list = [None] * len(vocab_map)
        for cid, idx in vocab_map.items():
            vocab_list[idx] = cid
        test_bow, _ = to_bow_dataframe(test_events, doc_spec=doc_spec, vocab=vocab_list)

        train_df = attach_label_column(train_bow, train_att)
        test_df = attach_label_column(test_bow, test_att)

        # 4) leakage strip: drop DAG-node type codes from TEST features.
        drop_idxs = {vocab_map[c] for c in node_ids if c in vocab_map}
        test_df = strip_test_features(test_df, drop_idxs)
        if strip_mode == "both":
            train_df = strip_test_features(train_df, drop_idxs)

        return CaseFindingBundle(
            train_df=train_df, test_df=test_df,
            dag={n: list(ps) for n, ps in label_dag.items()},
            vocab_map=vocab_map,
            name_by_id={n: v for n, v in (names or {}).items() if n in node_ids},
            ledger=ledger)
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
                                 holdout_frac=0.2, split_salt=_SPLIT_SALT,
                                 vocab_size, min_df, min_patient_count,
                                 doc_min_length=0,
                                 prior_obs_days=365, window_days=365,
                                 strip_mode="test_only", window_mode="forward",
                                 lookback_days=365, label_window_days=365):
    """End-to-end BQ assembly: load OMOP (person_mod sample), apply the
    `disease`+background cohort, build the disease's label DAG, and assemble the
    bundle.

    `window_mode="forward"` (default): one `window_days` window per patient via
    `apply_population_disease_cohort`; labels and features both come from that
    single window (`assemble_from_events`'s `label_events=None` path).

    `window_mode="lookback"`: an index date per patient
    (`cohorts.case_finding_index_table`) splits raw events into a pre-index
    feature frame (`lookback_days` back) and a forward label frame
    (`label_window_days` forward) via `cohorts.lookback_feature_label_events`.

    `disease` is the single knob. Its DAG anchors come from the cohort registry
    (cohorts.disease_anchors) — the SAME concept ancestors that define the
    foreground arm — so a single-disease name (diabetes, eds) roots the DAG at
    one anchor (a top-level node of the handed-over DAG) while a multi-disease name
    (rare6) builds a forest of the six anchors' subtrees (the synthetic forest root
    `_FOREST_ROOT_CID` is dropped: the anchors are top-level). Node ids are OMOP
    concept ids (int). Requires a live CDR; unit tests cover assemble_from_events
    and _condition_dag_from_frames directly."""
    from charmpheno.omop import load_omop_bigquery
    from charmpheno.omop.cohorts import (
        apply_population_disease_cohort,
        case_finding_index_table, lookback_feature_label_events,
    )
    from charmpheno.omop.doc_spec import PatientCohortDocSpec

    omop = load_omop_bigquery(
        spark=spark, cdr_dataset=cdr, billing_project=billing,
        person_sample_mod=person_mod, source_table=source_table)
    date_col = "condition_era_start_date"

    if window_mode == "lookback":
        # The lookback prior-observation gate is the fixed ≥1yr floor
        # (_LOOKBACK_PRIOR_OBS_DAYS), NOT the forward-mode prior_obs_days knob.
        index_df = case_finding_index_table(
            omop, disease=disease, spark=spark, cdr_dataset=cdr,
            billing_project=billing, date_col=date_col,
            prior_obs_days=_LOOKBACK_PRIOR_OBS_DAYS,
            label_window_days=label_window_days)
        feature_events, label_events = lookback_feature_label_events(
            omop, index_df, date_col=date_col,
            lookback_days=lookback_days, label_window_days=label_window_days)
        events, label_arg = feature_events, label_events
    elif window_mode == "forward":
        events = apply_population_disease_cohort(
            omop, disease=disease, window_days=window_days,
            spark=spark, cdr_dataset=cdr, billing_project=billing,
            date_col=date_col, prior_obs_days=prior_obs_days)
        label_arg = None
    else:
        raise ValueError(
            f"window_mode must be 'forward' or 'lookback', got {window_mode!r}")

    dag, names = snomed_label_dag(spark, disease=disease, cdr=cdr, billing=billing)
    doc_spec = PatientCohortDocSpec(min_doc_length=doc_min_length)
    return assemble_from_events(
        events, dag, names=names, doc_spec=doc_spec,
        holdout_frac=holdout_frac, split_salt=split_salt, vocab_size=vocab_size,
        min_df=min_df, min_patient_count=min_patient_count,
        strip_mode=strip_mode, label_events=label_arg)


def snomed_label_dag(spark, *, disease, cdr, billing):
    """`(dag, names)`: the disease's SNOMED label DAG in the estimator's user-id
    contract (`ConditionDag.user_dag`; int concept ids, the synthetic forest root
    dropped) and `{concept_id: name}`."""
    from charmpheno.omop.cohorts import disease_anchors
    anchors = disease_anchors(disease)
    root = _FOREST_ROOT_CID if len(anchors) > 1 else None
    cdag = load_condition_dag(spark, anchors=anchors, root=root, cdr=cdr,
                              billing=billing)
    dag = cdag.user_dag(synthetic_root=root)
    return dag, {n: cdag.names.get(n, str(n)) for n in dag}
