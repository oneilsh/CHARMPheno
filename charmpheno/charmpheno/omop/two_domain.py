"""Two-domain (MixEHR-style) corpus assembly for the multi-domain gated model.

A document carries TWO bag-of-words feature columns -- one per domain (e.g.
conditions and drugs) -- over two INDEPENDENT vocabularies. Per-modality
generative separation (each domain gets its own vocabulary and its own
observation model downstream) follows MixEHR (Li, Nair, Lu et al. 2020,
Nat. Commun. 11:2536): distinct EHR data types are modeled as distinct
"modalities" sharing patient-level topic structure while keeping separate
per-modality vocabularies. Here the shared structure is the gated topic
model's per-document theta; the gate itself is condition-only and orthogonal
to domain (arc design) -- this module only assembles the two feature
columns, it has no opinion about the gate.

This module is a thin two-domain layer over the domain-agnostic
`topic_prep.to_bow_dataframe`: it fits each domain's vocabulary and BOW
separately and joins the two feature columns per document. It does NOT
reimplement BOW/vocab fitting, and it does NOT touch the single-domain
`case_finding_assembly` path.
"""
from __future__ import annotations

from dataclasses import dataclass

from pyspark.ml.linalg import SparseVector, VectorUDT
from pyspark.sql import DataFrame
from pyspark.sql import functions as F


@dataclass(frozen=True)
class DomainVocabSpec:
    """Per-domain vocabulary controls. Independent per domain because the two
    domains have very different natural sizes (SP3b design). `vocab` pins a
    pre-built vocabulary (concept-ids in assignment order) for eval/reproduce;
    None fits from the data."""
    vocab_size: int | None
    min_df: int | float = 1
    min_patient_count: int = 1
    vocab: list[int] | None = None


def _empty_vec_udf(size: int):
    return F.udf(lambda: SparseVector(size, [], []), VectorUDT())


def two_domain_bow(events_a: DataFrame, events_b: DataFrame, *, doc_spec,
                    vocab_a: DomainVocabSpec, vocab_b: DomainVocabSpec):
    """Two aligned per-domain BOW columns joined per doc. Returns
    (df[doc_id, person_id, features_a, features_b], vocab_map_a, vocab_map_b).

    A doc present in only one domain gets an EMPTY vector (of that domain's
    vocab size) on the absent side, never a dropped row -- so every doc
    carries both columns and each per-domain vector size is CONSTANT across
    the corpus (SP3a's shim derives domainBounds from the first row and
    validates every row against it; a variable size would raise).
    """
    from charmpheno.omop.topic_prep import to_bow_dataframe

    bow_a, vm_a = to_bow_dataframe(
        events_a, doc_spec=doc_spec, token_col="concept_id",
        vocab_size=vocab_a.vocab_size, min_df=vocab_a.min_df,
        min_patient_count=vocab_a.min_patient_count, vocab=vocab_a.vocab)
    bow_b, vm_b = to_bow_dataframe(
        events_b, doc_spec=doc_spec, token_col="concept_id",
        vocab_size=vocab_b.vocab_size, min_df=vocab_b.min_df,
        min_patient_count=vocab_b.min_patient_count, vocab=vocab_b.vocab)

    va, vb = len(vm_a), len(vm_b)

    a = bow_a.select("doc_id", "person_id", F.col("features").alias("features_a"))
    b = bow_b.select("doc_id", F.col("features").alias("features_b"),
                      F.col("person_id").alias("person_id_b"))

    joined = a.join(b, on="doc_id", how="full_outer")

    # Coalesce person_id across the outer join (present on whichever side
    # actually has the doc); fill absent per-domain vectors with an empty
    # SparseVector of that domain's fixed vocab size so every row's per-domain
    # vector size is constant across the corpus.
    joined = (
        joined
        .withColumn("person_id", F.coalesce(F.col("person_id"), F.col("person_id_b")))
        .drop("person_id_b")
        .withColumn("features_a", F.coalesce(F.col("features_a"), _empty_vec_udf(va)()))
        .withColumn("features_b", F.coalesce(F.col("features_b"), _empty_vec_udf(vb)()))
    )

    return (
        joined.select("doc_id", "person_id", "features_a", "features_b"),
        vm_a,
        vm_b,
    )


@dataclass
class TwoDomainBundle:
    """The assembled two-domain case-finding corpus. Mirrors
    `CaseFindingBundle` (case_finding_assembly.py) but the docs carry TWO
    per-domain feature columns over TWO independent vocabularies instead of one:
    `train_df`/`test_df` are [person_id, doc_id, features_a, features_b, frontier
    (engine-ids), source_cohort]. `vocab_map_a`/`vocab_map_b` are the two
    {concept_id: vocab_idx} maps (domain A = conditions, domain B = drugs).

    The frontier is CONDITION-ONLY (the gate is condition-only, gate ⟂ domain --
    arc design), so the bridge/receipt fields are identical to the single-domain
    bundle: `parent_int`/`int2cid`/`cid2int` bridge engine <-> concept-id (Task 4's
    multi-domain fit consumes `parent_int`); `name_by_id` is {concept_id:
    concept_name}; `ledger` is the pruning receipt (kept/dropped/K_nodes +
    coarsening, train + test).

    NOTE (id space): as with CaseFindingBundle, `name_by_id` is keyed on
    CONCEPT-ids while dag_placement.render_profile keys `names` on ENGINE-ids;
    remap via {i: name_by_id[c] for i, c in int2cid.items() if c in name_by_id}
    before rendering."""
    train_df: object
    test_df: object
    parent_int: dict
    int2cid: dict
    cid2int: dict
    vocab_map_a: dict
    vocab_map_b: dict
    name_by_id: dict
    ledger: dict


def _frozen_vocab_spec(spec: DomainVocabSpec, vocab_map: dict) -> DomainVocabSpec:
    """A DomainVocabSpec that pins `vocab_map` as a frozen concept-id list (index
    order), so a TEST BOW is bag-of-worded against the TRAIN-fit vocabulary. The
    fit knobs are reset to defaults because `to_bow_dataframe` rejects a frozen
    `vocab` combined with non-default vocab_size/min_df/min_patient_count."""
    from dataclasses import replace

    vocab_list = [None] * len(vocab_map)
    for cid, idx in vocab_map.items():
        vocab_list[idx] = cid
    return replace(spec, vocab_size=None, min_df=1, min_patient_count=1,
                   vocab=vocab_list)


def assemble_two_domain_from_events(cond_events, drug_events, before_dag, *,
                                    doc_spec, min_n, vocab_a: DomainVocabSpec,
                                    vocab_b: DomainVocabSpec, holdout_frac=0.2,
                                    split_salt=None, n_bg=2, tpn=1,
                                    strip_mode="test_only",
                                    label_events=None) -> TwoDomainBundle:
    """Assemble the TWO-domain case-finding bundle from already-windowed events.

    A thin two-domain layer over the single-domain `assemble_from_events`
    orchestration (case_finding_assembly.py): it reuses that module's split,
    frontier, prune, ledger, and strip helpers verbatim, and swaps only the
    single BOW for the two-column `two_domain_bow`, applying the leakage strip
    PER DOMAIN. It does NOT reimplement the split/frontier/prune/strip and does
    NOT touch the single-domain path.

    Domain A is conditions (`cond_events`), domain B is drugs (`drug_events`).
    The frontier/label side is CONDITION-ONLY (the gate is condition-only, gate ⟂
    domain): the attestation frame is `cond_events`, or `label_events` when given
    (lookback mode: the frontier is read from a separate forward window while
    features are the pre-index history). A drug never enters a frontier.

    Split-first, leakage-free (mirrors the single-domain assembler): patients are
    split into train/test BEFORE anything else. The SAME salted-hash
    `split_train_test(holdout_frac, split_salt)` keyed on person_id is applied to
    the condition frame, the drug frame, AND the label frame, so a person's
    condition rows, drug rows, and label rows all land on the same side. The DAG
    is pruned on TRAIN condition-node counts; each domain's vocabulary is fit on
    TRAIN and frozen for TEST.

    Per-domain leakage strip: node-marker concept-ids ARE the DAG nodes, so they
    live in whichever domain's vocabulary contains them. The strip is applied to
    each domain independently -- for each domain d, the node-marker concept-ids
    are mapped through that domain's vocab_map to its vocab indices and stripped
    from features_d. Condition node markers only exist in the CONDITION vocab so
    in practice features_b (drug) is untouched, but the logic is symmetric, not
    hardcoded to domain A. `strip_mode="test_only"` (default) strips TEST only;
    `"both"` also strips TRAIN.
    """
    from pyspark.sql import functions as F
    from charmpheno.omop.condition_dag import prune_by_attestation, pruning_ledger
    from charmpheno.omop.case_finding_assembly import (
        _SPLIT_SALT, split_train_test, doc_attested_nodes, node_patient_counts,
        attach_frontiers, most_specific_cids, strip_test_features)
    from spark_vi.models.topic.dag_placement import DagLayout

    if split_salt is None:
        split_salt = _SPLIT_SALT
    if strip_mode not in ("test_only", "both"):
        raise ValueError(
            f"strip_mode must be 'test_only' or 'both', got {strip_mode!r}")

    node_cids = before_dag.nodes()

    # 1) split PATIENTS first, keyed on person_id with the SAME salted hash, so a
    #    person's condition, drug, and label rows land on the same side. The
    #    label frame == cond_events in forward mode (label_events None).
    train_cond, test_cond = split_train_test(
        cond_events, holdout_frac=holdout_frac, split_salt=split_salt)
    train_drug, test_drug = split_train_test(
        drug_events, holdout_frac=holdout_frac, split_salt=split_salt)
    if label_events is None:
        train_lab, test_lab = train_cond, test_cond
    else:
        train_lab, test_lab = split_train_test(
            label_events, holdout_frac=holdout_frac, split_salt=split_salt)

    # 2) condition-only frontier: prune the DAG on TRAIN attestation counts.
    train_att = doc_attested_nodes(train_lab, node_cids, doc_spec=doc_spec).cache()
    test_att = doc_attested_nodes(test_lab, node_cids, doc_spec=doc_spec).cache()
    try:
        counts = node_patient_counts(train_att)
        after_dag = prune_by_attestation(before_dag, counts, min_n)
        parent_int, int2cid, cid2int = after_dag.to_engine()
        lay = DagLayout(parent_int, n_bg=n_bg, tpn=tpn)
        keep = after_dag.nodes()

        # 3) ledger: TRAIN + TEST coarsening (same accounting as single-domain).
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

        # 4) frontiers for both sides via the TRAIN-pruned DAG.
        train_fr = attach_frontiers(train_att, before_dag, keep, cid2int, lay)
        test_fr = attach_frontiers(test_att, before_dag, keep, cid2int, lay)

        # 5) two-domain BOW: fit each domain's vocab on TRAIN, freeze for TEST.
        train_bow, vm_a, vm_b = two_domain_bow(
            train_cond, train_drug, doc_spec=doc_spec,
            vocab_a=vocab_a, vocab_b=vocab_b)
        test_bow, _, _ = two_domain_bow(
            test_cond, test_drug, doc_spec=doc_spec,
            vocab_a=_frozen_vocab_spec(vocab_a, vm_a),
            vocab_b=_frozen_vocab_spec(vocab_b, vm_b))

        def _label(bow, fr):
            return (bow.join(fr.select("doc_id", "frontier", "source_cohort"),
                             on="doc_id", how="left")
                    .withColumn("frontier",
                                F.coalesce(F.col("frontier"),
                                           F.array().cast("array<bigint>"))))
        train_df = _label(train_bow, train_fr)
        test_df = _label(test_bow, test_fr)

        # 6) per-domain leakage strip: for each domain, map the DAG node-marker
        #    concept-ids to THAT domain's vocab indices and strip its column. A
        #    condition marker only exists in vm_a so features_b is untouched, but
        #    the strip is applied symmetrically (not hardcoded to A).
        per_domain = [("features_a", vm_a), ("features_b", vm_b)]

        def _strip(df):
            for col, vm in per_domain:
                drop_idxs = {vm[c] for c in node_cids if c in vm}
                df = strip_test_features(df, drop_idxs, features_col=col)
            return df

        test_df = _strip(test_df)
        if strip_mode == "both":
            train_df = _strip(train_df)

        return TwoDomainBundle(
            train_df=train_df, test_df=test_df, parent_int=parent_int,
            int2cid=int2cid, cid2int=cid2int, vocab_map_a=vm_a, vocab_map_b=vm_b,
            name_by_id=dict(before_dag.names), ledger=ledger)
    finally:
        train_att.unpersist(); test_att.unpersist()
