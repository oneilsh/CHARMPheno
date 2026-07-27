"""Multi-domain (MixEHR-style) corpus assembly for the multi-domain gated model.

A document carries N bag-of-words feature columns -- one per domain (e.g.
conditions, drugs, observations) -- over N INDEPENDENT vocabularies. Per-modality
generative separation (each domain gets its own vocabulary and its own
observation model downstream) follows MixEHR (Li, Nair, Lu et al. 2020,
Nat. Commun. 11:2536): distinct EHR data types are modeled as distinct
"modalities" sharing patient-level topic structure while keeping separate
per-modality vocabularies. Here the shared structure is the gated topic
model's per-document theta; the gate itself is condition-only and orthogonal
to domain (arc design) -- this module only assembles the N feature
columns, it has no opinion about the gate.

This module is a thin N-domain layer over the domain-agnostic
`topic_prep.to_bow_dataframe`: it fits each domain's vocabulary and BOW
separately and joins the N feature columns per document. It does NOT
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
    """Per-domain vocabulary controls. Independent per domain because domains have
    very different natural sizes (SP3b/SP3c). `vocab` pins a pre-built vocabulary
    (concept-ids in assignment order) for eval/reproduce; None fits from data."""
    vocab_size: int | None
    min_df: int | float = 1
    min_patient_count: int = 1
    vocab: list[int] | None = None


def _empty_vec_udf(size: int):
    return F.udf(lambda: SparseVector(size, [], []), VectorUDT())


def multidomain_bow(domain_events, vocab_specs, *, doc_spec):
    """N aligned per-domain BOW columns joined per doc. Returns
    (df[doc_id, person_id, features_0 .. features_{N-1}], vocab_maps).

    domain_events[i] is bag-of-worded against vocab_specs[i]; domain 0 is
    conditions by convention. A doc present in only some domains gets an EMPTY
    vector (of the absent domain's vocab size) on each absent side -- never a
    dropped row -- so every doc carries all N columns and each per-domain vector
    size is CONSTANT across the corpus (SP3a's shim derives domainBounds from the
    first row and validates every row against it).
    """
    from charmpheno.omop.topic_prep import to_bow_dataframe

    if len(domain_events) != len(vocab_specs):
        raise ValueError(
            f"domain_events ({len(domain_events)}) and vocab_specs "
            f"({len(vocab_specs)}) must have the same length")

    bows, vms = [], []
    for ev, spec in zip(domain_events, vocab_specs):
        bow, vm = to_bow_dataframe(
            ev, doc_spec=doc_spec, token_col="concept_id",
            vocab_size=spec.vocab_size, min_df=spec.min_df,
            min_patient_count=spec.min_patient_count, vocab=spec.vocab)
        bows.append(bow)
        vms.append(vm)

    joined = bows[0].select(
        "doc_id", "person_id", F.col("features").alias("features_0"))
    for i in range(1, len(bows)):
        side = bows[i].select(
            "doc_id",
            F.col("features").alias(f"features_{i}"),
            F.col("person_id").alias(f"person_id_{i}"))
        joined = (joined.join(side, on="doc_id", how="full_outer")
                  .withColumn("person_id",
                              F.coalesce(F.col("person_id"), F.col(f"person_id_{i}")))
                  .drop(f"person_id_{i}"))

    for i, vm in enumerate(vms):
        col = f"features_{i}"
        joined = joined.withColumn(
            col, F.coalesce(F.col(col), _empty_vec_udf(len(vm))()))

    feat_cols = [f"features_{i}" for i in range(len(vms))]
    return joined.select("doc_id", "person_id", *feat_cols), vms


@dataclass
class MultiDomainBundle:
    """The assembled N-domain case-finding corpus. `train_df`/`test_df` carry
    feature columns features_0 .. features_{N-1} (domain 0 = conditions) plus
    `frontier` (engine-ids) and `source_cohort`. `vocab_maps` is the list of
    per-domain {concept_id: vocab_idx} maps in domain order.

    The frontier is CONDITION-ONLY (gate ⟂ domain): the bridge/receipt fields are
    identical to the single-domain bundle -- `parent_int`/`int2cid`/`cid2int`
    bridge engine <-> concept-id; `name_by_id` is {concept_id: concept_name};
    `ledger` is the pruning receipt. Clinical domain names live in the DRIVER, not
    here -- this bundle is index-based and domain-agnostic."""
    train_df: object
    test_df: object
    parent_int: dict
    int2cid: dict
    cid2int: dict
    vocab_maps: list
    name_by_id: dict
    ledger: dict


def lookback_feature_frames(domain_raws, index_df, date_cols, *,
                            lookback_days, label_window_days):
    """Split every domain's raw events against ONE shared index into pre-index
    FEATURE frames, and return the condition (domain 0) forward-window LABEL frame.

    domain_raws[0] MUST be conditions (the label/gate source). For each domain i,
    `cohorts.lookback_feature_label_events` splits domain_raws[i] on date_cols[i]:
    the pre-index [index - lookback_days, index) window is kept as that domain's
    feature frame. Only domain 0's forward [index, index + label_window_days)
    window is kept as the label frame -- the gate is condition-only, so a
    drug/observation event never defines a frontier. index_df carries
    source_cohort, which the join propagates onto every feature/label frame.
    """
    from charmpheno.omop.cohorts import lookback_feature_label_events

    if len(domain_raws) != len(date_cols):
        raise ValueError(
            f"domain_raws ({len(domain_raws)}) and date_cols ({len(date_cols)}) "
            f"must have the same length")

    feature_frames, cond_label = [], None
    for i, (raw, dc) in enumerate(zip(domain_raws, date_cols)):
        feat, lab = lookback_feature_label_events(
            raw, index_df, date_col=dc,
            lookback_days=lookback_days, label_window_days=label_window_days)
        feature_frames.append(feat)
        if i == 0:
            cond_label = lab
    return feature_frames, cond_label


def _frozen_vocab_spec(spec: DomainVocabSpec, vocab_map: dict) -> DomainVocabSpec:
    """A DomainVocabSpec that pins `vocab_map` as a frozen concept-id list (index
    order), so a TEST BOW is bag-of-worded against the TRAIN-fit vocabulary. The
    fit knobs reset to defaults because to_bow_dataframe rejects a frozen `vocab`
    combined with non-default vocab_size/min_df/min_patient_count."""
    from dataclasses import replace

    vocab_list = [None] * len(vocab_map)
    for cid, idx in vocab_map.items():
        vocab_list[idx] = cid
    return replace(spec, vocab_size=None, min_df=1, min_patient_count=1,
                   vocab=vocab_list)


def assemble_multidomain_from_events(cond_events, extra_events, before_dag, *,
                                     doc_spec, min_n, vocab_specs,
                                     holdout_frac=0.2, split_salt=None, n_bg=2,
                                     tpn=1, strip_mode="test_only",
                                     label_events=None) -> MultiDomainBundle:
    """Assemble the N-domain case-finding bundle from already-windowed events.

    A thin N-domain layer over the single-domain `assemble_from_events`
    orchestration: it reuses that module's split, frontier, prune, ledger, and
    strip helpers verbatim, swaps the single BOW for the N-column `multidomain_bow`,
    and applies the leakage strip PER DOMAIN over all N vocabularies.

    Domain 0 is conditions (`cond_events`); domains 1..N-1 are `extra_events` in
    order. `vocab_specs` has one spec per domain (len == 1 + len(extra_events)).
    The frontier/label side is CONDITION-ONLY: the attestation frame is
    `cond_events`, or `label_events` when given (lookback mode). A non-condition
    event never enters a frontier.

    Split-first, leakage-free: the SAME salted split (keyed on person_id) is
    applied to every domain frame AND the label frame, so a person's rows across
    all domains and labels land on the same side. The DAG is pruned on TRAIN
    condition-node counts; each domain's vocabulary is fit on TRAIN, frozen for
    TEST. Per-domain strip: node-marker concept-ids are mapped through EACH
    domain's vocab_map and stripped from that column (defensive; a condition marker
    is expected only in vocab 0, but the strip is symmetric across domains).
    `strip_mode="test_only"` (default) strips TEST only; `"both"` also strips TRAIN.
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

    domain_events = [cond_events, *extra_events]
    if len(vocab_specs) != len(domain_events):
        raise ValueError(
            f"vocab_specs ({len(vocab_specs)}) must equal number of domains "
            f"({len(domain_events)} = 1 condition + {len(extra_events)} extra)")

    node_cids = before_dag.nodes()

    # 1) split PATIENTS first, keyed on person_id with the SAME salted hash, so a
    #    person's rows across all domains + labels land on the same side.
    train_doms, test_doms = [], []
    for ev in domain_events:
        tr, te = split_train_test(ev, holdout_frac=holdout_frac, split_salt=split_salt)
        train_doms.append(tr)
        test_doms.append(te)
    if label_events is None:
        train_lab, test_lab = train_doms[0], test_doms[0]
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

        # 3) ledger: TRAIN + TEST coarsening.
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

        # 5) N-domain BOW: fit each domain's vocab on TRAIN, freeze for TEST.
        train_bow, vms = multidomain_bow(train_doms, vocab_specs, doc_spec=doc_spec)
        test_bow, _ = multidomain_bow(
            test_doms, [_frozen_vocab_spec(s, vm) for s, vm in zip(vocab_specs, vms)],
            doc_spec=doc_spec)

        def _label(bow, fr):
            return (bow.join(fr.select("doc_id", "frontier", "source_cohort"),
                             on="doc_id", how="left")
                    .withColumn("frontier",
                                F.coalesce(F.col("frontier"),
                                           F.array().cast("array<bigint>"))))
        train_df = _label(train_bow, train_fr)
        test_df = _label(test_bow, test_fr)

        # 6) per-domain leakage strip over ALL N vocabularies (defensive).
        per_domain = [(f"features_{i}", vms[i]) for i in range(len(vms))]

        def _strip(df):
            for col, vm in per_domain:
                drop_idxs = {vm[c] for c in node_cids if c in vm}
                df = strip_test_features(df, drop_idxs, features_col=col)
            return df

        test_df = _strip(test_df)
        if strip_mode == "both":
            train_df = _strip(train_df)

        return MultiDomainBundle(
            train_df=train_df, test_df=test_df, parent_int=parent_int,
            int2cid=int2cid, cid2int=cid2int, vocab_maps=vms,
            name_by_id=dict(before_dag.names), ledger=ledger)
    finally:
        train_att.unpersist(); test_att.unpersist()
