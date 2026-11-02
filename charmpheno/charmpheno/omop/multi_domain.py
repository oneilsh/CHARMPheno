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
    # binary=True tokenizes this domain with per-document PRESENCE (each token
    # counts once per doc regardless of within-doc repeats). The measurement
    # domain sets this: it has no OMOP era rollup and is extremely bursty (~924
    # rows/person, insight 0077), so binary presence reproduces for it what the
    # era tables already do for condition/drug (near-binary per patient-window).
    binary: bool = False


def _empty_vec_udf(size: int):
    return F.udf(lambda: SparseVector(size, [], []), VectorUDT())


def multidomain_bow(domain_events, vocab_specs, *, doc_spec):
    """N aligned per-domain BOW columns joined per doc. Returns
    (df[doc_id, person_id, features_0 .. features_{N-1}], vocab_maps).

    domain_events[i] is bag-of-worded against vocab_specs[i]; domain 0 is
    conditions by convention. A doc present in only some domains gets an EMPTY
    vector (of the absent domain's vocab size) on each absent side -- never a
    dropped row -- so every doc carries all N columns and each per-domain vector
    size is CONSTANT across the corpus (the estimator reads each domain's width
    from its vectors and validates every row against it).
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
            min_patient_count=spec.min_patient_count, vocab=spec.vocab,
            binary=spec.binary)
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
    `labels` (the attested label-DAG node ids; condition-only) and `source_cohort`.
    `vocab_maps` is the list of per-domain {concept_id: vocab_idx} maps in domain
    order.

    The labels are CONDITION-ONLY (gate ⟂ domain): `dag` is the label DAG
    `{node: [parents]}` (the estimator's `setDag`), `name_by_id` is {node id: name},
    `ledger` the receipt (`case_finding_assembly.label_ledger`). Clinical domain names
    live in the DRIVER, not here -- this bundle is index-based and domain-agnostic."""
    train_df: object
    test_df: object
    dag: dict
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
    drug/observation event never labels a document. index_df carries
    source_cohort AND index_date, which the join propagates onto every
    feature/label frame (index_date is what an EpisodeDocSpec keys the document on
    -- exp 0111 R5.2; the passthrough lives in `lookback_feature_label_events`, so
    this wrapper carries it with no change of its own).
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


def assemble_multidomain_from_events(cond_events, extra_events, label_dag, *,
                                     names=None, doc_spec, vocab_specs,
                                     holdout_frac=0.2, split_salt=None,
                                     strip_mode="test_only",
                                     label_events=None,
                                     attested_provider=None) -> MultiDomainBundle:
    """Assemble the N-domain case-finding bundle from already-windowed events.

    A thin N-domain layer over the single-domain `assemble_from_events`
    orchestration: it reuses that module's split, attestation, label-join and strip
    helpers, swaps the single BOW for the N-column `multidomain_bow`, and applies
    the leakage strip PER DOMAIN over all N vocabularies.

    Domain 0 is conditions (`cond_events`); domains 1..N-1 are `extra_events` in
    order. `vocab_specs` has one spec per domain (len == 1 + len(extra_events)).
    The label side is CONDITION-ONLY: the attestation frame is `cond_events`, or
    `label_events` when given (lookback mode). A non-condition event never labels.

    Split-first, leakage-free: the SAME salted split (keyed on person_id) is
    applied to every domain frame AND the label frame, so a person's rows across
    all domains and labels land on the same side. Each domain's vocabulary is fit
    on TRAIN, frozen for TEST. No node is dropped here: `label_dag` is handed over
    whole and the estimator learns the node set from the training documents.
    Per-domain strip: the DAG's node ids are mapped through EACH domain's
    vocab_map and stripped from that column (a SNOMED node is a condition concept;
    Mondo curies are in no vocabulary, so on the Mondo path the strip removes
    nothing). `strip_mode="test_only"` (default) strips TEST only; `"both"` also
    strips TRAIN.

    `attested_provider` (optional): a callable ``events_df -> attested_df``
    (columns doc_id, person_id, source_cohort, labels) that replaces the default
    exact-node attestation (`doc_attested_nodes(ev, node_ids, ...)`). This is the
    DAG-source seam: pass `label_dag` = the native Mondo subgraph
    (`charmpheno.mondo.native_dag.build_mondo_native_fit_inputs`) and
    `attested_provider` = its code-ladder provider
    (`native_dag.make_mondo_native_attested_provider`) to label patients with Mondo
    terms; everything else (split/BOW/strip) is unchanged.
    """
    from charmpheno.omop.case_finding_assembly import (
        _SPLIT_SALT, split_train_test, doc_attested_nodes, strip_test_features,
        attach_label_column, label_ledger)

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

    node_ids = set(label_dag)

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

    # 2) condition-only attestation. Default is exact-node (patient has the code ==
    #    the node); `attested_provider` overrides it (the Mondo code ladder),
    #    keeping the same [doc_id, person_id, source_cohort, labels] shape.
    def _attest(ev):
        if attested_provider is not None:
            return attested_provider(ev)
        return doc_attested_nodes(ev, node_ids, doc_spec=doc_spec)
    train_att = _attest(train_lab).cache()
    test_att = _attest(test_lab).cache()
    try:
        ledger = label_ledger(train_att, test_att, len(node_ids))

        # 3) N-domain BOW: fit each domain's vocab on TRAIN, freeze for TEST.
        train_bow, vms = multidomain_bow(train_doms, vocab_specs, doc_spec=doc_spec)
        test_bow, _ = multidomain_bow(
            test_doms, [_frozen_vocab_spec(s, vm) for s, vm in zip(vocab_specs, vms)],
            doc_spec=doc_spec)
        train_df = attach_label_column(train_bow, train_att)
        test_df = attach_label_column(test_bow, test_att)

        # 4) per-domain leakage strip over ALL N vocabularies.
        per_domain = [(f"features_{i}", vms[i]) for i in range(len(vms))]

        def _strip(df):
            for col, vm in per_domain:
                drop_idxs = {vm[c] for c in node_ids if c in vm}
                df = strip_test_features(df, drop_idxs, features_col=col)
            return df

        test_df = _strip(test_df)
        if strip_mode == "both":
            train_df = _strip(train_df)

        return MultiDomainBundle(
            train_df=train_df, test_df=test_df,
            dag={n: list(ps) for n, ps in label_dag.items()}, vocab_maps=vms,
            name_by_id={n: v for n, v in (names or {}).items() if n in node_ids},
            ledger=ledger)
    finally:
        train_att.unpersist(); test_att.unpersist()


# Domain names loadable as their own OMOP fact table (domain 0 is always
# conditions; extra domains are read single-domain via load_omop_bigquery, whose
# fused/measurement paths both emit the common `event_date` column). Measurement
# carries value-aware synthetic tokens (concept_id*state) and is tokenized with
# per-document binary presence (its events are bursty, no era rollup — 0077/0079).
_EXTRA_DOMAIN_DATE = "event_date"   # bigquery._FUSED_EVENT_DATE for a single non-cond load
_SUPPORTED_EXTRA_DOMAINS = ("drug", "procedure", "measurement")


def assemble_multidomain_case_finding_corpus(
        spark, *, disease="rare6", cdr, billing, extra_domains=("drug",),
        person_mod, holdout_frac=0.2, split_salt=None,
        vocab_size, min_df, min_patient_count, doc_min_length=0,
        strip_mode="test_only", lookback_days=365, label_window_days=365,
        label_dag=None, names=None, attested_provider=None, index_mode="disease",
        index_df=None, doc_spec=None):
    """End-to-end BQ assembly of the multi-domain case-finding bundle (LOOKBACK
    mode). Domain 0 is conditions (the label source); ``extra_domains`` are read
    single-domain and windowed against the SAME condition-derived index.

    The multi-domain sibling of ``case_finding_assembly.assemble_case_finding_corpus``
    — same disease DAG, same PatientCohortDocSpec, same lookback windowing — but it
    fits ONE vocabulary per domain (MixEHR-style) and emits per-domain
    ``features_0..features_{N-1}`` columns. Only lookback mode is supported (the
    forward per-patient window is condition-defined and does not cleanly window a
    second domain); pass an already-lookback config. Requires a live CDR
    (cluster-covered, like the single-domain lookback body).

    Label DAG: the disease's SNOMED anchor forest by default
    (`case_finding_assembly.snomed_label_dag`), or a caller-built `label_dag` +
    `names` (the native Mondo subgraph) with its `attested_provider` (the code
    ladder).

    Two injection seams (exp 0111 R7.1), on the `attested_provider` precedent so
    that ALL episode logic stays driver-owned and no episode code lands in this
    source-hashed module:

    - `index_df` + `index_mode="external"`: a driver-supplied
      `(person_id, index_date, source_cohort, ...)` index frame used VERBATIM as
      the index table. The two self-building modes ("population"/"disease")
      REJECT a non-None `index_df` -- they own their index -- and "external"
      REQUIRES one.
    - `doc_spec`: the document unit. `None` defaults to
      `PatientCohortDocSpec(min_doc_length=doc_min_length)`.
    """
    from charmpheno.omop import load_omop_bigquery
    from charmpheno.omop.cohorts import (
        case_finding_index_table, case_finding_population_index_table)
    from charmpheno.omop.case_finding_assembly import (
        _LOOKBACK_PRIOR_OBS_DAYS, snomed_label_dag)
    from charmpheno.omop.doc_spec import PatientCohortDocSpec

    bad = [d for d in extra_domains if d not in _SUPPORTED_EXTRA_DOMAINS]
    if bad:
        raise ValueError(
            f"extra_domains {bad} not supported (this MVP: {_SUPPORTED_EXTRA_DOMAINS}; "
            "measurement's value-aware loader is deferred — see the multi-domain PC plan)")

    # Domain 0: conditions (legacy single-domain load → condition_era_start_date).
    cond_date = "condition_era_start_date"
    cond = load_omop_bigquery(
        spark=spark, cdr_dataset=cdr, billing_project=billing,
        person_sample_mod=person_mod, source_table="condition_era")
    domain_raws, date_cols = [cond], [cond_date]
    # Extra domains: single-domain fused load → the common `event_date` column.
    for dom in extra_domains:
        d = load_omop_bigquery(
            spark=spark, cdr_dataset=cdr, billing_project=billing,
            person_sample_mod=person_mod, concept_types=(dom,))
        domain_raws.append(d)
        date_cols.append(_EXTRA_DOMAIN_DATE)

    # ONE shared index; every domain windowed against it. The lookback prior-obs
    # gate is the intrinsic ≥1yr floor (not the forward knob). "disease" anchors
    # the index on first-dx foreground + general background; "population" indexes
    # EVERY patient on a random event-anchored window (whole-Mondo / template-branch
    # fit, where the Mondo code ladder — not one disease — defines the labels);
    # "external" takes a driver-built index frame verbatim (the episode seam).
    if index_mode == "external":
        if index_df is None:
            raise ValueError(
                "index_mode='external' requires an index_df (the driver-supplied "
                "(person_id, index_date, source_cohort) index frame); none was "
                "passed.")
    elif index_mode == "population":
        if index_df is not None:
            raise ValueError(
                "index_mode='population' builds its own random-anchored index; it "
                "does not accept an index_df. Use index_mode='external' to supply "
                "one.")
        index_df = case_finding_population_index_table(
            cond, spark=spark, cdr_dataset=cdr, billing_project=billing,
            date_col=cond_date, prior_obs_days=_LOOKBACK_PRIOR_OBS_DAYS,
            label_window_days=label_window_days)
    elif index_mode == "disease":
        if index_df is not None:
            raise ValueError(
                "index_mode='disease' builds its own first-dx index; it does not "
                "accept an index_df. Use index_mode='external' to supply one.")
        index_df = case_finding_index_table(
            cond, disease=disease, spark=spark, cdr_dataset=cdr,
            billing_project=billing, date_col=cond_date,
            prior_obs_days=_LOOKBACK_PRIOR_OBS_DAYS, label_window_days=label_window_days)
    else:
        raise ValueError(
            f"index_mode must be 'disease', 'population' or 'external', "
            f"got {index_mode!r}")
    feature_frames, cond_label = lookback_feature_frames(
        domain_raws, index_df, date_cols,
        lookback_days=lookback_days, label_window_days=label_window_days)

    if label_dag is None:
        label_dag, names = snomed_label_dag(spark, disease=disease, cdr=cdr,
                                            billing=billing)
    if doc_spec is None:
        doc_spec = PatientCohortDocSpec(min_doc_length=doc_min_length)

    # One vocab spec per domain (same fit knobs; a real per-domain sweep is future).
    # Measurement is tokenized with per-document BINARY presence — it has no OMOP
    # era rollup and is extremely bursty, so presence reproduces for it what the
    # era tables already do for condition/drug (measurement_tokens docstring).
    domain_binary = [False] + [d == "measurement" for d in extra_domains]
    vocab_specs = [
        DomainVocabSpec(vocab_size=vocab_size, min_df=min_df,
                        min_patient_count=min_patient_count, binary=b)
        for b in domain_binary]
    return assemble_multidomain_from_events(
        feature_frames[0], feature_frames[1:], label_dag, names=names,
        doc_spec=doc_spec, vocab_specs=vocab_specs, holdout_frac=holdout_frac,
        split_salt=split_salt, strip_mode=strip_mode,
        label_events=cond_label, attested_provider=attested_provider)
