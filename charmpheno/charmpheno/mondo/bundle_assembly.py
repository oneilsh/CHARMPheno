"""What the native-Mondo bundle adds after the generic assembler: the all-ever gate
labels (`gate_labels`, label redesign R2 / spec 2026-10-30 R8) and the code ladder
carried as the bundle's `native_code_map`.

SOURCE-HASHED (AGENTS.md "cache-key landmine"): these functions run on a bundle cache
MISS and decide what the cached bundle holds, so this module's source is folded into
the Mondo bundle key (`_case_finding_cache.compute_bundle_cache_key`,
`mondo_bundle_src`). They lived in the fit driver, whose source no key folds, so an
edit there would have served a stale bundle under an unchanged key (review 2026-11-13
D4; moved 2026-11-24). Editing this module is an announced cache drop that re-pins
the tripwire hashes in the same commit.

Driver-side only (DataFrame operations, no UDFs), so it needs no `--py-files` entry
beyond the charmpheno archive every submit already ships.
"""
from __future__ import annotations


def native_code_map_of(code_map_sdf):
    """The `(std_cid, Mondo term)` code ladder as the sorted list of pairs the bundle
    carries (`bundle.native_code_map`; exp 0136 WP-C'), or None for a frame without
    `collect` (a test stub)."""
    collect = getattr(code_map_sdf, "collect", None)
    if collect is None:
        return None
    return sorted((int(r["std_cid"]), str(r["node_id"])) for r in collect())


def gate_ever_attested(docs, events, provider):
    """Per TRAIN document, the label nodes its person attests ANYWHERE in the record.

    `docs` = the bundle frame (`person_id`, `doc_id`); `events` = the full-history
    condition frame (`person_id`, `concept_id`, ...). Each event inherits its
    person's doc(s) — `source_cohort` is the first `:` component of the population
    doc_id (`PatientCohortDocSpec`) — and goes through the SAME `provider` the labels
    used (same code ladder), so the two columns differ only in WHEN, never in HOW a
    code becomes a node. Returns `(person_id, doc_id, gate_labels)`; a doc whose
    person attests nothing keeps a present, empty array."""
    from pyspark.sql import functions as F
    from charmpheno.omop.case_finding_assembly import GATE_LABELS_COL, LABELS_COL
    roster = docs.select("person_id", "doc_id").distinct().withColumn(
        "source_cohort", F.element_at(F.split(F.col("doc_id"), ":"), 1))
    ev = events.select("person_id", "concept_id").join(
        roster.select("person_id", "source_cohort").distinct(), "person_id")
    att = provider(ev)
    dt = att.schema[LABELS_COL].dataType.simpleString()
    att = att.select("doc_id", F.col(LABELS_COL).alias(GATE_LABELS_COL))
    return (roster.select("person_id", "doc_id").join(att, "doc_id", "left")
            .withColumn(GATE_LABELS_COL, F.coalesce(
                F.col(GATE_LABELS_COL), F.array().cast(dt))))


def gate_count_receipt(df, label_col, gate_col):
    """Pooled R2.3 receipt: attested (doc, node) cells and attested docs, label year
    vs all-ever, and the median per-node ratio of attesting train docs. Counts only."""
    from pyspark.sql import functions as F

    def per_node(col):
        return {r["n"]: int(r["c"]) for r in
                df.select(F.explode(col).alias("n")).groupBy("n")
                  .agg(F.count("*").alias("c")).collect()}
    b, a = per_node(label_col), per_node(gate_col)
    ratios = sorted(a.get(n, 0) / c for n, c in b.items() if c > 0)
    return {"cells_label": sum(b.values()), "cells_ever": sum(a.values()),
            "docs_gated_label": df.where(F.size(label_col) > 0).count(),
            "docs_gated_ever": df.where(F.size(gate_col) > 0).count(),
            "median_node_ratio": (ratios[len(ratios) // 2] if ratios else float("nan")),
            "n_nodes": len(b)}


def attach_gate_ever(spark, bundle, *, provider, cdr, billing, person_mod,
                      _events=None):
    """Label redesign R2 (spec 2026-10-13; R8 of spec 2026-10-30): add `gate_labels`
    to each TRAIN document — its person's ALL-EVER attestations through the same
    code ladder — beside the label-year `labels`, which stay untouched (they are
    the readout's); the test split is untouched (the test transform is ungated).
    The fit's gate reads `gate_labels` (`labelsCol`). MISS-only, inside the
    assemble closure, where the provider exists. One full-history `condition_era`
    read (the sidecar's load; ~70 s on 0137's corpus).

    Why the future may gate a TRAIN document: the gate shapes topics on the train
    split only; test documents are folded in ungated, so no test outcome reaches
    them (spec 2026-10-13, Why)."""
    from charmpheno.omop.case_finding_assembly import GATE_LABELS_COL, LABELS_COL
    events = _events
    if events is None:
        from charmpheno.omop import load_omop_bigquery
        events = load_omop_bigquery(spark=spark, cdr_dataset=cdr,
                                    billing_project=billing,
                                    person_sample_mod=person_mod,
                                    source_table="condition_era")
    before = bundle.train_df
    gate = gate_ever_attested(before, events, provider)
    after = attach_gate_column(before, gate, GATE_LABELS_COL).cache()
    rec = gate_count_receipt(after, LABELS_COL, GATE_LABELS_COL)
    print(f"[gate-ever] train gate, label year -> ever: attested (doc, node) cells "
          f"{rec['cells_label']:,} -> {rec['cells_ever']:,}; docs with any label "
          f"{rec['docs_gated_label']:,} -> {rec['docs_gated_ever']:,}; median "
          f"per-node ratio of attesting docs {rec['median_node_ratio']:.2f} over "
          f"{rec['n_nodes']} nodes", flush=True)
    bundle.train_df = after
    bundle.gate_ever_receipt = rec
    return bundle


def attach_gate_column(df, gate, col):
    """`df` with `col` joined from `gate` on the doc key, refusing a missing /
    duplicate / mismatched join LOUDLY (WP-D3 join integrity).

    `gate` carries `(person_id, doc_id, col)`, one row per document. The guards, in
    order: (1) the gate frame must be UNIQUE on (person_id, doc_id) — an ambiguous
    gate is a raise, never a silent pick; (2) a LEFT join must not change the
    document count; (3) NO document may come back without a gate — a missing join
    surfaces as a NULL (distinct from an EMPTY gate's present `[]`) and is a hard
    error, never an empty-gate fallback. Every other column rides through."""
    from pyspark.sql import functions as F

    tmp = f"_gate_{col}"
    g = gate.select("person_id", "doc_id", F.col(col).alias(tmp)).cache()
    try:
        n_gate = g.count()
        n_keys = g.select("person_id", "doc_id").distinct().count()
        if n_gate != n_keys:
            raise ValueError(
                f"gate join: {n_gate - n_keys} duplicate (person_id, doc_id) rows "
                f"in the {col!r} frame — refusing an ambiguous gate (one per "
                "document is required).")
        n_before = df.count()
        base = df.drop(col) if col in df.columns else df
        joined = base.join(g, on=["person_id", "doc_id"], how="left")
        n_after = joined.count()
        if n_after != n_before:
            raise ValueError(
                f"gate join: the {col!r} join changed the document count "
                f"({n_before} -> {n_after}) — a duplicate gate key inflated the "
                "bundle; refusing.")
        n_missing = joined.where(F.col(tmp).isNull()).count()
        if n_missing:
            raise ValueError(
                f"gate join: {n_missing} bundle document(s) received no {col!r} — "
                "a missing join is a hard error, never an empty-gate fallback.")
        return joined.withColumn(col, F.col(tmp)).drop(tmp)
    finally:
        g.unpersist()
